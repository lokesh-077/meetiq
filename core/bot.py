import asyncio
import logging
import uuid
from datetime import datetime
from pathlib import Path
from core.config import settings
from core.gemini import analyze_transcript_with_gemini
from core.pdf_generator import generate_meeting_pdf
from core.mailer import send_pdf_report_email

logger = logging.getLogger("meetiq.bot")

class GoogleMeetBot:
    def __init__(self, meeting_url: str, notify_emails: list = None, bot_name: str = None, target_language: str = "English"):
        self.meeting_url = meeting_url.strip()
        self.notify_emails = notify_emails or []
        self.bot_name = bot_name or settings.bot_name
        self.target_language = target_language or "English"
        self.session_id = str(uuid.uuid4())[:8]
        
        self.status = "idle" # idle | launching | joining | admitted | recording | processing | completed | error
        self.error_message = None
        self.transcript_entries = [] # [{"speaker": str, "text": str, "time": str}]
        self.analysis = {}
        self.pdf_path = None
        self.email_result = {}
        
        self._stop_event = asyncio.Event()
        self._browser = None
        self._context = None
        self._page = None

    async def start(self):
        """Asynchronously launch the Playwright Chromium bot and join Google Meet"""
        self.status = "launching"
        logger.info(f"[{self.session_id}] Launching MeetIQ Autonomous Bot for: {self.meeting_url}")
        
        try:
            from playwright.async_api import async_playwright
            playwright = await async_playwright().start()
            
            # Launch Chromium with media stream bypass (using installed Chrome)
            self._browser = await playwright.chromium.launch(
                channel="chrome",
                headless=True,
                args=[
                    "--use-fake-ui-for-media-stream",
                    "--use-fake-device-for-media-stream",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-blink-features=AutomationControlled"
                ]
            )
            
            self._context = await self._browser.new_context(
                permissions=["microphone", "camera"],
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            )
            
            self._page = await self._context.new_page()
            self.status = "joining"
            
            logger.info(f"[{self.session_id}] Navigating to Google Meet URL...")
            await self._page.goto(self.meeting_url, wait_until="networkidle", timeout=60000)
            await self._page.wait_for_timeout(3000)

            # Turn off microphone and camera before entering
            try:
                await self._page.keyboard.press("Control+e") # Camera
                await self._page.wait_for_timeout(500)
                await self._page.keyboard.press("Control+d") # Mic
                await self._page.wait_for_timeout(500)
            except Exception:
                pass

            # Fill bot name if there is an input field (guest join)
            name_inputs = await self._page.query_selector_all('input[type="text"]')
            for inp in name_inputs:
                if await inp.is_visible():
                    await inp.fill(self.bot_name)
                    await self._page.wait_for_timeout(500)
                    break

            # Click "Ask to join" or "Join now"
            join_btn = None
            selectors = [
                'button:has-text("Ask to join")',
                'button:has-text("Join now")',
                'button:has-text("Ask to Join")',
                'span:has-text("Ask to join")',
                'span:has-text("Join now")'
            ]
            for sel in selectors:
                btn = await self._page.query_selector(sel)
                if btn and await btn.is_visible():
                    join_btn = btn
                    break

            if join_btn:
                await join_btn.click()
                logger.info(f"[{self.session_id}] Clicked join request. Waiting for host admission...")
            else:
                # Try pressing Enter
                await self._page.keyboard.press("Enter")

            # Wait for admission (detect Leave call button or main meet interface)
            self.status = "waiting_admission"
            in_call = False
            for _ in range(120): # Wait up to 4 minutes for admission
                if self._stop_event.is_set():
                    break
                # Check for leave call button
                leave_btn = await self._page.query_selector('button[aria-label*="Leave call"], button[aria-label*="leave call"]')
                if leave_btn and await leave_btn.is_visible():
                    in_call = True
                    break
                await asyncio.sleep(2)

            if not in_call and not self._stop_event.is_set():
                raise TimeoutError("Host did not admit the bot within the timeout period.")

            self.status = "recording"
            logger.info(f"[{self.session_id}] Bot admitted to meeting! Activating live captions...")
            
            # Press 'c' to enable closed captions
            await self._page.keyboard.press("c")
            await asyncio.sleep(1)

            # Continuous caption capture loop
            last_text = ""
            while not self._stop_event.is_set():
                try:
                    # Check if call was ended
                    leave_btn = await self._page.query_selector('button[aria-label*="Leave call"]')
                    if not leave_btn:
                        logger.info(f"[{self.session_id}] Meeting ended by host or bot was disconnected.")
                        break

                    # Extract caption blocks from Google Meet DOM
                    caption_blocks = await self._page.evaluate("""() => {
                        const blocks = [];
                        // Google Meet caption containers
                        const containers = document.querySelectorAll('.nMx22c, .VbkSUe, .iOzk7, div[jsname="YSxPC"]');
                        containers.forEach(el => {
                            const nameEl = el.querySelector('.zs7F8d, .NWHXxf, span') || el.previousElementSibling;
                            const speaker = nameEl ? nameEl.innerText.trim() : 'Participant';
                            const text = el.innerText.trim();
                            if (text) {
                                blocks.push({ speaker, text });
                            }
                        });
                        return blocks;
                    }""")

                    if caption_blocks:
                        for b in caption_blocks:
                            combined = f"{b['speaker']}: {b['text']}"
                            if combined != last_text and b['text']:
                                self.transcript_entries.append({
                                    "speaker": b['speaker'] or "Speaker",
                                    "text": b['text'],
                                    "time": datetime.now().strftime("%H:%M:%S")
                                })
                                last_text = combined

                except Exception as e:
                    logger.debug(f"Caption polling error: {e}")

                await asyncio.sleep(2)

        except Exception as e:
            self.error_message = str(e)
            self.status = "error"
            logger.error(f"[{self.session_id}] Bot error: {e}")
        finally:
            await self._finalize_meeting()

    async def stop(self):
        """Signal bot to gracefully leave meeting and process report"""
        logger.info(f"[{self.session_id}] Received stop signal. Concluding meeting...")
        self._stop_event.set()

    async def _finalize_meeting(self):
        """Cleanup browser, synthesize meeting with Gemini, compile PDF, and email results"""
        self.status = "processing"
        
        # Close browser
        try:
            if self._context:
                await self._context.close()
            if self._browser:
                await self._browser.close()
        except Exception:
            pass

        # Build full transcript
        if not self.transcript_entries:
            logger.warning(f"[{self.session_id}] No speech captured during this session.")
            full_tx = "No speech or conversation was detected in this meeting session."
        else:
            full_tx = "\n".join([f"{entry['speaker']}: {entry['text']}" for entry in self.transcript_entries])

        logger.info(f"[{self.session_id}] Synthesizing transcript with Gemini 2.0 Flash...")
        try:
            self.analysis = analyze_transcript_with_gemini(full_tx, target_language=self.target_language)
        except Exception as e:
            logger.error(f"[{self.session_id}] Gemini analysis failed: {e}")
            self.analysis = {
                "meeting_overview": {"main_topic": "Meeting Intelligence Summary", "estimated_duration_minutes": 15},
                "summary": "Meeting concluded with transcript recorded.",
                "action_items": [],
                "decisions_made": []
            }

        # Generate PDF
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_filename = f"MeetIQ_Report_{timestamp_str}.pdf"
        self.pdf_path = str(settings.reports_dir / pdf_filename)
        
        logger.info(f"[{self.session_id}] Generating executive PDF report -> {self.pdf_path}")
        generate_meeting_pdf(self.analysis, self.pdf_path, transcript=full_tx)

        # Save metadata JSON for Archive browser and Interactive Viewer
        meta_path = settings.reports_dir / f"MeetIQ_Report_{timestamp_str}.json"
        try:
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({
                    "filename": pdf_filename,
                    "topic": self.analysis.get("meeting_overview", {}).get("main_topic") or "Google Meet Intelligence Summary",
                    "summary": self.analysis.get("summary", ""),
                    "action_count": len(self.analysis.get("action_items", [])),
                    "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    "language": self.target_language,
                    "analysis": self.analysis,
                    "transcript": full_tx
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to write metadata JSON: {e}")

        # Send Email
        logger.info(f"[{self.session_id}] Dispatching PDF report to: {self.notify_emails or settings.email_to}")
        self.email_result = send_pdf_report_email(self.notify_emails, self.pdf_path, self.analysis)

        self.status = "completed"
        logger.info(f"[{self.session_id}] All tasks complete! Report delivered silently via email.")
