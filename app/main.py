"""
MeetIQ v4.0 — Autonomous Meeting Intelligence System
- Autonomous Google Meet Bot (Playwright)
- Multilingual Gemini AI Engine (English, Portuguese, Spanish, French, etc.)
- Automated Executive PDF Generation with Verbatim Spoken Dialogue
- Silent PDF Delivery via Gmail SMTP (Zero terminal clutter)
- Interactive Meeting Archive & History Manager
"""

import os
import sys
import json
import uuid
import asyncio
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List

# Ensure parent directory is in sys.path so core imports resolve cleanly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from core.config import settings
from core.gemini import analyze_transcript_with_gemini
from core.pdf_generator import generate_meeting_pdf
from core.mailer import send_pdf_report_email
from core.audio import extract_audio, transcribe_audio_file
from core.bot import GoogleMeetBot

# ── LOGGING SETUP ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("meetiq.server")

# ── APP INITIALIZATION ────────────────────────────────────────────────────────
app = FastAPI(title="MeetIQ Intelligence API", version="4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True
)

active_bots = {} # session_id -> GoogleMeetBot
upload_jobs = {} # job_id -> dict

class BotJoinRequest(BaseModel):
    meeting_url: str
    emails: Optional[str] = ""
    bot_name: Optional[str] = None
    target_language: Optional[str] = "English"

class ResendEmailRequest(BaseModel):
    filename: str
    emails: str

# ── STATIC & FAVICON ROUTE ────────────────────────────────────────────────────
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    # Return 204 No Content to silence browser 404 warning
    return Response(status_code=204)

@app.get("/")
def serve_frontend():
    # Check app/index.html first, then root index.html
    index_file = Path(__file__).parent / "index.html"
    if not index_file.exists():
        index_file = PROJECT_ROOT / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "MeetIQ v4.0 API is active"}

# ── AUTONOMOUS BOT ROUTES ─────────────────────────────────────────────────────
@app.post("/api/bot/join")
async def bot_join(req: BotJoinRequest, background_tasks: BackgroundTasks):
    url = req.meeting_url.strip()
    if not ("meet.google.com" in url or "teams.microsoft.com" in url or "zoom.us" in url):
        raise HTTPException(400, "Please provide a valid meeting link (e.g., https://meet.google.com/xxx-yyyy-zzz)")

    notify = [e.strip() for e in req.emails.split(",") if e.strip()]
    bot = GoogleMeetBot(
        meeting_url=url,
        notify_emails=notify,
        bot_name=req.bot_name or settings.bot_name,
        target_language=req.target_language or "English"
    )
    
    active_bots[bot.session_id] = bot
    asyncio.create_task(bot.start())
    
    logger.info(f"Meeting bot initiated for URL: {url} (Session: {bot.session_id}, Lang: {bot.target_language})")
    return {
        "status": "launching",
        "session_id": bot.session_id,
        "message": f"MeetIQ Bot is launching and navigating to {url}"
    }

@app.post("/api/bot/leave/{session_id}")
async def bot_leave(session_id: str):
    if session_id not in active_bots:
        raise HTTPException(404, "Session not found")
    bot = active_bots[session_id]
    await bot.stop()
    return {"status": "stopping", "message": "Bot is concluding meeting and generating PDF report"}

@app.get("/api/bot/status/{session_id}")
def bot_status(session_id: str):
    if session_id not in active_bots:
        raise HTTPException(404, "Session not found")
    bot = active_bots[session_id]
    return {
        "session_id": session_id,
        "status": bot.status,
        "captions_captured": len(bot.transcript_entries),
        "recent_captions": bot.transcript_entries[-10:], # Live streaming captions to UI!
        "error": bot.error_message,
        "pdf_available": bool(bot.pdf_path and Path(bot.pdf_path).exists())
    }

@app.get("/api/bot/result/{session_id}")
def bot_result(session_id: str):
    if session_id not in active_bots:
        raise HTTPException(404, "Session not found")
    bot = active_bots[session_id]
    if bot.status != "completed":
        raise HTTPException(400, f"Session still in state: {bot.status}")
    
    pdf_filename = Path(bot.pdf_path).name if bot.pdf_path else None
    return {
        "session_id": session_id,
        "meeting_url": bot.meeting_url,
        "analysis": bot.analysis,
        "transcript_count": len(bot.transcript_entries),
        "pdf_filename": pdf_filename,
        "email_result": bot.email_result
    }

# ── FILE UPLOAD PIPELINE ──────────────────────────────────────────────────────
def process_uploaded_file(job_id: str, media_path: str, notify_emails: list, target_language: str = "English"):
    wav_path = str(settings.upload_dir / f"{job_id}.wav")
    try:
        upload_jobs[job_id].update({"status": "extracting_audio", "progress": 15})
        extract_audio(media_path, wav_path)

        upload_jobs[job_id].update({"status": "transcribing", "progress": 40})
        plain_tx, speaker_tx, labeled, lang = transcribe_audio_file(wav_path)
        upload_jobs[job_id].update({
            "language": lang,
            "transcript": plain_tx,
            "speaker_transcript": speaker_tx,
            "progress": 70
        })

        upload_jobs[job_id].update({"status": "analyzing_with_gemini", "progress": 80})
        analysis = analyze_transcript_with_gemini(speaker_tx or plain_tx, target_language=target_language)

        upload_jobs[job_id].update({"status": "generating_pdf", "progress": 90})
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        pdf_path = str(settings.reports_dir / f"MeetIQ_Report_{timestamp_str}.pdf")
        generate_meeting_pdf(analysis, pdf_path, transcript=speaker_tx or plain_tx)

        # Save metadata json for Archive browser and Interactive Viewer
        meta_path = settings.reports_dir / f"MeetIQ_Report_{timestamp_str}.json"
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "filename": Path(pdf_path).name,
                "topic": analysis.get("meeting_overview", {}).get("main_topic") or "Meeting Intelligence Summary",
                "summary": analysis.get("summary", ""),
                "action_count": len(analysis.get("action_items", [])),
                "created": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "language": target_language,
                "analysis": analysis,
                "transcript": speaker_tx or plain_tx
            }, f, indent=2)

        upload_jobs[job_id].update({"status": "sending_email", "progress": 95})
        email_result = send_pdf_report_email(notify_emails, pdf_path, analysis)

        upload_jobs[job_id].update({
            "status": "done",
            "progress": 100,
            "analysis": analysis,
            "pdf_filename": Path(pdf_path).name,
            "email_result": email_result,
            "completed": datetime.now().isoformat()
        })
        logger.info(f"Job {job_id} completed successfully. PDF report generated and dispatched.")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        upload_jobs[job_id].update({"status": "error", "error": str(e)})
    finally:
        for p in [media_path, wav_path]:
            try:
                if p and Path(p).exists():
                    os.remove(p)
            except Exception:
                pass

@app.post("/api/upload")
async def upload_meeting_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    emails: str = Form(""),
    target_language: str = Form("English")
):
    allowed_exts = {".mp4", ".mp3", ".wav", ".m4a", ".mkv", ".webm", ".mov", ".ogg"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_exts:
        raise HTTPException(400, f"Unsupported media format: {ext}")

    job_id = str(uuid.uuid4())[:8]
    saved_media_path = str(settings.upload_dir / f"{job_id}{ext}")
    
    with open(saved_media_path, "wb") as f:
        f.write(await file.read())

    notify = [e.strip() for e in emails.split(",") if e.strip()]
    upload_jobs[job_id] = {
        "job_id": job_id,
        "filename": file.filename,
        "status": "queued",
        "progress": 5,
        "created": datetime.now().isoformat(),
        "notify_emails": notify,
        "target_language": target_language
    }

    background_tasks.add_task(process_uploaded_file, job_id, saved_media_path, notify, target_language)
    logger.info(f"File upload job queued: {job_id} ({file.filename}, Target: {target_language})")
    return {"job_id": job_id, "message": "Processing started in background"}

@app.get("/api/status/{job_id}")
def job_status(job_id: str):
    if job_id not in upload_jobs:
        raise HTTPException(404, "Job not found")
    j = upload_jobs[job_id]
    return {
        "job_id": job_id,
        "status": j.get("status"),
        "progress": j.get("progress", 0),
        "error": j.get("error")
    }

@app.get("/api/result/{job_id}")
def job_result(job_id: str):
    if job_id not in upload_jobs:
        raise HTTPException(404, "Job not found")
    j = upload_jobs[job_id]
    if j.get("status") != "done":
        raise HTTPException(400, f"Job not finished: {j.get('status')}")
    return j

# ── REPORTS ARCHIVE & DOWNLOAD ────────────────────────────────────────────────
@app.get("/api/reports")
def list_reports():
    """List all generated PDF reports from the outputs/ directory with metadata"""
    reports = []
    pdf_files = sorted(settings.reports_dir.glob("*.pdf"), key=os.path.getmtime, reverse=True)
    
    for pdf in pdf_files:
        meta_file = pdf.with_suffix(".json")
        topic = pdf.stem.replace("_", " ")
        summary = ""
        action_count = 0
        created_str = datetime.fromtimestamp(pdf.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        lang = "English"

        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    m = json.load(f)
                    topic = m.get("topic", topic)
                    summary = m.get("summary", "")
                    action_count = m.get("action_count", 0)
                    created_str = m.get("created", created_str)
                    lang = m.get("language", lang)
            except Exception:
                pass

        reports.append({
            "filename": pdf.name,
            "topic": topic,
            "summary": summary,
            "action_count": action_count,
            "created": created_str,
            "size_kb": round(pdf.stat().st_size / 1024, 1),
            "language": lang,
            "download_url": f"/api/download-pdf/{pdf.name}"
        })
    return {"reports": reports}

@app.post("/api/reports/resend")
def resend_report(req: ResendEmailRequest):
    safe_name = Path(req.filename).name
    pdf_path = settings.reports_dir / safe_name
    if not pdf_path.exists():
        raise HTTPException(404, "Report not found")
    
    notify = [e.strip() for e in req.emails.split(",") if e.strip()]
    if not notify:
        raise HTTPException(400, "Please provide at least one recipient email")

    mock_analysis = {"meeting_overview": {"main_topic": safe_name.replace(".pdf", "")}, "summary": "Meeting report resent upon request."}
    res = send_pdf_report_email(notify, str(pdf_path), mock_analysis)
    return res

@app.get("/api/download-pdf/{filename}")
def download_pdf(filename: str):
    safe_name = Path(filename).name
    pdf_path = settings.reports_dir / safe_name
    if not pdf_path.exists():
        raise HTTPException(404, "Requested PDF report does not exist")
    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=safe_name
    )

@app.get("/api/reports/details/{filename}")
def report_details(filename: str):
    safe_name = Path(filename).name
    meta_name = Path(safe_name).with_suffix(".json")
    meta_path = settings.reports_dir / meta_name
    pdf_path = settings.reports_dir / safe_name
    
    if not pdf_path.exists():
        raise HTTPException(404, "Report not found")
        
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to read metadata for {filename}: {e}")
            
    return {
        "filename": safe_name,
        "topic": safe_name.replace(".pdf", "").replace("_", " "),
        "summary": "Executive PDF summary report is available for direct download.",
        "action_count": 0,
        "created": datetime.fromtimestamp(pdf_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
        "language": "English",
        "analysis": None,
        "transcript": None
    }

@app.get("/api/system/status")
def system_status():
    pdf_count = len(list(settings.reports_dir.glob("*.pdf")))
    return {
        "gemini_model": settings.gemini_model,
        "gemini_active": bool(settings.gemini_api_key),
        "whisper_model": settings.whisper_model,
        "email_sender": settings.email_sender or "loaloki2005@gmail.com",
        "reports_count": pdf_count,
        "bot_name": settings.bot_name,
        "server_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

# ── RUN SERVER ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting MeetIQ v4.0 Server...")
    logger.info(f"Gemini API Key: {'Configured' if settings.gemini_api_key else 'Missing'}")
    logger.info(f"Email Dispatch: {settings.email_sender or 'Not configured'}")
    logger.info(f"Dashboard available at: http://localhost:{settings.port}")
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)
