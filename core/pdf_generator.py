from fpdf import FPDF
from datetime import datetime
from pathlib import Path
import re

def clean_text(text: str) -> str:
    """Ensure text is Latin-1 compatible for standard PDF fonts, replacing unsupported chars"""
    if not text:
        return ""
    # Map common unicode chars
    replacements = {
        "–": "-", "—": "-", "“": '"', "”": '"', "‘": "'", "’": "'",
        "…": "...", "•": "*", "→": "->", "✅": "[YES]", "❌": "[NO]",
        "⚡": "[ACTION]", "📋": "[SUMMARY]", "👥": "[TEAM]", "⚠️": "[RISK]"
    }
    for orig, repl in replacements.items():
        text = text.replace(orig, repl)
    # Remove any remaining characters outside latin-1 to avoid fpdf encoding crash
    return text.encode('latin-1', 'replace').decode('latin-1')

class MeetIQPDF(FPDF):
    def __init__(self, topic: str):
        super().__init__()
        self.topic = clean_text(topic)

    def header(self):
        # Header banner
        self.set_fill_color(15, 23, 42) # Slate-900
        self.rect(0, 0, 210, 22, 'F')
        
        self.set_xy(10, 4)
        self.set_font("Helvetica", "B", 13)
        self.set_text_color(56, 189, 248) # Cyan-400
        self.cell(0, 6, "MEETIQ  |  INTELLIGENCE REPORT", ln=True)
        
        self.set_xy(10, 11)
        self.set_font("Helvetica", "", 8)
        self.set_text_color(148, 163, 184) # Slate-400
        self.cell(0, 6, f"{self.topic}  --  Generated on {datetime.now().strftime('%B %d, %Y')}", ln=True)
        self.ln(12)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184)
        self.cell(0, 8, f"Page {self.page_no()}/{{nb}}  |  Strictly Confidential  |  Powered by MeetIQ AI", align="C")

def generate_meeting_pdf(analysis: dict, output_path: str, transcript: str = "") -> str:
    """Generate professional PDF executive summary report including full spoken dialogue"""
    overview = analysis.get("meeting_overview", {})
    topic = overview.get("main_topic") or "Executive Meeting Summary"
    
    pdf = MeetIQPDF(topic=topic)
    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # ── 1. TITLE & METADATA ───────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(15, 23, 42)
    pdf.multi_cell(0, 8, clean_text(topic))
    pdf.ln(2)
    
    # Metadata Badge Row
    duration = overview.get("estimated_duration_minutes") or "N/A"
    mtype = (overview.get("meeting_type") or "Discussion").upper()
    speakers = len(overview.get("speakers_identified", [])) or "Multi"
    
    pdf.set_fill_color(241, 245, 249) # Slate-100
    pdf.set_draw_color(226, 232, 240) # Slate-200
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(71, 85, 105) # Slate-600
    
    meta_text = clean_text(f" DATE: {datetime.now().strftime('%Y-%m-%d')}   |   TYPE: {mtype}   |   DURATION: ~{duration} MIN   |   PARTICIPANTS: {speakers} ")
    pdf.cell(0, 8, meta_text, border=1, fill=True, ln=True, align="L")
    pdf.ln(5)

    # ── 2. EXECUTIVE SUMMARY ──────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 7, "1. Executive Summary", ln=True)
    
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(51, 65, 85) # Slate-700
    summary = clean_text(analysis.get("summary", "No summary recorded."))
    pdf.set_fill_color(248, 250, 252)
    pdf.multi_cell(0, 6, summary, border=1, fill=True)
    pdf.ln(5)

    # ── 3. ACTION ITEMS TABLE ─────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 7, "2. Key Action Items & Ownership", ln=True)
    
    actions = analysis.get("action_items", [])
    if actions:
        # Table Header
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_fill_color(30, 41, 59) # Slate-800
        pdf.set_text_color(255, 255, 255)
        pdf.cell(85, 7, " ACTION TASK", border=1, fill=True)
        pdf.cell(40, 7, " OWNER", border=1, fill=True)
        pdf.cell(30, 7, " PRIORITY", border=1, align="C", fill=True)
        pdf.cell(35, 7, " DEADLINE", border=1, align="C", fill=True)
        pdf.ln(7)

        # Table Rows
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(30, 41, 59)
        for i, act in enumerate(actions):
            fill = (i % 2 == 1)
            pdf.set_fill_color(248, 250, 252) if fill else pdf.set_fill_color(255, 255, 255)
            
            task = clean_text(act.get("task", ""))[:55]
            owner = clean_text(act.get("owner", "Unassigned"))[:22]
            prio = clean_text((act.get("priority") or "medium")).upper()
            deadline = clean_text(str(act.get("deadline") or "TBD"))[:18]
            
            pdf.cell(85, 6, f" {task}", border=1, fill=fill)
            pdf.cell(40, 6, f" {owner}", border=1, fill=fill)
            pdf.cell(30, 6, prio, border=1, align="C", fill=fill)
            pdf.cell(35, 6, deadline, border=1, align="C", fill=fill)
            pdf.ln(6)
    else:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 6, "No explicit action items detected.", ln=True)
    pdf.ln(5)

    # ── 4. DECISIONS MADE ─────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 7, "3. Decisions Reached", ln=True)
    
    decisions = analysis.get("decisions_made", [])
    if decisions:
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(51, 65, 85)
        for d in decisions:
            dec = clean_text(d.get("decision", ""))
            owner = clean_text(d.get("owner", "Team"))
            pdf.cell(0, 5, f"[x] {dec}  (Owner: {owner})", ln=True)
    else:
        pdf.set_font("Helvetica", "I", 9)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 6, "No formal decisions recorded.", ln=True)
    pdf.ln(5)

    # ── 5. DISCUSSION TOPICS & KEY POINTS ─────────────────────────────────────
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 7, "4. Key Discussion Points", ln=True)
    
    pts = analysis.get("key_discussion_points", [])
    if pts:
        for i, p in enumerate(pts, 1):
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(15, 23, 42)
            speaker_tag = f" [{clean_text(p.get('speaker'))}]" if p.get("speaker") else ""
            pdf.cell(0, 5, f"{i}. {clean_text(p.get('topic', 'Topic'))}{speaker_tag}", ln=True)
            
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(71, 85, 105)
            pdf.multi_cell(0, 4, f"   {clean_text(p.get('summary', ''))}")
            pdf.ln(2)
    pdf.ln(3)

    # ── 6. SENTIMENT & RISKS ──────────────────────────────────────────────────
    sa = analysis.get("sentiment_analysis", {})
    risks = analysis.get("risks", [])
    
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(15, 23, 42)
    pdf.cell(0, 7, "5. Sentiment & Risk Assessment", ln=True)
    
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(51, 65, 85)
    overall_sent = clean_text((sa.get("overall_sentiment") or "Neutral")).upper()
    energy = clean_text((sa.get("energy_level") or "Normal")).upper()
    score = sa.get("sentiment_score", 0)
    pdf.cell(0, 5, f"Overall Tone: {overall_sent} (Score: {score:+.2f})  |  Energy: {energy}", ln=True)
    
    tone = clean_text(sa.get("tone_description", ""))
    if tone:
        pdf.set_font("Helvetica", "I", 8)
        pdf.multi_cell(0, 4, f"Tone: {tone}")
        pdf.ln(2)
        
    if risks:
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(185, 28, 28) # Red-700
        pdf.cell(0, 5, "Identified Risks & Mitigations:", ln=True)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(51, 65, 85)
        for r in risks:
            risk_text = clean_text(r.get("risk", ""))
            mitigation = clean_text(r.get("mitigation", ""))
            pdf.multi_cell(0, 4, f" ! {risk_text} -> Mitigation: {mitigation}")
            pdf.ln(1)

    # ── 6. WHAT WAS SPOKEN (VERBATIM DIALOGUE & TRANSCRIPT) ───────────────────
    full_tx = transcript or analysis.get("transcript") or analysis.get("speaker_transcript") or ""
    if full_tx:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 8, "6. Spoken Dialogue & Full Transcript", ln=True)
        pdf.set_font("Helvetica", "I", 8)
        pdf.set_text_color(100, 116, 139)
        pdf.cell(0, 5, "Complete chronological record of all spoken words and exchanges in this meeting:", ln=True)
        pdf.ln(3)

        pdf.set_draw_color(226, 232, 240)
        
        for line in full_tx.split("\n"):
            line = line.strip()
            if not line:
                continue
            if ":" in line:
                spk, speech = line.split(":", 1)
                pdf.set_font("Helvetica", "B", 9)
                pdf.set_text_color(2, 132, 199) # Sky-600
                pdf.cell(0, 5, clean_text(spk.strip()) + ":", ln=True)
                pdf.set_font("Helvetica", "", 8)
                pdf.set_text_color(51, 65, 85)
                pdf.multi_cell(0, 4.5, clean_text(speech.strip()))
                pdf.ln(2)
            else:
                pdf.set_font("Helvetica", "", 8)
                pdf.set_text_color(51, 65, 85)
                pdf.multi_cell(0, 4.5, clean_text(line))
                pdf.ln(2)

    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    pdf.output(output_path)
    return output_path
