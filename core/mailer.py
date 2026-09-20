import smtplib
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from pathlib import Path
from datetime import datetime
from core.config import settings

logger = logging.getLogger("meetiq.mailer")

def send_pdf_report_email(to_emails: list, pdf_path: str, analysis: dict) -> dict:
    """Silently dispatch the meeting report PDF as an attachment via Gmail SMTP"""
    if not settings.email_sender or not settings.email_password:
        logger.info("Email credentials not configured in .env. Skipping email dispatch.")
        return {"status": "skipped", "reason": "No EMAIL_SENDER or EMAIL_PASSWORD set"}

    recipients = [e.strip() for e in to_emails if e and e.strip()]
    if not recipients and settings.email_to:
        recipients = [e.strip() for e in settings.email_to.split(",") if e.strip()]

    if not recipients:
        logger.info("No recipients provided. Skipping email dispatch.")
        return {"status": "skipped", "reason": "No recipients specified"}

    overview = analysis.get("meeting_overview", {})
    topic = overview.get("main_topic") or "Meeting Summary"
    summary = analysis.get("summary") or "Meeting completed successfully."
    actions = analysis.get("action_items", [])

    action_rows = "".join(
        f"<tr>"
        f"<td style='padding:8px 12px;border-bottom:1px solid #e2e8f0;font-size:13px;color:#1e293b;'>{t.get('task','')}</td>"
        f"<td style='padding:8px 12px;border-bottom:1px solid #e2e8f0;font-size:13px;color:#475569;'>{t.get('owner','TBD')}</td>"
        f"<td style='padding:8px 12px;border-bottom:1px solid #e2e8f0;font-size:11px;font-weight:700;color:#dc2626;'>{(t.get('priority') or 'medium').upper()}</td>"
        f"<td style='padding:8px 12px;border-bottom:1px solid #e2e8f0;font-size:13px;color:#475569;'>{t.get('deadline') or 'TBD'}</td>"
        f"</tr>"
        for t in actions[:5]
    )

    html_content = f"""
    <div style="font-family:'Segoe UI',Arial,sans-serif;max-width:640px;margin:0 auto;background:#f8fafc;padding:24px;border-radius:16px;">
      <div style="background:linear-gradient(135deg,#0284c7,#4f46e5);padding:24px;border-radius:12px;text-align:center;color:white;">
        <h1 style="margin:0;font-size:22px;letter-spacing:0.5px;">MeetIQ — Meeting Intelligence</h1>
        <p style="margin:6px 0 0;font-size:14px;color:rgba(255,255,255,0.9);">{topic}</p>
      </div>
      
      <div style="background:white;border-radius:12px;padding:20px;margin-top:16px;box-shadow:0 1px 3px rgba(0,0,0,0.05);">
        <p style="color:#64748b;font-size:12px;margin:0 0 10px;">
          📅 {datetime.now().strftime('%B %d, %Y')} &nbsp;|&nbsp; 📋 Executive Overview
        </p>
        <p style="color:#334155;line-height:1.7;font-size:14px;margin:0;">
          {summary}
        </p>
      </div>

      {f'''
      <div style="background:white;border-radius:12px;padding:20px;margin-top:16px;box-shadow:0 1px 3px rgba(0,0,0,0.05);">
        <h3 style="margin:0 0 12px;font-size:14px;color:#0f172a;text-transform:uppercase;letter-spacing:0.5px;">⚡ High Priority Action Items</h3>
        <table style="width:100%;border-collapse:collapse;">
          <tr style="background:#f1f5f9;text-align:left;">
            <th style="padding:8px 12px;font-size:12px;color:#64748b;">Task</th>
            <th style="padding:8px 12px;font-size:12px;color:#64748b;">Owner</th>
            <th style="padding:8px 12px;font-size:12px;color:#64748b;">Priority</th>
            <th style="padding:8px 12px;font-size:12px;color:#64748b;">Deadline</th>
          </tr>
          {action_rows}
        </table>
      </div>
      ''' if action_rows else ''}

      <div style="background:#e0f2fe;border:1px solid #bae6fd;border-radius:12px;padding:16px;margin-top:16px;text-align:center;">
        <p style="margin:0;color:#0369a1;font-size:13px;font-weight:600;">
          📎 Your complete multi-page PDF Meeting Report is attached below.
        </p>
      </div>

      <p style="text-align:center;color:#94a3b8;font-size:11px;margin-top:20px;">
        Sent automatically by MeetIQ AI Notetaker • Confidential & Automated
      </p>
    </div>
    """

    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        logger.error(f"Cannot send email: PDF file not found at {pdf_path}")
        return {"status": "failed", "error": "PDF file missing"}

    sent_list = []
    failed_list = []

    for to_addr in recipients:
        try:
            msg = MIMEMultipart()
            msg["Subject"] = f"📋 Meeting Report: {topic} — {datetime.now().strftime('%b %d')}"
            msg["From"] = settings.email_sender
            msg["To"] = to_addr
            
            # HTML body
            msg.attach(MIMEText(html_content, "html"))
            
            # Attach PDF
            with open(pdf_file, "rb") as f:
                pdf_attachment = MIMEApplication(f.read(), Name=pdf_file.name)
                pdf_attachment["Content-Disposition"] = f'attachment; filename="{pdf_file.name}"'
                msg.attach(pdf_attachment)

            with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as server:
                server.login(settings.email_sender, settings.email_password)
                server.sendmail(settings.email_sender, to_addr, msg.as_string())

            sent_list.append(to_addr)
            logger.info(f"Report emailed successfully to: {to_addr}")
        except Exception as e:
            logger.error(f"Failed to email {to_addr}: {str(e)}")
            failed_list.append({"email": to_addr, "error": str(e)})

    return {"status": "done", "sent": sent_list, "failed": failed_list}
