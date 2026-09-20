# 🔌 MeetIQ API Reference

MeetIQ exposes a high-performance REST API built on FastAPI.

---

## 🤖 Autonomous Bot Endpoints

### `POST /api/bot/join`
Deploys the autonomous bot to join a Google Meet call.
```json
{
  "meeting_url": "https://meet.google.com/xxx-yyyy-zzz",
  "bot_name": "MeetIQ Notetaker",
  "emails": "loaloki2005@gmail.com",
  "target_language": "English"
}
```

### `GET /api/bot/status/{session_id}`
Returns real-time session progress, participant captions count, and the last 10 streaming closed captions.

### `POST /api/bot/leave/{session_id}`
Signals the bot to conclude the meeting, run Gemini synthesis, generate the PDF, and email the report.

### `GET /api/bot/result/{session_id}`
Returns complete meeting analysis, action items, decisions, and PDF filename.

---

## 📁 File Upload Endpoints

### `POST /api/upload`
Uploads a `.mp4`, `.mp3`, `.wav`, or `.m4a` file with target language and recipient emails.

### `GET /api/status/{job_id}`
Returns upload processing stage (e.g., `extracting_audio`, `transcribing`, `analyzing_with_gemini`, `generating_pdf`, `done`).

---

## 📚 Reports Archive Endpoints

### `GET /api/reports`
Lists all generated PDF reports from `outputs/` with title, date, size, and language.

### `GET /api/reports/details/{filename}`
Retrieves full executive summary, action items, decisions, and verbatim transcript for the interactive in-browser previewer.

### `GET /api/download-pdf/{filename}`
Downloads the compiled PDF directly.

### `POST /api/reports/resend`
Resends an existing PDF report to a specified email address.
