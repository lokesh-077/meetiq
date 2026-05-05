"""
MeetIQ Backend v3.0 — Tier 1
✅ Speaker Detection
✅ Auto Email Summary
✅ Live WebSocket Analysis
✅ 99+ Languages → English
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, whisper, subprocess, json, os, socket, uuid, urllib.request, urllib.error
import shutil, smtplib, wave, numpy as np
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from datetime import datetime

app = FastAPI(title="MeetIQ API", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"], allow_credentials=False, expose_headers=["*"])

# ── CONFIG ── Fill these before running ─────────────────────────────────────
GROQ_API_KEY   = ""       
WHISPER_MODEL  = "small"  
UPLOAD_DIR     = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
EMAIL_SENDER   = ""       
EMAIL_PASSWORD = ""       
# ────────────────────────────────────────────────────────────────────────────

jobs = {}

ANALYSIS_PROMPT = """You are an expert multilingual meeting analyst.
The transcript may be in ANY language. Produce ALL output in ENGLISH ONLY.
Be very detailed. Extract EVERY important point.

TRANSCRIPT:
{transcript}

Return ONLY valid JSON (no markdown, no extra text):
{{
  "meeting_overview": {{
    "main_topic": "<one-line English description>",
    "estimated_duration_minutes": <number or null>,
    "participant_count": <number or null>,
    "meeting_type": "standup|planning|review|brainstorm|interview|other",
    "speakers_identified": ["<speaker label>"]
  }},
  "summary": "<3-5 sentence executive summary in English>",
  "speaker_contributions": [
    {{
      "speaker": "<label>",
      "key_points": ["<point in English>"],
      "sentiment": "positive|neutral|negative",
      "talk_time_percent": <number>
    }}
  ],
  "key_discussion_points": [
    {{
      "topic": "<English title>",
      "summary": "<2-3 sentence English summary>",
      "importance": "high|medium|low",
      "category": "technical|business|hr|finance|strategy|other",
      "speaker": "<who raised this>"
    }}
  ],
  "decisions_made": [
    {{
      "decision": "<English description>",
      "context": "<brief English context>",
      "owner": "<person or Team>",
      "deadline": "<deadline or null>",
      "decided_by": "<speaker>"
    }}
  ],
  "action_items": [
    {{
      "task": "<specific task in English>",
      "owner": "<person responsible>",
      "priority": "urgent|high|medium|low",
      "deadline": "<deadline or null>",
      "category": "follow_up|research|development|meeting|review|other",
      "estimated_hours": <number or null>,
      "assigned_by": "<speaker>"
    }}
  ],
  "optimization_suggestions": [
    {{
      "area": "<area>",
      "issue": "<problem identified>",
      "suggestion": "<actionable suggestion>",
      "impact": "high|medium|low"
    }}
  ],
  "sentiment_analysis": {{
    "overall_sentiment": "positive|neutral|mixed|negative",
    "sentiment_score": <float -1.0 to 1.0>,
    "tone_description": "<2-3 sentences in English>",
    "energy_level": "high|medium|low",
    "notable_moments": [
      {{"moment": "<description>", "sentiment": "positive|negative|neutral", "speaker": "<who>"}}
    ]
  }},
  "open_questions": ["<unresolved question>"],
  "risks": [
    {{"risk": "<risk>", "severity": "high|medium|low", "mitigation": "<suggestion>"}}
  ]
}}"""

LIVE_PROMPT = """Analyse this meeting transcript chunk. Be concise. English only.
Return ONLY valid JSON:
{{"key_points":["<point>"],"action_items":["<action>"],"sentiment":"positive|neutral|negative","summary":"<1-2 sentences>"}}
TRANSCRIPT: {transcript}"""


# ── AUDIO ────────────────────────────────────────────────────────────────────
def extract_audio(video_path, job_id):
    out = str(UPLOAD_DIR / f"{job_id}.wav")
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path, "-vn", "-ar", "16000", "-ac", "1",
         "-af", "highpass=f=200,lowpass=f=3000,volume=2.0", "-c:a", "pcm_s16le", out],
        capture_output=True
    )
    if r.returncode != 0:
        raise RuntimeError("ffmpeg failed: " + r.stderr.decode())
    return out


# ── SPEAKER DETECTION ─────────────────────────────────────────────────────────
def detect_speakers(audio_path, segments):
    try:
        with wave.open(audio_path, 'rb') as wf:
            frames = wf.readframes(wf.getnframes())
            fr = wf.getframerate()
            sw = wf.getsampwidth()
        samples = (np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                   if sw == 2 else np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128)
        samples /= (np.max(np.abs(samples)) + 1e-8)
        labeled = []
        cur = 1
        prev_end = 0.0
        hist = []
        for seg in segments:
            s = seg.get("start", 0)
            e = seg.get("end", s + 2)
            txt = seg.get("text", "").strip()
            chunk = samples[int(s * fr):int(e * fr)]
            energy = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0
            if s - prev_end > 1.5:
                cur = 3 - cur
            if hist and abs(energy - sum(hist[-5:]) / len(hist[-5:])) > 0.15:
                cur = 3 - cur
            hist.append(energy)
            labeled.append({"start": round(s, 2), "end": round(e, 2),
                             "speaker": f"Speaker {cur}", "text": txt})
            prev_end = e
        return labeled
    except Exception:
        return [{"start": s.get("start", 0), "end": s.get("end", 0),
                 "speaker": "Speaker 1", "text": s.get("text", "").strip()}
                for s in segments]


def build_speaker_tx(labeled):
    lines, prev, buf = [], None, []
    for seg in labeled:
        spk, txt = seg["speaker"], seg["text"].strip()
        if not txt:
            continue
        if spk != prev:
            if buf:
                lines.append(f"{prev}: {' '.join(buf)}")
            buf = []
            prev = spk
        buf.append(txt)
    if buf and prev:
        lines.append(f"{prev}: {' '.join(buf)}")
    return "\n".join(lines)


# ── TRANSCRIPTION ─────────────────────────────────────────────────────────────
def transcribe_audio(audio_path):
    model = whisper.load_model(WHISPER_MODEL)
    result = model.transcribe(audio_path, verbose=False, fp16=False,
                              task="transcribe", condition_on_previous_text=False,
                              temperature=0.2, best_of=5, beam_size=5)
    lang = result.get("language", "unknown").upper()
    clean = [seg for seg in result.get("segments", [])
             if seg["text"].strip() and not (
                 len(seg["text"].split()) > 4 and
                 len(set(seg["text"].split())) / len(seg["text"].split()) < 0.3)]
    labeled = detect_speakers(audio_path, clean)
    speaker_tx = build_speaker_tx(labeled)
    plain_tx = " ".join(s["text"] for s in clean).strip() or result["text"].strip()
    return plain_tx, speaker_tx, labeled, lang


# ── GROQ API ──────────────────────────────────────────────────────────────────
def call_groq(prompt, max_tokens=3000):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set in main.py")
    payload = json.dumps({
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "Multilingual meeting analyst. English output only. Valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.2,
        "max_tokens": max_tokens
    }).encode()
    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {GROQ_API_KEY}",
                 "User-Agent": "Mozilla/5.0", "Accept": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    raw = data["choices"][0]["message"]["content"].strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


def analyze_with_groq(speaker_tx):
    raw = call_groq(ANALYSIS_PROMPT.format(transcript=speaker_tx))
    try:
        return json.loads(raw)
    except Exception:
        return {"raw_response": raw}


# ── EMAIL ─────────────────────────────────────────────────────────────────────
def send_email(to_emails, analysis, filename, lang):
    if not EMAIL_SENDER or not EMAIL_PASSWORD:
        return {"status": "skipped"}
    A = analysis
    ov = A.get("meeting_overview", {})
    rows = "".join(
        f"<tr><td style='padding:8px;border-bottom:1px solid #eee'>{t.get('task','')}</td>"
        f"<td style='padding:8px;border-bottom:1px solid #eee'>{t.get('owner','TBD')}</td>"
        f"<td style='padding:8px;border-bottom:1px solid #eee;color:#dc2626;font-weight:600'>{t.get('priority','').upper()}</td>"
        f"<td style='padding:8px;border-bottom:1px solid #eee'>{t.get('deadline') or 'TBD'}</td></tr>"
        for t in A.get("action_items", [])
    )
    dec = "".join(
        f"<li style='margin-bottom:6px'>✅ {d.get('decision','')} — <b>{d.get('owner','')}</b></li>"
        for d in A.get("decisions_made", [])
    )
    spk_section = "".join(
        f"<li style='margin-bottom:8px'><b>{s.get('speaker','')}</b> ({s.get('talk_time_percent','?')}% talk): "
        f"{', '.join(s.get('key_points',[])[:2])}</li>"
        for s in A.get("speaker_contributions", [])
    )
    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:680px;margin:0 auto;background:#f9fafb;padding:20px">
      <div style="background:linear-gradient(135deg,#0ea5e9,#7c3aed);padding:24px;border-radius:12px;margin-bottom:16px;text-align:center">
        <h1 style="color:white;margin:0;font-size:22px">🎙️ MeetIQ Meeting Summary</h1>
        <p style="color:rgba(255,255,255,0.85);margin:6px 0 0">{ov.get('main_topic','Meeting Summary')}</p>
      </div>
      <div style="background:white;border-radius:12px;padding:20px;margin-bottom:14px">
        <p style="color:#6b7280;font-size:12px">📅 {datetime.now().strftime('%B %d, %Y')} | 🌐 {lang} → English | 📁 {filename}</p>
        <h2 style="font-size:15px;margin:12px 0 8px">📋 Executive Summary</h2>
        <p style="color:#374151;line-height:1.7;font-size:13px">{A.get('summary','')}</p>
      </div>
      {f'<div style="background:white;border-radius:12px;padding:20px;margin-bottom:14px"><h2 style="font-size:15px;margin:0 0 12px">👥 Speakers</h2><ul style="margin:0;padding-left:18px;font-size:13px">{spk_section}</ul></div>' if spk_section else ''}
      <div style="background:white;border-radius:12px;padding:20px;margin-bottom:14px">
        <h2 style="font-size:15px;margin:0 0 12px">⚡ Action Items</h2>
        <table style="width:100%;border-collapse:collapse;font-size:13px">
          <tr style="background:#f3f4f6">
            <th style="padding:8px;text-align:left;color:#6b7280">Task</th>
            <th style="padding:8px;text-align:left;color:#6b7280">Owner</th>
            <th style="padding:8px;text-align:left;color:#6b7280">Priority</th>
            <th style="padding:8px;text-align:left;color:#6b7280">Deadline</th>
          </tr>
          {rows or '<tr><td colspan="4" style="padding:8px;color:#9ca3af">No action items</td></tr>'}
        </table>
      </div>
      <div style="background:white;border-radius:12px;padding:20px">
        <h2 style="font-size:15px;margin:0 0 10px">✅ Decisions Made</h2>
        <ul style="margin:0;padding-left:18px;font-size:13px">
          {dec or '<li style="color:#9ca3af">No decisions recorded</li>'}
        </ul>
      </div>
      <p style="text-align:center;color:#9ca3af;font-size:11px;margin-top:14px">Generated by MeetIQ v3.0 · AI Meeting Intelligence</p>
    </div>"""
    results = []
    for to in to_emails:
        try:
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"📋 Meeting: {ov.get('main_topic','Summary')} — {datetime.now().strftime('%b %d')}"
            msg["From"] = EMAIL_SENDER
            msg["To"] = to
            msg.attach(MIMEText(html, "html"))
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
                smtp.sendmail(EMAIL_SENDER, to, msg.as_string())
            results.append({"email": to, "status": "sent"})
        except Exception as e:
            results.append({"email": to, "status": "failed", "error": str(e)})
    return {"status": "done", "results": results}


# ── PIPELINE ──────────────────────────────────────────────────────────────────
def process_meeting(job_id, video_path, notify_emails):
    try:
        jobs[job_id].update({"status": "extracting_audio", "progress": 10})
        audio = extract_audio(video_path, job_id)

        jobs[job_id].update({"status": "transcribing", "progress": 30})
        plain_tx, speaker_tx, labeled, lang = transcribe_audio(audio)
        jobs[job_id].update({"transcript": plain_tx, "speaker_transcript": speaker_tx,
                              "labeled_segments": labeled, "language": lang, "progress": 60})

        jobs[job_id].update({"status": "analyzing", "progress": 75})
        analysis = analyze_with_groq(speaker_tx)

        jobs[job_id].update({"status": "sending_email", "progress": 90})
        email_result = send_email(notify_emails, analysis, jobs[job_id]["filename"], lang) if notify_emails else {}

        jobs[job_id].update({"status": "done", "progress": 100, "analysis": analysis,
                              "email_result": email_result, "completed": datetime.now().isoformat()})
        try:
            os.remove(video_path)
            os.remove(audio)
        except Exception:
            pass
    except Exception as e:
        import traceback
        jobs[job_id].update({"status": "error", "error": str(e),
                              "traceback": traceback.format_exc()})


# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "MeetIQ API v3.0", "status": "running"}


@app.post("/upload")
async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...), emails: str = ""):
    allowed = {".mp4", ".mp3", ".wav", ".m4a", ".mkv", ".webm", ".avi", ".mov", ".ogg"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(400, f"Unsupported format: {ext}")
    job_id = str(uuid.uuid4())[:8]
    vpath = str(UPLOAD_DIR / f"{job_id}{ext}")
    with open(vpath, "wb") as f:
        shutil.copyfileobj(file.file, f)
    notify = [e.strip() for e in emails.split(",") if e.strip()]
    jobs[job_id] = {"job_id": job_id, "filename": file.filename, "status": "queued",
                    "progress": 0, "created": datetime.now().isoformat(), "notify_emails": notify}
    background_tasks.add_task(process_meeting, job_id, vpath, notify)
    return {"job_id": job_id, "message": "Processing started"}


@app.get("/status/{job_id}")
def status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    return {"job_id": job_id, "status": j["status"],
            "progress": j.get("progress", 0), "error": j.get("error")}


@app.get("/result/{job_id}")
def result(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Job not found")
    j = jobs[job_id]
    if j["status"] != "done":
        raise HTTPException(400, f"Not complete: {j['status']}")
    return {"job_id": job_id, "filename": j["filename"], "language": j.get("language", "?"),
            "transcript": j.get("transcript", ""), "speaker_transcript": j.get("speaker_transcript", ""),
            "labeled_segments": j.get("labeled_segments", []), "analysis": j.get("analysis", {}),
            "email_result": j.get("email_result", {}), "completed": j.get("completed")}


@app.get("/jobs")
def list_jobs():
    return [{"job_id": jid, "filename": j.get("filename"), "status": j.get("status"),
             "progress": j.get("progress", 0), "created": j.get("created")}
            for jid, j in jobs.items()]


@app.get("/latest")
def latest_result():
    """Get the most recent completed job result directly"""
    done = [(jid, j) for jid, j in jobs.items() if j.get("status") == "done"]
    if not done:
        raise HTTPException(404, "No completed jobs found")
    jid, j = sorted(done, key=lambda x: x[1].get("completed", ""))[-1]
    return {"job_id": jid, "filename": j["filename"], "language": j.get("language", "?"),
            "transcript": j.get("transcript", ""), "speaker_transcript": j.get("speaker_transcript", ""),
            "labeled_segments": j.get("labeled_segments", []), "analysis": j.get("analysis", {}),
            "email_result": j.get("email_result", {}), "completed": j.get("completed")}


@app.websocket("/live/{session_id}")
async def live(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session = {"transcript": []}
    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "transcript_chunk":
                chunk = msg.get("text", "").strip()
                if chunk:
                    session["transcript"].append(chunk)
                    recent = " ".join(session["transcript"][-5:])
                    try:
                        raw = call_groq(LIVE_PROMPT.format(transcript=recent), max_tokens=500)
                        ana = json.loads(raw)
                    except Exception:
                        ana = {"key_points": [], "action_items": [], "sentiment": "neutral", "summary": chunk}
                    await websocket.send_text(json.dumps({
                        "type": "analysis", "chunk": chunk, "analysis": ana,
                        "full_so_far": " ".join(session["transcript"])
                    }))
            elif msg.get("type") == "end_session":
                full = " ".join(session["transcript"])
                if full.strip():
                    try:
                        raw = call_groq(ANALYSIS_PROMPT.format(transcript=full))
                        ana = json.loads(raw)
                    except Exception:
                        ana = {}
                    await websocket.send_text(json.dumps({
                        "type": "final_analysis", "analysis": ana, "transcript": full
                    }))
                break
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    print("\n🚀 MeetIQ API v3.0 — Tier 1")
    print(f"   Groq Key : {'✅ set' if GROQ_API_KEY else '❌ MISSING — paste in line 18'}")
    print(f"   Email    : {'✅ set' if EMAIL_SENDER else '⚠️  not configured (optional)'}")
    print(f"   Whisper  : {WHISPER_MODEL}\n")
    start_port = int(os.environ.get("PORT", 8000))
    port = None
    for attempt in range(5):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("0.0.0.0", start_port + attempt))
                port = start_port + attempt
                break
        except OSError:
            continue
    if port is None:
        raise RuntimeError(f"No available port found in range {start_port}-{start_port + 4}")
    if port != start_port:
        print(f"⚠️ Port {start_port} unavailable, starting on {port} instead.")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)