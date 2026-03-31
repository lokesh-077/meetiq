"""
MeetIQ Web Backend v3.0 — FastAPI
✅ Speaker Detection  ✅ Live WebSocket  ✅ Auto Email  ✅ 99+ Languages
"""

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn, whisper, subprocess, json, os, uuid, urllib.request, urllib.error
import shutil, smtplib, wave, numpy as np
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from datetime import datetime
from typing import List

app = FastAPI(title="MeetIQ API", version="3.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"], allow_credentials=False, expose_headers=["*"])

# ── CONFIG ──────────────────────────────────────────────────────────────────
GROQ_API_KEY   = "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"        # 👈 paste your free Groq key (console.groq.com)
WHISPER_MODEL  = "tiny"   # small | medium
UPLOAD_DIR     = Path("uploads"); UPLOAD_DIR.mkdir(exist_ok=True)
EMAIL_SENDER   = "ulokesh.kannan2005@gmail.com"        # 👈 optional Gmail
EMAIL_PASSWORD = "tpag vhpy pqqj pqvm"        # 👈 optional Gmail App Password

jobs = {}

ANALYSIS_PROMPT = """You are an expert multilingual meeting analyst.
The transcript may be in ANY language. Produce ALL output in ENGLISH ONLY.
Be detailed — extract EVERY important point.

TRANSCRIPT:
{transcript}

Return ONLY valid JSON (no markdown):
{{
  "meeting_overview": {{
    "main_topic": "<one-line English>",
    "estimated_duration_minutes": <number or null>,
    "participant_count": <number or null>,
    "meeting_type": "standup|planning|review|brainstorm|interview|other",
    "speakers_identified": ["<speaker>"]
  }},
  "summary": "<3-5 sentence executive summary>",
  "speaker_contributions": [
    {{"speaker":"<label>","key_points":["<point>"],"sentiment":"positive|neutral|negative","talk_time_percent":<number>}}
  ],
  "key_discussion_points": [
    {{"topic":"<title>","summary":"<2-3 sentences>","importance":"high|medium|low","category":"technical|business|hr|finance|strategy|other","speaker":"<who>"}}
  ],
  "decisions_made": [
    {{"decision":"<description>","context":"<context>","owner":"<person>","deadline":"<or null>","decided_by":"<speaker>"}}
  ],
  "action_items": [
    {{"task":"<task>","owner":"<person>","priority":"urgent|high|medium|low","deadline":"<or null>","category":"follow_up|research|development|meeting|review|other","estimated_hours":<number or null>,"assigned_by":"<speaker>"}}
  ],
  "optimization_suggestions": [
    {{"area":"<area>","issue":"<problem>","suggestion":"<actionable>","impact":"high|medium|low"}}
  ],
  "sentiment_analysis": {{
    "overall_sentiment":"positive|neutral|mixed|negative",
    "sentiment_score":<float -1.0 to 1.0>,
    "tone_description":"<2-3 sentences>",
    "energy_level":"high|medium|low",
    "notable_moments":[{{"moment":"<desc>","sentiment":"positive|negative|neutral","speaker":"<who>"}}]
  }},
  "open_questions": ["<question>"],
  "risks": [{{"risk":"<risk>","severity":"high|medium|low","mitigation":"<suggestion>"}}]
}}"""

LIVE_PROMPT = """Analyse this meeting transcript chunk. Be concise.
Return ONLY valid JSON:
{{"key_points":["<point>"],"action_items":["<action>"],"sentiment":"positive|neutral|negative","summary":"<1-2 sentences>"}}
TRANSCRIPT: {transcript}"""

# ── HELPERS ──────────────────────────────────────────────────────────────────
def extract_audio(video_path, job_id):
    out = str(UPLOAD_DIR / f"{job_id}.wav")
    r = subprocess.run(["ffmpeg","-y","-i",video_path,"-vn","-ar","16000","-ac","1",
                        "-af","highpass=f=200,lowpass=f=3000,volume=2.0","-c:a","pcm_s16le",out],
                       capture_output=True)
    if r.returncode != 0: raise RuntimeError("ffmpeg: "+r.stderr.decode())
    return out

def detect_speakers(audio_path, segments):
    try:
        with wave.open(audio_path,'rb') as wf:
            frames=wf.readframes(wf.getnframes()); fr=wf.getframerate(); sw=wf.getsampwidth()
        samples=(np.frombuffer(frames,dtype=np.int16).astype(np.float32) if sw==2
                 else np.frombuffer(frames,dtype=np.uint8).astype(np.float32)-128)
        samples/=(np.max(np.abs(samples))+1e-8)
        labeled=[]; cur=1; prev_end=0.0; hist=[]
        for seg in segments:
            s=seg.get("start",0); e=seg.get("end",s+2); txt=seg.get("text","").strip()
            chunk=samples[int(s*fr):int(e*fr)]
            energy=float(np.sqrt(np.mean(chunk**2))) if len(chunk)>0 else 0
            if s-prev_end>1.5: cur=3-cur
            if hist and abs(energy-sum(hist[-5:])/len(hist[-5:]))>0.15: cur=3-cur
            hist.append(energy)
            labeled.append({"start":round(s,2),"end":round(e,2),"speaker":f"Speaker {cur}","text":txt})
            prev_end=e
        return labeled
    except:
        return [{"start":s.get("start",0),"end":s.get("end",0),"speaker":"Speaker 1","text":s.get("text","").strip()} for s in segments]

def build_speaker_tx(labeled):
    lines=[]; prev=None; buf=[]
    for seg in labeled:
        spk,txt=seg["speaker"],seg["text"].strip()
        if not txt: continue
        if spk!=prev:
            if buf: lines.append(f"{prev}: {' '.join(buf)}")
            buf=[]; prev=spk
        buf.append(txt)
    if buf and prev: lines.append(f"{prev}: {' '.join(buf)}")
    return "\n".join(lines)

def transcribe_audio(audio_path):
    model=whisper.load_model(WHISPER_MODEL)
    result=model.transcribe(audio_path,verbose=False,fp16=False,task="transcribe",
                            condition_on_previous_text=False,temperature=0.2,best_of=5,beam_size=5)
    lang=result.get("language","unknown").upper()
    clean=[seg for seg in result.get("segments",[])
           if seg["text"].strip() and not (len(seg["text"].split())>4 and len(set(seg["text"].split()))/len(seg["text"].split())<0.3)]
    labeled=detect_speakers(audio_path,clean)
    speaker_tx=build_speaker_tx(labeled)
    plain_tx=" ".join(s["text"] for s in clean).strip() or result["text"].strip()
    return plain_tx, speaker_tx, labeled, lang

def call_groq(prompt, max_tokens=3000):
    if not GROQ_API_KEY: raise RuntimeError("GROQ_API_KEY not set in main.py")
    payload=json.dumps({"model":"llama-3.3-70b-versatile",
        "messages":[{"role":"system","content":"Multilingual meeting analyst. English only. Valid JSON only."},
                    {"role":"user","content":prompt}],
        "temperature":0.2,"max_tokens":max_tokens}).encode()
    req=urllib.request.Request("https://api.groq.com/openai/v1/chat/completions",data=payload,
        headers={"Content-Type":"application/json","Authorization":f"Bearer {GROQ_API_KEY}",
                 "User-Agent":"Mozilla/5.0","Accept":"application/json"})
    with urllib.request.urlopen(req,timeout=120) as resp:
        data=json.loads(resp.read())
    raw=data["choices"][0]["message"]["content"].strip()
    if raw.startswith("```"): raw=raw.split("```")[1]; raw=raw[4:] if raw.startswith("json") else raw
    return raw.strip()

def analyze_with_groq(speaker_tx):
    raw=call_groq(ANALYSIS_PROMPT.format(transcript=speaker_tx))
    try: return json.loads(raw)
    except: return {"raw_response":raw}

def send_email(to_emails, analysis, filename, lang):
    if not EMAIL_SENDER or not EMAIL_PASSWORD: return {"status":"skipped"}
    A=analysis; ov=A.get("meeting_overview",{})
    rows="".join(f"<tr><td style='padding:8px;border-bottom:1px solid #eee'>{t.get('task','')}</td>"
                 f"<td style='padding:8px;border-bottom:1px solid #eee'>{t.get('owner','TBD')}</td>"
                 f"<td style='padding:8px;border-bottom:1px solid #eee;color:#dc2626'>{t.get('priority','').upper()}</td>"
                 f"<td style='padding:8px;border-bottom:1px solid #eee'>{t.get('deadline') or 'TBD'}</td></tr>"
                 for t in A.get("action_items",[]))
    dec="".join(f"<li style='margin-bottom:6px'>✅ {d.get('decision','')} — <b>{d.get('owner','')}</b></li>" for d in A.get("decisions_made",[]))
    html=f"""<div style="font-family:Arial,sans-serif;max-width:680px;margin:0 auto;background:#f9fafb;padding:20px">
      <div style="background:linear-gradient(135deg,#0ea5e9,#7c3aed);padding:24px;border-radius:12px;margin-bottom:16px;text-align:center">
        <h1 style="color:white;margin:0;font-size:20px">🎙️ MeetIQ Meeting Summary</h1>
        <p style="color:rgba(255,255,255,0.85);margin:4px 0 0">{ov.get('main_topic','')}</p>
      </div>
      <div style="background:white;border-radius:12px;padding:18px;margin-bottom:12px">
        <p style="color:#6b7280;font-size:12px">📅 {datetime.now().strftime('%B %d, %Y')} | 🌐 {lang} → English | 📁 {filename}</p>
        <h2 style="font-size:14px;margin:10px 0 8px">📋 Summary</h2>
        <p style="color:#374151;line-height:1.7;font-size:13px">{A.get('summary','')}</p>
      </div>
      <div style="background:white;border-radius:12px;padding:18px;margin-bottom:12px">
        <h2 style="font-size:14px;margin:0 0 12px">⚡ Action Items</h2>
        <table style="width:100%;border-collapse:collapse;font-size:13px">
          <tr style="background:#f3f4f6"><th style="padding:8px;text-align:left;color:#6b7280">Task</th><th style="padding:8px;text-align:left;color:#6b7280">Owner</th><th style="padding:8px;text-align:left;color:#6b7280">Priority</th><th style="padding:8px;text-align:left;color:#6b7280">Deadline</th></tr>
          {rows or '<tr><td colspan="4" style="padding:8px;color:#9ca3af">No action items</td></tr>'}
        </table>
      </div>
      <div style="background:white;border-radius:12px;padding:18px">
        <h2 style="font-size:14px;margin:0 0 10px">✅ Decisions</h2>
        <ul style="margin:0;padding-left:16px;font-size:13px">{dec or '<li style="color:#9ca3af">None</li>'}</ul>
      </div>
      <p style="text-align:center;color:#9ca3af;font-size:11px;margin-top:12px">Generated by MeetIQ v3.0</p>
    </div>"""
    results=[]
    for to in to_emails:
        try:
            msg=MIMEMultipart("alternative"); msg["Subject"]=f"📋 Meeting: {ov.get('main_topic','Summary')} — {datetime.now().strftime('%b %d')}"
            msg["From"]=EMAIL_SENDER; msg["To"]=to; msg.attach(MIMEText(html,"html"))
            with smtplib.SMTP_SSL("smtp.gmail.com",465) as smtp:
                smtp.login(EMAIL_SENDER,EMAIL_PASSWORD); smtp.sendmail(EMAIL_SENDER,to,msg.as_string())
            results.append({"email":to,"status":"sent"})
        except Exception as e:
            results.append({"email":to,"status":"failed","error":str(e)})
    return {"status":"done","results":results}

def process_meeting(job_id, video_path, notify_emails):
    try:
        jobs[job_id].update({"status":"extracting_audio","progress":10})
        audio=extract_audio(video_path,job_id)
        jobs[job_id].update({"status":"transcribing","progress":30})
        plain_tx,speaker_tx,labeled,lang=transcribe_audio(audio)
        jobs[job_id].update({"transcript":plain_tx,"speaker_transcript":speaker_tx,"labeled_segments":labeled,"language":lang,"progress":60})
        jobs[job_id].update({"status":"analyzing","progress":75})
        analysis=analyze_with_groq(speaker_tx)
        jobs[job_id].update({"status":"sending_email","progress":90})
        email_result=send_email(notify_emails,analysis,jobs[job_id]["filename"],lang) if notify_emails else {}
        jobs[job_id].update({"status":"done","progress":100,"analysis":analysis,"email_result":email_result,"completed":datetime.now().isoformat()})
        os.remove(video_path); os.remove(audio)
    except Exception as e:
        import traceback
        jobs[job_id].update({"status":"error","error":str(e),"traceback":traceback.format_exc()})

# ── ROUTES ──────────────────────────────────────────────────────────────────
@app.get("/")
def root(): return {"message":"MeetIQ API v3.0","status":"running","features":["speaker_detection","live_analysis","email_summary"]}

@app.post("/upload")
async def upload(background_tasks: BackgroundTasks, file: UploadFile = File(...), emails: str = ""):
    allowed={".mp4",".mp3",".wav",".m4a",".mkv",".webm",".avi",".mov",".ogg"}
    ext=Path(file.filename).suffix.lower()
    if ext not in allowed: raise HTTPException(400,f"Unsupported: {ext}")
    job_id=str(uuid.uuid4())[:8]; vpath=str(UPLOAD_DIR/f"{job_id}{ext}")
    with open(vpath,"wb") as f: shutil.copyfileobj(file.file,f)
    notify=[e.strip() for e in emails.split(",") if e.strip()]
    jobs[job_id]={"job_id":job_id,"filename":file.filename,"status":"queued","progress":0,"created":datetime.now().isoformat(),"notify_emails":notify}
    background_tasks.add_task(process_meeting,job_id,vpath,notify)
    return {"job_id":job_id,"message":"Processing started"}

@app.get("/status/{job_id}")
def status(job_id:str):
    if job_id not in jobs: raise HTTPException(404,"Not found")
    j=jobs[job_id]; return {"job_id":job_id,"status":j["status"],"progress":j.get("progress",0),"error":j.get("error")}

@app.get("/result/{job_id}")
def result(job_id:str):
    if job_id not in jobs: raise HTTPException(404,"Not found")
    j=jobs[job_id]
    if j["status"]!="done": raise HTTPException(400,f"Not done: {j['status']}")
    return {"job_id":job_id,"filename":j["filename"],"language":j.get("language","?"),
            "transcript":j.get("transcript",""),"speaker_transcript":j.get("speaker_transcript",""),
            "labeled_segments":j.get("labeled_segments",[]),"analysis":j.get("analysis",{}),
            "email_result":j.get("email_result",{}),"completed":j.get("completed")}

@app.get("/jobs")
def list_jobs(): return [{"job_id":jid,"filename":j.get("filename"),"status":j.get("status"),"progress":j.get("progress",0),"created":j.get("created")} for jid,j in jobs.items()]

@app.websocket("/live/{session_id}")
async def live(websocket: WebSocket, session_id: str):
    await websocket.accept()
    session={"transcript":[]}
    try:
        while True:
            data=await websocket.receive_text(); msg=json.loads(data)
            if msg.get("type")=="transcript_chunk":
                chunk=msg.get("text","").strip()
                if chunk:
                    session["transcript"].append(chunk)
                    recent=" ".join(session["transcript"][-5:])
                    try: ana=json.loads(call_groq(LIVE_PROMPT.format(transcript=recent),max_tokens=500))
                    except: ana={"key_points":[],"action_items":[],"sentiment":"neutral","summary":chunk}
                    await websocket.send_text(json.dumps({"type":"analysis","chunk":chunk,"analysis":ana,"full_so_far":" ".join(session["transcript"])}))
            elif msg.get("type")=="end_session":
                full=" ".join(session["transcript"])
                if full.strip():
                    try: ana=json.loads(call_groq(ANALYSIS_PROMPT.format(transcript=full)))
                    except: ana={}
                    await websocket.send_text(json.dumps({"type":"final_analysis","analysis":ana,"transcript":full}))
                break
    except WebSocketDisconnect: pass

if __name__=="__main__":
    print("\n🚀 MeetIQ API v3.0")
    print(f"  Groq Key : {'✅ set' if GROQ_API_KEY else '❌ MISSING — paste in line 16'}")
    print(f"  Email    : {'✅ set' if EMAIL_SENDER else '⚠️  not configured (optional)'}\n")
    uvicorn.run("main:app",host="0.0.0.0",port=8000,reload=False)