#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║         MeetIQ CLI v3.0 — Meeting Analyzer                  ║
║  ✅ Speaker Detection   ✅ Email Summary   ✅ 99+ Languages  ║
║  ✅ Task Automation     ✅ Optimization    ✅ Risk Detection  ║
╚══════════════════════════════════════════════════════════════╝
"""

import os, sys, json, subprocess, urllib.request, urllib.error, smtplib, wave
from pathlib import Path
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# ══════════════════════════════════════════════════════════════
#  CONFIG — Fill these in before running
# ══════════════════════════════════════════════════════════════
VIDEO_FILE     = "CEO Talk_1.mp4"   # 👈 your video file name
GROQ_API_KEY   = "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"                 
WHISPER_MODEL  = "small"            # small | medium (medium = best quality)
OUTPUT_FILE    = "Meeting_Report.txt"

# Optional — Auto Email Summary
EMAIL_SENDER   = "ulokesh.kannan2005@gmail.com"                 # 👈 your Gmail e.g. you@gmail.com
EMAIL_PASSWORD = ""                 # 👈 Gmail App Password
EMAIL_TO       = ["ulokeshkannan@gmail.com"]                 # 👈 list of emails e.g. ["a@x.com","b@x.com"]

# Language hint (helps Whisper accuracy)
# Common: "ta"=Tamil, "hi"=Hindi, "en"=English, None=auto-detect
LANGUAGE_HINT  = None               # 👈 set "ta" for Tamil, None for auto
# ══════════════════════════════════════════════════════════════

IMPORTANCE_EMOJI = {"high": "🔴", "medium": "🟡", "low": "🟢"}
SENTIMENT_EMOJI  = {"positive": "😊", "neutral": "😐", "mixed": "🤔", "negative": "😟"}
PRIORITY_EMOJI   = {"urgent": "🚨", "high": "🔴", "medium": "🟡", "low": "🟢"}

ANALYSIS_PROMPT = """You are an expert multilingual meeting analyst and productivity optimizer.
The transcript may be in ANY language (Tamil, Hindi, English, etc).
Analyse it fully and produce the ENTIRE output in ENGLISH ONLY.
Be very detailed — extract EVERY important point.

TRANSCRIPT (with speaker labels if available):
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
      "speaker": "<speaker label>",
      "key_points": ["<point in English>"],
      "sentiment": "positive|neutral|negative",
      "talk_time_percent": <estimated %>
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
      "deadline": "<deadline or null>"
    }}
  ],
  "action_items": [
    {{
      "task": "<specific task in English>",
      "owner": "<person responsible>",
      "priority": "urgent|high|medium|low",
      "deadline": "<deadline or null>",
      "category": "follow_up|research|development|meeting|review|other",
      "estimated_hours": <number or null>
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
  "open_questions": ["<unresolved question in English>"],
  "risks": [
    {{"risk": "<identified risk>", "severity": "high|medium|low", "mitigation": "<suggestion>"}}
  ]
}}"""


# ══════════════════════════════════════════════════════════════
#  STEP 0 — Dependency check
# ══════════════════════════════════════════════════════════════
def check_deps():
    missing = []
    try: import whisper
    except ImportError: missing.append("openai-whisper")
    try: import numpy
    except ImportError: missing.append("numpy")
    if missing:
        print("❌ Missing packages:", *missing)
        print("   Run: pip install", *missing)
        sys.exit(1)


# ══════════════════════════════════════════════════════════════
#  STEP 1 — Extract & clean audio
# ══════════════════════════════════════════════════════════════
def extract_audio(video_path: str) -> str:
    audio_path = Path(video_path).stem + "_audio.wav"
    print(f"🎬 Extracting audio from: {video_path}")
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", video_path,
         "-vn", "-ar", "16000", "-ac", "1",
         "-af", "highpass=f=200,lowpass=f=3000,volume=2.0",
         "-c:a", "pcm_s16le", audio_path],
        capture_output=True
    )
    if result.returncode != 0:
        print("❌ ffmpeg failed. Install ffmpeg:")
        print("   Windows : https://ffmpeg.org/download.html")
        print("   Mac     : brew install ffmpeg")
        print("   Ubuntu  : sudo apt install ffmpeg")
        sys.exit(1)
    mb = os.path.getsize(audio_path) / 1024 / 1024
    print(f"✅ Audio extracted → {audio_path} ({mb:.1f} MB)\n")
    return audio_path


# ══════════════════════════════════════════════════════════════
#  STEP 2 — Speaker detection (energy-based)
# ══════════════════════════════════════════════════════════════
def detect_speakers(audio_path: str, segments: list) -> list:
    """Energy-based speaker detection — no extra libraries needed"""
    try:
        import numpy as np
        with wave.open(audio_path, 'rb') as wf:
            frames    = wf.readframes(wf.getnframes())
            framerate = wf.getframerate()
            sampwidth = wf.getsampwidth()

        samples = (np.frombuffer(frames, dtype=np.int16).astype(np.float32)
                   if sampwidth == 2
                   else np.frombuffer(frames, dtype=np.uint8).astype(np.float32) - 128)
        samples /= (np.max(np.abs(samples)) + 1e-8)

        labeled      = []
        current_spk  = 1
        prev_end     = 0.0
        energy_hist  = []

        for seg in segments:
            start = seg.get("start", 0)
            end   = seg.get("end",   start + 2)
            text  = seg.get("text",  "").strip()

            s_idx  = int(start * framerate)
            e_idx  = int(end   * framerate)
            chunk  = samples[s_idx:e_idx]
            energy = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) > 0 else 0

            # Speaker switch: long pause OR big energy change
            gap = start - prev_end
            if gap > 1.5:
                current_spk = 3 - current_spk  # toggle 1↔2

            if energy_hist:
                avg = sum(energy_hist[-5:]) / len(energy_hist[-5:])
                if abs(energy - avg) > 0.15:
                    current_spk = 3 - current_spk

            energy_hist.append(energy)
            labeled.append({
                "start":   round(start, 2),
                "end":     round(end,   2),
                "speaker": f"Speaker {current_spk}",
                "text":    text
            })
            prev_end = end

        return labeled

    except Exception:
        return [{"start": s.get("start", 0), "end": s.get("end", 0),
                 "speaker": "Speaker 1", "text": s.get("text", "").strip()}
                for s in segments]


def build_speaker_transcript(labeled: list) -> str:
    lines, prev_spk, buf = [], None, []
    for seg in labeled:
        spk, txt = seg["speaker"], seg["text"].strip()
        if not txt: continue
        if spk != prev_spk:
            if buf: lines.append(f"{prev_spk}: {' '.join(buf)}")
            buf = []; prev_spk = spk
        buf.append(txt)
    if buf and prev_spk:
        lines.append(f"{prev_spk}: {' '.join(buf)}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════
#  STEP 3 — Transcribe with Whisper
# ══════════════════════════════════════════════════════════════
def transcribe(audio_path: str) -> tuple:
    import whisper
    print(f"🎙️  Loading Whisper ({WHISPER_MODEL})…")
    model = whisper.load_model(WHISPER_MODEL)

    print("📝 Transcribing… (may take a few minutes)")
    print("   Please wait, do not close the window…\n")

    kwargs = dict(verbose=True, fp16=False, task="transcribe",
                  condition_on_previous_text=False, temperature=0.2,
                  best_of=5, beam_size=5)
    if LANGUAGE_HINT:
        kwargs["language"] = LANGUAGE_HINT

    result       = model.transcribe(audio_path, **kwargs)
    detected     = result.get("language", "unknown").upper()
    raw_segments = result.get("segments", [])

    # Filter hallucinated repeating segments
    clean = []
    for seg in raw_segments:
        txt = seg["text"].strip()
        if not txt: continue
        words = txt.split()
        if len(words) > 4 and len(set(words)) / len(words) < 0.3:
            continue  # hallucination
        clean.append(seg)

    # Speaker detection
    labeled      = detect_speakers(audio_path, clean)
    speaker_tx   = build_speaker_transcript(labeled)
    plain_tx     = " ".join(s["text"] for s in clean).strip() or result["text"].strip()

    print(f"\n🌐 Detected language : {detected}")
    print(f"👥 Speakers detected : {len(set(s['speaker'] for s in labeled))}")
    print(f"✅ Transcription done ({len(plain_tx.split())} words)\n")

    if detected not in ("EN", "ENGLISH", "UNKNOWN"):
        print(f"🔄 Meeting in {detected} → Summary will be in ENGLISH\n")

    return plain_tx, speaker_tx, labeled, detected


# ══════════════════════════════════════════════════════════════
#  STEP 4 — Analyze with Groq AI
# ══════════════════════════════════════════════════════════════
def analyze_with_groq(speaker_tx: str) -> dict:
    if not GROQ_API_KEY:
        print("❌ GROQ_API_KEY is empty!")
        print("   Get your free key at: https://console.groq.com")
        print("   Then paste it in line 17 of this script.")
        sys.exit(1)

    print("🤖 Sending to Groq AI for analysis…")

    payload = json.dumps({
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "system", "content": "You are a multilingual meeting analyst. Always respond in English only. Return valid JSON only, no markdown."},
            {"role": "user",   "content": ANALYSIS_PROMPT.format(transcript=speaker_tx)}
        ],
        "temperature": 0.2,
        "max_tokens": 3000
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json"
        }
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print(f"❌ Groq API error {e.code}: {e.read().decode()}")
        sys.exit(1)

    raw = data["choices"][0]["message"]["content"].strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"): raw = raw[4:]
    try:    return json.loads(raw.strip())
    except: return {"raw_response": raw}


# ══════════════════════════════════════════════════════════════
#  STEP 5 — Auto Email Summary
# ══════════════════════════════════════════════════════════════
def send_email_summary(analysis: dict, lang: str):
    if not EMAIL_SENDER or not EMAIL_PASSWORD or not EMAIL_TO:
        print("📧 Email not configured — skipping (set EMAIL_SENDER, EMAIL_PASSWORD, EMAIL_TO)")
        return

    print(f"📧 Sending email summary to {', '.join(EMAIL_TO)}…")
    A  = analysis
    ov = A.get("meeting_overview", {})

    action_rows = "".join(
        f"<tr><td style='padding:8px;border-bottom:1px solid #eee'>{t.get('task','')}</td>"
        f"<td style='padding:8px;border-bottom:1px solid #eee'>{t.get('owner','TBD')}</td>"
        f"<td style='padding:8px;border-bottom:1px solid #eee;color:#dc2626;font-weight:600'>{t.get('priority','').upper()}</td>"
        f"<td style='padding:8px;border-bottom:1px solid #eee'>{t.get('deadline') or 'TBD'}</td></tr>"
        for t in A.get("action_items", [])
    )
    decision_li = "".join(
        f"<li style='margin-bottom:6px'>✅ {d.get('decision','')} — <b>{d.get('owner','Team')}</b></li>"
        for d in A.get("decisions_made", [])
    )

    html = f"""
    <div style="font-family:Arial,sans-serif;max-width:680px;margin:0 auto;background:#f9fafb;padding:20px">
      <div style="background:linear-gradient(135deg,#0ea5e9,#7c3aed);padding:24px;border-radius:12px;margin-bottom:20px;text-align:center">
        <h1 style="color:white;margin:0;font-size:22px">🎙️ MeetIQ Meeting Summary</h1>
        <p style="color:rgba(255,255,255,0.85);margin:6px 0 0">{ov.get('main_topic','Meeting Summary')}</p>
      </div>
      <div style="background:white;border-radius:12px;padding:20px;margin-bottom:14px">
        <p style="color:#6b7280;font-size:12px">📅 {datetime.now().strftime('%B %d, %Y')} &nbsp;|&nbsp; 🌐 {lang} → English</p>
        <h2 style="font-size:15px;margin:12px 0 8px">📋 Executive Summary</h2>
        <p style="color:#374151;line-height:1.7;margin:0">{A.get('summary','')}</p>
      </div>
      <div style="background:white;border-radius:12px;padding:20px;margin-bottom:14px">
        <h2 style="font-size:15px;margin:0 0 14px">⚡ Action Items</h2>
        <table style="width:100%;border-collapse:collapse;font-size:13px">
          <tr style="background:#f3f4f6">
            <th style="padding:8px;text-align:left;color:#6b7280">Task</th>
            <th style="padding:8px;text-align:left;color:#6b7280">Owner</th>
            <th style="padding:8px;text-align:left;color:#6b7280">Priority</th>
            <th style="padding:8px;text-align:left;color:#6b7280">Deadline</th>
          </tr>
          {action_rows or '<tr><td colspan="4" style="padding:10px;color:#9ca3af">No action items</td></tr>'}
        </table>
      </div>
      <div style="background:white;border-radius:12px;padding:20px;margin-bottom:14px">
        <h2 style="font-size:15px;margin:0 0 12px">✅ Decisions Made</h2>
        <ul style="margin:0;padding-left:18px;color:#374151;font-size:13px">
          {decision_li or '<li style="color:#9ca3af">No decisions recorded</li>'}
        </ul>
      </div>
      <p style="text-align:center;color:#9ca3af;font-size:11px">Generated by MeetIQ v3.0 · AI Meeting Intelligence</p>
    </div>"""

    for to in EMAIL_TO:
        try:
            msg            = MIMEMultipart("alternative")
            msg["Subject"] = f"📋 Meeting Summary: {ov.get('main_topic','Meeting')} — {datetime.now().strftime('%b %d')}"
            msg["From"]    = EMAIL_SENDER
            msg["To"]      = to
            msg.attach(MIMEText(html, "html"))
            with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
                smtp.login(EMAIL_SENDER, EMAIL_PASSWORD)
                smtp.sendmail(EMAIL_SENDER, to, msg.as_string())
            print(f"   ✅ Sent to {to}")
        except Exception as e:
            print(f"   ❌ Failed to send to {to}: {e}")


# ══════════════════════════════════════════════════════════════
#  STEP 6 — Print & save report
# ══════════════════════════════════════════════════════════════
def print_report(analysis: dict, plain_tx: str, speaker_tx: str, labeled: list, lang: str):
    lines = []
    a = lines.append

    a("=" * 68)
    a("          🎙️  MeetIQ v3.0 — MEETING ANALYSIS REPORT")
    a("=" * 68)
    a(f"  🌐 Language  : {lang}  →  Summary in : ENGLISH")
    a(f"  🤖 Powered by: Groq AI (Llama 3.3 70B) + Whisper {WHISPER_MODEL}")
    a(f"  📅 Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    if "raw_response" in analysis:
        a("\n" + analysis["raw_response"])
    else:
        ov = analysis.get("meeting_overview", {})

        # ── Overview
        a(f"\n{'─'*68}")
        a("  🗂️  OVERVIEW")
        a(f"{'─'*68}")
        a(f"  Topic    : {ov.get('main_topic','N/A')}")
        a(f"  Type     : {ov.get('meeting_type','N/A').upper()}")
        a(f"  Duration : {ov.get('estimated_duration_minutes','N/A')} min (estimated)")
        a(f"  Speakers : {len(set(s['speaker'] for s in labeled))} detected")

        # ── Executive Summary
        a(f"\n{'─'*68}")
        a("  📋 EXECUTIVE SUMMARY")
        a(f"{'─'*68}")
        summary = analysis.get("summary","")
        for line in summary.split(". "):
            if line.strip(): a(f"  {line.strip()}.")

        # ── Speaker Contributions
        speakers = analysis.get("speaker_contributions", [])
        if speakers:
            a(f"\n{'─'*68}")
            a("  👥 SPEAKER CONTRIBUTIONS")
            a(f"{'─'*68}")
            for s in speakers:
                spk   = s.get("speaker","")
                pct   = s.get("talk_time_percent","?")
                sent  = SENTIMENT_EMOJI.get(s.get("sentiment","neutral"),"")
                a(f"\n  {sent} {spk}  ({pct}% talk time)")
                for pt in s.get("key_points",[]):
                    a(f"     › {pt}")

        # ── Key Discussion Points
        a(f"\n{'─'*68}")
        a("  📌 KEY DISCUSSION POINTS")
        a(f"{'─'*68}")
        for i, pt in enumerate(analysis.get("key_discussion_points",[]), 1):
            imp  = IMPORTANCE_EMOJI.get(pt.get("importance","medium"),"•")
            spk  = f"  [{pt.get('speaker','')}]" if pt.get("speaker") else ""
            a(f"\n  {i}. {imp}  {pt.get('topic','')}{spk}")
            a(f"      {pt.get('summary','')}")

        # ── Action Items
        a(f"\n{'─'*68}")
        a("  ⚡ ACTION ITEMS")
        a(f"{'─'*68}")
        tasks = analysis.get("action_items",[])
        if tasks:
            for i, t in enumerate(tasks, 1):
                pri = PRIORITY_EMOJI.get(t.get("priority","medium"),"•")
                a(f"\n  {i}. {pri}  {t.get('task','')}")
                a(f"      Owner    : {t.get('owner','TBD')}")
                a(f"      Deadline : {t.get('deadline','TBD')}")
                a(f"      Est.hrs  : {t.get('estimated_hours','?')}h")
                a(f"      Category : {t.get('category','')}")
        else:
            a("  No action items found.")

        # ── Decisions
        a(f"\n{'─'*68}")
        a("  ✅ DECISIONS MADE")
        a(f"{'─'*68}")
        decisions = analysis.get("decisions_made",[])
        if decisions:
            for i, d in enumerate(decisions, 1):
                a(f"  {i}. {d.get('decision','')}")
                a(f"     Owner   : {d.get('owner','Team')}")
                a(f"     Context : {d.get('context','')}")
                if d.get("deadline"): a(f"     Deadline: {d['deadline']}")
        else:
            a("  No decisions recorded.")

        # ── Optimization
        a(f"\n{'─'*68}")
        a("  🚀 OPTIMIZATION SUGGESTIONS")
        a(f"{'─'*68}")
        opts = analysis.get("optimization_suggestions",[])
        if opts:
            for i, o in enumerate(opts, 1):
                imp = IMPORTANCE_EMOJI.get(o.get("impact","medium"),"•")
                a(f"  {i}. {imp}  [{o.get('area','')}] {o.get('issue','')}")
                a(f"      → {o.get('suggestion','')}")
        else:
            a("  None identified.")

        # ── Risks
        a(f"\n{'─'*68}")
        a("  ⚠️  RISKS IDENTIFIED")
        a(f"{'─'*68}")
        risks = analysis.get("risks",[])
        if risks:
            for i, r in enumerate(risks, 1):
                sev = IMPORTANCE_EMOJI.get(r.get("severity","medium"),"•")
                a(f"  {i}. {sev}  {r.get('risk','')}")
                a(f"      Mitigation: {r.get('mitigation','')}")
        else:
            a("  None identified.")

        # ── Sentiment
        a(f"\n{'─'*68}")
        a("  🎭 SENTIMENT ANALYSIS")
        a(f"{'─'*68}")
        sa      = analysis.get("sentiment_analysis",{})
        overall = sa.get("overall_sentiment","neutral")
        score   = sa.get("sentiment_score",0)
        emoji   = SENTIMENT_EMOJI.get(overall,"")
        a(f"  Overall : {emoji} {overall.upper()}  (score: {score:+.2f})")
        a(f"  Energy  : {sa.get('energy_level','N/A').upper()}")
        a(f"  Tone    : {sa.get('tone_description','')}")
        for nm in sa.get("notable_moments",[]):
            ne = SENTIMENT_EMOJI.get(nm.get("sentiment","neutral"),"•")
            spk = f"  — {nm['speaker']}" if nm.get("speaker") else ""
            a(f"    {ne} {nm.get('moment','')}{spk}")

        # ── Open Questions
        a(f"\n{'─'*68}")
        a("  ❓ OPEN / UNRESOLVED QUESTIONS")
        a(f"{'─'*68}")
        qs = analysis.get("open_questions",[])
        if qs:
            for i, q in enumerate(qs, 1): a(f"  {i}. {q}")
        else:
            a("  None.")

    # ── Speaker Transcript
    a(f"\n{'═'*68}")
    a("  👥 SPEAKER-LABELED TRANSCRIPT")
    a(f"{'═'*68}")
    for line in speaker_tx.split("\n"):
        a(f"  {line}")

    # ── Full plain transcript
    a(f"\n{'═'*68}")
    a("  📄 FULL TRANSCRIPT (Original Language)")
    a(f"{'═'*68}")
    words = plain_tx.split()
    for i in range(0, len(words), 18):
        a("  " + " ".join(words[i:i+18]))
    a("=" * 68)

    report = "\n".join(lines)
    print(report)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n💾 Report saved → {OUTPUT_FILE}")


# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
def main():
    print("\n" + "═"*68)
    print("       🎙️  MeetIQ CLI v3.0 — Meeting Analyzer")
    print("       ✅ Speaker Detection  ✅ Email  ✅ 99+ Languages")
    print("═"*68)

    check_deps()

    if not Path(VIDEO_FILE).exists():
        print(f"\n❌ File not found: {VIDEO_FILE}")
        print(f"   Put the video in the same folder as this script.")
        sys.exit(1)

    # Step 1
    audio_path = extract_audio(VIDEO_FILE)

    # Step 2+3
    plain_tx, speaker_tx, labeled, lang = transcribe(audio_path)

    # Step 4
    analysis = analyze_with_groq(speaker_tx)
    print("✅ Analysis complete!\n")

    # Step 5
    send_email_summary(analysis, lang)

    # Step 6
    print_report(analysis, plain_tx, speaker_tx, labeled, lang)

    # Cleanup audio
    try: os.remove(audio_path)
    except: pass


if __name__ == "__main__":
    main()