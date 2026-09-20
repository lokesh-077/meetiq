#  MeetIQ — Autonomous Meeting Intelligence System

> **MeetIQ** is like having your own personal, silent AI assistant that attends your Google Meet calls (or listens to recorded meetings/lectures), writes down who said what, highlights key action items, and automatically delivers a clean executive PDF report straight to your email inbox!

---

##  What is MeetIQ? (Explained Simply)

Imagine you are in a 1-hour college lecture or a busy team project meeting:
- You have to listen, take notes, remember who agreed to do which assignment, and keep track of deadlines.
- With **MeetIQ**, you don't have to stress about taking notes!
- Just paste your Google Meet link or drop a video/audio recording into MeetIQ.
- MeetIQ will listen to the conversation, identify different speakers, translate any foreign language (like Portuguese, Tamil, Spanish, etc.) into English, and create a **professional PDF report** with:
  1. **Executive Overview**: A quick recap of what the meeting was about.
  2. **Action Items Checklist**: Exactly who has to do what, and by when.
  3. **Key Decisions**: Decisions agreed upon during the discussion.
  4. **Section 6 (Verbatim Spoken Dialogue)**: Word-for-word transcript of what was spoken.
  5. **Direct Email Delivery**: The finished PDF is sent directly to your Gmail inbox!

---

##  Key Features

-  **Autonomous Google Meet Scribe**: Paste a Google Meet link, and MeetIQ launches a silent bot in Chrome that joins the meeting, reads live closed captions, and concludes automatically when you're done.
-  **Upload Any Recording**: Drag and drop existing `.mp4`, `.mp3`, `.wav`, or `.m4a` files.
-  **99+ Language Support**: Detects and translates foreign meetings (Portuguese, Spanish, French, German, Tamil, Hindi, Japanese, etc.) into your preferred report language.
-  **Executive PDF Generator**: Generates formatted, multi-section PDF reports ready to share with managers, professors, or team members.
-  **Silent Gmail Dispatcher**: Sends the PDF report silently via Gmail SMTP without dumping noisy text logs onto your screen.
-  **Futuristic Glassmorphic Dashboard**: Dark cyber UI with real-time audio visualizer, live closed captions stream, and an archive to preview and download past reports.

---

##  Tech Stack

MeetIQ is built with modern, reliable, and lightweight open-source technologies:

| Component | Technology | Why We Use It |
| :--- | :--- | :--- |
| **Backend API** | **Python 3.10+ & FastAPI** | Fast, asynchronous web server and REST API. |
| **AI Intelligence** | **Google Gemini 3.5 Flash** | Ultra-fast AI that synthesizes transcripts, extracts tasks, and translates. |
| **Speech-to-Text** | **OpenAI Whisper & FFmpeg** | High-fidelity 16kHz audio extraction and multi-language transcription. |
| **Browser Bot** | **Playwright (Native Chrome)** | Autonomously opens Google Meet, mutes mic/cam, and reads live captions. |
| **PDF Engine** | **FPDF2** | Generates executive PDF documents with tables, badges, and transcripts. |
| **Email Delivery** | **Python `smtplib` (SSL)** | Silently delivers PDF attachments via Gmail SMTP (`smtp.gmail.com:465`). |
| **Frontend UI** | **Vanilla HTML5, CSS3, JS** | High-performance cyber-glassmorphic UI with zero bulky framework dependencies. |

---

## 📁 Repository Structure

```text
meetiq/
├── 📁 app/               # Web Application Dashboard & FastAPI Server
│   ├── index.html        # Interactive Cyber-Glassmorphic UI
│   ├── main.py           # Application Entry Point
│   └── requirements.txt  # Python Dependencies
├── 📁 core/              # Core Intelligence Engines
│   ├── audio.py          # Whisper 16kHz Diarization
│   ├── bot.py            # Playwright Google Meet Automation
│   ├── config.py         # Settings & Config Loader
│   ├── gemini.py         # Gemini 3.5 Multilingual Synthesis
│   ├── mailer.py         # Silent Gmail SMTP Dispatcher
│   └── pdf_generator.py  # Multi-Section Executive PDF Generator
├── 📁 docs/              # Documentation & Architecture
│   ├── ARCHITECTURE.md   # Pipeline Flowcharts & System Design
│   └── API_REFERENCE.md  # REST API Endpoints Specification
├── 📁 outputs/           # Generated Executive PDF Reports (.gitkeep)
├── 📁 prompts/           # AI Prompt Schemas & Guidelines
│   └── meeting_analysis.json
├── 📁 samples/           # Sample Test Media (.gitkeep)
├── 📁 uploads/           # Audio Buffer (.gitkeep)
├── .env.example          # Configuration Template
├── .gitignore            # Security & Git Rules
└── README.md             # Master Documentation
```

---

##  How to Run MeetIQ (Step-by-Step)

Follow these simple steps to run MeetIQ on your computer:

### 1. Prerequisites
- [Python 3.10+](https://www.python.org/downloads/) installed.
- [Google Chrome](https://www.google.com/chrome/) installed.
- A free **Google Gemini API Key** (get one at [Google AI Studio](https://aistudio.google.com/app/apikey)).
- A **Gmail App Password** (for automated email delivery).

---

### 2. Setup the Project

1. **Clone the repository:**
   ```bash
   git clone https://github.com/lokesh-077/meetiq.git
   cd meetiq
   ```

2. **Create and activate a virtual environment:**
   * **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   * **macOS / Linux:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r app/requirements.txt
   playwright install
   ```

---

### 3. Configure Environment Variables (`.env`)

Create a `.env` file in the project root by copying `.env.example`:
```bash
cp .env.example .env
```

Open `.env` in any text editor and fill in your keys:
```env
# Google Gemini API Key (Free from https://aistudio.google.com/app/apikey)
GEMINI_API_KEY=your_gemini_api_key_here

# Gmail SMTP Settings (Your email + 16-character Google App Password)
EMAIL_SENDER=your_email@gmail.com
EMAIL_PASSWORD=your_16_digit_app_password
EMAIL_TO=recipient@example.com

# Server Settings (Defaults work out of the box)
PORT=8000
HOST=0.0.0.0
BOT_NAME=MeetIQ Notetaker
WHISPER_MODEL=small
```

---

### 4. Start the Application

Run the server with Python:
```bash
python app/main.py
```

Now open your web browser and navigate to:
👉 **`http://localhost:8000`**

---

##  How to Use the App

1. **Autonomous Google Meet Bot**:
   - Go to the **Autonomous Bot** tab.
   - Paste a Google Meet link (e.g., `https://meet.google.com/xxx-yyyy-zzz`).
   - Select your target language (e.g., English or Portuguese).
   - Click **"Send Bot to Google Meet"**.
   - When the meeting finishes, click **"Conclude & Compile PDF"** to receive your report by email!

2. **Upload Meeting Recording**:
   - Go to the **Upload Recording** tab.
   - Drag and drop your `.mp4` or `.mp3` file.
   - Click **"Process & Deliver PDF"**.

3. **Browse Past Reports**:
   - Open the **Report Archive** tab to preview executive summaries, interactive action checklists, and full transcripts on screen, or download PDFs directly.

---

##  Privacy & Security

- **No Secret Leaks**: All credentials remain local in your `.env` file (protected by `.gitignore`).
- **Clean Storage**: Meeting audio and generated files are stored locally and never shared with third parties.
- **Silent Operation**: All reports are delivered privately via email.

---

##  License
This project is open-source and available under the [MIT License](LICENSE).
