import json
import logging
import requests
from core.config import settings

logger = logging.getLogger("meetiq.gemini")

ANALYSIS_SYSTEM_INSTRUCTION = """You are an expert multilingual meeting intelligence analyst and corporate productivity optimizer.
The provided transcript may be in any language (English, Tamil, Hindi, Spanish, mixed language, etc.).
Extract EVERY important detail and produce the ENTIRE output in ENGLISH ONLY.
Be meticulous, professional, and comprehensive.

Return a valid JSON object matching this exact schema:
{
  "meeting_overview": {
    "main_topic": "<Clear, professional 1-line English title of the meeting>",
    "estimated_duration_minutes": <number or null>,
    "participant_count": <number or null>,
    "meeting_type": "standup|planning|review|brainstorm|interview|presentation|other",
    "speakers_identified": ["<speaker name or label>"]
  },
  "summary": "<3-5 sentence detailed executive summary in English capturing core objectives, outcomes, and progress>",
  "speaker_contributions": [
    {
      "speaker": "<Speaker name or label>",
      "key_points": ["<Key contribution in English>"],
      "sentiment": "positive|neutral|negative",
      "talk_time_percent": <estimated percentage 0-100>
    }
  ],
  "key_discussion_points": [
    {
      "topic": "<Discussion topic title>",
      "summary": "<2-3 sentence summary of what was discussed>",
      "importance": "high|medium|low",
      "category": "technical|business|operations|finance|hr|other",
      "speaker": "<who raised or led this>"
    }
  ],
  "decisions_made": [
    {
      "decision": "<Clear statement of decision>",
      "context": "<Why this was decided>",
      "owner": "<Person or team responsible>",
      "deadline": "<Date, timeframe or null>",
      "decided_by": "<Speaker>"
    }
  ],
  "action_items": [
    {
      "task": "<Specific actionable task in English>",
      "owner": "<Person responsible or TBD>",
      "priority": "urgent|high|medium|low",
      "deadline": "<Specific deadline or null>",
      "category": "development|research|follow_up|meeting|review|other",
      "estimated_hours": <number or null>,
      "assigned_by": "<Speaker>"
    }
  ],
  "optimization_suggestions": [
    {
      "area": "<Process or technical area>",
      "issue": "<Identified friction or inefficiency>",
      "suggestion": "<Actionable improvement>",
      "impact": "high|medium|low"
    }
  ],
  "sentiment_analysis": {
    "overall_sentiment": "positive|neutral|mixed|negative",
    "sentiment_score": <float from -1.0 to 1.0>,
    "tone_description": "<2-3 sentences describing team dynamic, engagement, and atmosphere>",
    "energy_level": "high|medium|low",
    "notable_moments": [
      {"moment": "<Specific exchange>", "sentiment": "positive|neutral|negative", "speaker": "<who>"}
    ]
  },
  "open_questions": ["<Unresolved question or blocker that needs follow up>"],
  "risks": [
    {"risk": "<Identified risk or challenge>", "severity": "high|medium|low", "mitigation": "<Proposed mitigation>"}
  ]
}
"""

def analyze_transcript_with_gemini(transcript: str, target_language: str = "English") -> dict:
    """Analyze full meeting transcript using Google Gemini, translating output to target_language"""
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not configured in .env")

    models_to_try = [
        "gemini-3.5-flash",
        "gemini-3.7-flash",
        "gemini-3.5-flash-lite",
        "gemini-flash-lite-latest"
    ]

    instruction = ANALYSIS_SYSTEM_INSTRUCTION.replace("ENGLISH ONLY", f"{target_language.upper()} ONLY")
    prompt = f"{instruction}\n\nTARGET OUTPUT LANGUAGE: {target_language}\n\nTRANSCRIPT:\n{transcript}"

    last_error = None
    for model_name in models_to_try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={settings.gemini_api_key}"
        payload = {
            "contents": [
                {
                    "parts": [{"text": prompt}]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json"
            }
        }
        
        try:
            logger.info(f"Calling Google Gemini API ({model_name})...")
            response = requests.post(url, json=payload, timeout=90)
            if response.status_code == 200:
                data = response.json()
                candidate_text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
                result = json.loads(candidate_text)
                logger.info(f"Gemini analysis completed successfully with {model_name}")
                return result
            else:
                err_msg = f"Gemini {model_name} HTTP {response.status_code}: {response.text}"
                logger.warning(err_msg)
                last_error = err_msg
        except Exception as e:
            logger.warning(f"Error calling {model_name}: {str(e)}")
            last_error = str(e)

    raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")
