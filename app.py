import os
import streamlit as st
import streamlit.components.v1 as components
import cv2
import groq
import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, AudioProcessorBase, RTCConfiguration
import av
import json
import numpy as np
import queue
import wave
import threading
import base64

# ==========================================
#           PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Mock Interviewer",
    page_icon="🎙️",
    layout="wide"
)

# ==========================================
#           CUSTOM CSS
# ==========================================

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        border-radius: 10px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: bold;
        border: none;
        width: 100%;
    }
    .stButton>button:hover { background-color: #ff2020; }
    .question-box {
        background-color: #1e2130;
        border-left: 4px solid #ff4b4b;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        font-size: 18px;
        color: white;
    }
    .feedback-box {
        background-color: #1e2130;
        border-left: 4px solid #00cc66;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        color: white;
    }
    .ideal-box {
        background-color: #1e2130;
        border-left: 4px solid #ffaa00;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
#           SESSION STATE
# ==========================================

if 'interview_started' not in st.session_state:
    st.session_state.job_role             = ""
    st.session_state.topic                = "General / Mixed"
    st.session_state.interview_started    = False
    st.session_state.current_question     = ""
    st.session_state.question_number      = 0
    st.session_state.max_questions        = 5
    st.session_state.current_level        = 1
    st.session_state.evaluation_data      = []
    st.session_state.conversation_history = []
    st.session_state.last_score           = None
    st.session_state.last_feedback        = ""
    st.session_state.last_ideal           = ""
    st.session_state.interview_done       = False
    st.session_state.recording            = False
    st.session_state.tts_b64              = ""
    st.session_state.speak_question       = False
    st.session_state.show_feedback        = False

# ==========================================
#           OPENCV FACE DETECTION SETUP
# ==========================================

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

# ==========================================
#           GROQ SETUP
# ==========================================

# ==========================================
#           GROQ SETUP
# ==========================================

import streamlit as st

# Try to get API key from Streamlit secrets first, then environment
try:
    GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", "")
except:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found. Please add it to Streamlit Secrets.")
    st.stop()

LEVELS = {
    1: "beginner - ask very basic conceptual questions only",
    2: "elementary - ask simple practical questions",
    3: "intermediate - ask moderate difficulty questions",
    4: "advanced - ask complex problem solving questions",
    5: "expert - ask deep architectural and system design questions"
}

def get_client():
    return groq.Groq(api_key=GROQ_API_KEY)

def get_interviewer_prompt(topic):
    topic_instructions = {
        "HR & Behavioral":                    "Focus on behavioral questions — tell me about yourself, strengths, weaknesses, conflict resolution, teamwork, leadership, career goals.",
        "General / Mixed":                    "Ask a balanced mix of role-specific technical and behavioral questions.",
        "Data Structures & Algorithms (DSA)": "Focus on arrays, strings, trees, graphs, sorting, searching, recursion, time and space complexity.",
        "System Design":                      "Focus on designing scalable systems, APIs, databases, load balancing, caching, microservices.",
        "Frontend Development":               "Focus on HTML, CSS, JavaScript, React/Vue/Angular, performance, accessibility, responsive design.",
        "Backend Development":                "Focus on APIs, server architecture, authentication, caching, databases, scalability.",
        "Database & SQL":                     "Focus on SQL queries, joins, normalization, indexes, transactions, NoSQL vs SQL.",
        "Machine Learning / AI":              "Focus on ML algorithms, model evaluation, overfitting, neural networks, feature engineering.",
        "DevOps & Cloud":                     "Focus on CI/CD, Docker, Kubernetes, AWS/GCP/Azure, monitoring, deployment pipelines.",
        "Marketing & Sales":                  "Focus on marketing strategies, campaign planning, lead generation, customer acquisition, brand building, sales techniques, CRM.",
        "Finance & Accounting":               "Focus on financial statements, budgeting, forecasting, accounting principles, taxation, auditing, financial analysis.",
        "Human Resources":                    "Focus on recruitment, employee relations, performance management, HR policies, onboarding, payroll, labor laws.",
        "Project Management":                 "Focus on project planning, risk management, agile/scrum, stakeholder communication, timelines, budgets, team coordination.",
        "Business Analysis":                  "Focus on requirements gathering, process mapping, stakeholder management, data analysis, business cases, SWOT analysis.",
        "Customer Service":                   "Focus on handling difficult customers, conflict resolution, communication skills, empathy, problem solving, service standards.",
        "Healthcare & Nursing":               "Focus on patient care, medical procedures, clinical knowledge, emergency response, ethics, documentation, teamwork.",
        "Teaching & Education":               "Focus on lesson planning, classroom management, student engagement, assessment methods, curriculum design, communication.",
        "Law & Legal":                        "Focus on legal principles, case analysis, contract law, client communication, research, ethics, courtroom procedures.",
        "Design & UX":                        "Focus on design principles, user research, wireframing, prototyping, usability testing, design tools, visual hierarchy.",
        "Operations & Logistics":             "Focus on supply chain management, inventory control, process optimization, vendor management, logistics planning.",
    }
    instruction = topic_instructions.get(topic, topic_instructions["General / Mixed"])
    return f"""You are a professional job interviewer conducting a mock interview.
Rules:
- Ask exactly ONE clear, practical interview question at a time
- NEVER ask personal questions like hobbies or favorite things
- Topic focus: {instruction}
- Level 1: Very basic conceptual questions only
- Level 2: Simple practical questions
- Level 3: Moderate practical usage questions
- Level 4: Complex problem solving questions
- Level 5: Deep architectural / expert questions
- Do NOT repeat questions already asked
- Be warm, supportive and professional
"""

EVALUATOR_PROMPT = """You are a fair and encouraging interview evaluator for ANY job field.
Return ONLY a JSON object like this:
{
    "score": 7,
    "level_recommendation": "increase",
    "feedback": "Good understanding. Could elaborate more on X.",
    "keywords_mentioned": ["keyword1"],
    "missing_concepts": ["concept1"]
}
Scoring: 8-10 excellent, 6-7 good, 4-5 partial, 2-3 weak, 1 off-topic.
level_recommendation: "increase" if score>=7, "same" if 4-6, "decrease" if <=3
Return ONLY JSON, no other text."""

# ==========================================
#           AI FUNCTIONS
# ==========================================

def ask_question(job_role, level, conversation_history, question_number, max_questions):
    client   = get_client()
    messages = [
        {"role": "system", "content": get_interviewer_prompt(st.session_state.topic)},
        *conversation_history,
        {"role": "user", "content": f"""
Job role: {job_role}
Difficulty: {LEVELS[level]}
Question {question_number + 1} of {max_questions}
Ask one clear practical interview question. Keep it concise.
"""}
    ]
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        max_tokens=200,
        temperature=0.7
    )
    return response.choices[0].message.content

def evaluate_answer(question, answer, job_role):
    client   = get_client()
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": EVALUATOR_PROMPT},
            {"role": "user",   "content": f"Job Role: {job_role}\nQuestion: {question}\nAnswer: {answer}"}
        ],
        max_tokens=300,
        temperature=0.3
    )
    raw = response.choices[0].message.content
    try:
        start = raw.find('{')
        end   = raw.rfind('}') + 1
        return json.loads(raw[start:end])
    except:
        return {"score": 5, "level_recommendation": "same",
                "feedback": raw, "keywords_mentioned": [], "missing_concepts": []}

def get_ideal_answer(question, job_role):
    client   = get_client()
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are an expert interview coach. Give a concise ideal answer in 3-4 sentences."},
            {"role": "user",   "content": f"Job Role: {job_role}\nQuestion: {question}\nIdeal answer:"}
        ],
        max_tokens=200,
        temperature=0.3
    )
    return response.choices[0].message.content

# ==========================================
#           TTS
# ==========================================

ELEVENLABS_API_KEY = os.environ.get("ELEVENLABS_API_KEY", "")
ELEVENLABS_VOICE_ID = "EXAVITQu4vr4xnSDxMaL"

def generate_tts_b64(text):
    if not ELEVENLABS_API_KEY:
        return ""
    try:
        import urllib.request
        url     = f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}"
        payload = json.dumps({
            "text": text[:400],
            "model_id": "eleven_turbo_v2",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}
        }).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={
            "xi-api-key":   ELEVENLABS_API_KEY,
            "Content-Type": "application/json",
            "Accept":       "audio/mpeg"
        })
        with urllib.request.urlopen(req) as resp:
            audio_bytes = resp.read()
        return base64.b64encode(audio_bytes).decode("utf-8")
    except Exception:
        return ""

def play_audio_in_browser(b64_audio, fallback_text=""):
    if b64_audio:
        components.html(f"""
        <audio autoplay style="display:none">
            <source src="data:audio/mpeg;base64,{b64_audio}" type="audio/mpeg">
        </audio>
        """, height=0)
    elif fallback_text:
        safe = fallback_text.replace("\\", "\\\\").replace("'", "\\'").replace("\n", " ")
        components.html(f"""
        <script>
        (function() {{
            window.speechSynthesis.cancel();
            var msg = new SpeechSynthesisUtterance('{safe}');
            msg.rate = 0.90; msg.pitch = 1.05; msg.volume = 1.0; msg.lang = 'en-US';
            function speak() {{
                var voices = window.speechSynthesis.getVoices();
                var preferred = voices.find(function(v) {{
                    return v.name === 'Google US English' || v.name.includes('Google');
                }});
                if (preferred) msg.voice = preferred;
                window.speechSynthesis.speak(msg);
            }}
            if (window.speechSynthesis.getVoices().length > 0) speak();
            else window.speechSynthesis.onvoiceschanged = speak;
        }})();
        </script>
        """, height=0)

# ==========================================
#           WEBRTC AUDIO PROCESSOR
# ==========================================

audio_queue = queue.Queue()

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.recording   = False
        self.chunks      = []
        self._lock       = threading.Lock()
        self.sample_rate = 48000

    def recv(self, frame: av.AudioFrame):
        with self._lock:
            if self.recording:
                try:
                    f16 = frame.reformat(format="s16", layout="mono")
                    pcm = f16.to_ndarray()
                    if pcm.ndim > 1:
                        pcm = pcm[0]
                    self.chunks.append(pcm.copy())
                    self.sample_rate = frame.sample_rate or 48000
                except Exception:
                    pass
        return frame

    def start_recording(self):
        with self._lock:
            self.chunks    = []
            self.recording = True

    def stop_and_get(self):
        with self._lock:
            self.recording = False
            chunks         = list(self.chunks)
            src_rate       = self.sample_rate
            self.chunks    = []
        return chunks, src_rate

# ==========================================
#           OPENCV VIDEO PROCESSOR
# ==========================================

def process_frame_opencv(frame):
    """Use OpenCV Haar cascades instead of MediaPipe."""
    try:
        if frame is None:
            return None, False, False, "neutral"
        
        h, w    = frame.shape[:2]
        frame   = cv2.flip(frame, 1)
        gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_detected = False
        eye_contact   = False
        expression    = "neutral"

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        if len(faces) > 0:
            face_detected = True
            x, y, fw, fh = faces[0]
            cv2.rectangle(frame, (x, y), (x + fw, y + fh), (0, 255, 0), 2)

            # Detect eyes within the face region
            face_gray   = gray[y:y+fh, x:x+fw]
            face_color  = frame[y:y+fh, x:x+fw]
            eyes        = eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5)

            if len(eyes) >= 2:
                eye_contact = True
                for (ex, ey, ew, eh) in eyes[:2]:
                    cv2.rectangle(face_color, (ex, ey), (ex+ew, ey+eh), (255, 100, 0), 2)

            # Simple expression heuristic: face height/width ratio
            ratio = fh / fw if fw > 0 else 1
            if ratio > 1.35:
                expression = "confident"
            elif ratio < 1.1:
                expression = "nervous"
            else:
                expression = "neutral"

        if not face_detected:
            cv2.putText(frame, "No face detected!", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            ec_color = (0, 255, 0) if eye_contact else (0, 165, 255)
            cv2.putText(frame, "Eye Contact: GOOD" if eye_contact else "Look at Camera!",
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ec_color, 2)
            exp_map = {
                "confident": ("Confident", (0, 255, 0)),
                "neutral":   ("Neutral",   (255, 255, 0)),
                "nervous":   ("Nervous",   (0, 0, 255))
            }
            cv2.putText(frame, exp_map[expression][0], (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, exp_map[expression][1], 2)

        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), eye_contact, face_detected, expression
    
    except Exception as e:
        st.error(f"Frame processing error: {e}")
        return None, False, False, "neutral"


class InterviewVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.eye_contact   = False
        self.face_detected = False
        self.expression    = "neutral"

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            result = process_frame_opencv(img)
            if result[0] is None:
                return frame
            processed, ec, fd, exp = result
            self.eye_contact   = ec
            self.face_detected = fd
            self.expression    = exp
            return av.VideoFrame.from_ndarray(processed, format="rgb24")
        except Exception as e:
            print(f"Video processor error: {e}")
            return frame

# ==========================================
#           HELPERS
# ==========================================

def score_color(s):
    if s >= 8: return "#00cc66"
    if s >= 6: return "#ffaa00"
    if s >= 4: return "#ff6600"
    return "#ff0000"

def score_emoji(s):
    if s >= 8: return "🔥"
    if s >= 6: return "✅"
    if s >= 4: return "⚠️"
    return "❌"

def level_badge(l):
    return {1:"🟢 Beginner",2:"🟡 Elementary",3:"🟠 Intermediate",
            4:"🔴 Advanced",5:"🔥 Expert"}.get(l,"🟢 Beginner")

def submit_answer(answer, job_role, level, max_q):
    st.session_state.conversation_history.append({"role": "user", "content": answer})
    with st.spinner("📊 Evaluating your answer..."):
        ev    = evaluate_answer(st.session_state.current_question, answer, job_role)
        ideal = get_ideal_answer(st.session_state.current_question, job_role)

    score = ev.get('score', 5)
    fb    = ev.get('feedback', '')
    rec   = ev.get('level_recommendation', 'same')

    st.session_state.last_score    = score
    st.session_state.last_feedback = fb
    st.session_state.last_ideal    = ideal
    st.session_state.last_answer   = answer

    st.session_state.evaluation_data.append({
        'question': st.session_state.current_question,
        'answer':   answer,
        'score':    score,
        'feedback': fb,
        'level':    level,
        'missing':  ev.get('missing_concepts', [])
    })

    if rec == "increase" and level < 5:
        st.session_state.current_level += 1
    elif rec == "decrease" and level > 1:
        st.session_state.current_level -= 1

    st.session_state.recording      = False
    st.session_state.show_feedback  = True
    st.rerun()

# ==========================================
#           HEADER
# ==========================================

st.markdown("""
<h1 style='text-align:center; color:#ff4b4b;'>🎙️ AI Mock Interviewer</h1>
<p style='text-align:center; color:#888; font-size:16px;'>
    Adaptive difficulty &bull; Real-time feedback &bull; Face detection
</p>
""", unsafe_allow_html=True)
st.divider()

# ==========================================
#           SCREEN 1 — SETUP
# ==========================================

if not st.session_state.interview_started:
    _, col, _ = st.columns([1, 2, 1])
    with col:
        st.markdown("### 👔 Setup Your Interview")
        job_role = st.text_input("Enter your target job role:",
                                 placeholder="e.g. Frontend Developer, Nurse, Marketing Manager...")
        num_q    = st.slider("Number of questions:", 3, 10, 5)
        topic    = st.selectbox("Interview Topic:", [
            "General / Mixed", "HR & Behavioral",
            "Data Structures & Algorithms (DSA)", "System Design",
            "Frontend Development", "Backend Development",
            "Database & SQL", "Machine Learning / AI", "DevOps & Cloud",
            "Marketing & Sales", "Finance & Accounting", "Human Resources",
            "Project Management", "Business Analysis", "Customer Service",
            "Healthcare & Nursing", "Teaching & Education", "Law & Legal",
            "Design & UX", "Operations & Logistics",
        ])
        st.markdown("---")
        st.markdown("**What to expect:**")
        st.markdown("- 🟢 Starts at beginner, gets harder as you improve")
        st.markdown("- 🎤 Speak your answer directly into the mic")
        st.markdown("- 🔊 AI reads questions aloud")
        st.markdown("- 📊 Scored after every answer")
        st.markdown("- 📄 Full report with chart at the end")

        if st.button("🚀 Start Interview"):
            if job_role.strip():
                st.session_state.job_role             = job_role
                st.session_state.topic                = topic
                st.session_state.max_questions        = num_q
                st.session_state.interview_started    = True
                st.session_state.interview_done       = False
                st.session_state.question_number      = 0
                st.session_state.current_level        = 1
                st.session_state.evaluation_data      = []
                st.session_state.conversation_history = []
                st.session_state.last_score           = None
                st.session_state.last_feedback        = ""
                st.session_state.last_ideal           = ""
                st.session_state.current_question     = ""
                st.session_state.recording            = False
                st.rerun()
            else:
                st.error("Please enter a job role!")

# ==========================================
#           SCREEN 2 — INTERVIEW
# ==========================================

elif st.session_state.interview_started and not st.session_state.interview_done:

    job_role = st.session_state.job_role
    q_num    = st.session_state.question_number
    max_q    = st.session_state.max_questions
    level    = st.session_state.current_level

    st.progress(q_num / max_q,
                text=f"Question {q_num}/{max_q} — {level_badge(level)}")

    # ---- FEEDBACK SCREEN ----
    if st.session_state.get("show_feedback", False):
        score = st.session_state.last_score
        color = score_color(score)
        emoji = score_emoji(score)

        st.markdown(f"""
        <div style='background:{color}22; border:2px solid {color};
                    padding:20px; border-radius:12px; margin:10px 0; text-align:center;'>
            <h2 style='color:{color};'>{emoji} Score: {score}/10</h2>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='question-box'>
            <b>❓ Question:</b><br>{st.session_state.current_question}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style='background:#1e2130; border-left:4px solid #888;
                    padding:15px; border-radius:8px; margin:8px 0; color:white;'>
            <b>🎤 Your Answer:</b><br>{st.session_state.get("last_answer", "")}
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class='feedback-box'>
            💬 <b>Feedback:</b><br>{st.session_state.last_feedback}
        </div>
        """, unsafe_allow_html=True)

        with st.expander("💡 See Ideal Answer", expanded=True):
            st.markdown(f"""
            <div class='ideal-box'>{st.session_state.last_ideal}</div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        q_num_after = q_num + 1
        if q_num_after >= max_q:
            if st.button("📄 Finish & Get Report"):
                st.session_state.question_number  += 1
                st.session_state.current_question  = ""
                st.session_state.show_feedback     = False
                st.session_state.interview_done    = True
                st.rerun()
        else:
            if st.button(f"➡️ Next Question ({q_num_after}/{max_q})"):
                st.session_state.question_number  += 1
                st.session_state.current_question  = ""
                st.session_state.tts_b64           = ""
                st.session_state.show_feedback     = False
                st.rerun()

        st.stop()

    cam_col, interview_col = st.columns([1, 1])

    # ---- CAMERA ----
    with cam_col:
        st.markdown("### 📷 Live Camera")

        RTC_CONFIG = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })

        ctx = webrtc_streamer(
            key="interview",
            video_processor_factory=InterviewVideoProcessor,
            audio_processor_factory=AudioProcessor,
            rtc_configuration=RTC_CONFIG,
            media_stream_constraints={"video": True, "audio": True},
            async_processing=False
        )

        if ctx.video_processor:
            p = ctx.video_processor
            st.markdown("---")
            m1, m2, m3 = st.columns(3)
            m1.metric("Face",        "✅" if p.face_detected else "❌")
            m2.metric("Eye Contact", "Good 👁️" if p.eye_contact else "Away 👀")
            m3.metric("Expression",  p.expression.capitalize())
        else:
            st.info("👆 Click START to turn on camera & mic")

    # ---- INTERVIEW ----
    with interview_col:
        st.markdown("### 🤖 Interview")

        if not st.session_state.current_question:
            with st.spinner("AI is thinking of a question..."):
                q = ask_question(job_role, level,
                                 st.session_state.conversation_history,
                                 q_num, max_q)
                st.session_state.current_question = q
                st.session_state.conversation_history.append(
                    {"role": "assistant", "content": q}
                )
            with st.spinner("🔊 Generating voice..."):
                st.session_state.tts_b64 = generate_tts_b64(q)
                st.session_state.speak_question = True

        if st.session_state.get("speak_question", False):
            play_audio_in_browser(st.session_state.get("tts_b64", ""), st.session_state.current_question)
            st.session_state.speak_question = False

        st.markdown(f"""
        <div class='question-box'>
            <b>Question {q_num + 1}:</b><br><br>
            {st.session_state.current_question}
        </div>
        """, unsafe_allow_html=True)

        if st.button("🔊 Replay Question"):
            play_audio_in_browser(st.session_state.get("tts_b64", ""), st.session_state.current_question)

        if st.session_state.last_score is not None:
            score = st.session_state.last_score
            color = score_color(score)
            emoji = score_emoji(score)
            st.markdown(f"""
            <div style='background:{color}22; border-left:4px solid {color};
                        padding:12px; border-radius:8px; margin:8px 0;'>
                <b>{emoji} Previous Score: {score}/10</b>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("#### 🎤 Your Answer")

        components.html("""
        <div style="font-family:sans-serif;padding:10px;">
            <button id="startBtn" onclick="startRec()"
                style="background:#ff4b4b;color:white;border:none;padding:10px 20px;
                       border-radius:8px;font-size:15px;cursor:pointer;margin:5px;">
                🎤 Start Recording
            </button>
            <button id="stopBtn" onclick="stopRec()" disabled
                style="background:#555;color:white;border:none;padding:10px 20px;
                       border-radius:8px;font-size:15px;cursor:pointer;margin:5px;">
                ⏹️ Stop
            </button>
            <div id="status" style="margin-top:8px;color:#aaa;font-size:13px;">
                Press Start and speak your answer
            </div>
            <audio id="preview" controls style="margin-top:10px;width:100%;display:none;"></audio>
            <div id="dlWrapper" style="margin-top:8px;"></div>
        </div>
        <script>
        let mediaRecorder, chunks = [];
        async function startRec() {
            chunks = [];
            const stream = await navigator.mediaDevices.getUserMedia({audio:true});
            mediaRecorder = new MediaRecorder(stream, {mimeType:'audio/webm'});
            mediaRecorder.ondataavailable = e => chunks.push(e.data);
            mediaRecorder.onstop = () => {
                const blob = new Blob(chunks, {type:'audio/webm'});
                const url  = URL.createObjectURL(blob);
                document.getElementById('preview').src = url;
                document.getElementById('preview').style.display = 'block';
                const a = document.createElement('a');
                a.href = url; a.download = 'answer.webm';
                a.innerHTML = '⬇️ Download answer.webm (then upload below)';
                a.style.cssText = 'display:inline-block;background:#00cc66;color:white;padding:8px 16px;border-radius:6px;text-decoration:none;margin-top:4px;';
                document.getElementById('dlWrapper').innerHTML = '';
                document.getElementById('dlWrapper').appendChild(a);
                document.getElementById('status').textContent = '✅ Done! Download the file then upload it below.';
                stream.getTracks().forEach(t => t.stop());
            };
            mediaRecorder.start();
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
            document.getElementById('stopBtn').style.background = '#ff4b4b';
            document.getElementById('status').textContent = '🔴 Recording... speak now!';
        }
        function stopRec() {
            mediaRecorder.stop();
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            document.getElementById('stopBtn').style.background = '#555';
        }
        </script>
        """, height=240)

        uploaded = st.file_uploader(
            "📁 Upload your recorded answer.webm here:",
            type=["webm", "wav", "mp3", "m4a", "ogg"],
            key=f"audio_upload_{q_num}"
        )

        if uploaded:
            st.audio(uploaded)
            if st.button("✅ Submit Answer"):
                with st.spinner("🤖 Transcribing with Groq Whisper..."):
                    try:
                        import tempfile
                        suffix = "." + uploaded.name.split(".")[-1]
                        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                            tmp.write(uploaded.read())
                            tmp_path = tmp.name
                        client = get_client()
                        with open(tmp_path, "rb") as f:
                            result = client.audio.transcriptions.create(
                                model="whisper-large-v3",
                                file=(uploaded.name, f, "audio/webm"),
                                language="en",
                                response_format="text"
                            )
                        os.remove(tmp_path)
                        answer = result.strip() if isinstance(result, str) else result.text.strip()
                    except Exception as e:
                        st.error(f"Transcription error: {e}")
                        answer = ""

                if answer and len(answer) > 3:
                    st.info(f"📝 **You said:** {answer}")
                    submit_answer(answer, job_role, level, max_q)
                else:
                    st.warning("⚠️ Couldn't transcribe. Try speaking louder or re-recording.")

        st.markdown("---")

        c1, c2 = st.columns(2)
        with c1:
            if st.button("⏭️ Skip Question"):
                st.session_state.question_number  += 1
                st.session_state.current_question  = ""
                st.session_state.recording         = False
                if st.session_state.question_number >= max_q:
                    st.session_state.interview_done = True
                st.rerun()
        with c2:
            if st.button("🛑 End & Get Report"):
                st.session_state.interview_done = True
                st.rerun()

# ==========================================
#           SCREEN 3 — REPORT
# ==========================================

elif st.session_state.interview_done:
    st.markdown("## 📄 Interview Complete!")

    data = st.session_state.evaluation_data

    if data:
        avg   = sum(e['score'] for e in data) / len(data)
        ready = "🟢 READY" if avg >= 7 else "🟡 ALMOST READY" if avg >= 5 else "🔴 NEEDS MORE PREP"

        c1, c2, c3 = st.columns(3)
        c1.metric("Average Score",      f"{avg:.1f}/10")
        c2.metric("Questions Answered", len(data))
        c3.metric("Status",             ready)

        st.divider()
        st.markdown("### 📊 Score Breakdown")

        labels = [f"Q{i+1}" for i in range(len(data))]
        scores = [e['score'] for e in data]
        levels = [e['level'] for e in data]
        colors = [score_color(s) for s in scores]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=labels, y=scores, marker_color=colors,
                             name="Your Score", text=scores, textposition='outside'))
        fig.add_hline(y=avg, line_dash="dash", line_color="white",
                      annotation_text=f"Avg: {avg:.1f}", annotation_position="top right")
        fig.add_trace(go.Scatter(x=labels, y=levels, mode='lines+markers',
                                 name="Difficulty", line=dict(color='#ff4b4b', width=2),
                                 yaxis='y2'))
        fig.update_layout(
            title="Performance vs Difficulty",
            paper_bgcolor='#0e1117', plot_bgcolor='#1e2130',
            font=dict(color='white'),
            yaxis=dict(title="Score (0-10)", range=[0, 11], gridcolor='#333'),
            yaxis2=dict(title="Difficulty", overlaying='y', side='right',
                        range=[0, 6], gridcolor='#333'),
            legend=dict(bgcolor='#1e2130'), height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        for i, e in enumerate(data):
            color = score_color(e['score'])
            emoji = score_emoji(e['score'])
            st.markdown(f"""
            <div style='background:{color}22; border-left:4px solid {color};
                        padding:12px; border-radius:8px; margin:8px 0;'>
                <b>Q{i+1} (Level {e['level']}):</b> {emoji} {e['score']}/10<br>
                <small><b>Q:</b> {e['question'][:100]}...</small><br>
                <small>💬 {e['feedback']}</small>
            </div>
            """, unsafe_allow_html=True)

        st.divider()
        st.markdown("### 📋 Full AI Report")
        with st.spinner("Generating report..."):
            all_fb = "\n".join([
                f"Q{i+1} (Level {e['level']}): Score {e['score']}/10 - {e['feedback']}"
                for i, e in enumerate(data)
            ])
            client   = get_client()
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are an expert career coach. Write a detailed encouraging interview performance report."},
                    {"role": "user",   "content": f"""
Job Role: {st.session_state.job_role}
Topic: {st.session_state.topic}
Average Score: {avg:.1f}/10
Questions: {len(data)}
Evaluations:
{all_fb}

Write a report with:
1. Overall Performance Summary
2. Key Strengths
3. Areas for Improvement
4. Study Recommendations
5. Interview Readiness Verdict
"""}
                ],
                max_tokens=700
            )
            report = response.choices[0].message.content

        st.markdown(report)

        report_txt = f"""AI MOCK INTERVIEW REPORT
{'='*50}
Job Role : {st.session_state.job_role}
Topic    : {st.session_state.topic}
Score    : {avg:.1f}/10
Status   : {ready}

SCORE BREAKDOWN:
{'='*50}
"""
        for i, e in enumerate(data):
            report_txt += f"Q{i+1} (Level {e['level']}): {e['score']}/10\n"
            report_txt += f"   Q: {e['question']}\n"
            report_txt += f"   A: {e['answer']}\n"
            report_txt += f"   Feedback: {e['feedback']}\n\n"
        report_txt += f"\nFULL REPORT:\n{'='*50}\n{report}\n"

        st.download_button("📥 Download Full Report",
                           data=report_txt,
                           file_name="interview_report.txt",
                           mime="text/plain")
    else:
        st.warning("No questions were answered.")

    st.divider()
    if st.button("🔄 Start New Interview"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
