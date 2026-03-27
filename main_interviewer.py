import cv2
import mediapipe as mp
import groq
import whisper
import pyaudio
import wave
import edge_tts
import asyncio
import os
from gtts import gTTS
import playsound
import json
import threading
import time

# ==========================================
#           SETUP
# ==========================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = groq.Groq(api_key=GROQ_API_KEY)
# Text to Speech setup
# 0=male, 1=female
def speak(text):
    """Speak text using Edge TTS - fast and natural"""
    def _speak():
        try:
            import asyncio
            import edge_tts
            
            async def _generate():
                communicate = edge_tts.Communicate(text[:200], voice="en-US-JennyNeural")
                await communicate.save("temp_speech.mp3")
            
            asyncio.run(_generate())
            playsound.playsound("temp_speech.mp3")
            if os.path.exists("temp_speech.mp3"):
                os.remove("temp_speech.mp3")
        except Exception as e:
            pass
    
    t = threading.Thread(target=_speak, daemon=True)
    t.start()
    return t
client = groq.Groq(api_key=GROQ_API_KEY)
whisper_model = whisper.load_model("base")

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmarks
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

# ==========================================
#           INTERVIEW CONFIG
# ==========================================

LEVELS = {
    1: "beginner - ask very basic conceptual questions only",
    2: "elementary - ask simple practical questions",
    3: "intermediate - ask moderate difficulty questions",
    4: "advanced - ask complex problem solving questions",
    5: "expert - ask deep architectural and system design questions"
}

INTERVIEWER_PROMPT = """You are a friendly and encouraging job interviewer.
Rules:
- Ask exactly ONE clear practical question at a time
- Level 1-2: Simple common questions only
- Level 3: Moderate practical usage questions
- Level 4-5: Complex problem solving questions
- Be warm, supportive and professional
- Do NOT repeat questions already asked
"""

EVALUATOR_PROMPT = """You are a fair and encouraging interview evaluator.
Return ONLY a JSON object:
{
    "score": 7,
    "level_recommendation": "increase",
    "feedback": "Good understanding. Could elaborate more on X.",
    "keywords_mentioned": ["html", "css"],
    "missing_concepts": ["responsive design"]
}
Scoring: 8-10 correct/clear, 6-7 mostly correct, 4-5 partial, 2-3 vague, 1 wrong/no answer.
Be generous — partial understanding = at least 5.
level_recommendation: increase if >=7, same if 4-6, decrease if <=3
Return ONLY JSON."""

# ==========================================
#           CAMERA MODULE
# ==========================================

# Shared state between camera thread and main thread
camera_state = {
    "running": False,
    "eye_contact": False,
    "face_detected": False,
    "feedback": "",
    "expression": "neutral",
    "recording": False
}

def check_eye_contact(landmarks, w, h):
    """Check if person is looking at camera"""
    left_iris  = landmarks[LEFT_IRIS[0]]
    right_iris = landmarks[RIGHT_IRIS[0]]
    left_corner  = landmarks[33]
    right_corner = landmarks[133]
    r_left_corner  = landmarks[362]
    r_right_corner = landmarks[263]

    left_eye_w  = abs(int(right_corner.x * w) - int(left_corner.x * w))
    right_eye_w = abs(int(r_right_corner.x * w) - int(r_left_corner.x * w))

    if left_eye_w == 0 or right_eye_w == 0:
        return True

    left_ratio  = (int(left_iris.x * w) - int(left_corner.x * w)) / left_eye_w
    right_ratio = (int(right_iris.x * w) - int(r_left_corner.x * w)) / right_eye_w
    avg = (left_ratio + right_ratio) / 2
    return 0.35 <= avg <= 0.65

def detect_expression(landmarks, w, h):
    """
    Detect facial expression using mouth and eyebrow landmarks
    Returns: 'confident', 'neutral', or 'nervous'
    """
    # Mouth landmarks
    upper_lip = landmarks[13]   # upper lip center
    lower_lip = landmarks[14]   # lower lip center
    left_mouth = landmarks[61]  # left mouth corner
    right_mouth = landmarks[291] # right mouth corner

    # Eyebrow landmarks
    left_brow  = landmarks[70]  # left eyebrow
    left_eye_top = landmarks[159] # left eye top
    right_brow = landmarks[300]  # right eyebrow
    right_eye_top = landmarks[386] # right eye top

    # Calculate mouth open ratio (smile detection)
    mouth_width  = abs(right_mouth.x - left_mouth.x)
    mouth_height = abs(lower_lip.y - upper_lip.y)
    mouth_ratio  = mouth_height / mouth_width if mouth_width > 0 else 0

    # Smile corners — if corners are raised it's a smile
    mouth_center_y = (upper_lip.y + lower_lip.y) / 2
    left_corner_y  = left_mouth.y
    right_corner_y = right_mouth.y
    smile_score = mouth_center_y - (left_corner_y + right_corner_y) / 2

    # Eyebrow raise — raised brows = surprised/nervous
    left_brow_dist  = abs(left_brow.y  - left_eye_top.y)
    right_brow_dist = abs(right_brow.y - right_eye_top.y)
    avg_brow_dist   = (left_brow_dist + right_brow_dist) / 2

    # Classify expression
    if smile_score > 0.01:
        return "confident", (0, 255, 0)      # green
    elif avg_brow_dist < 0.02:
        return "nervous", (0, 0, 255)        # red
    else:
        return "neutral", (255, 255, 0)      # yellow

def get_expression_emoji(expression):
    emojis = {
        "confident": "😊 Confident",
        "neutral":   "😐 Neutral",
        "nervous":   "😟 Nervous"
    }
    return emojis.get(expression, "😐 Neutral")

def camera_thread():
    """Runs camera in background thread"""
    cap = cv2.VideoCapture(0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while camera_state["running"]:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        eye_contact = False
        face_detected = False

        if results.multi_face_landmarks:
            face_detected = True
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark
                eye_contact = check_eye_contact(landmarks, w, h)

                # Expression detection
                expression, exp_color = detect_expression(landmarks, w, h)
                camera_state["expression"] = expression

                # Draw iris dots
                for idx in LEFT_IRIS + RIGHT_IRIS:
                    x = int(landmarks[idx].x * w)
                    y = int(landmarks[idx].y * h)
                    cv2.circle(frame, (x, y), 3, (255, 0, 0), -1)

                # Show expression on camera
                cv2.putText(frame, get_expression_emoji(expression), (20, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, exp_color, 2)

        camera_state["eye_contact"] = eye_contact
        camera_state["face_detected"] = face_detected

        # Display status on camera
        if not face_detected:
            cv2.putText(frame, "No face detected!", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        elif eye_contact:
            cv2.putText(frame, "Eye Contact: GOOD", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Look at Camera!", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Recording indicator
        if camera_state["recording"]:
            cv2.circle(frame, (w - 30, 30), 15, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (w - 60, 65),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Show camera feedback
        if camera_state["feedback"]:
            cv2.putText(frame, camera_state["feedback"], (20, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("AI Mock Interviewer", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            camera_state["running"] = False
            break

    cap.release()
    cv2.destroyAllWindows()

# ==========================================
#           AI MODULE
# ==========================================

conversation_history = []
evaluation_data = []
current_level = 1
question_number = 0
max_questions = 8

def ask_question(job_role, level):
    difficulty = LEVELS[level]
    messages = [
        {"role": "system", "content": INTERVIEWER_PROMPT},
        *conversation_history,
        {"role": "user", "content": f"""
Job role: {job_role}
Difficulty level: {difficulty}
Question number: {question_number + 1} of {max_questions}
Ask a straightforward practical interview question for this level.
Keep it simple and clear.
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
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": EVALUATOR_PROMPT},
            {"role": "user", "content": f"Job Role: {job_role}\nQuestion: {question}\nAnswer: {answer}"}
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
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are an expert interview coach. Give a concise ideal answer in 3-5 sentences."},
            {"role": "user", "content": f"Job Role: {job_role}\nQuestion: {question}\nGive the ideal interview answer."}
        ],
        max_tokens=200,
        temperature=0.3
    )
    return response.choices[0].message.content

# ==========================================
#           AUDIO MODULE
# ==========================================

def record_audio(filename="answer.wav", duration=20):
    CHUNK   = 1024
    FORMAT  = pyaudio.paInt16
    CHANNELS = 1
    RATE    = 44100

    p = pyaudio.PyAudio()
    camera_state["recording"] = True
    print(f"\n🎤 Recording {duration}s... SPEAK NOW!")

    stream = p.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                input_device_index=1,
                frames_per_buffer=CHUNK)
    frames = []

    for i in range(0, int(RATE / CHUNK * duration)):
        frames.append(stream.read(CHUNK))
        seconds_left = duration - int(i * CHUNK / RATE)
        print(f"\r⏱️  {seconds_left}s remaining...", end="")

    print("\n✅ Done recording!")
    camera_state["recording"] = False

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return filename

def transcribe(filename):
    """Convert audio to text with better accuracy"""
    print("🤖 Transcribing...")
    result = whisper_model.transcribe(
        filename,
        language="en",
        fp16=False,
        temperature=0.0,        # more deterministic
        no_speech_threshold=0.6, # ignore silence
        condition_on_previous_text=False  # don't guess based on previous
    )
    os.remove(filename)
    text = result["text"].strip()
    return text
# ==========================================
#           FINAL REPORT
# ==========================================

def generate_final_report(job_role):
    if not evaluation_data:
        return "No data to report.", 0

    avg_score = sum(e['score'] for e in evaluation_data) / len(evaluation_data)
    all_feedback = "\n".join([
        f"Q{i+1} (Level {e['level']}): Score {e['score']}/10 - {e['feedback']}"
        for i, e in enumerate(evaluation_data)
    ])

    # Eye contact summary
    eye_contact_score = sum(1 for e in evaluation_data if e.get('eye_contact', True))
    eye_pct = int(eye_contact_score / len(evaluation_data) * 100)

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are an expert career coach. Write a detailed interview performance report."},
            {"role": "user", "content": f"""
Job Role: {job_role}
Average Score: {avg_score:.1f}/10
Eye Contact: {eye_pct}% of the time
Questions: {len(evaluation_data)}

Per-question evaluation:
{all_feedback}

Write a report with:
1. Overall Performance Summary
2. Key Strengths
3. Areas for Improvement
4. Study Recommendations
5. Interview Readiness (Ready / Almost Ready / Needs More Prep)
"""}
        ],
        max_tokens=600,
        temperature=0.5
    )
    return response.choices[0].message.content, avg_score

# ==========================================
#           MAIN PROGRAM
# ==========================================

print("=" * 55)
print("     🎙️  AI MOCK INTERVIEWER — FULL VERSION")
print("=" * 55)
print("📷 Camera + 🎤 Speech + 🤖 AI — all in one!")
print("=" * 55)

job_role = input("\n👔 Enter job role: ")
print(f"\n✅ Starting interview for: {job_role}")
print("\nA camera window will open. Keep your face visible!")
input("Press ENTER to start...\n")

# Start camera in background thread
camera_state["running"] = True
cam_thread = threading.Thread(target=camera_thread, daemon=True)
cam_thread.start()
time.sleep(2)  # give camera time to start

print("📷 Camera started! You should see your face in a window.")
print("🔴 Red dot = recording  |  🟢 Green = good eye contact\n")

# ---- INTERVIEW LOOP ----
while question_number < max_questions and camera_state["running"]:

    print(f"\n{'='*55}")
    level_names = {1:"🟢 BEGINNER", 2:"🟡 ELEMENTARY",
                   3:"🟠 INTERMEDIATE", 4:"🔴 ADVANCED", 5:"🔥 EXPERT"}
    print(f"Question {question_number+1}/{max_questions}  {level_names[current_level]}")
    print('='*55)

    # Get question
    print("🤖 Thinking of a question...")
    current_question = ask_question(job_role, current_level)
    print(f"\n🤖 INTERVIEWER: {current_question}\n")
    time.sleep(0.5)
    speak(current_question)

    conversation_history.append({"role": "assistant", "content": current_question})

    # Check face before recording
    if not camera_state["face_detected"]:
        print("⚠️  No face detected! Please sit in front of camera.")

    user_input = input("Press ENTER to answer (or 'quit' for report): ")
    if user_input.lower() in ['quit', 'q', 'exit', 'stop']:
        print("\n⏹️  Ending interview...")
    # Stop any ongoing speech
    for thread in threading.enumerate():
        if thread.name != "MainThread":
            pass  # daemon threads auto-stop
        break

    # Track eye contact during answer
    eye_contact_checks = []

    # Record in background while checking eye contact
    def record_and_track():
        audio_file = record_audio(duration=20)
        answer = transcribe(audio_file)
        return answer

    # Record answer
    answer_result = [None]
    def recording_job():
        answer_result[0] = record_and_track()

    rec_thread = threading.Thread(target=recording_job)
    rec_thread.start()

    # Track eye contact while recording
    # Track eye contact AND expression while recording
    expression_checks = []
    start_time = time.time()
    while rec_thread.is_alive():
        eye_contact_checks.append(camera_state["eye_contact"])
        expression_checks.append(camera_state["expression"])
        time.sleep(0.5)

    rec_thread.join()
    answer = answer_result[0]

    # Eye contact percentage during this answer
    eye_pct = sum(eye_contact_checks) / max(len(eye_contact_checks), 1) * 100

    print(f"\n📝 YOUR ANSWER: {answer}")
    print(f"👁️  Eye Contact: {eye_pct:.0f}% during answer")
    # Expression summary
    confident_pct = expression_checks.count("confident") / max(len(expression_checks), 1) * 100
    nervous_pct   = expression_checks.count("nervous")   / max(len(expression_checks), 1) * 100
    neutral_pct   = expression_checks.count("neutral")   / max(len(expression_checks), 1) * 100
    
    dominant_exp = max(
        [("confident", confident_pct), ("neutral", neutral_pct), ("nervous", nervous_pct)],
        key=lambda x: x[1]
    )[0]
    
    print(f"😊 Expression: {get_expression_emoji(dominant_exp)} ({confident_pct:.0f}% confident, {nervous_pct:.0f}% nervous)")
    
    # Give tip if nervous
    if nervous_pct > 50:
        print("💡 TIP: Try to relax! Take a deep breath before answering.")
        speak("Try to relax and take a deep breath before your next answer.")
    elif confident_pct > 50:
        print("🌟 Great confident expression!")

    if len(answer) < 5:
        print("⚠️  Couldn't hear clearly. Skipping.")
        question_number += 1
        continue

    conversation_history.append({"role": "user", "content": answer})

    # Evaluate
    print("\n📊 Evaluating...")
    evaluation = evaluate_answer(current_question, answer, job_role)

    score          = evaluation.get('score', 5)
    feedback       = evaluation.get('feedback', '')
    recommendation = evaluation.get('level_recommendation', 'same')
    missing        = evaluation.get('missing_concepts', [])

    # Store with eye contact data
    evaluation_data.append({
        'question': current_question,
        'answer': answer,
        'score': score,
        'feedback': feedback,
        'level': current_level,
        'eye_contact': eye_pct >= 50,
        'expression': dominant_exp,
        'confident_pct': confident_pct
    })
    # Show feedback
    score_emoji = "🔥" if score>=8 else "✅" if score>=6 else "⚠️" if score>=4 else "❌"
    print(f"\n{score_emoji} SCORE: {score}/10")
    print(f"💬 FEEDBACK: {feedback}")
    # Short spoken feedback only
    speak(f"Score {score} out of 10. {feedback[:100]}")
    if missing:
        print(f"📚 MISSED: {', '.join(missing)}")
    if eye_pct < 50:
        print(f"👁️  TIP: Try to maintain more eye contact during your answer!")

    # Ideal answer
    print(f"\n💡 IDEAL ANSWER:")
    print(get_ideal_answer(current_question, job_role))

    # Adjust level
    if recommendation == "increase" and current_level < 5:
        current_level += 1
        print(f"\n⬆️  Moving to {level_names[current_level]}!")
    elif recommendation == "decrease" and current_level > 1:
        current_level -= 1
        print(f"\n⬇️  Back to {level_names[current_level]}")
    else:
        print(f"\n➡️  Staying at {level_names[current_level]}")

    question_number += 1

# ---- STOP CAMERA ----
camera_state["running"] = False
time.sleep(1)

# ---- FINAL REPORT ----
if evaluation_data:
    print(f"\n{'='*55}")
    print("         📄 GENERATING FINAL REPORT...")
    print('='*55)

    report, avg_score = generate_final_report(job_role)

    print(f"\n📊 SCORE BREAKDOWN:")
    for i, e in enumerate(evaluation_data):
        bar = "█" * e['score'] + "░" * (10 - e['score'])
        eye = "👁️" if e['eye_contact'] else "👀"
        exp = "😊" if e['expression'] == "confident" else "😐" if e['expression'] == "neutral" else "😟"
        print(f"  Q{i+1}: [{bar}] {e['score']}/10  {eye}  Level {e['level']}")

    avg = sum(e['score'] for e in evaluation_data) / len(evaluation_data)
    readiness = "🟢 READY" if avg>=7 else "🟡 ALMOST READY" if avg>=5 else "🔴 NEEDS MORE PREP"

    print(f"\n  Average: {avg:.1f}/10  |  Status: {readiness}")
    print(f"\n{'='*55}")
    print("📋 FULL REPORT:")
    print('='*55)
    print(report)

    with open("interview_report.txt", "w", encoding="utf-8") as f:
        f.write("AI MOCK INTERVIEW REPORT\n")
        f.write(f"Job Role: {job_role}\n")
        f.write(f"Average Score: {avg:.1f}/10\n")
        f.write(f"Status: {readiness}\n\n")
        f.write("SCORE BREAKDOWN:\n")
        for i, e in enumerate(evaluation_data):
            f.write(f"Q{i+1}: {e['score']}/10 - {e['question']}\n")
        f.write(f"\nFULL REPORT:\n{report}\n")

    print(f"\n✅ Report saved to interview_report.txt")

print("\n🎯 Thanks for using AI Mock Interviewer! Good luck! 🚀")