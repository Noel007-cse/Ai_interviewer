import groq
import whisper
import pyaudio
import wave
import os
import json

# ---- SETUP ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = groq.Groq(api_key=GROQ_API_KEY) # your gsk_... key here

client = groq.Groq(api_key=GROQ_API_KEY)
whisper_model = whisper.load_model("base")

# ---- DIFFICULTY LEVELS ----
LEVELS = {
    1: "beginner - ask very basic conceptual questions only",
    2: "elementary - ask simple practical questions",
    3: "intermediate - ask moderate difficulty questions",
    4: "advanced - ask complex problem solving questions",
    5: "expert - ask deep architectural and system design questions"
}

# ---- INTERVIEW STATE ----
current_level = 1
question_number = 0
max_questions = 8
conversation_history = []
evaluation_data = []  # stores each Q&A with score

# ---- SYSTEM PROMPT ----
INTERVIEWER_PROMPT = """You are a friendly and encouraging job interviewer conducting a mock interview.

Rules:
- Ask exactly ONE clear, practical question at a time
- Level 1-2: Ask simple, common interview questions (what is HTML, what is CSS, etc.)
- Level 3: Ask moderate questions about practical usage
- Level 4-5: Ask complex problem solving questions
- NEVER ask trick questions or obscure terminology questions at beginner level
- Be warm, supportive and professional
- Do NOT repeat questions already asked
"""

EVALUATOR_PROMPT = """You are a fair and encouraging interview evaluator.
Evaluate the candidate's answer and return ONLY a JSON object like this:
{
    "score": 7,
    "level_recommendation": "increase",
    "feedback": "Good understanding of basics. Could elaborate more on X.",
    "keywords_mentioned": ["html", "css"],
    "missing_concepts": ["responsive design"]
}

Scoring guide:
- 8-10: Answer is correct and clear
- 6-7: Answer is mostly correct, missing some details
- 4-5: Answer shows some understanding but is incomplete
- 2-3: Answer is vague or mostly incorrect
- 1: No answer or completely wrong

Be GENEROUS and ENCOURAGING. If the candidate shows even partial understanding, score at least 5.
level_recommendation: "increase" if score>=7, "same" if score 4-6, "decrease" if score<=3
Return ONLY the JSON, no other text."""

def ask_question(job_role, level, context=""):
    """Get next interview question from AI"""
    difficulty = LEVELS[level]
    
    messages = [
        {"role": "system", "content": INTERVIEWER_PROMPT},
        *conversation_history,
        {"role": "user", "content": f"""
Job role: {job_role}
Difficulty level: {difficulty}
Question number: {question_number + 1} of {max_questions}

Ask a straightforward, practical interview question appropriate for this level.
At beginner level, ask common questions like explaining basic concepts.
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
    """Evaluate candidate's answer and return score + feedback"""
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": EVALUATOR_PROMPT},
            {"role": "user", "content": f"""
Job Role: {job_role}
Question: {question}
Candidate's Answer: {answer}

Evaluate this answer.
"""}
        ],
        max_tokens=300,
        temperature=0.3
    )
    
    raw = response.choices[0].message.content
    
    # Parse JSON response
    try:
        # Clean up response in case there's extra text
        start = raw.find('{')
        end = raw.rfind('}') + 1
        json_str = raw[start:end]
        return json.loads(json_str)
    except:
        # Fallback if JSON parsing fails
        return {
            "score": 5,
            "level_recommendation": "same",
            "feedback": raw,
            "keywords_mentioned": [],
            "missing_concepts": []
        }

def record_audio(filename="answer.wav", duration=20):
    """Record audio from microphone"""
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    p = pyaudio.PyAudio()
    print(f"\n🎤 Recording for {duration} seconds... SPEAK NOW!")

    stream = p.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True, frames_per_buffer=CHUNK)

    frames = []
    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
        seconds_left = duration - int(i * CHUNK / RATE)
        print(f"\r⏱️  {seconds_left} seconds remaining...", end="")

    print("\n✅ Recording done!")
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
    """Convert audio to text"""
    print("🤖 Transcribing your answer...")
    result = whisper_model.transcribe(filename, language="en", fp16=False)
    os.remove(filename)
    return result["text"].strip()

def generate_final_report(job_role, evaluation_data):
    """Generate comprehensive final interview report"""
    
    total_score = sum(e['score'] for e in evaluation_data)
    avg_score = total_score / len(evaluation_data) if evaluation_data else 0
    
    # Collect all feedback
    all_feedback = "\n".join([
        f"Q{i+1}: Score {e['score']}/10 - {e['feedback']}"
        for i, e in enumerate(evaluation_data)
    ])
    
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are an expert career coach. Write a detailed interview performance report."},
            {"role": "user", "content": f"""
Job Role: {job_role}
Average Score: {avg_score:.1f}/10
Number of Questions: {len(evaluation_data)}

Per-question evaluation:
{all_feedback}

Write a comprehensive final report with:
1. Overall Performance Summary
2. Key Strengths (what they did well)
3. Areas for Improvement (what to study)
4. Specific Study Recommendations
5. Interview Readiness (Ready / Almost Ready / Needs More Prep)
"""}
        ],
        max_tokens=600,
        temperature=0.5
    )
    return response.choices[0].message.content, avg_score

def get_level_emoji(level):
    emojis = {1: "🟢", 2: "🟡", 3: "🟠", 4: "🔴", 5: "🔥"}
    return emojis.get(level, "⚪")

# ==========================================
#           MAIN INTERVIEW LOOP
# ==========================================

print("=" * 55)
print("        🎙️  AI MOCK INTERVIEWER v2.0")
print("=" * 55)
print("📋 Adaptive difficulty — starts easy, gets harder")
print("📊 Real-time scoring after each answer")
print("📄 Full evaluation report at the end")
print("=" * 55)

job_role = input("\n👔 Enter job role (e.g. Frontend Developer): ")
print(f"\n✅ Starting adaptive interview for: {job_role}")
print(f"📝 Total questions: {max_questions}")
print("\nPress ENTER after each question to record your answer.")
print("Type 'quit' to end early and get your report.\n")
input("Press ENTER to begin the interview...")

current_question = ""

while question_number < max_questions:
    
    print(f"\n{'='*55}")
    print(f"Question {question_number + 1}/{max_questions}  {get_level_emoji(current_level)} Level {current_level}: {LEVELS[current_level].split('-')[0].strip().upper()}")
    print('='*55)
    
    # Get question from AI
    print("🤖 Thinking of a question...")
    current_question = ask_question(job_role, current_level)
    print(f"\n🤖 INTERVIEWER: {current_question}\n")
    
    # Add to conversation history
    conversation_history.append({
        "role": "assistant",
        "content": current_question
    })
    
    # Get candidate's answer
    user_input = input("Press ENTER to answer (or type 'quit' for report): ")
    if user_input.lower() == 'quit':
        break
    
    # Record and transcribe
    audio_file = record_audio(duration=20)
    answer = transcribe(audio_file)
    print(f"\n📝 YOUR ANSWER: {answer}\n")
    
    if len(answer) < 5:
        print("⚠️  Couldn't hear your answer clearly. Moving to next question.")
        question_number += 1
        continue
    
    # Add to conversation history
    conversation_history.append({
        "role": "user",
        "content": answer
    })
    
    # Evaluate the answer
    print("📊 Evaluating your answer...")
    evaluation = evaluate_answer(current_question, answer, job_role)
    
    score = evaluation.get('score', 5)
    feedback = evaluation.get('feedback', 'No feedback available')
    recommendation = evaluation.get('level_recommendation', 'same')
    missing = evaluation.get('missing_concepts', [])
    
    # Store evaluation
    evaluation_data.append({
        'question': current_question,
        'answer': answer,
        'score': score,
        'feedback': feedback,
        'level': current_level
    })
    
    # Show instant feedback
    score_emoji = "🔥" if score >= 8 else "✅" if score >= 6 else "⚠️" if score >= 4 else "❌"
    print(f"\n{score_emoji} SCORE: {score}/10")
    print(f"💬 FEEDBACK: {feedback}")
    if missing:
        print(f"📚 MISSED: {', '.join(missing)}")
    # Show ideal answer
    print(f"\n💡 IDEAL ANSWER:")
    ideal_response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "You are an expert interview coach. Give a concise ideal answer for this interview question in 3-5 sentences. Be clear and to the point."},
            {"role": "user", "content": f"Job Role: {job_role}\nQuestion: {current_question}\nGive the ideal interview answer."}
        ],
        max_tokens=200,
        temperature=0.3
    ).choices[0].message.content
    print(ideal_response)
    # Adjust difficulty level
    old_level = current_level
    if recommendation == "increase" and current_level < 5:
        current_level += 1
        print(f"\n⬆️  Great answer! Moving to Level {current_level} {get_level_emoji(current_level)}")
    elif recommendation == "decrease" and current_level > 1:
        current_level -= 1
        print(f"\n⬇️  Let's try a slightly easier question. Level {current_level} {get_level_emoji(current_level)}")
    else:
        print(f"\n➡️  Staying at Level {current_level} {get_level_emoji(current_level)}")
    
    question_number += 1

# ==========================================
#           FINAL REPORT
# ==========================================

if evaluation_data:
    print(f"\n{'='*55}")
    print("         📄 GENERATING FINAL REPORT...")
    print('='*55)
    
    report, avg_score = generate_final_report(job_role, evaluation_data)
    
    # Score summary
    print(f"\n📊 SCORE SUMMARY:")
    for i, e in enumerate(evaluation_data):
        bar = "█" * e['score'] + "░" * (10 - e['score'])
        print(f"  Q{i+1}: [{bar}] {e['score']}/10  (Level {e['level']})")
    
    print(f"\n  Average: {avg_score:.1f}/10")
    
    readiness = "🟢 READY" if avg_score >= 7 else "🟡 ALMOST READY" if avg_score >= 5 else "🔴 NEEDS MORE PREP"
    print(f"  Status:  {readiness}")
    
    print(f"\n{'='*55}")
    print("📋 DETAILED REPORT:")
    print('='*55)
    print(report)
    
    # Save report to file
    with open("interview_report.txt", "w", encoding="utf-8") as f:
        f.write(f"AI MOCK INTERVIEW REPORT\n")
        f.write(f"Job Role: {job_role}\n")
        f.write(f"Average Score: {avg_score:.1f}/10\n")
        f.write(f"Status: {readiness}\n\n")
        f.write("SCORE BREAKDOWN:\n")
        for i, e in enumerate(evaluation_data):
            f.write(f"Q{i+1}: {e['score']}/10 - {e['question']}\n")
        f.write(f"\nDETAILED REPORT:\n{report}\n")
    
    print(f"\n✅ Report saved to: interview_report.txt")
else:
    print("\n⚠️  No questions answered — no report generated.")

print("\n🎯 Thanks for using AI Mock Interviewer! Good luck! 🚀")