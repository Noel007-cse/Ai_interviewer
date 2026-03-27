import whisper
import pyaudio
import wave
import os

# Load Whisper model (tiny = fastest, base = more accurate)
print("Loading Whisper model...")
model = whisper.load_model("base")
print("Model loaded!")

def record_audio(filename="answer.wav", duration=10):
    """Record audio from microphone for given duration"""
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000

    p = pyaudio.PyAudio()

    print(f"\n🎤 Recording for {duration} seconds... SPEAK NOW!")

    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )

    frames = []
    for i in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

        # Show countdown
        seconds_left = duration - int(i * CHUNK / RATE)
        print(f"\r⏱️  {seconds_left} seconds remaining...", end="")

    print("\n✅ Recording done!")

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save to WAV file
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    return filename

def transcribe_audio(filename="answer.wav"):
    """Convert audio file to text using Whisper"""
    print("\n🤖 Transcribing your answer...")
    result = model.transcribe(filename, language="en", fp16=False)
    return result["text"]

# --- Main Program ---
print("=" * 50)
print("   AI INTERVIEWER - Speech to Text Test")
print("=" * 50)

# Ask a sample interview question
question = "Tell me about yourself and your background."
print(f"\n📋 QUESTION: {question}")
print("\nPress ENTER when ready to answer...")
input()

# Record answer
audio_file = record_audio(duration=30)

# Transcribe
answer = transcribe_audio(audio_file)

print("\n" + "=" * 50)
print("📝 YOUR ANSWER (transcribed):")
print("=" * 50)
print(answer)
print("=" * 50)

# Clean up audio file
os.remove(audio_file)