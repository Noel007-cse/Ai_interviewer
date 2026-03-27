import pyaudio
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
DURATION = 5
MIC_INDEX = 1  # change this to test different mics

p = pyaudio.PyAudio()
print(f"Testing mic index {MIC_INDEX}...")
print("🎤 Speak for 5 seconds...")

stream = p.open(format=FORMAT, channels=CHANNELS,
                rate=RATE, input=True,
                input_device_index=1,
                frames_per_buffer=CHUNK)

frames = []
for i in range(0, int(RATE / CHUNK * DURATION)):
    frames.append(stream.read(CHUNK))

print("✅ Done! Playing back...")
stream.stop_stream()
stream.close()

wf = wave.open("mic_test.wav", 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
p.terminate()
print("Saved to mic_test.wav — open it and check if your voice is recorded!")