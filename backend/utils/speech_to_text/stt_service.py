import speech_recognition as sr
import pyaudio
import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize
import webrtcvad
import collections
import threading
import time

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_DURATION_MS = 20
PADDING_DURATION_MS = 600
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)
PADDING_CHUNKS = int(PADDING_DURATION_MS / CHUNK_DURATION_MS)

# VAD parameters
VAD_MODE = 0# Aggressiveness mode (0-3)

def enhance_audio(audio_segment):
    # Normalize audio
    audio_segment = normalize(audio_segment)

    # Apply a high-pass filter to reduce low-frequency noise
    audio_segment = audio_segment.high_pass_filter(80)

    # Apply a low-pass filter to reduce high-frequency noise
    audio_segment = audio_segment.low_pass_filter(10000)

    return audio_segment

class AudioStreamer:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE,
        )
        self.vad = webrtcvad.Vad(VAD_MODE)
        self.is_recording = False
        self.frames = []

    def start_recording(self):
        self.is_recording = True
        self.frames = []
        ring_buffer = collections.deque(maxlen=PADDING_CHUNKS)
        triggered = False

        while self.is_recording:
            chunk = self.stream.read(CHUNK_SIZE)
            is_speech = self.vad.is_speech(chunk, RATE)

            print("Speech detected" if is_speech else "No speech")  # Debug print

            if not triggered:
                ring_buffer.append((chunk, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > 0.9 * ring_buffer.maxlen:
                    triggered = True
                    print("Speech started")  # Debug print
                    self.frames.extend([f for f, _ in ring_buffer])
                    ring_buffer.clear()
            else:
                self.frames.append(chunk)
                ring_buffer.append((chunk, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > 0.9 * ring_buffer.maxlen:
                    triggered = False
                    print("Speech ended")  # Debug print
                    yield b''.join(self.frames)
                    self.frames = []
                    ring_buffer.clear()

    def stop_recording(self):
        self.is_recording = False

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

def transcribe_audio(audio_data):
    recognizer = sr.Recognizer()
    try:
        # Convert raw audio data to AudioData object
        audio = sr.AudioData(audio_data, RATE, 2)
        # Try to recognize with Google (supports multiple languages)
        text = recognizer.recognize_google(audio, language="hi-IN")  # Change to "en-IN" for Indian English
        print("Transcription successful")  # Debug print
        print(f"Transcription: {text}")
    except sr.UnknownValueError:
        text = "Could not understand audio"
        print("Transcription failed: Could not understand audio")  # Debug print
    except sr.RequestError as e:
        text = f"Could not request results: {e}"
        print(f"Transcription failed: {e}")  # Debug print

    return text

def continuous_stt():
    print("Starting advanced continuous STT service with automatic speech detection.")
    print("Start speaking, and the system will automatically detect and transcribe your speech.")
    print("Press Ctrl+C to stop the service.")

    audio_streamer = AudioStreamer()
    recording_thread = threading.Thread(target=audio_streamer.start_recording)
    recording_thread.start()

    try:
        for audio_data in audio_streamer.start_recording():
            print("Processing audio chunk")  # Debug print
            audio_segment = AudioSegment(
                data=audio_data,
                sample_width=2,
                frame_rate=RATE,
                channels=CHANNELS
            )
            enhanced_audio = enhance_audio(audio_segment)
            transcription = transcribe_audio(enhanced_audio.raw_data)
    except KeyboardInterrupt:
        print("Stopping STT service.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        audio_streamer.stop_recording()
        recording_thread.join()
        audio_streamer.close()

if __name__ == "__main__":
    continuous_stt()
