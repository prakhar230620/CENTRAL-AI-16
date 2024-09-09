import asyncio
import logging
import speech_recognition as sr
import pyaudio
import numpy as np
from pydub import AudioSegment
from pydub.effects import normalize
import webrtcvad
import collections
import threading

# Audio recording parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_DURATION_MS = 20
PADDING_DURATION_MS = 600
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)
PADDING_CHUNKS = int(PADDING_DURATION_MS / CHUNK_DURATION_MS)

# VAD parameters
VAD_MODE = 0  # Aggressiveness mode (0-3)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def enhance_audio(audio_segment):
    audio_segment = normalize(audio_segment)
    audio_segment = audio_segment.high_pass_filter(80)
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

            if not triggered:
                ring_buffer.append((chunk, is_speech))
                num_voiced = len([f for f, speech in ring_buffer if speech])
                if num_voiced > 0.9 * ring_buffer.maxlen:
                    triggered = True
                    self.frames.extend([f for f, _ in ring_buffer])
                    ring_buffer.clear()
            else:
                self.frames.append(chunk)
                ring_buffer.append((chunk, is_speech))
                num_unvoiced = len([f for f, speech in ring_buffer if not speech])
                if num_unvoiced > 0.9 * ring_buffer.maxlen:
                    triggered = False
                    yield b''.join(self.frames)
                    self.frames = []
                    ring_buffer.clear()

    def stop_recording(self):
        self.is_recording = False

    def close(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

async def transcribe_audio(audio_data):
    recognizer = sr.Recognizer()
    try:
        audio = sr.AudioData(audio_data, RATE, 2)
        text = recognizer.recognize_google(audio, language="en-US")
        logger.info(f"Transcription: {text}")
    except sr.UnknownValueError:
        text = "Could not understand audio"
        logger.error("Transcription failed: Could not understand audio")
    except sr.RequestError as e:
        text = f"Could not request results: {e}"
        logger.error(f"Transcription failed: {e}")

    return text

class InputProcessor:
    def __init__(self):
        self.audio_streamer = AudioStreamer()
        self.is_listening = False

    async def process_voice_input(self, callback):
        logger.info("Starting voice input processing.")
        self.is_listening = True
        recording_thread = threading.Thread(target=self.audio_streamer.start_recording)
        recording_thread.start()

        try:
            for audio_data in self.audio_streamer.start_recording():
                if not self.is_listening:
                    break
                logger.info("Processing audio chunk")
                audio_segment = AudioSegment(
                    data=audio_data,
                    sample_width=2,
                    frame_rate=RATE,
                    channels=CHANNELS
                )
                enhanced_audio = enhance_audio(audio_segment)
                transcription = await transcribe_audio(enhanced_audio.raw_data)
                await callback(transcription)
        except Exception as e:
            logger.error(f"An error occurred during voice processing: {str(e)}")
        finally:
            self.audio_streamer.stop_recording()
            recording_thread.join()
            self.audio_streamer.close()

    def stop_voice_input(self):
        logger.info("Stopping voice input processing.")
        self.is_listening = False
        self.audio_streamer.stop_recording()

    async def process_text_input(self, callback):
        logger.info("Starting text input processing.")
        while True:
            try:
                user_input = input("You: ")
                if user_input.lower() == 'quit':
                    break
                await callback(user_input)
            except Exception as e:
                logger.error(f"An error occurred during text processing: {str(e)}")

    async def start_processing(self, callback, input_mode='text'):
        if input_mode == 'voice':
            await self.process_voice_input(callback)
        else:
            await self.process_text_input(callback)

if __name__ == "__main__":
    async def example_callback(text):
        print(f"Processed input: {text}")

    async def main():
        processor = InputProcessor()
        await processor.start_processing(example_callback, input_mode='text')

    asyncio.run(main())