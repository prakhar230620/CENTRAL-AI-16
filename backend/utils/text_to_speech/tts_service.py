# backend/utils/text_to_speech/tts_service.py

import pyttsx3
import asyncio
from typing import Optional

class TTSService:
    def __init__(self, language: str = 'en', gender: str = 'female'):
        self.language = language
        self.gender = gender
        self.engine = pyttsx3.init()
        voices = self.engine.getProperty('voices')
        self.engine.setProperty('voice', voices[0].id)  # Default voice

    async def synthesize(self, text: str, output_format: str = 'wav') -> Optional[bytes]:
        loop = asyncio.get_event_loop()
        audio_data = await loop.run_in_executor(None, self._synthesize_sync, text, output_format)
        return audio_data

    def _synthesize_sync(self, text: str, output_format: str) -> bytes:
        output_file = f'output.{output_format}'
        self.engine.save_to_file(text, output_file)
        self.engine.runAndWait()
        with open(output_file, 'rb') as f:
            return f.read()