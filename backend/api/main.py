# main.py

import asyncio
import logging
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from backend.core.input_handler.input_processor import InputProcessor
from backend.core.main_brain.llama_integration import LLamaBrain
from backend.utils.auth_manager import register_user, authenticate_user, create_access_token
from backend.core.main_brain.llama_output_analyzer import EnhancedLlamaOutputAnalyzer
from database.database import SessionLocal, engine, Base

# Initialize FastAPI app
app = FastAPI()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize database
Base.metadata.create_all(bind=engine)

# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Initialize LLamaBrain, InputProcessor, and EnhancedLlamaOutputAnalyzer
llama_brain = LLamaBrain()
input_processor = InputProcessor()
output_analyzer = EnhancedLlamaOutputAnalyzer()

@app.post("/register")
async def register(username: str, email: str, password: str, db: Session = Depends(get_db)):
    return await register_user(db, username, email, password)

@app.post("/login")
async def login(username: str, password: str, db: Session = Depends(get_db)):
    user = await authenticate_user(db, username, password)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": user.username})
    return {"access_token": access_token, "token_type": "bearer"}


async def process_input(text):
    logger.info(f"Input received: {text}")
    llama_response = await llama_brain.process_input(text)
    logger.info(f"LLama response: {llama_response}")

    is_relevant, confidence, filtered_output, audio_data = await output_analyzer.analyze_output(text, llama_response)

    if is_relevant:
        logger.info(f"Relevant output (confidence: {confidence:.2f}): {filtered_output}")
        print(f"LLama: {filtered_output}")

        if audio_data:
            # Here you can save the audio data to a file or stream it to the user
            with open("output.wav", "wb") as audio_file:
                audio_file.write(audio_data)
            logger.info("Audio output saved as output.wav")
    else:
        logger.info(f"Output not relevant (confidence: {confidence:.2f})")
        await output_analyzer.route_to_other_module(text)

    @app.on_event("startup")
    async def startup_event():
        # This will run when the FastAPI app starts
        asyncio.create_task(start_input_processing())

    async def start_input_processing():
        input_mode = input("Choose input mode (text/voice): ").lower()
        if input_mode not in ['text', 'voice']:
            logger.error("Invalid input mode. Defaulting to text.")
            input_mode = 'text'

        try:
            await input_processor.start_processing(process_input, input_mode=input_mode)
        except KeyboardInterrupt:
            logger.info("Stopping the application.")
        finally:
            if input_mode == 'voice':
                input_processor.stop_voice_input()

    if __name__ == "__main__":
        import uvicorn
        uvicorn.run(app, host="0.0.0.0", port=8000)
