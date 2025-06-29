import openai
from gtts import gTTS
import uuid
import os
from pathlib import Path
from dotenv import load_dotenv
from services.file_handler import get_filename

# Load API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

async def apply_openai_style(audio_path: str, style_prompt: str) -> str:
    """
    Transforms audio by:
    1. Transcribing speech using Whisper
    2. Modifying text style using GPT
    3. Re-generating speech using gTTS
    """
    # Step 1: Transcribe
    with open(audio_path, "rb") as f:
        transcript = openai.Audio.transcribe("whisper-1", f)["text"]

    # Step 2: GPT style transformation
    style_query = (
        f"Transform the following sentence into the style of: {style_prompt}.\n\n"
        f"Original: \"{transcript}\"\nStyled:"
    )

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": style_query}],
        max_tokens=200
    )
    styled_text = response["choices"][0]["message"]["content"]

    # Step 3: TTS
    tts = gTTS(text=styled_text)
    new_filename = f"{get_filename(audio_path)}_{uuid.uuid4().hex[:6]}_styled.wav"
    new_path = str(Path("data/processed") / new_filename)
    tts.save(new_path)

    return new_path
