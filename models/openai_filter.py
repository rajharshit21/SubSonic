# models/openai_filter.py

import os
import openai
from gtts import gTTS
import uuid
from pathlib import Path
from dotenv import load_dotenv
from fastapi import HTTPException
from services.file_handler import get_filename

# Set your OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def transform_voice_with_style(transcript: str, style_prompt: str) -> str:
    prompt = f"Convert this voice transcript to match this style: {style_prompt}\n\nTranscript: {transcript}\n\nStyled Output:"
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {str(e)}"


async def apply_openai_style(audio_path: str, style_prompt: str) -> str:
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = openai.Audio.transcribe("whisper-1", audio_file)
        original_text = transcript["text"]

        style_query = (
            f"Transform the following sentence into the style of: {style_prompt}.\n\n"
            f"Original: \"{original_text}\"\nStyled:"
        )

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": style_query}],
            max_tokens=200
        )
        styled_text = response["choices"][0]["message"]["content"]

        tts = gTTS(text=styled_text)
        new_filename = f"{get_filename(audio_path)}_{uuid.uuid4().hex[:6]}_styled.wav"
        new_path = str(Path("data/processed") / new_filename)
        tts.save(new_path)

        return new_path

    except Exception as e:
        print(f"🔥 OpenAI Style Transform Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
