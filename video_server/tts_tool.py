import os
import asyncio
from pathlib import Path
from typing import Optional
from mcp.server.fastmcp import FastMCP
import edge_tts

server = FastMCP("tts-tool")

OUTPUT_DIR = Path(__file__).parent.parent / "output" / "audio"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

VOICE_MAPPING = {
    "authoritative": "en-US-GuyNeural",
    "professional": "en-GB-RyanNeural",
    "energetic": "en-US-AmberNeural",
    "calm": "en-US-AriaNeural",
    "deep": "en-US-GuyNeural",
    "female_professional": "en-GB-SoniaNeural",
    "female_energetic": "en-US-JennyNeural",
    "male_casual": "en-US-BrandonNeural"
}

@server.tool()
def generate_voiceover(text: str, voice_tone: str = "authoritative") -> dict:
    """
    Generate high-fidelity voiceover using Edge TTS.
    
    Args:
        text: Script text to convert to speech
        voice_tone: Voice style (authoritative, professional, energetic, calm, deep, 
                   female_professional, female_energetic, male_casual)
    
    Returns:
        dict with status, audio file path, and metadata
    """
    
    if not text or not text.strip():
        return {
            "success": False,
            "error": "Text cannot be empty"
        }
    
    voice = VOICE_MAPPING.get(voice_tone, VOICE_MAPPING["authoritative"])
    
    filename = f"voiceover_{voice_tone}_{hash(text) % 10000}.mp3"
    filepath = OUTPUT_DIR / filename
    
    try:
        async def generate_audio():
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(str(filepath))
        
        asyncio.run(generate_audio())
        
        if not filepath.exists():
            return {
                "success": False,
                "error": f"Audio file was not created at {filepath}"
            }
        
        file_size = filepath.stat().st_size
        
        word_count = len(text.split())
        estimated_duration = word_count / 2.5
        
        return {
            "success": True,
            "filename": filename,
            "path": str(filepath),
            "voice": voice,
            "voice_tone": voice_tone,
            "text_length": len(text),
            "word_count": word_count,
            "file_size_bytes": file_size,
            "estimated_duration_seconds": round(estimated_duration, 2),
            "output_directory": str(OUTPUT_DIR)
        }
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to generate voiceover: {str(e)}",
            "voice": voice,
            "voice_tone": voice_tone
        }
