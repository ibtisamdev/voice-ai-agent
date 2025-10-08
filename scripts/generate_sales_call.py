#!/usr/bin/env python3
"""
Generate a realistic sales call conversation with TTS.
Creates separate audio files for Sarah (sales rep) and Michael (customer),
then optionally merges them into a complete conversation.
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import List, Tuple

# Add backend to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from ai.voice.tts_service import TTSService, TTSEngine

# Sales call transcript
CONVERSATION: List[Tuple[str, str, str]] = [
    ("Sarah", "female", "Hi, is this Michael?"),
    ("Michael", "male", "Yes, this is Michael."),
    ("Sarah", "female", "Hi Michael, this is Sarah calling from TechFlow Systems. How are you today?"),
    ("Michael", "male", "I'm okay, pretty busy actually. What's this about?"),
    ("Sarah", "female", "I totally understand. I'll be quick. We work with marketing agencies like yours to automate their reporting process. I saw your company just expanded to the Chicago market - congratulations on that."),
    ("Michael", "male", "Oh, thanks. Yeah, we just opened that office last month."),
    ("Sarah", "female", "That's exciting. I imagine that's created some additional workload for your team?"),
    ("Michael", "male", "You could say that. We're pretty stretched right now, honestly."),
    ("Sarah", "female", "I hear you. That's actually why I'm calling. We help agencies cut their reporting time in half so teams can focus on client work instead of spreadsheets. Are you currently handling client reports manually or do you have a system?"),
    ("Michael", "male", "We use a mix. Google Data Studio for some clients, manual Excel reports for others. It works but it's... yeah, it's time-consuming."),
    ("Sarah", "female", "How many hours would you say your team spends on that each week?"),
    ("Michael", "male", "Probably 15 to 20 hours total across the team."),
    ("Sarah", "female", "Okay, so that's significant. And with the Chicago expansion, that number's probably going up?"),
    ("Michael", "male", "It already has. We hired someone just to help with reporting."),
    ("Sarah", "female", "Got it. So here's what we do - our platform connects to all your data sources automatically and generates branded reports in about 5 minutes. Most of our clients get that 15-20 hours down to maybe 3 or 4."),
    ("Michael", "male", "That sounds good, but what's the cost? We're watching our budget pretty closely right now."),
    ("Sarah", "female", "I understand. Our pricing starts at $299 per month for up to 25 clients. Given that you hired someone for reporting, you're probably paying what, $3,000-$4,000 a month for that role?"),
    ("Michael", "male", "More like $4,500."),
    ("Sarah", "female", "Right, so even if this just reduces their workload by half, you're looking at real ROI. Plus they could focus on higher-value work."),
    ("Michael", "male", "Maybe. I'd need to see how it actually works though."),
    ("Sarah", "female", "Absolutely. How about this - I can set up a 20-minute demo where I'll connect to your actual data and show you exactly what the reports would look like. No commitment, just see if it fits. Does this Thursday at 2pm work, or is Friday morning better?"),
    ("Michael", "male", "Thursday's tough. What time Friday?"),
    ("Sarah", "female", "How's 10am?"),
    ("Michael", "male", "Yeah, I can do 10am Friday."),
    ("Sarah", "female", "Perfect. I'll send you a calendar invite right now with the Zoom link. I'll also send a quick prep email - if you can have your Google Analytics login ready, I can show you a real report during the call."),
    ("Michael", "male", "Okay, sounds good."),
    ("Sarah", "female", "Great Michael. Looking forward to showing you this on Friday. Have a great rest of your day."),
    ("Michael", "male", "You too, thanks."),
    ("Sarah", "female", "Bye now."),
]


async def generate_sales_call(output_dir: str = "output/sales_call"):
    """Generate sales call audio files."""

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"🎙️  Generating sales call conversation...")
    print(f"📁 Output directory: {output_path.absolute()}\n")

    # Initialize TTS service
    tts = TTSService()
    print("Initializing TTS service...")
    await tts.initialize()

    # Check available engines and voices
    available_engines = tts.get_available_engines()
    print(f"Available engines: {[e.value for e in available_engines]}\n")

    # Get voices
    voices = tts.get_available_voices()
    print(f"Available voices: {len(voices)} total\n")

    # Select best voices for Sarah and Michael
    sarah_voice = None
    michael_voice = None

    # Prioritize ElevenLabs > Azure > Coqui
    for engine in [TTSEngine.ELEVENLABS, TTSEngine.AZURE, TTSEngine.COQUI_TTS]:
        if engine in available_engines:
            engine_voices = tts.get_available_voices(engine)

            if engine == TTSEngine.ELEVENLABS:
                # Use specific ElevenLabs voices (adjust IDs as needed)
                sarah_voice = "21m00Tcm4TlvDq8ikWAM"  # Rachel - professional female
                michael_voice = "TxGEqnHWrfWFTfGW9XjX"  # Josh - professional male
                selected_engine = TTSEngine.ELEVENLABS
                break

            elif engine == TTSEngine.AZURE:
                sarah_voice = "en-US-JennyNeural"
                michael_voice = "en-US-GuyNeural"
                selected_engine = TTSEngine.AZURE
                break

            elif engine == TTSEngine.COQUI_TTS:
                sarah_voice = "coqui_en_female"
                michael_voice = "coqui_en_female"  # Coqui only has one voice
                selected_engine = TTSEngine.COQUI_TTS
                break

    if not sarah_voice:
        print("❌ No suitable TTS engine available!")
        return

    print(f"✅ Using {selected_engine.value} engine")
    print(f"   Sarah voice: {sarah_voice}")
    print(f"   Michael voice: {michael_voice}\n")

    # Generate audio for each line
    all_audio_files = []

    for idx, (speaker, gender, text) in enumerate(CONVERSATION, 1):
        voice_id = sarah_voice if gender == "female" else michael_voice

        print(f"[{idx:02d}] {speaker}: {text[:60]}{'...' if len(text) > 60 else ''}")

        try:
            result = await tts.synthesize(
                text=text,
                voice_id=voice_id,
                engine=selected_engine,
                speed=1.0,
                use_cache=False
            )

            # Save audio file
            file_ext = result.audio_format
            filename = f"{idx:02d}_{speaker.lower()}_{gender}.{file_ext}"
            filepath = output_path / filename

            with open(filepath, "wb") as f:
                f.write(result.audio_data)

            all_audio_files.append(str(filepath))

            print(f"     ✓ Saved: {filename} ({result.duration_ms:.0f}ms)")

        except Exception as e:
            print(f"     ✗ Error: {e}")
            continue

    # Print summary
    print(f"\n{'='*70}")
    print(f"✅ Generated {len(all_audio_files)} audio files")
    print(f"📁 Location: {output_path.absolute()}")

    # Show TTS statistics
    stats = tts.get_stats()
    print(f"\n📊 TTS Statistics:")
    print(f"   Total syntheses: {stats['syntheses_completed']}")
    print(f"   Total characters: {stats['total_characters_synthesized']}")
    print(f"   Avg processing time: {stats['average_processing_time_ms']:.1f}ms")
    print(f"   Engine usage: {stats['engine_usage']}")

    print(f"\n💡 Next steps:")
    print(f"   1. Listen to individual files in: {output_path}")
    print(f"   2. Merge files: use audio editing software (Audacity, ffmpeg)")
    print(f"   3. Add silence between turns for natural pacing")

    # Create a manifest file
    manifest_path = output_path / "conversation_manifest.txt"
    with open(manifest_path, "w") as f:
        f.write("# Sales Call Conversation Manifest\n\n")
        for idx, (speaker, gender, text) in enumerate(CONVERSATION, 1):
            file_ext = "mp3" if selected_engine == TTSEngine.ELEVENLABS else "wav"
            filename = f"{idx:02d}_{speaker.lower()}_{gender}.{file_ext}"
            f.write(f"{filename}\n")
            f.write(f"  Speaker: {speaker}\n")
            f.write(f"  Text: {text}\n\n")

    print(f"   4. View manifest: {manifest_path}\n")


async def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate realistic sales call audio")
    parser.add_argument(
        "--output-dir",
        default="output/sales_call",
        help="Output directory for audio files (default: output/sales_call)"
    )

    args = parser.parse_args()

    await generate_sales_call(args.output_dir)


if __name__ == "__main__":
    asyncio.run(main())
