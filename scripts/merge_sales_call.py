#!/usr/bin/env python3
"""
Merge individual sales call audio files into a complete conversation.
Adds natural pauses between speakers.
"""

import os
import sys
from pathlib import Path
from pydub import AudioSegment
from pydub.playback import play


def merge_audio_files(input_dir: str, output_file: str = "sales_call_complete.mp3", pause_ms: int = 800):
    """
    Merge audio files in order with pauses.

    Args:
        input_dir: Directory containing numbered audio files
        output_file: Output filename
        pause_ms: Pause duration between speakers in milliseconds
    """

    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"❌ Directory not found: {input_dir}")
        return

    # Find all audio files
    audio_files = sorted([
        f for f in input_path.glob("*")
        if f.suffix.lower() in ['.mp3', '.wav', '.m4a', '.flac']
        and not f.name.startswith('.')
    ])

    if not audio_files:
        print(f"❌ No audio files found in {input_dir}")
        return

    print(f"🎙️  Merging {len(audio_files)} audio files...")
    print(f"⏱️  Pause between speakers: {pause_ms}ms\n")

    # Create silence segment
    silence = AudioSegment.silent(duration=pause_ms)

    # Merge all audio files
    combined = AudioSegment.empty()

    for idx, audio_file in enumerate(audio_files, 1):
        print(f"[{idx:02d}] Adding: {audio_file.name}")

        try:
            audio = AudioSegment.from_file(str(audio_file))
            combined += audio

            # Add pause after each segment (except last)
            if idx < len(audio_files):
                combined += silence

        except Exception as e:
            print(f"     ✗ Error loading {audio_file.name}: {e}")
            continue

    # Export final audio
    output_path = input_path / output_file
    print(f"\n💾 Exporting to: {output_path}")

    combined.export(
        str(output_path),
        format=output_file.split('.')[-1],
        bitrate="192k"
    )

    duration_sec = len(combined) / 1000
    print(f"✅ Complete! Duration: {duration_sec:.1f}s")
    print(f"📁 Saved: {output_path.absolute()}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Merge sales call audio files")
    parser.add_argument(
        "--input-dir",
        default="output/sales_call",
        help="Input directory containing audio files"
    )
    parser.add_argument(
        "--output",
        default="sales_call_complete.mp3",
        help="Output filename (default: sales_call_complete.mp3)"
    )
    parser.add_argument(
        "--pause",
        type=int,
        default=800,
        help="Pause between speakers in milliseconds (default: 800)"
    )

    args = parser.parse_args()

    merge_audio_files(args.input_dir, args.output, args.pause)


if __name__ == "__main__":
    main()
