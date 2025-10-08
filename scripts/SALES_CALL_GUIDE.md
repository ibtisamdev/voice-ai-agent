# Generate Realistic Sales Call Audio

Create a realistic sales conversation with TTS using your existing Voice AI Agent infrastructure.

## Quick Start

### Option 1: Using ElevenLabs (Best Quality)

1. **Get ElevenLabs API Key**: https://elevenlabs.io
   - Sign up and get your API key from Settings
   - Browse voices at https://elevenlabs.io/voice-library

2. **Configure Environment**:
```bash
cd /Users/ibtisam/Documents/voice-ai-agent
cp .env.example .env

# Edit .env and add:
TTS_ENGINE=elevenlabs
ELEVENLABS_API_KEY=your_api_key_here
```

3. **Run Generator**:
```bash
# Start services
make dev

# Generate audio (in Docker)
docker-compose -f docker/docker-compose.yml run --rm api \
  python /app/../scripts/generate_sales_call.py

# Or run directly
python scripts/generate_sales_call.py --output-dir output/sales_call
```

### Option 2: Using Azure TTS (Good Quality, Many Voices)

1. **Get Azure API Key**: https://portal.azure.com
   - Create Cognitive Services resource
   - Get your API key and region

2. **Configure**:
```bash
TTS_ENGINE=azure
AZURE_SPEECH_KEY=your_key
AZURE_SPEECH_REGION=eastus
```

3. **Select Voices**:
   - Sarah (female): `en-US-JennyNeural`, `en-US-AriaNeural`
   - Michael (male): `en-US-GuyNeural`, `en-US-DavisNeural`

### Option 3: Using Coqui TTS (Free, Local)

```bash
TTS_ENGINE=coqui
# No API key needed - runs locally
```

Note: Coqui has limited voice options but is completely free.

## Output

The script generates:
- Individual audio files: `01_sarah_female.mp3`, `02_michael_male.mp3`, etc.
- Manifest file: `conversation_manifest.txt`
- Location: `output/sales_call/`

## Merge Audio Files

Combine individual files into one conversation:

```bash
# Install pydub
pip install pydub

# Merge with 800ms pauses
python scripts/merge_sales_call.py \
  --input-dir output/sales_call \
  --output sales_call_complete.mp3 \
  --pause 800
```

## Customize Conversation

Edit `scripts/generate_sales_call.py` and modify the `CONVERSATION` variable:

```python
CONVERSATION = [
    ("Sarah", "female", "Your custom text here"),
    ("Michael", "male", "Response text"),
    # Add more lines...
]
```

## Voice Selection Tips

### ElevenLabs Voices (Recommended)
Best voices for sales calls:
- **Sarah (Sales Rep)**: Rachel (`21m00Tcm4TlvDq8ikWAM`) - professional, friendly
- **Michael (Customer)**: Josh (`TxGEqnHWrfWFTfGW9XjX`) - natural, conversational

Browse more at: https://elevenlabs.io/voice-library

### Azure Neural Voices
- **Sarah**: `en-US-JennyNeural` (friendly), `en-US-SaraNeural` (professional)
- **Michael**: `en-US-GuyNeural` (neutral), `en-US-JasonNeural` (casual)

Full list: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/language-support

## Advanced Options

### Adjust Speech Parameters

Modify in `generate_sales_call.py`:

```python
result = await tts.synthesize(
    text=text,
    voice_id=voice_id,
    speed=1.1,      # Faster (0.5 - 2.0)
    pitch=0.0,      # Higher pitch (-1.0 to 1.0)
    volume=1.0,     # Volume (0.0 - 1.0)
    emotion="professional",  # ElevenLabs only
    use_cache=False
)
```

### Add Background Noise (Realistic Phone Quality)

Use audio editing software:
1. Open in Audacity
2. Generate > Noise
3. Mix at -40dB to -50dB
4. Apply EQ: reduce 0-100Hz (remove rumble), boost 2-4kHz (phone clarity)

### Export Formats

Supported formats: WAV, MP3, M4A, FLAC

Change output format in merge script:
```bash
python scripts/merge_sales_call.py --output sales_call.wav
```

## Cost Estimates

- **ElevenLabs**: ~$0.30 per 1000 characters (this call: ~$0.60)
- **Azure**: ~$0.016 per 1000 characters (this call: ~$0.03)
- **Coqui**: Free

## Troubleshooting

### "No TTS engines available"
- Check `.env` configuration
- Verify API keys are valid
- Run `make voice-test` to diagnose

### "Module not found"
```bash
# Install dependencies
pip install torch torchaudio TTS pydub
```

### Poor audio quality
- Use ElevenLabs or Azure for best quality
- Increase bitrate when merging: `bitrate="320k"`
- Use higher sample rate voices

## Next Steps

1. **Test different voices**: Browse voice libraries
2. **Adjust pacing**: Modify pause duration between speakers
3. **Add emotions**: Use SSML or voice styles (Azure/ElevenLabs)
4. **Create variations**: Generate multiple versions with different voices
5. **Post-process**: Add background music, normalize volume

## Examples

Generate and merge in one go:
```bash
# Generate audio
python scripts/generate_sales_call.py --output-dir output/sales_call

# Merge files
python scripts/merge_sales_call.py --input-dir output/sales_call --pause 1000

# Play result
ffplay output/sales_call/sales_call_complete.mp3
```

## Support

- ElevenLabs docs: https://elevenlabs.io/docs
- Azure TTS docs: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/
- Coqui TTS: https://github.com/coqui-ai/TTS
