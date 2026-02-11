# ไพลิน (Pailin) - AI VTuber Application

[![CI](https://github.com/Krx-21/AI_Vtube/actions/workflows/ci.yml/badge.svg)](https://github.com/Krx-21/AI_Vtube/actions/workflows/ci.yml)

A Python-based AI VTuber application that combines chatbot, speech-to-text, and text-to-speech functionality to create an interactive virtual assistant. The application features ไพลิน (Pailin), a cheerful, friendly young girl personality that communicates in Thai language with a female voice. The name "ไพลิน" (Pailin) means "sapphire" in Thai, representing brightness and vibrancy.

## Features

- **Chatbot**: Uses Google's Gemini models to generate responses in Thai language with a cheerful, friendly personality
- **Speech-to-Text**: Converts user's speech to text using Google's speech recognition (configured for Thai language)
- **Text-to-Speech**: Converts the AI's responses to speech with a female Thai voice using Microsoft Edge TTS
- **Async/Await**: Fully asynchronous implementation for better performance
- **Monorepo Structure**: Organized into shared core and application-specific packages
- **Special Character Handling**: Automatically filters out special characters to ensure natural speech output
- **Response Caching**: Caches TTS responses to improve performance and reduce API calls
- **Cross-platform**: Works on Windows, Linux, and macOS

## Project Structure

```
AI_Vtube/
├── packages/
│   ├── pailin-core/           # Shared core functionality
│   │   ├── pailin_core/
│   │   │   ├── __init__.py
│   │   │   ├── config.py      # Shared configuration
│   │   │   ├── personality.py # Pailin personality prompts
│   │   │   ├── ai/
│   │   │   │   ├── __init__.py
│   │   │   │   └── chatbot.py # Async Gemini chatbot
│   │   │   ├── speech/
│   │   │   │   ├── __init__.py
│   │   │   │   ├── stt.py     # Async speech-to-text
│   │   │   │   └── tts.py     # Async text-to-speech (edge-tts)
│   │   │   └── text/
│   │   │       ├── __init__.py
│   │   │       └── sanitizer.py # Text processing utilities
│   │   ├── tests/
│   │   └── pyproject.toml
│   ├── pailin-vtube/          # VTuber application
│   │   ├── pailin_vtube/
│   │   │   ├── __init__.py
│   │   │   ├── app.py         # Main application & orchestrator
│   │   │   └── audio/
│   │   │       ├── __init__.py
│   │   │       ├── devices.py # Audio device detection
│   │   │       └── playback.py # Audio playback utilities
│   │   ├── tests/
│   │   └── pyproject.toml
│   └── pailin-discord/        # Discord bot (placeholder)
│       ├── pailin_discord/
│       │   └── __init__.py
│       ├── tests/
│       └── pyproject.toml
├── .github/
│   └── workflows/
│       └── ci.yml             # CI/CD pipeline
├── .env.example
├── .gitignore
├── README.md
└── pyproject.toml             # Root workspace config
```

## How It Works

1. User sends text or voice input to the application
2. If input is voice, it's converted to text using Speech-to-Text
3. Text is sent to the Gemini API for processing
4. Gemini API returns a text response
5. Text response is converted to speech using Text-to-Speech
6. The AI VTuber responds with voice that the user can hear

## Requirements

- Python 3.10+
- Google Gemini API key
- PortAudio (for microphone input)

## Installation

### Option 1: Install from monorepo (recommended)

1. Clone this repository:
   ```bash
   git clone https://github.com/Krx-21/AI_Vtube.git
   cd AI_Vtube
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On macOS/Linux
   ```

3. Install all packages in development mode:
   ```bash
   # Install pailin-core
   pip install -e "./packages/pailin-core[dev]"
   
   # Install pailin-vtube (includes pailin-core as dependency)
   pip install -e "./packages/pailin-vtube[dev]"
   
   # Install development tools
   pip install ruff mypy pytest pytest-asyncio
   ```

4. Create a `.env` file with your Gemini API key:
   ```bash
   cp .env.example .env
   ```
   Then edit the `.env` file to add your actual API key.

5. (Optional) Install PortAudio for microphone support:
   - **Windows**: Download and install from [PortAudio website](http://www.portaudio.com/)
   - **macOS**: `brew install portaudio`
   - **Linux**: `sudo apt-get install portaudio19-dev python3-pyaudio`

### Option 2: Legacy installation (deprecated)

The old `src/ai_vtube/` structure is deprecated. Please use the monorepo structure above.

## Usage

Run the VTuber application:

```bash
# Using the package entry point
pailin-vtube

# Or run the module directly
python -m pailin_vtube.app

# With custom cache size
pailin-vtube --cache-size 30
```

The application will:
1. Initialize the chatbot, speech-to-text, and text-to-speech components
2. Adjust the microphone for ambient noise
3. Listen for your voice input
4. Process your input through the chatbot
5. Speak the response
6. Cache the audio response for future use

To exit the application, say "exit", "quit", or "bye" (in any language).

## Running Tests

### Test all packages:
```bash
pytest packages/pailin-core/tests -v
pytest packages/pailin-vtube/tests -v
```

### Test a specific package:
```bash
cd packages/pailin-core
pytest tests -v
```

### Run async tests:
```bash
pytest --asyncio-mode=auto
```

## Development

### Linting with Ruff:
```bash
ruff check packages/
ruff check packages/ --fix  # Auto-fix issues
```

### Type checking with mypy:
```bash
mypy packages/pailin-core/pailin_core --ignore-missing-imports
mypy packages/pailin-vtube/pailin_vtube --ignore-missing-imports
```

### Format code:
```bash
ruff format packages/
```

## Customization

### Language Settings

The application is configured to use Thai as the default language. You can change this by modifying the constants in `packages/pailin-core/pailin_core/config.py` or passing different parameters to the component classes.

Note that ไพลิน (Pailin) is designed with a cheerful, friendly young girl personality that responds in Thai. Changing the language will require updating the personality prompt in `packages/pailin-core/pailin_core/personality.py` as well for consistency.

### Gemini Models

Available models (configure in `packages/pailin-core/pailin_core/config.py`):

- **gemini-2.0-flash** (default): Newer model with improved capabilities
- **gemini-1.5-flash**: Fast and versatile
- **gemini-1.5-pro**: More powerful for complex reasoning

Change the model by modifying `DEFAULT_MODEL` in `config.py` or passing `model=` when creating a `Chatbot` instance.

### Text-to-Speech Voice

The application uses Microsoft Edge TTS with Thai female voice (`th-TH-PremwadeeNeural`). You can change the voice by modifying the `voice` parameter in the `TextToSpeech` class.

To see available Thai voices:
```bash
python -m pailin_core.speech.tts
```

## Migration from Old Structure

If you're migrating from the old `src/ai_vtube/` structure:

1. The old structure is deprecated and will be removed in a future version
2. Update your imports:
   - `from ai_vtube.config import` → `from pailin_core.config import`
   - `from ai_vtube.chatbot import` → `from pailin_core.ai.chatbot import`
   - `from ai_vtube.text_utils import` → `from pailin_core.text.sanitizer import`
3. Update your code to use async/await:
   - `chatbot.chat_with_gemini(text)` → `await chatbot.chat_with_gemini(text)`
   - `stt.listen_for_speech()` → `await stt.listen_for_speech()`
   - `tts.speak(text)` → `await tts.speak(text)`
4. The TTS now uses `edge-tts` instead of `gTTS` for better async support

## Troubleshooting

### Microphone Issues
- Make sure your microphone is properly connected and set as the default input device.
- If you get a "Could not find PyAudio" error, install portaudio: `pip install PyAudio`

### Audio Output Issues
- If you don't hear any audio, check your speakers/headphones and volume settings.
- The application now uses the system default audio device.

### API Key Issues
- Make sure you've created a `.env` file with your Gemini API key.
- Verify that the API key is valid and has the necessary permissions.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)

## Acknowledgements

- [Google Gemini API](https://ai.google.dev/) for the chatbot functionality
- [Microsoft Edge TTS](https://github.com/rany2/edge-tts) for text-to-speech conversion
- [SpeechRecognition](https://pypi.org/project/SpeechRecognition/) for speech-to-text conversion
- [pygame](https://www.pygame.org/) for audio playback
- [python-dotenv](https://pypi.org/project/python-dotenv/) for environment variable management
- [PyAudio](https://pypi.org/project/PyAudio/) for audio input/output
