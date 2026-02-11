# ไพลิน (Pailin) - AI VTuber Application

A Python-based AI VTuber application that combines chatbot, speech-to-text, and text-to-speech functionality to create an interactive virtual assistant. The application features ไพลิน (Pailin), a cheerful, friendly young girl personality that communicates in Thai language with a female voice. The name "ไพลิน" (Pailin) means "sapphire" in Thai, representing brightness and vibrancy.

## Features

- **Chatbot**: Uses Google's Gemini models to generate responses in Thai language with a cheerful, friendly personality
- **Speech-to-Text**: Converts user's speech to text using Google's speech recognition (configured for Thai language)
- **Text-to-Speech**: Converts the AI's responses to speech with a female Thai voice
- **Audio Output**: Configured to use CABLE Input as the audio output device for integration with VTuber software
- **Special Character Handling**: Automatically filters out special characters to ensure natural speech output
- **Response Caching**: Caches TTS responses to improve performance and reduce API calls
- **Cross-platform**: Audio device detection works on Windows, Linux, and macOS

## Project Structure

```
AI_Vtube/
├── src/
│   └── ai_vtube/           # Main package
│       ├── __init__.py
│       ├── config.py        # Shared configuration & constants
│       ├── chatbot.py       # Gemini chatbot with Pailin personality
│       ├── speech_to_text.py  # Google Speech Recognition wrapper
│       ├── text_to_speech.py  # gTTS + pygame playback with caching
│       ├── text_utils.py    # Special character removal
│       └── main.py          # Application entry point & orchestrator
├── tests/                   # Unit tests
│   ├── test_text_utils.py
│   ├── test_config.py
│   └── test_chatbot.py
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
└── pyproject.toml
```

## How It Works

1. User sends text or voice input to the application
2. If input is voice, it's converted to text using Speech-to-Text
3. Text is sent to the Gemini API for processing
4. Gemini API returns a text response
5. Text response is converted to speech using Text-to-Speech
6. The AI VTuber responds with voice that the user can hear

## Requirements

- Python 3.9+
- Google Gemini API key

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Krx-21/AI_Vtube.git
   cd AI_Vtube
   ```

2. Create a virtual environment (recommended):
   ```
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On macOS/Linux
   ```

3. Install the package in development mode:
   ```
   pip install -e ".[dev]"
   ```

   Or install dependencies only:
   ```
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your Gemini API key:
   ```
   cp .env.example .env
   ```
   Then edit the `.env` file to add your actual API key.

5. (Optional) Install VB-Audio Virtual Cable for routing audio to VTuber software:
   - Download from [VB-Audio website](https://vb-audio.com/Cable/)
   - Install following the provided instructions

## Usage

Run the main application:
```bash
# Using the package entry point
ai-vtube

# Or run the module directly
python -m ai_vtube.main

# With custom cache size
ai-vtube --cache-size 30
```

The application will:
1. Initialize the chatbot, speech-to-text, and text-to-speech components
2. Adjust the microphone for ambient noise
3. Listen for your voice input
4. Process your input through the chatbot
5. Speak the response
6. Cache the audio response for future use

To exit the application, say "exit", "quit", or "bye" (in any language).

## Individual Components

You can also run each component separately for testing:

```bash
python -m ai_vtube.chatbot
python -m ai_vtube.speech_to_text
python -m ai_vtube.text_to_speech
```

## Running Tests

```bash
pytest
```

## Customization

### Audio Output Device

The application is configured to use "CABLE Input" as the default audio output device. You can change this when creating a `TextToSpeech` instance:

```python
from ai_vtube.text_to_speech import TextToSpeech
tts = TextToSpeech(device_name="Your Device Name")
```

### Language Settings

The application is configured to use Thai as the default language. You can change this by modifying the constants in `src/ai_vtube/config.py` or passing different parameters to the component classes.

Note that ไพลิน (Pailin) is designed with a cheerful, friendly young girl personality that responds in Thai. Changing the language will require updating the personality prompt as well for consistency.

### Gemini Models

Available models:

- **gemini-1.5-flash**: Fast and versatile
- **gemini-1.5-pro**: More powerful for complex reasoning
- **gemini-2.0-flash** (default): Newer model with improved capabilities

Change the model by modifying `DEFAULT_MODEL` in `config.py` or passing `model=` when creating a `Chatbot` instance.

## Troubleshooting

### Microphone Issues
- Make sure your microphone is properly connected and set as the default input device.
- If you get a "Could not find PyAudio" error, install it separately: `pip install PyAudio`

### Audio Output Issues
- If you don't hear any audio, check your speakers/headphones and volume settings.
- Run `python -m ai_vtube.text_to_speech` to list available audio devices.

### API Key Issues
- Make sure you've created a `.env` file with your Gemini API key.
- Verify that the API key is valid and has the necessary permissions.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[MIT License](LICENSE)

## Acknowledgements

- [Google Gemini API](https://ai.google.dev/) for the chatbot functionality
- [gTTS](https://gtts.readthedocs.io/) for text-to-speech conversion
- [SpeechRecognition](https://pypi.org/project/SpeechRecognition/) for speech-to-text conversion
- [pygame](https://www.pygame.org/) for audio playback
- [VB-Audio](https://vb-audio.com/) for Virtual Cable software
- [python-dotenv](https://pypi.org/project/python-dotenv/) for environment variable management
- [PyAudio](https://pypi.org/project/PyAudio/) for audio input/output
