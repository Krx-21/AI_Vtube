# ไพลิน (Pailin) - AI VTuber Application

A Python-based AI VTuber application that combines chatbot, speech-to-text, and text-to-speech functionality to create an interactive virtual assistant. The application features ไพลิน (Pailin), a cheerful, friendly young girl personality that communicates in Thai language with a female voice. The name "ไพลิน" (Pailin) means "sapphire" in Thai, representing brightness and vibrancy.

## Features

- **Chatbot**: Uses Google's Gemini models to generate responses in Thai language with a cheerful, friendly personality
- **Speech-to-Text**: Converts user's speech to text using Google's speech recognition (configured for Thai language)
- **Text-to-Speech**: Converts the AI's responses to speech with a female Thai voice
- **Audio Output**: Configured to use CABLE Input as the audio output device for integration with VTuber software
- **Special Character Handling**: Automatically filters out special characters to ensure natural speech output
- **Response Caching**: Caches TTS responses to improve performance and reduce API calls

## How It Works

1. User sends text or voice input to the application
2. If input is voice, it's converted to text using Speech-to-Text
3. Text is sent to the Gemini API for processing
4. Gemini API returns a text response
5. Text response is converted to speech using Text-to-Speech
6. The AI VTuber responds with voice that the user can hear

## Requirements

- Python 3.7+
- Google Gemini API key

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Krx-21/AI_Vtube.git
   cd AI_Vtube
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv .venv
   .venv\Scripts\activate  # On Windows
   source .venv/bin/activate  # On macOS/Linux
   ```

3. Install the required packages:
   ```
   pip install -r Requirements.txt
   ```

   Or install packages individually:
   ```
   pip install google-generativeai python-dotenv SpeechRecognition pygame gTTS PyAudio
   ```

4. Create a `.env` file with your Gemini API key (you can copy from the example file):
   ```
   cp .env.example .env
   ```
   Then edit the `.env` file to add your actual API key.

5. (Optional) Install VB-Audio Virtual Cable for routing audio to VTuber software:
   - Download from [VB-Audio website](https://vb-audio.com/Cable/)
   - Install following the provided instructions

## Usage

Run the main application:
```
python main.py
```

Command-line options:
```
python main.py --cache-size 30  # Set the maximum number of TTS responses to cache (default: 20)
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

- Chatbot: `python chatbot.py`
- Speech-to-Text: `python speech2text.py`
- Text-to-Speech: `python text2speech.py`

## Customization

### Audio Output Device

The application is configured to use "CABLE Input" as the default audio output device. This allows you to route the audio to VTuber software. You can change this in the `text2speech.py` file:

```python
# Change the device_name parameter to use a different audio output device
tts = TextToSpeech(device_name="Your Device Name")
```

To list available audio devices, run `python text2speech.py` and check the console output. The application will automatically display all available audio devices and provide setup instructions.

### Language Settings

The application is configured to use Thai as the default language. You can change this by:

- Modifying the `language` parameter in the `SpeechToText` class initialization in `speech2text.py` (e.g., change "th-TH" to "en-US" for English)
- Updating the system prompt in the `Chatbot` class in `chatbot.py` to specify a different language and personality
- Changing the `language` parameter in the `TextToSpeech` class in `text2speech.py`

Note that the AI VTuber ไพลิน (Pailin) is designed with a cheerful, friendly young girl personality that responds in Thai. Changing the language will require updating the personality prompt as well for consistency.

### Gemini Models

The application uses the Gemini API for the chatbot functionality. The following models are available:

- **gemini-1.5-flash**: Fast and versatile performance across a diverse variety of tasks
- **gemini-1.5-pro**: More powerful model for complex reasoning tasks requiring more intelligence
- **gemini-2.0-flash** (default): Newer model with improved capabilities, including a 1M token context window

You can change the model by modifying the `model` parameter when initializing the Chatbot class in `main.py` or when creating a Chatbot instance directly.

## Audio Management

The application uses a caching system to store TTS responses in memory and as temporary files. This improves performance by reusing previously generated speech for repeated phrases. The cache size can be configured using the `--cache-size` parameter when starting the application.

## Troubleshooting

### Microphone Issues

- If the application can't detect your microphone, make sure it's properly connected and set as the default input device.
- If you get a "Could not find PyAudio" error, install it separately with `pip install PyAudio`.
- If ambient noise calibration fails, try running in a quieter environment.

### Audio Output Issues

- If you don't hear any audio, check that your speakers or headphones are connected and working.
- If using CABLE Input, make sure it's properly installed and configured in your system.
- Check the volume settings in both the application and your system.
- To list available audio devices, run `python text2speech.py` and check the console output.
- The application automatically attempts to use "CABLE Input" as the output device for integration with VTuber software.

### API Key Issues

- If you get an error about the API key, make sure you've created a `.env` file with your Gemini API key.
- Verify that the API key is valid and has the necessary permissions.

### Speech Recognition Issues

- If speech recognition is inaccurate, try speaking more clearly and in a quieter environment.
- Check that you're using the correct language code for your speech (default is "th-TH" for Thai).
- The application will automatically provide friendly error messages in Thai if it can't understand your speech.
- If you're speaking in a language other than Thai, modify the `language` parameter in `speech2text.py`.

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
