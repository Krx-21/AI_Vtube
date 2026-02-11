# Migration Guide: Phase 1 Foundation Refactor

This document describes the migration from the old `src/ai_vtube/` structure to the new monorepo structure.

## What Changed

### 1. Project Structure

**Before:**
```
AI_Vtube/
├── src/
│   └── ai_vtube/
│       ├── __init__.py
│       ├── config.py
│       ├── chatbot.py
│       ├── speech_to_text.py
│       ├── text_to_speech.py
│       ├── text_utils.py
│       └── main.py
├── tests/
├── requirements.txt
└── pyproject.toml
```

**After:**
```
AI_Vtube/
├── packages/
│   ├── pailin-core/
│   │   ├── pailin_core/
│   │   │   ├── ai/chatbot.py
│   │   │   ├── speech/stt.py, tts.py
│   │   │   ├── text/sanitizer.py
│   │   │   ├── config.py
│   │   │   └── personality.py
│   │   ├── tests/
│   │   └── pyproject.toml
│   ├── pailin-vtube/
│   │   ├── pailin_vtube/
│   │   │   ├── app.py
│   │   │   └── audio/
│   │   ├── tests/
│   │   └── pyproject.toml
│   └── pailin-discord/ (placeholder)
├── .github/workflows/ci.yml
└── pyproject.toml (workspace)
```

### 2. Import Changes

| Old Import | New Import |
|------------|------------|
| `from ai_vtube.config import *` | `from pailin_core.config import *` |
| `from ai_vtube.chatbot import Chatbot` | `from pailin_core.ai.chatbot import Chatbot` |
| `from ai_vtube.text_utils import remove_special_characters` | `from pailin_core.text.sanitizer import remove_special_characters` |
| `from ai_vtube.speech_to_text import SpeechToText` | `from pailin_core.speech.stt import SpeechToText` |
| `from ai_vtube.text_to_speech import TextToSpeech` | `from pailin_core.speech.tts import TextToSpeech` |
| `from ai_vtube.main import AIVtuber` | `from pailin_vtube.app import AIVtuber` |

### 3. API Changes (Async)

All main methods are now async:

```python
# Old (synchronous)
response = chatbot.chat_with_gemini("Hello")
text = stt.listen_for_speech()
tts.speak("Hello")
vtuber.run()

# New (asynchronous)
response = await chatbot.chat_with_gemini("Hello")
text = await stt.listen_for_speech()
await tts.speak("Hello")
await vtuber.run()

# Entry point updated to use asyncio
import asyncio
asyncio.run(vtuber.run())
```

### 4. TTS Library Change

- **Old:** `gTTS` (synchronous)
- **New:** `edge-tts` (asynchronous)

The new TTS uses Microsoft Edge's TTS service (via edge-tts library) which provides:
- Async support
- Better voice quality
- Faster generation
- No API key required

### 5. Installation Changes

**Old:**
```bash
pip install -r requirements.txt
# OR
pip install -e .
```

**New:**
```bash
# Install pailin-core
pip install -e "./packages/pailin-core[dev]"

# Install pailin-vtube (includes pailin-core)
pip install -e "./packages/pailin-vtube[dev]"

# Install dev tools
pip install ruff mypy pytest pytest-asyncio
```

### 6. Running the Application

**Old:**
```bash
ai-vtube
# OR
python -m ai_vtube.main
```

**New:**
```bash
pailin-vtube
# OR
python -m pailin_vtube.app
```

### 7. Running Tests

**Old:**
```bash
pytest
```

**New:**
```bash
# Test all packages
pytest packages/pailin-core/tests -v
pytest packages/pailin-vtube/tests -v

# Test specific package
cd packages/pailin-core
pytest tests -v
```

### 8. Development Tools

**New tools added:**
- **Ruff:** Fast Python linter and formatter
  ```bash
  ruff check packages/
  ruff check packages/ --fix
  ```

- **Mypy:** Static type checker
  ```bash
  mypy packages/pailin-core/pailin_core --ignore-missing-imports
  mypy packages/pailin-vtube/pailin_vtube --ignore-missing-imports
  ```

- **Pytest-asyncio:** Async test support
  ```bash
  pytest --asyncio-mode=auto
  ```

### 9. CI/CD

New GitHub Actions workflow (`.github/workflows/ci.yml`):
- Python 3.10, 3.11, 3.12 matrix
- Ruff linting
- Mypy type checking
- Pytest testing

### 10. Documentation

- Updated README.md with new structure
- Added CI badge
- Added migration guide
- Updated installation and usage instructions

## Breaking Changes

1. **Synchronous to Asynchronous:** All main APIs are now async
2. **Import paths changed:** Package names updated
3. **TTS library changed:** gTTS → edge-tts
4. **Python version:** Minimum version is now 3.10+
5. **Entry point:** `ai-vtube` → `pailin-vtube`

## Benefits

1. **Better organization:** Separation of core logic from application
2. **Async performance:** Better concurrency and responsiveness
3. **Type safety:** Full type annotations with mypy checking
4. **Code quality:** Automated linting with ruff
5. **Testing:** Async test support with pytest-asyncio
6. **CI/CD:** Automated testing on multiple Python versions
7. **Extensibility:** Easy to add new applications (e.g., Discord bot)

## Verification

All changes have been verified:
- ✅ 19/19 tests passing
- ✅ All ruff lint checks pass
- ✅ No CodeQL security alerts
- ✅ Packages install correctly
- ✅ Entry points work correctly
