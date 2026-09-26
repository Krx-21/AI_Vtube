# Prior art & orchestration lessons — component brief

_Research snapshot: 2026-09-25. Verified facts carry a source; unverified items are marked._

## Recommendation

DECISION: build our own small, single-process asyncio "brain" core. Do not fork Open-LLM-VTuber (OLV) and do not build on pipecat, LiveKit Agents or AIRI. Take their algorithms and parameter defaults instead. Everything below comes from open-source clones. None of it is a fact about how Neuro-sama is built. Neuro facts come only from the corpus summaries (T1/T2).

WHY NOT ADOPT EACH ONE:
(1) OLV v1.2.1 is in maintenance mode. Its README says the team is focusing on "v2.0 — a complete rewrite… in its early discussion and planning phase" and asks for no new feature issues or PRs on v1. The v1.2.1 tag commit is dated 2025-08-26 and the last main commit 2026-05-15. Its design is one user and one browser.
  - Each WebSocket connection gets its own ServiceContext, so a second client means a second brain. A Sept 2026 user report: LLM, MCP and TTS all ran twice and the audio doubled.
  - Live chat is a FIFO queue, one full turn per message.
  - Sentence splitting relies on punctuation. I tested it: a 150-character Thai reply comes out as ONE chunk, so TTS waits for the whole LLM reply.
  - There is no output filter.
  - On Windows the lockfile resolves CPU-only torch from PyPI, and GPU ASR needs a manual CUDA Toolkit and cuDNN install.
(2) pipecat 1.11.0 (BSD-2):
  - Fast release churn: 12 minor releases between 1.0.0 (2026-04-14) and 2026-09-17.
  - Its core pins onnxruntime~=1.24.3, which clashes with onnxruntime-gpu 1.30 and sherpa-onnx.
  - It is built for one user talking to one bot, with no multi-source livestream arbitration.
  - No Thai support where it matters: its sentencex segmenter returns a whole Thai paragraph as one sentence (tested); Smart Turn v3.2 has no Thai data; the min-words interruption check uses text.split().
  - LocalAudioTransport (pyaudio) has no echo cancellation (AEC).
(3) LiveKit Agents 1.8.3 (Apache-2.0):
  - Best turn-handling settings of the group, but it needs a LiveKit SFU server and the worker/room model.
  - Its turn-detector model covers 14 languages, not Thai, under a custom licence.
  - Its turn-handling constructor arguments were deprecated in favour of turn_handling={...}, another sign of API churn.
(4) AIRI v0.12.0-beta.5 is a TypeScript/Vue/Electron pnpm monorepo, not Python. Its memory ("Memory Alaya") is still marked WIP. Use it only as a design reference for the speech pipeline and the renderer.
(5) No maintained Thai AI VTuber project exists. The closest is ppirch/thai-realtime-voice: a macOS/MLX prototype first committed today (2026-09-25). Borrow its Thai chunking and prompt lessons.

WHAT TO BUILD:
- One Brain task, serial: one decision at a time. Stimuli that arrive mid-decision are queued and merged afterwards (Neuro T1).
- One Speaker: a queue of text chunks where TTS for chunk k+1 is prefetched while chunk k plays (lookahead 1–2; TTS concurrency 1 on the shared 12 GB GPU).
- Speech preemption at chunk boundaries using Neuro SDK priority meanings (T1):
  - low: wait until she finishes speaking.
  - medium: finish the current utterance sooner, i.e. stop after the current chunk.
  - high: shorten the utterance and start the new decision now.
  - critical: stop audio at once, with a 20–30 ms fade.
- Input sources are adapters that push Stimulus(kind, priority, ttl):
  - streamer voice: HIGH;
  - chat: LOW, as a limited window of the last ~20 filtered messages from which the LLM picks one;
  - game action force: its own priority;
  - idle timer: LOW, fires after about 20–40 s of silence, with backoff;
  - operator: CRITICAL.
- The renderer (VTube Studio or our own Live2D page) and the operator panel are subscribers over a local WebSocket. They are never owners, so restarting the renderer does not stop the core (Neuro's 2026 rewrite principle, T1).
- Streaming path: LLM tokens → Thai-aware chunker (first chunk "boosted") → tier-1 keyword filter per chunk → TTS → playback. Blocked output is visibly replaced by the literal "Filtered." and the rest of the utterance is dropped (Neuro T4/T5 behaviour, kimjammer implementation).
- Barge-in: VAD start plus at least 0.5 s of speech, or at least N Thai characters of STT text that is not a backchannel → cut playback. Store only the text actually heard, suffixed "…" plus an interruption marker (OLV pattern).
- Memory: SQLite. A small set of curated long-term slots written by an explicit remember() tool that the model chooses to call (Neuro T1 "depends what you choose to remember"; AIRI's FlowChat experiment), plus per-stream episode summaries. FTS5 trigram for Thai recall; vector search optional later.
- Packaging: uv with a lockfile, CPU/CUDA torch extras marked as conflicts, Python 3.12, setup.ps1 and run.bat, a doctor command, typed layered config, and GPU components in separate processes wherever their CUDA major versions differ.

TL;DR BORROW LIST:
- Neuro (corpus): serial loop; 4-level speech priority; "Filtered."; operator kill switch; chat window; per-character bank.
- kimjammer: Prompter policy function; Injection(text, priority) prompt assembly with context-budget trimming; moderator actions cancel_next and abort_current.
- OLV: ordered parallel TTS by sequence number; heard-text truncation on interrupt; first-comma boost; per-20 ms RMS lip-sync arrays; config versioning and migration.
- AIRI: intent queue/interrupt/replace model; chunker settings boost=2, min 4, max 12 words; in-band <|ACT|>/<|DELAY|>/<|CALL|> tokens synced to speech segments.
- LiveKit: endpointing min 0.5 s / max 3.0 s; interruption min 0.5 s; false-interruption 2.0 s with resume; backchannel boundary 1.0 s; AEC warmup 3 s; preemptive LLM without TTS.
- pipecat: system frames that bypass queues; idle controller semantics; time-to-first-byte (TTFB) metrics; speculation gate.
- thai-realtime-voice: space-based Thai chunks; unspeakable-chunk guard; spoken-Thai prompt rules; low reasoning effort for voice turns; keep-alive HTTP; --text mode for CI.

## Alternatives

### Custom asyncio core in our repo (recommended)
- **Pros:** Built around the Neuro-like serial brain with 4-level speech preemption, multi-source arbitration (voice, chat, game, idle, operator) and chunk-level filter gating, none of which are first-class in any framework. Thai-specific chunking, endpointing and filtering throughout. Minimal dependencies (asyncio, httpx, sounddevice, pythainlp, sqlite3), so uv.lock stays stable. Fully testable on Linux CI with fakes (sketch tests pass without audio or GPU).
- **Cons:** We own the turn-taking edge cases (false interruptions, AEC, drift) that LiveKit and pipecat already solved. More initial code, roughly 1.5–3k lines for the core.
- **When:** Default choice for Pailin: a streamer-centric VTuber with chat and games on a Windows PC.

### Fork Open-LLM-VTuber v1.2.1
- **Pros:** Working end-to-end quickly: Live2D web and Electron frontend, many ASR/TTS/LLM adapters, browser AEC barge-in, MCP tools, group (multi-character) chat, MIT licence.
- **Cons:** v1 is in maintenance while v2 is a planned full rewrite. One brain per WebSocket. Chat is a FIFO with no arbitration. Punctuation-only chunking gives one chunk for Thai. No output filter; memory backends removed (Mem0/MemGPT) or external (Letta). Windows needs a manual CUDA/cuDNN setup. Heavy dependency list (anthropic, azure, cartesia, elevenlabs, groq, letta-client and more).
- **When:** Only as a code reading reference, or a quick demo to show the user what is possible. Not as a base.

### pipecat 1.11 as the voice layer
- **Pros:** Mature frame-based pipeline, interruption frames, idle controller, user-mute strategies, speculation gate, TTFB metrics, many service adapters, BSD-2 licence.
- **Cons:** Python ≥3.11. High release churn. Core pins onnxruntime~=1.24.3. 1:1 conversation model with no livestream arbitration. No Thai sentence segmentation or turn detection; whitespace word counting. Local transport has no AEC.
- **When:** If we later add a Discord or phone-style 1:1 voice-chat mode as a separate service. Even then, wrap it behind our Stimulus/Speaker interfaces.

### LiveKit Agents 1.8 (+ livekit-server)
- **Pros:** Best turn-handling knobs (adaptive interruption with backchannels, false-interruption resume, preemptive generation, AEC warm-up). Console mode with WebRTC APM. Apache-2.0.
- **Cons:** Needs an SFU server and the room/worker model. Turn detector has no Thai and a custom licence. Heavy framework for a single-PC stream.
- **When:** Do not adopt. Copy its parameter defaults. Optionally depend only on the 11 MB `livekit` rtc wheel for AudioProcessingModule AEC.

### AIRI as renderer or stage + Python brain over its plugin protocol
- **Pros:** Polished Live2D/VRM stage, desktop 'tamagotchi', speech pipeline with priorities and intents, in-band ACT/DELAY/CALL tokens, active development (commits 2026-09-25).
- **Cons:** TypeScript/Electron monorepo; pnpm build complexity. Beta releases (0.12.0-beta.5). WebView memory problems (VRM 724 MB RAM in their own mobile benchmarks). Memory still WIP. Tight coupling to their stores.
- **When:** Only if the avatar team decides against VTube Studio or a bespoke web Live2D renderer, and wants a ready stage app. Talk to it over a WebSocket bridge.

### huggingface/speech-to-speech 1.0 as a component server
- **Pros:** Modular VAD→STT→LLM→TTS pipeline exposing OpenAI Realtime GA events over WebSocket/WebRTC, so it can be driven by a standard protocol.
- **Cons:** Defaults are English and Mac-oriented (Parakeet, Qwen3-TTS). Thai support unverified. Another process with its own conversation loop, which conflicts with a single serial brain.
- **When:** Reference for exposing our core through an OpenAI-Realtime-style event API later. Not as the brain.

## Verified facts

- ✅ OLV pyproject version is 1.2.1. The tag v1.2.1 points to a commit dated 2025-08-26 and the last main commit is 2026-05-15. The README states v2.0 is a complete rewrite in early planning and asks users not to open feature issues or PRs on v1. The GitHub page on 2026-09-25 showed about 13.9k stars, 1.7k forks, 111 open issues and an MIT licence.  
  Source: git clone https://github.com/Open-LLM-VTuber/Open-LLM-VTuber (pyproject.toml, README.md, git log) + https://github.com/Open-LLM-VTuber/Open-LLM-VTuber
- ✅ OLV pipeline: agent.chat() yields sentences through a decorator chain (sentence_divider → actions_extractor → display_processor → tts_filter). TTSTaskManager starts one asyncio task per sentence with no concurrency limit and delivers results in order using sequence numbers. Each payload is {type:'audio', audio: base64 WAV, volumes: per-20ms normalised RMS, slice_length:20, display_text, actions:{expressions}, forwarded}. The end of a turn waits for a 'frontend-playback-complete' message from the client.  
  Source: olv/src/open_llm_vtuber/conversations/{tts_manager.py,conversation_utils.py}, utils/stream_audio.py, agent/transformers.py
- ✅ OLV faster_first_response cuts the first sentence at its first comma. After that it splits only on END_PUNCTUATIONS ['.','!','?','。','！','？','...','。。。']. pysbd does not support Thai, and langdetect runs on every buffer. Empirical test: a 150-character Thai stream with spaces but no punctuation produced exactly 1 chunk, flushed at end of stream.  
  Source: olv/src/open_llm_vtuber/utils/sentence_divider.py + local test scratchpad/research/prior_art/thai_chunk_test.py
- ✅ How OLV does barge-in 'without headphones' (README: 'AI won't hear its own voice'). Capture and playback both happen in the browser. @ricky0123/vad-web 0.0.24 MicVAD forces getUserMedia {echoCancellation:true, autoGainControl:true, noiseSuppression:true}. Silero v5 frames are 512 samples (32 ms). When onSpeechRealStart fires in the 'thinking-speaking' state, the page stops audio and sends {type:'interrupt-signal', text: fullResponse-heard-so-far}. The backend cancels the conversation task and handle_interrupt() rewrites the last assistant message as heard+'...' and appends '[Interrupted by user]' with role 'user' or 'system' (interrupt_method).  
  Source: olvweb/src/renderer/src/context/vad-context.tsx, hooks/utils/use-interrupt.ts; https://cdn.jsdelivr.net/npm/@ricky0123/vad-web@0.0.24/dist/real-time-vad.js; olv/src/open_llm_vtuber/agent/agents/basic_memory_agent.py
- ✅ OLV VAD defaults. Frontend: positiveSpeechThreshold 0.50, negativeSpeechThreshold 0.35, redemptionFrames 35 (about 1.12 s end-of-speech), preSpeechPadFrames 20. Backend Silero: prob_threshold 0.4, db_threshold 60, required_hits 3 (about 0.1 s), required_misses 24 (about 0.77 s).  
  Source: olvweb vad-context.tsx; olv/src/open_llm_vtuber/vad/silero.py
- ✅ OLV live chat (Bilibili): each danmaku is forwarded as {type:'text-input'} into ProxyMessageQueue, a FIFO deque consumed only when no conversation is active, one full turn per message. There is no batching, selection or content filter. Proactive speech comes from a frontend idle timer (default 5 s) that sends 'ai-speak-signal'. The backend then uses the prompt 'Please say something that would be engaging and appropriate for the current context.' with skip_memory and skip_history set.  
  Source: olv/src/open_llm_vtuber/proxy_message_queue.py, live/bilibili_live.py, conversations/conversation_handler.py; olvweb proactive-speak-context.tsx
- ✅ OLV memory: basic_memory_agent is an in-RAM message list reloaded from chat_history. The Letta agent needs a separately run Letta server. MemGPT is marked 'temporarily removed'. Issue #256 (Aug 2025) says Mem0 and long-term memory had been removed 5 months earlier and that Letta is hard to configure.  
  Source: olv config_templates/conf.default.yaml; https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/issues/256
- ✅ OLV and AIRI contain no output moderation or blocklist code; a grep for moderation/blacklist/profanity found nothing. OLV only preprocesses TTS text (drops brackets, parentheses, asterisks, angle brackets, emoji).  
  Source: grep over olv/src and airi/{packages,apps,plugins}
- ✅ On Windows, OLV's uv.lock resolves torch 2.10.0 from PyPI. The win_amd64 wheel is 113,723,116 bytes (CPU-only) against 915,607,863 bytes for the Linux CUDA wheel. The uv docs confirm that PyPI hosts CPU-only torch wheels for Windows and macOS.  
  Source: olv/uv.lock; https://raw.githubusercontent.com/astral-sh/uv/main/docs/guides/integration/pytorch.md
- ✅ OLV Windows install path: uv (the docs note that before v1.0.0 they used conda and 'received a significant number of Python-related questions'), git with the frontend as a git submodule (missing submodule gives {"detail":"Not Found"}), `winget install ffmpeg`, and a manual CUDA Toolkit + cuDNN copy into Program Files for GPU. Chrome is the only recommended browser. The unsigned Electron client triggers SmartScreen. Issue #219: sherpa-onnx GPU is not on by default ('cuda.dll' problems on Windows).  
  Source: https://github.com/Open-LLM-VTuber/open-llm-vtuber.github.io (docs quick-start.md, faq.md); https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/issues/219
- ✅ OLV issue #444 (2026-09-01): gemini-3.5-flash-lite multi-turn function calling fails with HTTP 400 'Function call is missing a thought_signature' because the app drops Google's thought_signature from stored history.  
  Source: https://github.com/Open-LLM-VTuber/Open-LLM-VTuber/issues/444
- ✅ A 2026-09-07 user report of OLV running locally on an RTX 4070 Ti SUPER 16 GB (single machine): LLM first token 0.63 s, full text 1.57 s, TTS 1.30 s (124 chars → 19.5 s of audio), speech start 2.87 s, LLM+TTS VRAM about 13.7 GB. Gemma-4-26B used 15.7 GB, so the author switched to gemma-4-12b Q5_K_M at 11.8 GB. Windows OpenSSH starts processes in Session 0, where CUDA contexts fail (llama-server exits silently). Running .venv\Scripts\python.exe directly left uvx off PATH. mcp 2.x caused a TypeError, so the author pinned mcp[cli]<2. Launching the client twice gave two WebSocket connections and every service ran twice.  
  Source: https://note.com/suishin_ai/n/nd6a985a45b19 (raw page parsed)
- ✅ kimjammer/Neuro: last commit 2025-01-16, a 7-day recreation, MIT. Built on an RTX 4070 12 GB, Windows 11, Python 3.11.9, torch 2.2.2+cu118. LLM: Llama-3-8B-Instruct EXL2 4.0bpw via text-generation-webui. STT: RealtimeSTT with faster-whisper tiny.en. TTS: RealtimeTTS with Coqui XTTSv2 + DeepSpeed (Windows wheels from AllTalk). Audio goes to VTube Studio over VB-Cable. Device selection uses hard-coded indices (INPUT_DEVICE_INDEX=1, OUTPUT_DEVICE_INDEX=7).  
  Source: https://github.com/kimjammer/Neuro (README.md, constants.py, tts.py, stt.py)
- ✅ kimjammer Prompter polls every 0.1 s and prompts when STT and TTS are ready, nobody is speaking or thinking, and one of: a new STT message, pending Twitch messages, or idle time over PATIENCE=60 s. The LLM streams fully before TTS starts (tts.play(full_message)). Output filter: any(bad_word in text.lower().split()) replaces the message with 'Filtered.'. blacklist.txt contains only 'turkey'. Moderator API: cancel_next (drop the generation before it is spoken) and abort_current (stop TTS).  
  Source: kimneuro/prompter.py, llmWrappers/abstractLLMWrapper.py, socketioServer.py, blacklist.txt
- ✅ kimjammer Injection(text, priority): injections are sorted ascending so the highest priority lands at the end of the prompt. Values in code: system 10, memory 60, chat history 100, Twitch 150. The prompt is trimmed by dropping the oldest messages until it is under 90% of CONTEXT_SIZE (8192). Twitch keeps about the last 10 messages of up to 300 chars and adds the instruction 'Pick the highest quality message with the most potential for an interesting answer and respond to them.'  
  Source: kimneuro/modules/injection.py, llmWrappers/abstractLLMWrapper.py, modules/twitchClient.py
- ✅ kimjammer memory: a ChromaDB PersistentClient (telemetry had to be disabled). Every 20 new messages it runs a reflection ('3 most salient high level questions' → Q&A pairs) and stores them as 'short-term'. Recall takes the top 5 by querying with the last 5 messages plus chat, injected at priority 60. Supports JSON import/export.  
  Source: kimneuro/modules/memory.py, memories/readme.md, constants.py
- ✅ kimjammer's self-reported latency, devlog 3 (02 Jan 2024): 'speech happens only after ~1 second after input speech ends'. Devlog 2: switching to ExLlamav2_HF went from about 10 to about 50 tok/s, and 12 GB VRAM was nearly full.  
  Source: https://blog.kimjammer.com/neuro-dev-log-3/ , https://blog.kimjammer.com/neuro-dev-log-2/
- ✅ AIRI: root package version 0.12.0-beta.5, MIT, TS/Vue/Electron pnpm monorepo, last commit 2026-09-25. The README memory checklist shows 'Memory Alaya (WIP)' unchecked. packages/pipelines-audio has priorities critical 300 / high 200 / normal 100 / low 0, intent behaviours 'queue'|'interrupt'|'replace', and ttsMaxConcurrent default 4. The chunker uses boost=2, minimumWords=4, maximumWords=12, word counting via Intl.Segmenter, and splits only at punctuation. In-band control tokens: <|ACT {...}|>, <|DELAY n|>, <|CALL ["name", {...}]|>.  
  Source: git clone https://github.com/moeru-ai/airi (package.json, README.md, packages/pipelines-audio/src/*)
- ✅ AIRI DevLog 2026-01-01, FlowChat memory experiment: '1. Create a memory table. 2. Provide the LLM with a tool function. When it determines something needs to be remembered, it summarizes what to remember in a declarative sentence, then calls this tool function. 3. When requesting a new reply each time, concatenate all memories into the system prompt.'  
  Source: airi/docs/content/en/blog/DevLog-2026.01.01/index.md
- ✅ pipecat-ai 1.11.0 was uploaded to PyPI 2026-09-18 and requires Python >=3.11 (BSD-2). 1.0.0 was released 2026-04-14. Core dependencies pin onnxruntime~=1.24.3 and numba. LocalAudioTransport uses pyaudio. VAD defaults: confidence 0.7, start 0.2 s, stop 0.2 s, min_volume 0.6. SpeechTimeout user_speech_timeout defaults to 0.6 s. SmartTurn uses STOP_SECS 3, PRE_SPEECH_MS 500, MAX_DURATION 8 s. Sentence end detection uses sentencex==1.0.31; a Thai paragraph came back as 1 sentence (tested). MinWordsUserTurnStartStrategy counts words with len(frame.text.split()).  
  Source: https://pypi.org/pypi/pipecat-ai/json; git clone https://github.com/pipecat-ai/pipecat (pyproject.toml, CHANGELOG.md, src/pipecat/...); local sentencex test
- ✅ Smart Turn v3.2 (pipecat-ai/smart-turn-v3, BSD-2): 8M parameters, Whisper-tiny encoder, 8 MB int8 ONNX. Benchmarked on 23 languages, with no Thai. Tonal languages score lowest: Vietnamese 79.38%, Chinese 85.79%.  
  Source: https://huggingface.co/pipecat-ai/smart-turn-v3 (README.md, benchmarks/smart-turn-v3.2-cpu.md)
- ✅ livekit-agents 1.8.3 (PyPI 2026-09-23), Apache-2.0. TurnHandlingOptions defaults:
- endpointing: fixed, min_delay 0.5, max_delay 3.0 (streaming STT 0.3 / 2.5)
- interruption: min_duration 0.5, min_words 0, false_interruption_timeout 2.0, resume_false_interruption True, backchannel_boundary (1.0, 1.0), discard_audio_if_uninterruptible True
- preemptive_generation: enabled, preemptive_tts False, max_speech_duration 10.0, max_retries 3
- session: user_away_timeout 15.0, aec_warmup_duration 3.0 (interruptions ignored while AEC converges)  
  Source: git clone https://github.com/livekit/agents (livekit-agents/livekit/agents/voice/turn.py, agent_session.py)
- ✅ The LiveKit multilingual turn-detector model supports 14 languages (en, es, fr, de, it, pt, nl, zh, ja, ko, id, tr, ru, hi), none of them Thai, under license 'livekit-model-license' (other).  
  Source: https://huggingface.co/livekit/turn-detector (README + API cardData)
- ✅ The livekit Python package 1.1.20 ships an 11.0 MB win_amd64 wheel (deps: protobuf, aiofiles, numpy) that exposes the WebRTC APM: rtc.AudioProcessingModule(echo_cancellation, noise_suppression, high_pass_filter, auto_gain_control) with process_stream, process_reverse_stream and set_stream_delay_ms on 10 ms int16 frames. LiveKit's console mode uses it with sounddevice. My synthetic Linux test with a broadband far-end signal and 30 ms delay: echo went from -29.2 dB to -66.6 dB; near-end during double-talk went from -20.0 dB to -23.5 dB.  
  Source: https://pypi.org/pypi/livekit/1.1.20/json; lkagents/livekit-agents/livekit/agents/cli/_legacy.py; scratchpad/research/prior_art/aec_test2.py
- ✅ ppirch/thai-realtime-voice (macOS/MLX, first and last commits 2026-09-25):
- Thai chunker: hard split on punctuation, soft split at a space once 30 chars are buffered (min 12, max 60).
- Benchmark: LLM time-to-first-chunk dominates (11.8 s average against 0.3 s TTS per chunk and 0.4 s STT).
- reasoning_effort=low cut first audio from 2.4 s to 1.4 s.
- MMS/VITS Thai overflows on digit- or punctuation-only chunks ('3.').
- A spoken-Thai prompt (1–2 short sentences, numbers written as words, no lists, markdown or emoji) removed unspeakable chunks.  
  Source: git clone https://github.com/ppirch/thai-realtime-voice (README.md, benchmark/RESULTS.md, src/realtime_voice/text.py)
- ✅ No maintained open-source Thai AI VTuber project was found. proj-airi/awesome-ai-vtubers (updated 2026-04-15) lists none. thunyoubun/TTS-VTuberAI-Anime describes itself as 'no AI yet'.  
  Source: https://github.com/proj-airi/awesome-ai-vtubers ; web searches 2026-09-25
- ✅ pythaiasr 2.1.0 (2026-09-14, Apache-2.0) defaults to Typhoon ASR as FastConformer RNN-T on ONNX Runtime, avoiding NeMo. It supports stream_asr for live microphone input.  
  Source: https://github.com/PyThaiNLP/pythaiasr README; https://pypi.org/pypi/pythaiasr/json
- ✅ The RealtimeSTT troubleshooting docs say 'Parakeet/NeMo and Qwen vLLM are Linux-oriented; use WSL2 for those real-model paths on a Windows workstation'. Scripts need an `if __name__ == "__main__":` guard on Windows because RealtimeSTT uses multiprocessing.  
  Source: https://raw.githubusercontent.com/KoljaB/RealtimeSTT/master/docs/troubleshooting.md and README.md
- ✅ torch 2.14.0 (PyPI 2026-09-02) has Windows CUDA wheels only on the cu126, cu130 and cu132 indexes (cp310–cp315); cu128 and cu129 have none. onnxruntime-gpu 1.30.0's [cuda] extra requires nvidia-cuda-runtime~=13.0. faster-whisper 1.2.1 (via ctranslate2 4.8.2) needs cuBLAS for CUDA 12 and cuDNN 9.  
  Source: https://download.pytorch.org/whl/{cu126,cu128,cu129,cu130,cu132}/torch/ ; https://pypi.org/pypi/onnxruntime-gpu/1.30.0/json ; https://raw.githubusercontent.com/SYSTRAN/faster-whisper/master/README.md
- ✅ Official uv PyTorch guidance: add explicit indexes (explicit = true), route torch through [tool.uv.sources] by extra or marker, and declare the extras as [tool.uv] conflicts. `--torch-backend=auto` exists only in the `uv pip` interface. The Windows uv installer puts uv in $HOME/.local/bin.  
  Source: https://raw.githubusercontent.com/astral-sh/uv/main/docs/guides/integration/pytorch.md ; https://astral.sh/uv/install.ps1
- ✅ The huggingface_hub cache on Windows without Developer Mode or admin falls back to a no-symlink mode that stores files directly in snapshots/, which is less efficient.  
  Source: https://raw.githubusercontent.com/huggingface/huggingface_hub/main/docs/source/en/guides/manage-cache.md
- ✅ Neuro SDK priority semantics (Vedal-authored SPECIFICATION.md, T1): low = wait until she finishes speaking; medium = finish the current utterance sooner; high = process immediately, shortening the utterance; critical = interrupt speech and respond at once. The speech_finished message is {isFinal, cancelled?, reason?}.  
  Source: https://raw.githubusercontent.com/VedalAI/neuro-sdk/main/API/SPECIFICATION.md (copy in scratchpad/research/neuro-sdk-repo)
- ✅ PyThaiNLP 5.3.8 newmm tokenizer: first call about 330 ms (dictionary load); warm calls about 0.11 ms per 100 characters, measured in the Linux container. Thai keyword filtering: whitespace-split matching misses Thai insults, raw substring matching over-blocks (a 2-character profanity is a prefix of หีบ 'box'), and matching on newmm tokens is correct.  
  Source: scratchpad/research/prior_art/thai_filter.py and inline tests
- ✅ SQLite 3.45 FTS5 with tokenize='trigram' finds Thai substrings (e.g. 'มันไก่') and case-insensitive Latin text in mixed Thai/English memory rows. Queries need at least 3 characters.  
  Source: local sqlite3 test (Linux, sqlite 3.45.1)
- ✅ huggingface/speech-to-speech 1.0.0 (PyPI 2026-09-06) is a modular VAD→STT→LLM→TTS pipeline that exposes the OpenAI Realtime GA event set over WebSocket and WebRTC.  
  Source: https://raw.githubusercontent.com/huggingface/speech-to-speech/main/README.md ; https://pypi.org/pypi/speech-to-speech/json
- ⚠️ unverified — The minimum NVIDIA Windows driver for cu130 wheels is R580 or newer; for cu126 it is R560 or newer.  
  Source: general NVIDIA CUDA compatibility knowledge; not re-checked this session

## Install (Windows)

```
Scope: the orchestration core only. STT, TTS, LLM and avatar installs belong to the other component reports. Target: Windows 11, RTX 4070, Python 3.12.

1) Prerequisites
- Up-to-date NVIDIA driver. cu130 wheels need R580 or newer (not re-verified); otherwise use the cu126 extra.
- Git.
- No system CUDA Toolkit and no manual cuDNN copy. OLV's docs require both; we do not.
- Start everything from a desktop session, never over SSH or as a Windows service. Session 0 cannot create a CUDA context (note.com 2026-09 report).

2) Install uv and Python
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"   # installs to %USERPROFILE%\.local\bin
  (or: winget install --id=astral-sh.uv -e)   then restart the terminal
  uv python install 3.12

3) Set up the repo
  git clone https://github.com/Krx-21/AI_Vtube && cd AI_Vtube   # no submodules: ship the built renderer in the repo or download it on first run with a checksum
  uv sync --frozen --extra cu130      # user's GPU PC
  uv sync --frozen --extra cpu        # CI on ubuntu-latest and windows-latest
  uv run pailin doctor                # audio devices by NAME, CUDA visible, VRAM free, model files, config valid, ports free
  uv run pailin run --text            # keyboard-in / speaker-out smoke test (thai-realtime-voice pattern)

4) pyproject.toml snippet (official uv pattern)
  [project]
  requires-python = ">=3.12,<3.13"
  [project.optional-dependencies]
  cpu   = ["torch==2.14.0"]
  cu130 = ["torch==2.14.0"]
  [tool.uv]
  conflicts = [[{ extra = "cpu" }, { extra = "cu130" }]]
  [tool.uv.sources]
  torch = [{ index = "pytorch-cpu", extra = "cpu" }, { index = "pytorch-cu130", extra = "cu130" }]
  [[tool.uv.index]]
  name = "pytorch-cpu"
  url = "https://download.pytorch.org/whl/cpu"
  explicit = true
  [[tool.uv.index]]
  name = "pytorch-cu130"
  url = "https://download.pytorch.org/whl/cu130"
  explicit = true

  Notes:
  - torch 2.14.0 Windows CUDA wheels exist only for cu126, cu130 and cu132; there is no cu128.
  - If a component uses ctranslate2/faster-whisper (CUDA 12 + cuDNN 9) while torch or onnxruntime-gpu use CUDA 13, run that component in its own process (its own venv or exe) or add the pip nvidia-*-cu12 wheels via os.add_dll_directory. Never rely on one process finding both.

5) setup.ps1 (one-click)
  $ErrorActionPreference='Stop'
  if (-not (Get-Command uv -EA SilentlyContinue)) { irm https://astral.sh/uv/install.ps1 | iex; $env:Path="$env:USERPROFILE\.local\bin;$env:Path" }
  uv python install 3.12
  $extra = if (Get-Command nvidia-smi -EA SilentlyContinue) {'cu130'} else {'cpu'}
  uv sync --frozen --extra $extra
  $env:HF_HOME="$PSScriptRoot\models\hf"   # keep multi-GB models off %USERPROFILE%; without Developer Mode the HF cache copies instead of symlinking
  uv run pailin doctor

6) run.bat
  @echo off
  cd /d "%~dp0"
  set HF_HOME=%~dp0models\hf
  uv run --frozen pailin run %*
  pause

7) Configuration layout
- config/defaults.toml: committed, versioned with schema_version.
- config/user.toml: small overrides written by the user.
- .env: secrets (TYPHOON_API_KEY, GEMINI_API_KEY, TWITCH_*).
- characters/<id>.toml: persona, voice, filter overrides, memory DB path.
- Validate with pydantic-settings at startup and print human-readable Thai/English errors.
- Include a `pailin config migrate` command, modelled on OLV's upgrade_codes/config_sync.py with backups.
```

## API notes

All snippets are tested unless marked otherwise. Files are in /tmp/claude-0/-home-user-AI-Vtube/02efebb2-c032-5a2c-81bb-f71e8a5461a3/scratchpad/research/prior_art/ (thai_chunker.py, thai_filter.py, sketch/core_loop.py, sketch/test_core_loop.py with 2 passing pytest tests, aec_test2.py, thai_chunk_test.py).

A) Thai-aware streaming chunker. Fixes the OLV/pipecat/AIRI failure where Thai text gets one chunk because they split only on punctuation. Tested: a 150-char Thai stream gives 4 chunks in 0.2 ms; an unspaced run gets a word-safe cut via newmm.
```python
import re
from typing import Iterable, Iterator
from pythainlp.tokenize import word_tokenize   # warm up once at startup (cold 330 ms, warm 0.11 ms)
HARD=set(".!?。！？\n…"); SOFT=set(",;:，、")
PARTICLE_END=re.compile(r"(ค่ะ|คะ|ครับ|นะ|น้า|จ้า|จ้ะ|เลย|ด้วย|แหละ|สิ|ล่ะ|หรอ|เหรอ|มั้ย|ไหม)$")
def _safe_cut(buf,mx):
    pos=0
    for w in word_tokenize(buf[:mx+20],engine="newmm",keep_whitespace=True):
        if pos+len(w)>mx: break
        pos+=len(w)
    return pos or mx
def speech_chunks(tokens:Iterable[str],first_min=6,first_target=14,min_chars=12,target=40,max_chars=90)->Iterator[str]:
    buf,first="",True
    for tok in tokens:
        buf+=tok
        while True:
            lo,tgt=(first_min,first_target) if first else (min_chars,target); cut=None
            for i,ch in enumerate(buf):
                if ch in HARD and i+1>=lo: cut=i+1; break
            if cut is None and len(buf)>=tgt:
                sp=[m.start() for m in re.finditer(" ",buf) if m.start()>=lo]
                strong=[s for s in sp if PARTICLE_END.search(buf[:s])]
                pick=(strong or sp or [None])[0]
                if pick is not None and pick<=max_chars: cut=pick+1
                elif any(c in SOFT for c in buf[lo:]): cut=next(i for i,c in enumerate(buf) if i>=lo and c in SOFT)+1
            if cut is None and len(buf)>=max_chars: cut=_safe_cut(buf,max_chars)
            if cut is None: break
            chunk,buf=buf[:cut],buf[cut:]
            if chunk.strip(): yield chunk.strip(); first=False
    if buf.strip(): yield buf.strip()
```
Also drop or merge chunks that contain no Thai or Latin letters, e.g. '3.' (VITS overflow seen in thai-realtime-voice). Strip [emotion], <|ACT|>, (…) and *…* before TTS but keep them for the avatar (OLV tts_preprocessor, AIRI special segments).

B) Serial brain plus preemptible speaker (full file: sketch/core_loop.py). Core API:
```python
class Prio(IntEnum): LOW=0; MEDIUM=1; HIGH=2; CRITICAL=3   # Neuro SDK meaning (T1)
@dataclass
class Stimulus: kind:str; text:str; prio:Prio=Prio.LOW; ts:float=field(default_factory=time.monotonic); ttl:float=60.0
class Speaker:            # say(async_iter_of_chunks); TTS of chunk k+1 prefetched while k plays; .heard = chunks fully played
    def preempt(self,p):  # CRITICAL -> task.cancel() (fade 20-30 ms, flush device); MEDIUM/HIGH -> stop after current chunk
class Brain:              # submit(stimulus) -> inbox + speaker.preempt(prio); run(): drain inbox -> drop expired ->
                          # pick top non-chat stimulus; if its prio < HIGH also attach last <=20 chat msgs (LLM picks one);
                          # idle stimulus when nothing pending for idle_after s; one decision at a time; history gets
                          # ' '.join(heard) + ' …[ถูกขัดจังหวะ]' when cut (OLV heard-text truncation)
```
Tests: chat arriving mid-decision is merged into the next decision; CRITICAL preempts; MEDIUM cuts at a chunk boundary and marks history. Implementation notes:
- HIGH should also start the next LLM call immediately, in parallel with the last chunk.
- Game forces carry their own ttl. The SDK allows about 20 s for action results.

C) State machine. Suggested defaults: LiveKit/pipecat values as the baseline, adjusted for Thai.
- States: IDLE → LISTENING (VAD start ≥ 0.2–0.3 s) → ENDPOINTING (silence) → THINKING (LLM before first chunk) → SPEAKING (chunks playing) → IDLE. INTERRUPTED is transient.
- Endpointing:
  - silence min 0.6 s / max 2.5 s (LiveKit 0.5/3.0; pipecat 0.6; OLV about 0.8–1.1 s).
  - Thai: extend by +0.3 s when the partial transcript ends in a continuation word (และ, แต่, ว่า, ที่, เพราะ, คือ) and cut to 0.4 s after a final particle (ค่ะ, ครับ, นะ, จ้า).
  - Smart Turn and LiveKit EOU have no Thai; use them only as an optional tie-break after validation.
- Barge-in: VAD speech ≥ 0.5 s (LiveKit min_duration) OR STT text ≥ 6 Thai chars that is not a backchannel.
  - Backchannel set: {อืม, อือ, อ๋อ, ค่ะ, ครับ, จ้า, เหรอ, จริงดิ, 555}, suppressed within 1.0 s of the start or end of the bot's utterance (LiveKit backchannel_boundary).
  - Ignore barge-in during the first 3 s of the session while AEC converges (LiveKit aec_warmup_duration).
  - If no transcript arrives within 2.0 s after a pause, treat it as a false interruption and resume or restart the remaining chunks (LiveKit resume_false_interruption).
  - Do not count words with str.split(); use PyThaiNLP or character counts.
- Preemptive generation: start the LLM on a stable interim transcript but hold TTS until the endpoint is confirmed; cancel if the final transcript differs (LiveKit preemptive_tts=False, max_speech_duration 10 s, max_retries 3; pipecat SpeculationGate).
- Idle chatter: the timer starts at the bot's playback end, is cancelled by any speech, and is suppressed during a user turn (pipecat UserIdleController semantics). Base 25 s with jitter; 60 s in kimjammer; 5 s in OLV is too spammy for a stream. Keep proactive lines in history so she does not repeat herself; OLV skips memory for proactive turns.

D) Echo cancellation in Python. Needed only if the streamer uses speakers, or if the AI must hear game or voice-chat audio. Keep capture and playback in the same process: OLV works 'without headphones' only because the browser does both.
```python
from livekit import rtc; import numpy as np, sounddevice as sd
apm=rtc.AudioProcessingModule(echo_cancellation=True,noise_suppression=True,high_pass_filter=True,auto_gain_control=True)
SR,F=48000,480      # 10 ms frames, int16 mono, mandatory
def to_frame(x_i16): return rtc.AudioFrame(data=x_i16.tobytes(),samples_per_channel=F,sample_rate=SR,num_channels=1)
# playback callback: for each 10 ms slice of what you are about to play -> apm.process_reverse_stream(to_frame(slice))
# capture callback:  apm.set_stream_delay_ms(int((out_latency+in_latency)*1000)); fr=to_frame(mic_slice); apm.process_stream(fr)
#                    cleaned = np.frombuffer(bytes(fr.data), np.int16)  -> VAD/STT
```
In the synthetic test echo dropped by about 37 dB and near-end speech survived. Validate on the real room and USB mic. Clock drift between separate input and output devices degrades AEC.

E) Prompt assembly (kimjammer Injection pattern, extended)
```python
@dataclass(order=True)
class Injection: priority:int; text:str=field(compare=False)
# persona 10 (stable prefix -> llama.cpp prompt cache), long-term memory slots 30, stream summary 40,
# history 100, game state 120, chat window 150, current stimulus 200 (last = most attended)
def build(injs, n_tokens, ctx, budget=0.9):
    parts=sorted(injs)
    while n_tokens(parts) > budget*ctx: drop_oldest_history(parts)   # kimjammer: drop oldest until <90%
    return "".join(p.text for p in parts)
```
Show chat as quoted data, never as instructions:
<chat_window> 1) [ต้นกล้า] "..." 2) [Mira_TH] "..." </chat_window>
Then: 'เลือกตอบข้อความที่น่าสนใจที่สุดเพียง 1 ข้อความ ถ้าไม่มีก็คุยต่อได้' ("pick the one most interesting message to answer; if none, keep chatting"). Cap each message at 200–300 chars, dedupe, and filter before inclusion.

F) Internal event and renderer WebSocket schema (proposal; merges OLV message types with Neuro SDK speech semantics). One JSON message per event, with ids for ordering.
- {"type":"state","value":"idle|listening|thinking|speaking","turn_id":"t42"}
- {"type":"speech.segment","turn_id":"t42","seq":3,"text":"...","display":"...","audio_ms":1840,"volumes":[...20ms RMS...],"expressions":["joy"]}   (OLV sends base64 WAV plus volumes; we play audio in the core and send only visemes/volumes to the renderer)
- {"type":"speech.finished","turn_id":"t42","is_final":true,"cancelled":false,"reason":null}   (Neuro SDK speech_finished)
- {"type":"filtered","turn_id":"t42","seq":4,"tier":"keyword|classifier|operator"}   (the renderer shows "Filtered.")
- operator → core: {"type":"op.mute"} {"type":"op.skip"} {"type":"op.cancel_next"} {"type":"op.say","text":"..."} {"type":"op.llm","enabled":false} {"type":"op.filter.reload"} (kimjammer control panel set)
- Two characters: every message carries "character":"pailin|<twin>"; Neuro SDK uses characterId per connection.

G) Memory. SQLite in stdlib, WAL, FTS5 trigram; tested on Thai.
```sql
CREATE TABLE memory(id INTEGER PRIMARY KEY, character TEXT NOT NULL, kind TEXT NOT NULL CHECK(kind IN('core','fact','episode','viewer')),
  subject TEXT, text TEXT NOT NULL, importance INTEGER NOT NULL DEFAULT 3 CHECK(importance BETWEEN 1 AND 5),
  created_at REAL NOT NULL, last_used_at REAL, uses INTEGER NOT NULL DEFAULT 0, source TEXT, pinned INTEGER NOT NULL DEFAULT 0);
CREATE VIRTUAL TABLE memory_fts USING fts5(text, subject, content='memory', content_rowid='id', tokenize='trigram');
```
Tool the LLM calls (AIRI FlowChat / Neuro 'choose what to remember'):
{"name":"remember","parameters":{"type":"object","properties":{"text":{"type":"string","description":"one declarative Thai sentence"},"subject":{"type":"string"},"importance":{"type":"integer","minimum":1,"maximum":5}},"required":["text"]}}
- Always inject: all 'core' slots (cap about 20–30 lines, like Evil's 3 slots) plus viewer facts for usernames in the current chat window (exact subject match) plus the top 3 FTS hits.
- End-of-stream job: summarise the stream into 1 'episode' row and dedupe or merge facts. kimjammer's 20-message Q&A reflection is a reference.
- Keep a per-character DB file.

H) Thai keyword gate (tier 1; tier 2 is an async LLM classifier on the complete utterance or a sampled subset)
```python
from pythainlp.tokenize import word_tokenize; from pythainlp.util import normalize as th_normalize
import re, unicodedata
_ZW=re.compile(r"[​-‍⁠﻿]")
def norm(s): s=th_normalize(unicodedata.normalize("NFC",_ZW.sub("",s))); return re.sub(r"(.)\1{2,}",r"\1\1",s.lower())
def hit(text, token_words:set, substrings:tuple):
    t=norm(text)
    return next((s for s in substrings if s in t), None) or next((w for w in word_tokenize(t,engine="newmm") if w in token_words), None)
```
Apply to (a) each chat message before it enters the window and (b) each speech chunk before TTS. On a hit: skip the chunk, emit "Filtered." (display, plus an optional short pre-recorded voice line), cancel the rest of the utterance, and log for the operator.

I) Reference configs from the other projects, for parity tests:
- AIRI tts-chunker: boost=2, minimumWords=4, maximumWords=12.
- LiveKit AgentSession(turn_handling={"endpointing":{"min_delay":0.5,"max_delay":3.0},"interruption":{"min_duration":0.5,"false_interruption_timeout":2.0,"resume_false_interruption":True,"backchannel_boundary":(1.0,1.0)},"preemptive_generation":{"enabled":True,"preemptive_tts":False}}).
- pipecat VADParams(confidence=0.7,start_secs=0.2,stop_secs=0.2,min_volume=0.6).
- OLV vad-web positive 0.50, negative 0.35, redemptionFrames 35.

## Latency & resources

Time-to-first-audio (TTFA), measured from the end of the streamer's speech, is the sum of:
t_endpoint (silence wait) + t_stt_final + t_prompt + t_llm_ttft + t_first_chunk_tokens + t_filter + t_tts_first_audio + t_output_buffer

Reference points (prior art, not measured on the user's PC):
- kimjammer (English, RTX 4070 12 GB, 7B EXL2 at about 50 tok/s, faster-whisper tiny.en via RealtimeSTT, XTTSv2 streaming): about 1 s after speech ends, self-reported Jan 2024. The LLM finished before TTS started; replies were about one sentence.
- OLV (Japanese, RTX 4070 Ti SUPER, gemma-4 12B Q5_K_M + OmniVoice, Sept 2026): LLM TTFT 0.63 s, full text 1.57 s, TTS 1.30 s, speech start 2.87 s. With LLM and TTS on a second PC: 2.35 s. LLM+TTS VRAM about 13.7 GB.
- thai-realtime-voice (Thai, API LLM, MMS-TTS on Apple MPS): first audio 11–14 s with a slow reasoning endpoint; 2.3 s after keep-alive plus max_tokens 1000; 1.4 s with reasoning_effort=low. Endpoint pause about 1 s on top of that. TTS about 0.3 s per 30-char chunk; STT 0.37–0.41 s.
- Neuro (corpus): about 700 ms average in 2024 (wiki-relayed); 'seconds, not milliseconds' for the full voice-chat cycle (T1 2026); Vedal accepts added latency for reasoning.

Target budget for Pailin on the RTX 4070 (design targets, to be benchmarked):
| Stage | Target |
|---|---|
| endpoint | 0.4–0.8 s (Thai particle-aware) |
| streaming STT final | ≤ 0.2 s after endpoint |
| prompt build | < 5 ms (keep the persona prefix stable for KV/prompt-cache reuse) |
| local LLM TTFT | ≤ 0.3 s warm; cloud API 0.5–2 s |
| first-chunk tokens | about 6–14 Thai chars ≈ 3–8 tokens ≈ 0.05–0.15 s |
| keyword filter | < 1 ms (newmm warm 0.11 ms) |
| TTS first audio | 0.15–0.4 s |
| output buffer | 20–60 ms |
| TOTAL | about 1.2–2.0 s local, 1.8–3 s with a cloud LLM |

Overlap rules that deliver this:
1) Stream LLM tokens into the chunker; never wait for the full reply (kimjammer did).
2) Boost the first chunk: small minimum, cut at the first particle-space or comma (OLV faster_first_response, AIRI boost=2).
3) Prefetch: synthesise chunk k+1 while chunk k plays. Lookahead 1–2 with TTS concurrency 1 on the shared GPU. OLV's unbounded per-sentence tasks and AIRI's default of 4 suit cloud TTS, not a 12 GB GPU shared with the LLM.
4) Keep ordering with sequence numbers (OLV TTSTaskManager).
5) Preemptive LLM start on a stable interim transcript, no TTS until the endpoint is confirmed (LiveKit).
6) Warm everything at startup: LLM prompt cache, TTS first call, PyThaiNLP dictionary (330 ms cold), VAD session.
7) Use keep-alive HTTP clients (httpx.AsyncClient).
8) No or low reasoning for voice replies; reserve reasoning for game decisions or out-of-band tasks.
9) Play audio from the core process via sounddevice/WASAPI in 10–20 ms blocks. Send only volumes or visemes to the renderer; OLV's base64-WAV-per-sentence adds encode and transfer time plus file I/O per sentence.
10) Log per-turn timestamps for every stage (pipecat TTFB metrics; thai-realtime-voice per-turn stderr timings) and add a benchmark CLI.

Resources:
- The core itself is about 50–150 MB RAM and negligible CPU.
- Budget VRAM for the 12 GB card. Kimjammer's 12 GB was 'just about' filled by an 8B LLM plus XTTS plus Whisper; the OLV user needed 11.8 GB for a 12B Q5_K_M alone. Plan on LLM ≤ 7–8 GB, TTS ≤ 2–3 GB, STT on CPU (Typhoon ASR ONNX via pythaiasr, or streaming on CPU), about 1 GB headroom for OBS NVENC.
- The WebRTC APM costs a few percent of one core at 48 kHz.

## Pitfalls

- Thai sentence splitting. OLV (pysbd/regex), pipecat (sentencex 1.0.31), LiveKit (blingfire/basic) and AIRI all split only on punctuation, and Thai LLM output rarely has any. Result: one TTS chunk per reply and TTFA equal to the whole LLM time (verified for OLV and sentencex). Use a space- and particle-aware chunker with word-safe cuts.
- Thai word counting. pipecat's MinWords interruption uses text.split(), and kimjammer's blacklist uses text.lower().split(). Both break for Thai because words are not separated by spaces. Use PyThaiNLP newmm, and warm it up at startup (330 ms cold).
- Thai turn detection. Smart Turn v3.2 (23 languages) and LiveKit EOU (14 languages) include no Thai, and tonal languages score worst (Vietnamese 79%). Default to silence plus Thai particle heuristics and validate any semantic model on recordings of the user.
- One brain per connection (OLV ServiceContext per WebSocket). A second browser or Electron client doubles LLM, TTS and MCP work and plays audio twice (reported Sept 2026). Keep a single core; renderers and control panels are subscribers.
- Unbounded chat FIFO (OLV ProxyMessageQueue) creates a growing backlog and answers go stale. Use a bounded window with TTL, dedupe, a per-user rate limit, and let the LLM pick one message (kimjammer, and Neuro's 'limited window' T1).
- Waiting for the full LLM reply before TTS (kimjammer) is acceptable only for one-sentence replies. Long or reasoning replies multiply latency.
- Unbounded parallel TTS on a shared GPU (OLV spawns one task per sentence) competes with LLM decoding. Cap concurrency at 1–2 and prefetch only one chunk ahead.
- Barge-in without AEC. pipecat LocalAudioTransport (pyaudio) and kimjammer have none, so the AI interrupts itself when using speakers. OLV avoids this only because the browser owns both mic and playback. If Python plays TTS while the browser captures the mic, browser AEC has no reference. Use headphones, or keep capture and playback in one process with the WebRTC APM (livekit rtc), and ignore barge-in during AEC warm-up (LiveKit 3 s).
- Heard-text accounting. After an interrupt, store only what was actually played (OLV: heard + '...' + '[Interrupted by user]', with role 'system' or 'user' depending on the provider). Otherwise the model believes it said things the audience never heard.
- Proactive or idle lines excluded from memory (OLV skip_memory) and very short idle timers (OLV default 5 s) cause repetitive filler. Keep proactive turns in history, use 20–40 s with jitter, and back off when chat is active.
- Hard-coded audio device indices (kimjammer: input 1, output 7) break when USB devices re-enumerate. Store device NAME plus host API (WASAPI), resolve at startup, and let `doctor` list the devices.
- Windows torch trap: PyPI torch on Windows is CPU-only (OLV's lock pulls a 113 MB wheel). Use uv explicit indexes with extras and conflicts. torch 2.14 Windows CUDA wheels exist only for cu126, cu130 and cu132.
- CUDA major-version mixing: ctranslate2/faster-whisper need CUDA 12 + cuDNN 9, while torch cu130 and onnxruntime-gpu ≥1.27 use CUDA 13. Isolate such components in separate processes or manage DLL directories explicitly. Never require a system CUDA Toolkit or cuDNN copy (OLV docs do).
- NeMo on Windows: RealtimeSTT docs say to use WSL2 for Parakeet/NeMo. For Typhoon ASR prefer the ONNX path (pythaiasr 2.1.0) or a separate process, not nemo-toolkit in the core venv.
- Windows process model: GPU services started over OpenSSH or as a service run in Session 0 and cannot create a CUDA context (llama-server exits silently). Running .venv\\Scripts\\python.exe directly can leave tools like uvx off PATH. Always launch with run.bat or `uv run` from the desktop.
- Libraries that use multiprocessing (RealtimeSTT) need an `if __name__ == '__main__':` guard on Windows.
- Dependency churn: pipecat shipped 12 minor versions in about 5 months and pins onnxruntime~=1.24.3; LiveKit deprecated its turn arguments in favour of turn_handling; mcp 2.x broke OLV (users pinned mcp<2). Pin everything with uv.lock `--frozen` and wrap third-party APIs behind our own interfaces.
- Provider-specific history fields: Gemini 3.x needs thought_signature echoed back in multi-turn function calling (OLV #444, HTTP 400). Store raw provider message parts, not a lossy OpenAI-shaped history.
- Git submodules and unsigned binaries: a missing OLV frontend submodule gives {"detail":"Not Found"}, and the unsigned Electron app triggers SmartScreen. Ship the renderer as static files in the repo and avoid building an exe.
- Huge single YAML (OLV conf.yaml is about 500 lines covering every provider) plus a separate model_dict.json produces mismatches (OLV FAQ: 'NoneType has no attribute emo_str'). Use typed, layered config, validation, and migrations.
- Hugging Face cache on Windows without Developer Mode falls back to copying files instead of symlinking (more disk). Set HF_HOME inside the project or on a data drive.
- Output moderation is absent in OLV and AIRI, and kimjammer only matches words after full generation. With streaming TTS the filter must gate every chunk before synthesis. Neuro's lessons: two tiers (keyword plus LLM classifier), a visible 'Filtered.', an operator mute/kill switch, and a human moderator present.
- Thai filter false positives: raw substring matching blocks innocent words (a profanity that is a prefix of หีบ 'box'). Match on newmm tokens after normalisation (zero-width characters, repeated letters, Thai vowel order) and keep substring matching only for unambiguous strings such as URLs and invite links.
- VITS-family Thai TTS crashes or overflows on chunks made only of digits or punctuation ('3.'). Guard these chunks and prompt the LLM to write numbers as Thai words with no lists or markdown.
- Reasoning models: thought tokens delay first audio (thai-realtime-voice: 2.4 s at default effort vs 1.4 s at low), and a small max_tokens can starve the answer entirely. Use low or no reasoning for chat replies, keep reasoning for games, and never speak <think> content (OLV think-tag handling).
- Treat clone designs as clone designs. kimjammer's Injection object, RAG memory, faster-whisper/XTTS stack and VB-Cable routing are not evidence about Neuro-sama (see corpus C05/C11).

## Open questions

- Does the streamer wear headphones on stream? If yes, AEC is optional and barge-in can be simple VAD. If they use speakers, we need in-process WebRTC APM with capture and playback in the same process.
- Where does the TTS audio play: the Python core via WASAPI, the browser renderer, or into VTube Studio through a virtual cable? This decides where lip-sync volumes are computed and where the AEC reference comes from.
- Thai endpointing thresholds (silence 0.4–0.8 s, particle rules) and the backchannel list need tuning on 10–20 minutes of the streamer's real speech. Should we record a calibration set?
- Chat platforms and volume (Twitch, YouTube, TikTok, Facebook Live?). This sets the window size, TTL and per-user rate limits.
- Is a human moderator present during streams to use the operator panel (Neuro: 'there currently needs to be a human there')? Which default Thai blocklist and LLM-classifier policy should we use?
- Should STT, TTS and LLM run as separate processes, for crash isolation and to allow mixing CUDA 12 and 13? This would also match Neuro's decoupled core and renderer. What IPC: local HTTP, WebSocket or ZeroMQ?
- Idle-chatter policy: frequency, allowed topics, and whether idle lines may reference long-term memories or recent chat.
- Memory scope: how many core slots per character, whether viewer facts need consent or opt-out, and whether the end-of-stream consolidation runs automatically.
- Second character (Evil-twin style): same core with two personas taking turns (OLV group-conversation round robin), or two cores sharing a GPU lock?
- Should voice replies use a reasoning model at all? Suggested: none or low for chat, reasoning only for game decisions within the SDK's roughly 20 s budget.
- Not verified this session: NVIDIA driver minimums for cu130 and cu126 wheels, the bundled SQLite version (FTS5 trigram needs 3.34+) in Windows Python 3.12, and whether WebRTC APM behaves on the user's actual mic and speaker clock-drift setup.
