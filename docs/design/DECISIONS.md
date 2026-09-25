# Design decisions (ADR index)

## Base architecture

**Decision:** Use Draft 2's robustness-first skeleton: launcher, core, voice worker and llama-server as separate processes. Graft in Draft 1's cache discipline and speculation (speculation behind M2 flags) and Draft 3's contracts, game design and memory quarantine.

**Rationale:** Robustness is the part that is hard to add later and decides whether a 3-hour stream survives. Latency tricks and parity features layer cleanly onto a supervised multi-process skeleton.

## Python and packaging

**Decision:** Python 3.12 on the PC (requires-python >=3.11,<3.13). One package, aivtube. uv with a frozen lockfile. No torch in the main venv.

**Rationale:** onnxruntime 1.30, av 18 and websockets 17 all need Python 3.11 or newer. VoxCPM needs <3.13. Leaving torch out avoids the Windows trap where PyPI serves CPU-only torch, and avoids mixing CUDA 12 and 13 in one process.

## Process model

**Decision:** A stdlib-only launcher with a Job Object and an emergency hard-kill endpoint on 8779. The core runs asyncio. The voice worker owns audio I/O, AEC, VAD, STT, TTS and lip tracks, and runs at ABOVE_NORMAL priority. llama-server is adopted if it is already running.

**Rationale:** This isolates native crashes and GIL stalls from the brain. AEC needs capture and playback in one process. The hard kill works even when the core is wedged. The LLM's KV cache survives core restarts.

## Timestamps

**Decision:** Every timestamp is time.perf_counter(). Sub-20 ms pacing uses a PrecisionTicker thread. asyncio.sleep is used only for timers of 50 ms or more.

**Rationale:** On Windows with Python 3.12, time.monotonic() and loop.time() come from GetTickCount64, which has ~15.6 ms resolution. perf_counter is system-wide since Python 3.10, so timestamps compare across processes.

## Core-voice IPC

**Decision:** WebSocket JSON on 127.0.0.1:8771, authenticated by a token. Only filtered text segments go to the voice worker. Events and lip tracks come back.

**Rationale:** websockets is already a dependency. PCM never crosses the link. Invariant I7 holds structurally, and the worker goes silent if the link drops (I1).

## LLM backend and model

**Decision:** llama.cpp llama-server b11177, CUDA 13.4 zip (12.4 if the driver is older than R580), serving the mradermacher Typhoon2.5-Qwen3-30B-A3B Q4_K_M GGUF. The Typhoon2.5-4B Q4_K_M is a cold standby and the light profile.

**Rationale:** Typhoon 2.5 is Thai-tuned, Apache-2.0, non-thinking, and supports Hermes tools. Only llama.cpp supports expert-only MoE offload on a 12 GB card. The mradermacher GGUFs embed the chat template; without it, tools are silently dropped.

## LLM placement

**Decision:** Start with --fit on, -c 24576 and --fit-target 3584. After bench llm --tune, pin -fit off -ngl all --n-cpu-moe N. On a load OOM, raise N by 2 and retry. Always pass --slot-save-path. Assert /props supports_tool_calls at startup.

**Rationale:** Pinning N before anything is measured risks a crash loop. --fit only measures VRAM at launch, so OBS scenes or games opened later need about 3.5 GB of headroom.

## LLM fallback and privacy

**Decision:** Fallback chain: local-30b → local-4b → typhoon-api → gemini → canned line. Cloud providers are skipped unless privacy.cloud_llm_consent is set. No hedging by default. Fallback happens only before the first emitted event. promote/rollback plus auto-rollback.

**Rationale:** The Typhoon API may train on inputs, and Gemini's free tier data is used by Google. Changing provider mid-utterance changes the voice or persona. Auto-rollback mirrors Neuro's T1 July 2026 pattern of keeping the old model ready.

## Prompt cache discipline

**Decision:** Layout goes stable to volatile: static persona and tools, then an epoch context block, then append-only compact history, then a small volatile tail. Epoch rebuilds are prewarmed on an inactive, double-buffered speak slot. Assistant history text is frozen at first render. A CI prefix byte-stability test and a panel cache alarm guard it.

**Rationale:** A cold prefill on the 30B takes seconds. Double-buffered slots let the context block change without ever putting a cold cache on the critical path.

## STT and VAD

**Decision:** A Silero VAD v6 ONNX numpy wrapper feeds Typhoon ASR Realtime int8 ONNX in sherpa-onnx on the CPU, one pass per segment. PyThaiASR is the fallback. The Typhoon API is opt-in. A typhoon-whisper-turbo CT2 worker is an M2 option.

**Rationale:** Measured at 140–240 ms on a weak CPU, with 0 VRAM, no NeMo or torch, and native Windows wheels. We self-export with CC-BY attribution because the third-party ONNX copy has no licence.

## ORT coexistence

**Decision:** Spike S3 and a Windows CI test load the python onnxruntime package and sherpa-onnx together in one process. If that fails, vad.backend becomes silero_sherpa (with our own pre-roll), or STT moves to its own process.

**Rationale:** sherpa-onnx bundles its own onnxruntime DLL. On Windows the first DLL loaded wins.

## Endpointing

**Decision:** M1 uses a 600 ms silence endpoint with 300 ms pre-roll. A forced split at 15 s decodes the audio but does not end the turn (60 s maximum). Particle-aware endpointing and speculation arrive in M2 behind flags, after calibration.

**Rationale:** No semantic end-of-turn model supports Thai. The particle heuristic is untested. A mid-monologue split must never trigger a reply.

## Barge-in

**Decision:** The reflex runs locally in the voice worker: duck −12 dB, confirm (≥500 ms of speech plus ≥3 non-backchannel characters from a quick decode), then cut. The core only sets policy. Headphones by default, livekit AEC3 with speakers, energy_dtd or half_duplex as fallbacks.

**Rationale:** Cutting locally reaches silence in ≤100 ms regardless of core load. The confirm step avoids stopping on coughs and backchannels. AEC3 measured best in synthetic tests.

## Voice addressing

**Decision:** Mic modes open/ptt/deafened switchable by panel, HTTP hotkeys and console. Addressing is always (co-host) or name_or_question. Read-aloud dedupe matches transcripts against recent chat. No decision starts while the streamer is speaking.

**Rationale:** A co-host that answers everything will reply to the streamer reading chat aloud or talking to teammates.

## TTS

**Decision:** edge-tts th-TH-PremwadeeNeural as the premwadee identity, with same-voice Azure substitution per segment. Piper is allowed only for a whole utterance. Captions overlay as the last resort. Streaming PyAV decode, prefetch 2, 2 s and 4 s first-audio timeouts, cached Filtered./filler phrases. Azure F0 is token-bucketed with min_chars 60.

**Rationale:** Premwadee is the only female Thai Edge voice and exists on Azure too, so the identity is preserved. From a datacentre, novel-text latency was bad, so S1 decides the order. Piper drops Latin text and its voice is non-commercial.

## Thai text pipeline

**Decision:** The TTS brief's incremental ThaiSpeechChunker, with first chunk 8–60 chars and later chunks 40–160, a stall flush, and no cuts before combining marks. Normalise after chunking. Emotion tags only when the emotion changes. No forced opener, but opener repetition is tracked.

**Rationale:** Chunkers that split on punctuation only produce one chunk per Thai reply. The first-chunk boost comes from the chunker, not from formulaic openers.

## Avatar

**Decision:** VTube Studio driven by our own websockets client (proxy=None, no pings, responses matched on requestID plus messageType). Parameters injected at 60 Hz by the PrecisionTicker, with lip tracks from our own PCM and a 40 ms lead. Expressions via ExpressionActivationRequest. Custom parameter fallback on error 454.

**Rationale:** No rendering code to maintain, licensing handled by VTS, and the renderer can restart without affecting the core (T1). pyvts lacks request matching.

## Chat in M1

**Decision:** Twitch anonymous IRC by default. YouTube list polling with an API key if S9 shows YouTube is primary. Generic POST /api/event for donation tools. EventSub, Helix and alert adapters come in M3.

**Rationale:** Anonymous IRC needs zero setup and is verified working. Thai streamers may be YouTube-first. Donations often arrive through third-party tools.

## Memory

**Decision:** One SQLite file per character (WAL, FTS5 trigram). Short-term turns plus epochs. 16 core slots, each ≤120 characters, written only when the model calls remember, with SlotsFull forcing a choice. Viewer facts. Episode summaries. Chat-sourced writes quarantined. Automatic backups.

**Rationale:** Mirrors Neuro's model-chosen, slot-like, cross-stream memory (T1/T5) while blocking viewer memory poisoning. Trigram FTS handles Thai substrings.

## Safety

**Decision:** Tier-0 synchronous filter: newmm tokens plus Aho-Corasick plus regex. Despaced and leet forms, prev_tail(40) + chunk on output, checks on tool and memory arguments. The Filtered. clip, with blocked text never stored in history or memory. The monarchy_112 category fails closed. Tier-1 typhoon2-safety ONNX in M2. Auto-strict on by default, auto-freeze opt-in.

**Rationale:** Covers split-phrase and letter-spelling bypasses, Thai legal exposure, and the visible Neuro behaviour. Auto-freeze can be baited by trolls.

## Kill ladder

**Decision:** SKIP, MUTE, FREEZE (≤150 ms), HARD KILL via the launcher (TerminateProcess on voice, held until REARM, ≤300 ms, independent of the core), then OBS as a last resort.

**Rationale:** Human moderation plus a force-mute panel is Neuro's model (T1/T5). A wedged core must still be stoppable.

## Ports

**Decision:** 8000 Pailin SDK, 8010 twin SDK, 8080/8081 llama, 8765/8766 reserved for the renderer, 8770 panel, 8771 IPC, 8779 emergency. Never bind 8001–8009 (VTube Studio). Checked by doctor.

**Rationale:** Fixes the port collision with VTS and keeps the panel clear of the browser renderer's ports.

## Games (M4)

**Decision:** neuro_compat on localhost:8000. A static game_action tool with available actions listed in the tail. Forced actions via a sanitised json_schema grammar with action-first early dispatch, falling back to tools with tool_choice=required. Retries configurable (default 3).

**Rationale:** Keeps the prompt cache intact. The game is frozen until the action arrives. llama.cpp's schema-to-grammar conversion silently drops unsupported features.

## Milestone order

**Decision:** M0 contracts and spikes; M1 core loop; M2 latency and safety pack; M3 tools and channel actions; M4 games; M5 twin; M6 local TTS and renderer; M7 vision and search; M8 singing.

**Rationale:** Latency is what makes it feel like Neuro. Tier-1 safety must precede moderation tools. Games reuse the validated tool and argument path. The twin needs a stable core. Local TTS competes with the 30B for VRAM.

## Testing

**Decision:** Every Protocol has a fake and a contract suite. FakeClock for timers. A sim harness drives the real core and voice worker against fakes over localhost with structural latency invariants. CI on ubuntu {3.11, 3.12} and windows 3.12, with Windows-only tests for the Job Object and ORT coexistence. Nightly model tests.

**Rationale:** Lets coding agents build modules in parallel without GPU or audio while still catching latency and cancellation regressions deterministically.

## Config

**Decision:** TOML layered in this order: defaults, then profile, then character, then user.toml, then env, then CLI. Validated with pydantic v2 and bilingual errors. Secrets only in .env. Profiles: stream, light, gaming, text, offline, ci.

**Rationale:** Avoids one huge YAML with mismatched files (an Open-LLM-VTuber lesson), and makes backend profiles for this PC explicit.

# Milestones

## M0 Contracts, fakes and on-PC spikes

Code: contracts, events, contract modules, config, infra, all fakes and contract suites, packaging and the CI skeleton, plus `aivtube run --text --fake-llm`. Spikes on the streamer's PC: S1 TTS TTFA from a Thai IP, S2 llama tune with OBS+VTS running, S3 onnxruntime+sherpa coexistence, S4 WASAPI latency and underflows, S5 VTS injection jitter and error 454, S6 Typhoon RT ONNX export, S7 speech calibration recording, S8 KV prefill check, S9 chat platform and donation tool.

**Done when:** CI is green on ubuntu and windows (including the ORT coexistence test). Contracts are frozen at 1.0. The text console runs on fakes. An ADR is written for each spike, with gates G1 (TTS order) and G2 (default LLM profile) decided and written to user.toml.

## M1 Smallest real thing

Thai voice loop: mic → VAD → Typhoon RT → serial brain → streaming 30B/4B → Thai chunker → tier-0 gate → edge/Azure → WASAPI. Barge-in with duck/confirm/cut. Mic modes: PTT, deafen, addressing. VTS lip-sync, emotions and idle motion. One chat source (Twitch anonymous IRC, or YouTube polling) plus /api/event. Text console. Control panel with kill ladder and captions overlay. Short- and long-term memory with remember/forget and quarantine. Launcher supervision and hard kill. setup, doctor, bench and report.

**Done when:** All 10 M1 exit criteria in ARCHITECTURE.md §1 pass on the PC, including: voice p50 ≤1.7 s (30B) or ≤1.3 s (4B) and p95 ≤3.0 s; FREEZE ≤150 ms; HARD KILL ≤300 ms with a wedged core; the chaos drill recovers; the red-team set has 0 leaks; cache ratio ≥0.85; 2 h soak passes; CI and the sim e2e are green.

## M2 Latency & safety pack

Speculative endpointing with gated playback. Particle endpointing after calibration. KV prefill at speech start (if S8 passes). Pre-think pipelining. Idle precompute. Tier-1 typhoon2-safety ONNX classifier. Review mode. Optional Whisper accuracy worker.

**Done when:** Voice turn p50 ≤1.3 s and p95 ≤2.5 s on the 30B profile. Speculation hit rate ≥70% with no false-endpoint increase above 5%. The tier-1 false-positive rate is measured on a labelled Thai slang set and thresholds are set. Sim speculation hit/miss scenarios are green.

## M3 Tools & channel actions

Twitch Device Code Flow, EventSub and Helix. Tools: set_stream_title, create_poll, timeout_user (≤600 s, never a ban), spin_model, set_talking_speed, play_sound. Redeems. YouTube streamList and actions. Alert-service adapters (after a spike).

**Done when:** The tool_abuse sim scenario is blocked by policy (caps, protected mods/VIPs, argument filtering, approval queue). Every tool and sound can be disabled live, and timeouts can be undone. A 30-minute live test with tools enabled has zero incidents.

## M4 Games (Neuro-SDK-compatible)

neuro_compat hub on localhost:8000, the static game_action tool, grammar-forced actions with schema sanitiser and early dispatch, tools-strategy fallback, retries, speech_finished, and the voice-socket decline.

**Done when:** Conformance against the neuro-api 4.0.0 client, Tony's test game and the Unity TicTacToe example. Force→action p50 ≤2 s. The game is never left frozen by a filtered or muted utterance.

## M5 Twin

Second CharacterRuntime, FloorManager, second VTS via UDP discovery, SDK port 8010, call_twin, Azure Achara voice, separate memory and filter overlay per character.

**Done when:** No overlapping speech. Floor hand-off ≤300 ms. The anti-ping-pong limit works. M1 latency targets still hold with both characters loaded.

## M6 Local TTS + renderer alternatives

VoxCPM2 sidecar in its own venv (paired with the light or cloud LLM profile), browser renderer fallback, TikTok read.

**Done when:** The local voice streams its first chunk in <0.5 s on the 4070 with 12 GB fitting the chosen profile. The browser sink passes the avatar contract suite.

## M7 Vision + search

Screen captioning as VISION stimuli; web_search tool with filtered results.

**Done when:** Vision never delays speech. Search results pass the input filter. Cloud VLM use is gated by consent.

## M8 Singing/karaoke

Offline stems, song mode with barge-in off and vocal-stem lip-sync, duets.

**Done when:** FREEZE stops a song within 150 ms. Lip-sync follows the vocal stem. Duets use the floor manager.

# Risks

- edge-tts Premwadee is slow or flaky on novel text (1.6–5 s TTFA and ~25% failures measured from a datacenter), and the endpoint is unofficial and grey-area. Mitigation: spike S1/gate G1 from the streamer's Thai connection, then Azure with the same voice, prefetch, a filler at 1.2 s, cached phrases, a pinned version, and a captions overlay as the last resort.
- Typhoon2.5-30B-A3B may be too slow on this RAM and GPU split (only an estimate of 30–50 tok/s exists), or VRAM may be contended by OBS, VTS or games. Mitigation: spike S2/gate G2, light and gaming profiles, hot-swap from the panel, 3.5 GB of fit headroom, and automatic n-cpu-moe stepping on out-of-memory.
- Prompt-cache invalidation would cost 4–10 s of cold prefill on the 30B. Mitigation: epochs with double-buffered speak slots, static tools, game actions in the tail only, history text frozen at first render, a CI test for prefix byte-stability, a panel alarm on cache_n, and slot save/restore.
- The onnxruntime package and sherpa-onnx's bundled ORT DLL may conflict in one Windows process. Mitigation: spike S3 plus a Windows CI test, a silero_sherpa VAD fallback, or moving STT to its own process.
- Typhoon RT garbles English words and misspells ไพลิน. Mitigation: an alias map, a prompt that tolerates misspellings, the M2 Whisper worker, and a CER eval set built from the streamer's VODs.
- False barge-ins from game audio, laughter or echo, and replies when the streamer was not talking to her. Mitigation: headphones by default, confirm-before-cut with a backchannel filter, a 2 s false-interruption resume, AEC3 with warm-up, PTT/deafen hotkeys, addressing modes, read-aloud dedupe, and no decisions while the streamer is speaking.
- Harmful output on stream, Thai §112 exposure, or a Twitch ban (Neuro was banned for 2 weeks in Jan 2023, T1). Mitigation: tier-0 on every chunk including the previous tail and despaced/leet forms, the monarchy category fails closed, the Filtered. path, tier-1 in M2, a human moderator, the kill ladder, review mode for the first streams, and an audit log.
- Prompt injection and memory poisoning via chat, display names or game strings. Mitigation: quoted untrusted blocks, role-token stripping, policy enforced in code, quarantine of chat-sourced memory, and write caps.
- Tool misuse (timeouts, polls, titles, sounds). Mitigation: tools ship only in M3, after tier-1; caps and rate limits; mods/VIPs protected; never bans; an approval queue; undo; per-tool and per-sound disable.
- Native crashes or GIL contention in the voice worker (PortAudio, livekit, sherpa, PyAV, TTS network code). Mitigation: process isolation at ABOVE_NORMAL priority, a 1 ms switch interval, allocation-free callbacks, the S4 soak test, and the option to move TTS to its own process.
- Windows timer resolution (15.6 ms for monotonic and loop.time on Py3.12) breaks lip-sync and latency timing. Mitigation: perf_counter everywhere and a PrecisionTicker thread with a 30 Hz fallback.
- The official typhoon-ai GGUFs lack a chat template, so tools are silently ignored. Mitigation: use the mradermacher GGUFs (or --chat-template-file) and assert the /props tool capabilities at startup.
- Azure F0 quota (20 transactions/min, not adjustable) is exhausted mid-stream. Mitigation: a token bucket, min_chars 60, and no hedging or speculative synthesis on F0.
- Cloud fallbacks leak stream data (the Typhoon API may train on inputs; the Gemini free tier is used by Google). Mitigation: explicit opt-in consent and a local-only default.
- Dependency and API churn (openai 3.x with httpx2, llama.cpp flags, websockets proxy defaults, Neuro SDK spec changes). Mitigation: uv.lock --frozen, binaries pinned by sha256, thin adapters, contract tests, nightly integration, and configurable retry counts.
- Licensing: Typhoon ASR is CC-BY (needs attribution), the Piper voice is non-commercial, monetised streams need the paid VTS version, Live2D sample notices apply, TikTokLive is AGPL, and the Neuro name is trademarked. Mitigation: a NOTICE file, doctor flags, Cubism Core never committed, and M4 described only as 'compatible with the Neuro Game SDK protocol'.
- Unverified alert-service APIs (StreamElements/Streamlabs). Mitigation: M1 relies on a generic POST /api/event, and M3 adapters need a spike and an ADR first.
- Parallel coding agents drift apart. Mitigation: contracts frozen at M0, ADRs for changes, contract suites, and sim scenarios exercising every seam on every PR.
