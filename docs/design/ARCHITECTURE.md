# AI_Vtube: Final Architecture (v1.0, 2026-09-25)

- **Package:** `aivtube`. **Repo:** `Krx-21/AI_Vtube`, rebuilt from scratch (the old code is deleted).
- **Default character:** ไพลิน (Pailin), a cheerful Thai girl who speaks Thai mixed with English words.
- **Runtime target:** Windows 11, RTX 4070 12 GB, i7-14700KF (8 P-cores + 12 E-cores), 64 GB DDR5.
- **Dev and CI:** a Linux container with no GPU and no audio devices, plus GitHub Actions on ubuntu and windows.

How this design was assembled:
- **Skeleton:** Draft 2 (robust): a launcher, a separate voice worker, a filtered-only link between core and voice, and a PRE_SHOW → Go Live gate.
- **From Draft 1 (latency):** prompt-cache discipline and slot adoption. Its speculation and prefill features are grafted in behind flags and ship in M2.
- **From Draft 3 (parity):** contract suites, the game design, memory quarantine and filter overlays.
- **Review fixes:** every judge must-fix item is fixed. Appendix D maps each one to where it is handled.

**Provenance.** Statements about Neuro-sama carry an evidence tier: T1 means Vedal's own words or docs, T2 means official repositories, T4/T5 means wiki-grade sources. Everything else here is our own design choice. Designs taken from clone projects (Open-LLM-VTuber, kimjammer, AIRI, LiveKit, pipecat) are named as such and are never presented as how Neuro works. Neuro's LLM, STT, TTS and memory mechanism are all unknown.

## 0. Decisions at a glance

| Area | Default on this PC | Fallback / alternative |
|---|---|---|
| Runtime | Python 3.12 (3.11 is the floor). One package, `aivtube`, with an asyncio core. `uv` with a frozen lockfile. **No torch in the main venv.** | – |
| Processes | `launcher` (stdlib only, Windows Job Object, out-of-process hard kill) supervises `core` (the brain), `voice` (all audio I/O, VAD, STT, TTS, lip tracks) and `llama-server.exe`. VTube Studio (VTS) and OBS are external peers. | GPU Python sidecars (Whisper in M2, VoxCPM2 in M6) run as separate processes |
| LLM | llama.cpp `llama-server` build b11177, CUDA 13.4 zip (12.4 zip if the driver is older than R580), running **Typhoon2.5-Qwen3-30B-A3B Q4_K_M** (mradermacher GGUF, which embeds the chat template). Starts with `--fit on --fit-target 3584`; `--n-cpu-moe` is pinned only after `bench llm --tune` | 4B Q4_K_M as a cold standby, then Typhoon API (opt-in), then Gemini 3.5 Flash-Lite (opt-in), then a canned line |
| STT/VAD | Silero VAD v6 ONNX through a numpy/onnxruntime wrapper, then **Typhoon ASR Realtime int8 ONNX** in sherpa-onnx 1.13.8 on the CPU (0 VRAM). End of utterance = 600 ms of silence | PyThaiASR; Typhoon ASR API (opt-in); typhoon-whisper-turbo CT2 worker (M2) |
| TTS | edge-tts 7.2.8, voice `th-TH-PremwadeeNeural` (+20Hz, +8%), streamed through a PyAV decoder, 2 segments prefetched, 2 s first-audio timeout | Azure with the same voice (per segment, token-bucketed), then Piper (whole utterance only, non-commercial voice), then a captions overlay |
| Audio | sounddevice 0.5.6 on WASAPI shared mode, an always-open 48 kHz player, markers timed to when audio reaches the DAC. Assumes headphones; uses livekit AEC3 when the streamer uses speakers | energy double-talk detection, half-duplex |
| Avatar | VTS through our own ~150-line client. Parameters injected at 60 Hz by a precision ticker thread. Lip-sync is computed from our own PCM | Browser renderer (M6) |
| Chat (M1) | Twitch anonymous IRC, or YouTube `liveChatMessages.list` polling if spike S9 shows YouTube is primary. Also `POST /api/event` for donation/alert tools | EventSub/Helix, YouTube streamList, alert-service adapters (M3) |
| Memory | One SQLite file per character (WAL, FTS5 trigram). Turns are the short-term memory. 16 long-term slots, written only when the model calls a `remember` tool. Prompt epochs. Writes that originate from chat are quarantined | Vector search later, behind the same Protocol |
| Safety | Tier-0 Thai-aware keyword/regex filter on chat input, on **every output chunk plus the previous 40 characters**, and on tool and memory arguments. Blocked output is replaced by the literal "Filtered.". A four-step kill ladder with a hard kill outside the core | Tier-1 typhoon2-safety ONNX classifier (M2) |
| Latency | Voice turn p50 ≤ 1.7 s on the 30B, ≤ 1.3 s on the 4B. The M2 latency pack brings the 30B to ≤ 1.3 s | – |

### 0.1 Neuro-parity matrix

| Neuro behaviour (evidence tier) | Our seam | Milestone |
|---|---|---|
| Serial decision loop: "she can only process one thing at once". Context that arrives mid-decision is merged afterwards (T1) | `Brain`, `DecisionSlot`, `Arbiter.drain_context` | M1 |
| Speech priorities low/medium/high/critical; "critical will interrupt her speech"; speech in segments with a final marker (T1, SDK) | `decide_preemption`, `SpeechOutput.stop`, `SegmentDone` / `UtteranceDone` events | M1 (SDK wire format in M4) |
| Short- and long-term memory across streams: "you shouldn't forget anything that happens in a stream", "it depends what you choose to remember" (T1). Evil reportedly had 3 long-term slots (T5) | SQLite turns, 16 slots written through the `remember` tool, epoch summaries | M1 |
| Filters "some powered by AI … needs to be a human there to moderate" (T1). Blocked output shows the literal "Filtered." (T4/T5). Keyword plus LLM tiers (T5). Input filtering since 2021 (T5). A force-mute panel (T5). An extra filter per platform (T4). Speech bugs "hard coded out of her speech" (T1) | `SafetyGate` tiers, replace rules, per-platform and per-character overlays, kill ladder | M1 / M2 |
| "Picks what to respond to within a limited window" of chat (T1). How many messages per turn is unknown | `ChatWindow` offers k=3 candidates and the LLM picks one (k is our choice) | M1 |
| Core decoupled from the renderer: "Unity can be restarted perfectly fine" (T1, 2026) | VTS is only a sink. Core, voice and llama-server each restart on their own | M1 |
| LLM hot-swap with the old model "ready to switch back at a moment's notice" (T1, Jul 2026) | `LLMRouter.promote()` / `rollback()` plus automatic rollback | M1 |
| "Iteration 18" tools: stream title, talking speed, spin model, timeouts, polls, soundboard, call twin, Discord, Google search (T4/T5). The "pipes" sound was removed after abuse (T5) | `ToolRegistry`, with every tool and every sound individually disableable | M3 (twin call in M5, search in M7) |
| Neuro Game SDK: plain-JSON WebSocket, forced actions, ~20 s result timeout, automatic retries (T1/T2 docs, 2026) | `neuro_compat` hub on port 8000 | M4 |
| Twins share one machine and one GPU ("you have to share it", T1), each with their own voice, filter and memory (T4/T5) | One `CharacterRuntime` per character, a shared router, `FloorManager` | M5 |
| Redeems and karaoke duets were still missing at the 2026 cutover (T1) | Redeems become SUPPORT stimuli (M3); a separate singing subsystem (M8) | – |
| During malfunctions she repeated "Someone tell Vedal there is a problem with my AI" (T5) | A canned "brain freeze" line plus a panel alarm | M1 |

## 1. Goals, non-goals, milestones

### Goals
- **G1 Reliability.** A 2–4 h stream runs with nobody touching a terminal. Any single component failure degrades the stream and recovers on its own; none ends the stream.
- **G2 Latency.** The §5 targets are met.
- **G3 Operator control.** FREEZE silences her within 150 ms. HARD KILL works even when the core's event loop is wedged. Every decision can be reconstructed from the logs.
- **G4 Hobbyist install.** One script, one command. No admin rights, no system-wide CUDA, no ffmpeg, no compiler.
- **G5 Neuro-like behaviour,** per §0.1.
- **G6 Parallel buildability.** Contracts are frozen at M0. Every Protocol has a fake and a contract suite. Everything runs without a GPU or audio device.

### Non-goals (v1)
- Training or fine-tuning models.
- SaaS hosting, or a Linux/macOS runtime (those are development environments only).
- Shipping our own Live2D renderer by default. A renderer that loads arbitrary models counts as a Cubism "Expandable Application" and needs Live2D approval.
- Real-time singing synthesis.
- High-APM game control inside the LLM loop.
- Running unattended without a human moderator.
- Claiming parity with Neuro's engines.

### M0: contracts, fakes and on-PC spikes (about 1 week)

Code deliverables: §3 contracts, config, infra, every fake and contract suite, the CI skeleton, and `aivtube run --text --fake-llm`.

Spikes run on the streamer's PC. Each spike ends in an ADR in `docs/adr/`.

| Spike | What is measured | Gate / decision |
|---|---|---|
| S1 TTS | `bench tts`: 40 random Thai sentences from the Thai residential connection, edge vs Azure | **G1:** if edge p50 time-to-first-audio (TTFA) on novel text is above 0.7 s, or more than 5 % of requests fail, Azure becomes the first backend of the `premwadee` identity. On Azure F0, `min_chars` is 60 |
| S2 LLM | `bench llm --tune`: llama-bench sweeping `-ncmoe` 28..44 and `-t` 6/8/12, then warm time-to-first-token (TTFT) with a 300-token uncached suffix. **OBS and VTS must be running** | **G2:** if the 30B's warm TTFT p50 is above 0.6 s, or generation is below 30 tok/s, the default profile becomes `light` (4B), with the 30B available by hot-swap. The tuned N is written to `user.toml` |
| S3 ORT coexistence | The python `onnxruntime` package (running Silero) and sherpa-onnx (which bundles its own onnxruntime) loaded and run in one Windows process. Also a CI test | If it fails: `vad.backend = "silero_sherpa"` (sherpa's own runtime plus our own pre-roll ring buffer), or move STT to its own process |
| S4 Audio | WASAPI `stream.latency` and underflow count at 0.04 s / 480-sample blocks, over 10 minutes with LLM, STT and TTS active | Sets `output_latency_s` (0.04–0.06) |
| S5 VTS | Token auth, jitter of the 60 Hz ticker, error 454, whether VoiceA/I/U/E/O can be injected | Mouth parameter set; run at 60 Hz or 30 Hz |
| S6 STT export | NeMo 3.0.0 export to int8 ONNX (CI job), published to the project's own HF repo with CC-BY-4.0 attribution; latency on the i7 | STT `num_threads`. PyThaiASR is the fallback until the export is published |
| S7 Calibration | 10–20 minutes of the streamer's real speech | Backchannel list, endpoint thresholds, data for the M2 particle heuristic |
| S8 KV prefill | `/completion` with `n_predict:1` on `id_slot`, then read `cache_n` on the next chat request | Turn the M2 prefill trick on or off |
| S9 Platform | Which chat platform is primary (Twitch, YouTube, FB, TikTok), and which donation tool the streamer uses | Selects the M1 chat adapter |

### M1: the smallest real thing (weeks 2–5)

Scope:
- Thai voice conversation: mic → VAD → Typhoon RT → brain → streaming LLM → Thai chunker → tier-0 gate → edge/Azure → WASAPI playback.
- Barge-in: duck, confirm, then cut.
- Push-to-talk, deafen and addressing modes.
- VTS lip-sync, emotions and idle motion.
- Reading one chat platform, plus `/api/event`.
- A text console mode.
- A control panel with the kill ladder, health lights, latency waterfall, memory editor and a captions overlay.
- Short- and long-term memory.
- Launcher supervision.
- The `setup`, `doctor`, `bench` and `report` commands.

**Exit criteria** (measured on the PC):
1. `setup.ps1` runs without admin rights, and the 4B works before the 30B has finished downloading.
2. A 60-minute live test (voice + chat + VTS) needs zero interventions.
3. Over at least 50 voice turns, the G2-selected profile meets §5: p50 ≤ 1.7 s (30B) or ≤ 1.3 s (4B), p95 ≤ 3.0 s. Chat turns reach first audio within p50 ≤ 1.0 s of the decision starting.
4. Kill and barge-in timings:
   - FREEZE to silence ≤ 150 ms.
   - HARD KILL ≤ 300 ms, including when the core is deliberately wedged with `aivtube debug wedge-core`.
   - A confirmed barge-in reaches silence ≤ 100 ms.
5. Chaos drill: kill voice, kill llama-server, close VTS, unplug the headset, drop the network for 30 s. Everything recovers within the §2.8 targets, and no unfiltered text is spoken.
6. A remembered fact survives a restart and is used again. A `remember` call induced by chat lands in quarantine.
7. A 200-line Thai/English red-team set leaks nothing from any blocked category, including phrases split across chunks and letter-spelled variants.
8. Median `cache_n / prompt_n` ≥ 0.85 over a session.
9. A 2-hour soak: fewer than 1 underflow per 10 minutes, and core RSS grows by less than 300 MB.
10. CI is green on ubuntu and windows, including the ORT coexistence test and the simulated end-to-end suite.

### M2+ (ordered)

| M | Content | Why here |
|---|---|---|
| M2 Latency & safety pack | Speculative endpointing: the LLM and TTS start at 300 ms of silence, playback is gated until commit. Thai particle endpointing (only after S7 calibration). KV prefill at speech start (if S8 passes). Pre-think pipelining. Precomputed idle lines. Tier-1 classifier. Review mode. Optional Whisper accuracy worker | Latency is what makes her feel like Neuro. Tier-1 must exist before she gets moderation powers |
| M3 Tools & channel actions | Twitch Device Code Flow + EventSub + Helix. Tools: `timeout_user` (≤ 600 s, never a ban), `create_poll`, `set_stream_title`, `spin_model`, `set_talking_speed`, soundboard. Redeems. YouTube streamList and actions. Alert-service adapters | Reuses M1's tool calling. Needs tier-1 checks on arguments |
| M4 Games | Neuro-SDK-compatible server on `localhost:8000`, a static `game_action` tool, grammar-forced actions with a schema sanitiser, `speech_finished` | Neuro's signature content. Reuses the argument pipeline validated in M3 |
| M5 Twin | A second `CharacterRuntime`, `FloorManager`, a second VTS instance found by UDP discovery, SDK port 8010, `call_twin`, Azure voice `th-TH-AcharaNeural` | Doubles the operational surface, so it comes after the core is stable |
| M6 Local TTS + renderer | VoxCPM2 sidecar in its own venv (~8 GB VRAM, so it pairs with the `light` or `cloud` LLM profile). Browser renderer (the user downloads Cubism Core). TikTok read | Competes with the 30B for VRAM |
| M7 Vision + search | Periodic screen caption fed in as a low-rank VISION stimulus. `web_search` tool with filtered results | Off the critical path. Neuro's vision took ~5 s in 2024 (T5) |
| M8 Singing | Stems rendered offline. Song mode: barge-in off, lip-sync driven by the vocal stem. Duets | A separate subsystem that bypasses the LLM (T1/T4) |
| Later | SDK voice side-channel, Discord calls, low-level game controllers | Demand-driven |

## 2. Process & thread model

### 2.1 Topology

```
run.bat → P0 launcher (stdlib only; Job Object KILL_ON_JOB_CLOSE; console keys; emergency HTTP 127.0.0.1:8779;
          │            nvidia-smi poller; keep-awake)
          ├─ P1 llama-server.exe :8080 30B  (slots 0,1 = speak double-buffer, 2 = background, +3 game in M4)
          │     [on demand] :8081 4B cold standby / light profile
          ├─ P2 aivtube core (asyncio): Brain · Intake · Arbiter · PromptBuilder · ReplyPipeline · ToolFlow
          │     SafetyGate(tier-0, sync) · LLMRouter → P1 / opt-in cloud · Memory (SQLite thread)
          │     AvatarDriver → VTS :8001 (external) · chat adapters · panel+overlays :8770 · IPC server :8771
          │     [M2] tier-1 classifier thread · [M4] Neuro SDK :8000
          └─ P3 aivtube voice (ABOVE_NORMAL, sys.setswitchinterval(0.001)):
                PortAudio out+in callbacks (WASAPI 48 kHz/10 ms) · mic thread: [AEC3]→soxr→VAD→Endpointer
                STT thread (sherpa, releases GIL) · notifier thread (DAC-timed marks)
                asyncio: IPC client · SpeechQueue · TTSRouter · PyAV · LipSync · BargeInController
Later: whisper worker :8091 (CUDA 12, M2) · voxcpm worker :8093 (own venv, M6)
```

### 2.2 Ports

All ports bind loopback only, are configurable, and are checked by `doctor`.

| Port | Owner |
|---|---|
| 8000 | Pailin's Neuro SDK hub (M4). Binds `localhost`, so both IPv4 and IPv6 |
| 8010 | Twin's SDK hub (M5). **Never 8001** |
| 8001, 8002, … | VTube Studio instances. Found via UDP 47779 discovery, never hard-coded |
| 8080 / 8081 | llama-server 30B / 4B |
| 8765 / 8766 | Reserved for the M6 browser renderer |
| 8770 | Control panel and OBS overlays |
| 8771 | Core ↔ worker IPC |
| 8779 | Launcher emergency endpoint |
| 8091 / 8093 | Workers (Whisper, VoxCPM2) |

Our own clients always use `127.0.0.1`, and websockets connections pass `proxy=None`.

### 2.3 Why each boundary exists

**Launcher.** Something must never crash and must be able to stop audio even when the core is wedged. The launcher uses only the standard library (subprocess, ctypes, threading, http.server, urllib). Its Job Object guarantees that no orphaned process keeps holding the mic or VRAM after it exits.

**Voice worker:**
- (a) The native audio stack (PortAudio, livekit FFI, onnxruntime, sherpa, PyAV) is the part most likely to segfault. If it crashes, only the worker restarts (~3 s).
- (b) The core's GIL-heavy work (pythainlp, JSON, prompt building) cannot starve the audio callbacks.
- (c) AEC needs mic capture and playback in one process on one clock.
- (d) A hard kill is simply `TerminateProcess(voice)`, which silences her immediately.

**llama-server.** Keeps its warm KV cache when the core restarts. The launcher also **adopts** an already-running server if `/props` reports the expected alias and model.

**GPU Python** (Whisper on CUDA 12, VoxCPM2 on torch). Each runs in its own process so CUDA 12 and CUDA 13 libraries never load into the same process. In M1 the only GPU user is llama-server.exe with its bundled cudart.

### 2.4 Threads

**Core:**
- One asyncio loop (Proactor).
- Blocking calls go to named single-worker executors: `sqlite`, `nlp`, and `guard` (M2).
- Nothing may block the loop for more than 5 ms. A lag monitor warns at 250 ms of lag and dumps all stacks with `faulthandler` at 2 s.
- Every fire-and-forget coroutine goes through `TaskSupervisor.track()`, which keeps a strong reference so the task is not garbage-collected.

**Voice worker:**
- The PortAudio callbacks only copy data; they never allocate.
- The mic thread runs AEC, soxr, VAD and the endpointer.
- One STT thread with a priority queue, so barge-in "quick" decodes jump ahead.
- A notifier thread.
- Asyncio on the main thread.

**llama-server** runs with `-t 8` (the P-cores). STT uses 2 threads and VAD/AEC use 1.

### 2.5 Time and pacing rules

- **All timestamps come from `time.perf_counter()`.** On Windows this has been system-wide since Python 3.10, and on Linux it is `CLOCK_MONOTONIC`, so values compare directly across processes.
- On Python 3.12 for Windows, `time.monotonic()` and `loop.time()` are GetTickCount64, with ~15.6 ms resolution. Never use them for audio markers, lip-sync or latency measurements.
- The audio brief's `audio_io.py` gets an injected `clock=time.perf_counter` in place of every `time.monotonic()`.
- The IPC handshake measures round-trip time and clock offset only as a sanity check: it warns and applies a correction if the offset exceeds 2 ms.
- Periodic work faster than 20 ms (the 60 Hz VTS driver) is paced by `PrecisionTicker`. This is a thread using `time.sleep`, which is a high-resolution waitable timer on Windows since 3.11, and it posts to the loop with `call_soon_threadsafe`. `asyncio.sleep` is only used for timers of 50 ms or more.
- If the ticker's p95 jitter exceeds 8 ms, the driver drops to 30 Hz.

### 2.6 Startup and readiness

1. **Launcher preflight.** Checks that ports are free, model files have the right size and sha256, the config is valid, the process is not in Session 0 (a CUDA context cannot be created there), and how much VRAM is free. On any failure it prints a hint in Thai and English and exits with code 2. The runbook says to start OBS and VTS first, because `--fit` measures VRAM at launch.
2. **llama-server.** The launcher adopts a running server or starts a new one. Core and voice start in parallel.
3. **Core.** Loads config, opens the databases, resumes a session younger than 6 h, then starts the bus, panel, IPC and supervisors. The brain enters **PRE_SHOW**, where only operator and voice input are accepted.
4. **Voice worker.** Sends `hello` with its token, loads VAD and STT, warms them up, verifies the cached phrases (including "Filtered."), then reports READY.
5. **LLM ready.** `/health` returns 200. The core **asserts `/props → chat_template_caps.supports_tool_calls == true`** and treats anything else as a hard error: a GGUF without a chat template silently ignores tools. It then restores the speak slot from `data/kv/<char>-<prefix_hash>.bin`, or prewarms the epoch prefix.
6. **Go Live.** When the panel is green (or amber with degradations the operator accepted), the operator presses **Go Live**, or `auto_live=true` does it. Chat intake and the idle timer switch on.

### 2.7 Invariants (each has a test)

- **I1.** The voice worker speaks only segments it received over the authenticated link. If the link drops, it finishes the current segment (which was already filtered), clears its queue, stops listening, and waits.
- **I2.** Every await on external I/O has a deadline.
- **I3.** Every long-running task is supervised. A crash loop marks the component FAILED; it never propagates.
- **I4.** Operator commands use the control channel. It never drops messages, is served before bus events, and has a synchronous fast path for FREEZE, SKIP and MUTE.
- **I5.** Conversation state can be rebuilt from SQLite.
- **I6.** `run --safe` works with no network, no GPU and no audio.
- **I7.** No text reaches the voice worker without passing the output gate, and the gate runs in the core.
- **I8.** Cloud providers receive no stream data unless `privacy.cloud_llm_consent` (or `cloud_stt_consent`) is true.

### 2.8 Failure and recovery

| Failure | Detection | Automatic response | On stream | Target |
|---|---|---|---|---|
| llama-server crash or hang | Process exit; 3 failed `/health` checks (every 2 s); 2 first-token timeouts | Circuit opens; the chain moves to the 4B (cold start) or opt-in cloud. The launcher restarts the 30B, restores the slot KV, and switches back after 60 s healthy. An out-of-memory at load increases N (§4.11) | The current reply ends with "…". If nothing can serve: the canned line, then silence, and the panel turns red | Fallback ≤ 5 s; 30B back ≤ 60 s |
| Voice worker crash | IPC drop, exit code, or 3 missed heartbeats | Launcher restarts it (backoff starting at 0.5 s). Core marks the utterance `voice_restart`, keeps only the heard text, and re-sends configuration | 3–5 s of silence | ≤ 5 s |
| Hard kill | Operator | Voice stays DOWN until REARM. Core enters FREEZE | Silence | ≤ 300 ms |
| Audio device lost | `finished_callback`, or no callback for 1 s | Reopen **by name** every 2 s. If the device cannot be found, restart the worker (which re-initialises PortAudio) | Silence | Until the device returns |
| STT error or decode > 3 s | Exception or timeout | Next backend in the chain; voice marked DEGRADED | One utterance lost | Immediate |
| TTS slow or failing | No first audio within 2 s (first segment) or 4 s (later segments) | Substitute a backend with the same voice for that segment. If the whole identity is exhausted, the rest of the utterance is captions-only and the next utterance uses the next identity. Per-backend breaker: 3 failures in 60 s takes the backend out for 120 s | The voice may change **between** utterances, never within one | Per segment |
| VTS closed | WebSocket closed | Reconnect with 1→10 s backoff, UDP discovery, stored token, re-sync expressions | Avatar freezes; audio continues | ≤ 2 s after VTS is back |
| Chat adapter | Exception, or 5 minutes of silence | Restart with backoff, dedupe messages, skip the backlog | Some chat missed | ≤ 30 s |
| Core crash | Exit code | Restart the core. llama-server and voice survive (I1). Short-term memory restored | 3–5 s pause | ≤ 5 s |
| Core loop wedged | Launcher heartbeat gets no answer for 5 s | Dump stacks with faulthandler, kill the core, restart it | As above | ≤ 8 s |
| Decision stuck | 30 s watchdog | Cancel it and dump the flight recorder | Short silence | 30 s |
| SQLite busy or corrupt | busy_timeout, or the open fails | Retry; otherwise switch to an in-memory store and turn the panel red | Memory not persisted | Manual |
| Crash loop | More than 5 restarts in 120 s | Component marked FAILED; dependent features degrade | Alarm | Manual |
| Internet outage | edge, cloud and IRC all fail | Local LLM keeps working; TTS falls back to Piper (if enabled) or captions; chat keeps retrying | Degraded | Automatic |

### 2.9 Supervision and shutdown

- **Backoff.** Restarts back off from 0.5 s to 30 s. A crash loop is 5 restarts within 120 s.
- **Critical tasks.** A failing `critical=True` task (the brain loop or the IPC server) makes the process exit with code 70.
- **Graceful stop.** Send `CTRL_BREAK_EVENT` (children are started with `CREATE_NEW_PROCESS_GROUP`); `TerminateProcess` after 5 s.
- **Shutdown order:**
  1. Stop chat intake.
  2. Stop new decisions.
  3. Let the current utterance finish (up to 5 s).
  4. Queue the episode-summary job.
  5. Save the slot KV.
  6. Disconnect VTS.
  7. Stop voice.
  8. Stop llama-server (unless adopted and `keep_llm` is set).
  9. Checkpoint the WAL and take a backup.
- **Keep awake.** `SetThreadExecutionState(ES_CONTINUOUS|ES_SYSTEM_REQUIRED|ES_DISPLAY_REQUIRED)` stops the PC from sleeping mid-stream.

### 2.10 Observability

- **Logs.** A logging QueueHandler writes `logs/<date>/<proc>.log` and `.jsonl`, rotated at 20 MB × 5 and kept for 14 days. A redaction filter removes keys, `oauth:` strings and Bearer tokens. `faulthandler` writes a `.fault` file per process.
- **TurnTrace** per turn:
  - stage marks: `stimulus_in, vad_end, stt_final, decision_start, prompt_built, llm_first_token, first_chunk, filter_done, tts_first_audio, first_audible, last_audible, done`;
  - also: provider, `prompt_n`/`cache_n`, tok/s, TTS backend and identity, fallbacks taken, whether speculation hit, and the opener (first 6 characters of the reply).

  Traces are stored in `ops.db`. The panel shows a waterfall of the last 20 turns and p50/p95 badges against the §5 budget.
- **Metrics** (1 Hz): underflows, drops, restarts, filter hits, fallbacks, loop lag, tok/s, cache ratio, opener repetition. The launcher alarms when free VRAM falls below 500 MiB.
- **Flight recorder.** Keeps the last 2000 events and the last 50 LLM request summaries. It is dumped on any error, crash or watchdog trip. `aivtube report` zips logs, redacted config, doctor output, versions and the flight dump.

### 2.11 Operator control: the kill ladder

| Level | Trigger | Effect | Target |
|---|---|---|---|
| SKIP | Panel key `S`, or `POST /api/cmd` | Cuts the current utterance (30 ms fade). The brain carries on | ≤ 100 ms |
| MUTE | `M` | Output gain set to 0; segments are dropped and logged | ≤ 100 ms |
| FREEZE | Big red button, `F`, or an HTTP hotkey | Cancels the LLM and stops audio. Brain goes PAUSED; chat intake and tools are switched off; the avatar goes neutral. Only RESUME undoes it | ≤ 150 ms |
| HARD KILL | Launcher console `K`, or `POST 127.0.0.1:8779/hardkill` with the token | `TerminateProcess(voice)`, held down until REARM; the core goes to FREEZE. **Does not depend on the core.** The panel's JS calls this endpoint directly if the core has not answered within 300 ms | ≤ 300 ms |
| OBS | A human | Mute the Desktop Audio source, or switch to a BRB scene | – |

**Mic control.** `mic.mode` is `open`, `ptt` or `deafened`. It can be switched from the panel, `POST /api/mic`, `GET /hotkey/<name>?token=` (loopback only, for AutoHotkey or Stream Deck HTTP actions), or launcher console keys. "Deafen" is the "I'm talking to chat" toggle.

**Automatic reactions to filter trips.** Many on-stream filter trips are deliberate comedy (T5), and trolls can bait them, so the defaults avoid stopping the stream:
- **Auto-strict (on by default).** Three "Filtered." events within 5 minutes switch on strict mode: temperature 0.4, tools off, tier-1 on every segment (M2). The chatters whose stimuli fed those decisions are muted from her view for 10 minutes, and the operator gets an alert.
- **Auto-freeze (off by default, opt-in).**

Every operator action is written to `op_audit`.

## 3. Contracts: public interfaces (authoritative)

All contracts live in `src/aivtube/contracts/` and are frozen at M0. Changing one needs an ADR and a bump of `CONTRACTS_VERSION`.

Rules:
- Implementations import only `contracts`, `config` and their own package.
- Protocols are structural, so fakes need no base class.
- Dataclasses are `frozen=True, slots=True`.
- PCM is numpy float32 in [-1, 1] or int16, always with an explicit sample rate.
- Every Protocol has a fake in `aivtube.testing.fakes` and a parametrised suite in `aivtube.testing.contracts`.

### 3.1 contracts/types.py
```python
CONTRACTS_VERSION = "1.0"
class Priority(IntEnum): LOW = 0; MEDIUM = 1; HIGH = 2; CRITICAL = 3          # Neuro SDK semantics (T1)
class Rank(IntEnum): OPERATOR = 0; VOICE = 10; FORCE_URGENT = 20; SUPPORT = 30; MENTION = 40
                     FORCE = 50; CHAT = 60; GAME_CONTEXT = 70; CHARACTER = 75; VISION = 80; IDLE = 90   # lower wins
class StimulusKind(StrEnum): OPERATOR, VOICE, GAME_FORCE, SUPPORT, MENTION, CHAT, GAME_CONTEXT, CHARACTER, VISION, IDLE
class Platform(StrEnum): TWITCH, YOUTUBE, TIKTOK, CONSOLE, ALERT
class MsgKind(StrEnum): TEXT, DONATION, SUB, GIFT_SUB, RAID, REDEEM, SYSTEM
@dataclass(frozen=True, slots=True)
class ChatUser: platform: Platform; id: str; name: str; is_broadcaster: bool = False; is_mod: bool = False
                is_vip: bool = False; is_sub: bool = False; sub_months: int = 0; is_verified: bool = False
@dataclass(frozen=True, slots=True)
class ChatMessage: platform: Platform; id: str; user: ChatUser; text: str; ts: float; received: float
    kind: MsgKind = MsgKind.TEXT; amount: float = 0.0; currency: str = ""; value_usd: float = 0.0
    first_msg: bool = False; reply_to: str | None = None; source_channel: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)
@dataclass(frozen=True, slots=True, kw_only=True)
class Stimulus: id: str; kind: StimulusKind; character: str; text: str; created: float
    priority: Priority = Priority.LOW; rank: Rank = Rank.CHAT; ttl_s: float | None = 30.0
    source: str = ""; speaker: str | None = None; addressed: bool = False
    payload: Mapping[str, Any] = field(default_factory=dict); trace_id: str = ""   # text is ALREADY input-filtered
class HealthState(StrEnum): STARTING, OK, DEGRADED, DOWN, DISABLED, FAILED
@dataclass(frozen=True, slots=True)
class Health: component: str; state: HealthState; detail: str = ""; since: float = 0.0
@dataclass(frozen=True, slots=True)
class VoiceSpec: identity: str; voice: str; rate: str = "+0%"; pitch: str = "+0Hz"; volume: str = "+0%"
SegmentKind = Literal["speech", "filtered", "filler", "operator", "canned"]
@dataclass(frozen=True, slots=True)
class Segment: utt_id: str; seq: int; text: str; caption: str; emotion: str | None = None
               last: bool = False; kind: SegmentKind = "speech"
@dataclass(frozen=True, slots=True)
class Transcript: text: str; is_final: bool; audio_s: float; latency_ms: float; engine: str
```
### 3.2 contracts/events.py
```python
@dataclass(frozen=True, slots=True, kw_only=True)       # every subclass uses the same decorator
class Event: ts: float = 0.0; character: str | None = None; turn_id: str | None = None   # ts = perf_counter
# input
UserSpeechStarted(barge: bool) · UserSpeechEnded(audio_s: float)
UserTranscript(text: str, engine: str, latency_ms: float, audio_s: float, parts: int = 1)
BargeInCandidate() · BargeInConfirmed(text: str, cut_local: bool) · BargeInRejected()
ChatReceived(message: ChatMessage) · ChatDropped(message_id: str, reason: str) · SupportReceived(message: ChatMessage)
StimulusQueued(stimulus: Stimulus) · StimulusExpired(stimulus_id: str)
# brain / speech
StateChanged(old: str, new: str) · DecisionStarted(stimulus_id: str, merged_ids: tuple[str, ...], provider: str, speculative: bool = False)
DecisionAborted(reason: str) · LLMFirstToken(provider: str, ttft_ms: float, prompt_n: int | None, cache_n: int | None)
UtteranceStarted(utt_id: str, stimulus_id: str) · SegmentQueued(utt_id: str, seq: int, caption: str)
SegmentStarted(utt_id: str, seq: int, t_audible: float, duration_s: float | None, backend: str, silent: bool, caption: str, emotion: str | None)
SegmentDone(utt_id: str, seq: int, heard: bool, heard_text: str)
UtteranceDone(utt_id: str, heard_text: str, cancelled: bool, reason: str | None, filtered: bool)
Filtered(direction: str, tier: str, category: str | None, rule: str | None, ref: str | None)
ToolRequested(tool: str, args: Mapping[str, Any]) · ToolExecuted(tool: str, ok: bool, content: str, dry_run: bool = False)
ToolRejected(tool: str, reason: str) · MemoryWritten(memory_id: int, kind: str, status: str) · EmotionChanged(emotion: str)
# infra
HealthChanged(health: Health) · ComponentRestarted(name: str, count: int) · ProviderSwitched(kind: str, old: str, new: str, reason: str)
OperatorAction(kind: str, args: Mapping[str, Any], ok: bool, latency_ms: float) · LatencyMark(stage: str)
TurnTraceReady(trace: Mapping[str, Any]) · Alert(level: Literal["info", "warn", "error"], message: str)
# games (M4)
GameConnected(game: str) · GameDisconnected(game: str) · GameContextReceived(game: str, text: str, silent: bool)
GameForceReceived(game: str, force_id: str, priority: Priority) · GameActionSent(game: str, name: str, force_id: str | None)
GameActionResult(game: str, name: str, ok: bool, message: str)
EVENT_TYPES: Mapping[str, type[Event]]
def event_to_json(e: Event) -> dict[str, Any]: ...
def event_from_json(d: Mapping[str, Any]) -> Event: ...
```
### 3.3 contracts/infra.py
```python
class Clock(Protocol):
    def now(self) -> float: ...                 # time.perf_counter()
    def wall(self) -> float: ...
    async def sleep(self, seconds: float) -> None: ...
class Component(Protocol):
    name: str
    async def start(self) -> None: ...
    async def aclose(self) -> None: ...
    def health(self) -> Health: ...
RestartPolicy = Literal["never", "on_error", "always"]
class TaskSupervisor(Protocol):
    def spawn(self, name: str, factory: Callable[[], Awaitable[None]], *, restart: RestartPolicy = "on_error",
              backoff: tuple[float, float] = (0.5, 30.0), breaker: tuple[int, float] = (6, 120.0), critical: bool = False) -> None: ...
    def track(self, coro: Awaitable[Any], *, name: str) -> asyncio.Task[Any]: ...   # strong ref + exception logging
    async def restart(self, name: str) -> None: ...
    def status(self) -> list[Health]: ...
    async def aclose(self, timeout: float = 5.0) -> None: ...
class Overflow(StrEnum): DROP_OLDEST = "drop_oldest"; DROP_NEWEST = "drop_newest"
class Subscription(Protocol):
    name: str; dropped: int
    def __aiter__(self) -> AsyncIterator[Event]: ...
    def close(self) -> None: ...
class EventBus(Protocol):
    def publish(self, event: Event) -> None: ...                 # loop thread; never blocks/raises
    def publish_threadsafe(self, event: Event) -> None: ...
    def subscribe(self, *types: type[Event], name: str, maxsize: int = 1024, overflow: Overflow = Overflow.DROP_OLDEST) -> Subscription: ...
```
### 3.4 contracts/voice.py (used inside the voice worker)
```python
F32 = npt.NDArray[np.float32]; I16 = npt.NDArray[np.int16]
MarkCallback = Callable[[bool, float], None]          # (heard, t_audible perf_counter), notifier thread
class AudioBackend(Protocol):                          # subset of `sounddevice`; FakeSD implements it
    def query_hostapis(self, index: int | None = None) -> Any: ...
    def query_devices(self, device: int | str | None = None, kind: str | None = None) -> Any: ...
    def OutputStream(self, **kw: Any) -> Any: ...
    def InputStream(self, **kw: Any) -> Any: ...
    def WasapiSettings(self, **kw: Any) -> Any: ...
class AudioOut(Protocol):
    sample_rate: int; output_latency_s: float; reference: collections.deque[tuple[float, F32]]; stats: Mapping[str, int]
    def start(self) -> None: ...
    def play(self, pcm: F32, sample_rate: int) -> None: ...
    def mark(self, cb: MarkCallback) -> None: ...
    def cancel(self, fade_ms: float = 30.0) -> float: ...
    def set_gain(self, gain: float, ramp_ms: float = 20.0) -> None: ...
    def is_speaking(self, tail_s: float = 0.25) -> bool: ...
    def close(self) -> None: ...
class AudioIn(Protocol):
    sample_rate: int; block_samples: int; stats: Mapping[str, int]
    def start(self, on_frame: Callable[[F32, float], None]) -> None: ...
    def close(self) -> None: ...
class EchoCanceller(Protocol):
    def feed_reference(self, block: F32) -> None: ...
    def process(self, mic_block: F32) -> F32: ...
class VoiceActivityDetector(Protocol):
    sample_rate: int; frame_samples: int               # 16000, 512
    def reset(self) -> None: ...
    def prob(self, frame: F32) -> float: ...
@dataclass(frozen=True, slots=True)
class EndpointerConfig: threshold: float = 0.5; neg_threshold: float = 0.35; barge_threshold: float = 0.6
    min_speech_ms: int = 250; end_silence_ms: int = 600; preroll_ms: int = 300; max_segment_s: float = 15.0
    max_turn_s: float = 60.0; particle_endpointing: bool = False; final_particle_ms: int = 420; continuation_ms: int = 900
@dataclass(frozen=True, slots=True)
class VadStart: t: float; barge: bool
@dataclass(frozen=True, slots=True)
class VadPartial: t: float; audio: F32               # forced split: decode, but the turn continues
@dataclass(frozen=True, slots=True)
class VadEnd: t: float; audio: F32                   # real end of utterance (incl. 300 ms pre-roll)
VadEvent = VadStart | VadPartial | VadEnd
class Endpointer(Protocol):
    def push(self, frame16k: F32, t: float, ai_speaking: bool) -> list[VadEvent]: ...
    def set_tail_hint(self, text: str) -> None: ...  # M2 particle endpointing
    def reset(self) -> None: ...
class SpeechRecognizer(Protocol):
    name: str; sample_rate: int
    def warmup(self) -> None: ...
    def transcribe(self, pcm16k: F32, *, quick: bool = False) -> Transcript: ...   # blocking; STT thread only
    def close(self) -> None: ...
class TranscriptPostProcessor(Protocol):
    def __call__(self, t: Transcript, recent_tts_text: str) -> Transcript | None: ...
@dataclass(frozen=True, slots=True)
class AudioChunk: pcm: I16; sample_rate: int
@dataclass(frozen=True, slots=True)
class WordMark: text: str; offset_s: float; duration_s: float
class TTSUnavailable(Exception): ...
@dataclass(frozen=True, slots=True)
class QuotaSpec: max_requests: int; per_s: float
class TTSBackend(Protocol):
    name: str; normalizer: Literal["cloud", "local"]; quota: QuotaSpec | None
    async def warmup(self) -> None: ...
    def synth(self, text: str, voice: VoiceSpec, *, first_audio_timeout: float, idle_timeout: float) -> AsyncIterator[AudioChunk | WordMark]: ...
    async def aclose(self) -> None: ...
class PhraseCache(Protocol):
    def get(self, key: tuple[str, ...]) -> AudioChunk | None: ...
    def put(self, key: tuple[str, ...], audio: AudioChunk) -> None: ...
```
### 3.5 contracts/speech.py (used in the core)
```python
MicMode = Literal["open", "ptt", "deafened"]; BargePolicy = Literal["interrupt", "duck_only", "off"]
EchoMode = Literal["auto", "aec", "energy_dtd", "half_duplex", "none"]; StopMode = Literal["now", "after_segment"]
@dataclass(frozen=True, slots=True)
class VoicePolicy: mic_mode: MicMode = "open"; ptt_active: bool = False; barge_in: BargePolicy = "interrupt"
                   echo_mode: EchoMode = "auto"; listening: bool = True
@dataclass(frozen=True, slots=True)
class TTSConstraints: first_min_chars: int; min_chars: int; max_chars: int; backend: str; identity: str
class SpeechOutput(Protocol):          # BusSpeechOutput | ConsoleSpeechOutput | FakeSpeechOutput
    def ready(self) -> bool: ...
    def constraints(self, character: str) -> TTSConstraints: ...
    async def begin(self, utt_id: str, character: str, *, filler_after_s: float | None = None, gate_open: bool = True) -> None: ...
    async def segment(self, seg: Segment) -> bool: ...        # False = backpressure; never unfiltered text (I7)
    async def open_gate(self, utt_id: str) -> None: ...       # M2 speculation / review mode
    async def stop(self, utt_id: str | None, mode: StopMode, reason: str, fade_ms: int = 30) -> None: ...
    async def duck(self, gain: float, ramp_ms: int = 30) -> None: ...
    async def mute(self, on: bool) -> None: ...
    async def set_policy(self, policy: VoicePolicy) -> None: ...
    async def set_voice_rate(self, character: str, percent: int) -> None: ...
    async def play_canned(self, key: str, character: str) -> None: ...   # "filtered", fillers, brain-freeze line
# results arrive as SegmentStarted/SegmentDone/UtteranceDone events; LipTrack goes straight to the AvatarDriver
```
### 3.6 contracts/llm.py
```python
@dataclass(frozen=True, slots=True)
class ToolSpec: name: str; description: str; parameters: Mapping[str, Any]      # JSON Schema, type: object
SlotRole = Literal["speak", "game", "background"]
@dataclass(frozen=True, slots=True)
class ChatRequest: messages: tuple[Mapping[str, Any], ...]; purpose: SlotRole = "speak"; tools: tuple[ToolSpec, ...] = ()
    tool_choice: Literal["auto", "required", "none"] = "auto"; response_schema: Mapping[str, Any] | None = None
    max_tokens: int = 256; temperature: float = 0.6; character: str = ""; turn_id: str = ""
    slot: int | None = None; first_token_timeout_s: float | None = None                 # string content only
@dataclass(frozen=True, slots=True)
class TextDelta: text: str
@dataclass(frozen=True, slots=True)
class ToolCall: id: str; name: str; arguments: Mapping[str, Any] | None; raw_arguments: str; extra: Mapping[str, Any] | None
@dataclass(frozen=True, slots=True)
class Done: provider: str; finish_reason: str | None; ttft_ms: float; prompt_n: int | None; cache_n: int | None
            completion_tokens: int | None; assistant_message: Mapping[str, Any]
LLMEvent = TextDelta | ToolCall | Done               # ToolCalls are yielded after the stream ends
@dataclass(frozen=True, slots=True)
class ProviderCaps: tools: bool; parallel_tools: bool; json_schema: bool; prompt_cache: bool; cloud: bool
                    keeps_thought_signatures: bool; reasoning: Literal["none", "effort", "template_kwarg"]
class ProviderFailed(Exception):
    def __init__(self, msg: str, *, emitted: bool) -> None: ...
class LLMProvider(Protocol):
    name: str; caps: ProviderCaps
    async def probe(self) -> Health: ...
    def stream(self, req: ChatRequest) -> AsyncIterator[LLMEvent]: ...   # aclose() -> stream.close() -> server cancels <= 0.2 s
    async def prefill(self, req: ChatRequest) -> float: ...              # warm KV; cloud: no-op 0.0
@dataclass(frozen=True, slots=True)
class ProviderStatus: name: str; healthy: bool; active: bool; enabled: bool; cloud: bool; down_until: float; fails: int; ttft_p50_ms: float | None
class LLMRouter(Protocol):
    def stream(self, req: ChatRequest) -> AsyncIterator[LLMEvent]: ...   # falls back only before the first event
    async def prefill(self, req: ChatRequest) -> None: ...
    def promote(self, name: str) -> None: ...
    def rollback(self) -> None: ...
    def active(self) -> str: ...
    def status(self) -> list[ProviderStatus]: ...
class LocalServerManager(Protocol):
    async def ensure_running(self, server: str, timeout_s: float) -> bool: ...
    async def stop(self, server: str) -> None: ...
    async def props(self, server: str) -> Mapping[str, Any]: ...
    async def save_slot(self, server: str, slot: int, filename: str) -> bool: ...
    async def restore_slot(self, server: str, slot: int, filename: str) -> bool: ...
```
### 3.7 contracts/avatar.py
```python
AvatarState = Literal["idle", "listening", "thinking", "speaking", "paused"]
@dataclass(frozen=True, slots=True)
class LipTrack: utt_id: str; seq: int; t0: float; fps: int; mouth: tuple[float, ...]; form: tuple[float, ...]; final: bool = False
class AvatarSink(Protocol):            # VTSSink | BrowserSink (M6) | NullSink | FakeSink
    name: str
    @property
    def connected(self) -> bool: ...
    async def run(self) -> None: ...                                    # connect/auth/reconnect loop
    def set_params(self, values: Mapping[str, float]) -> None: ...     # fire-and-forget, <=8 in flight
    async def set_emotion(self, emotion: str, fade_s: float = 0.3) -> None: ...
    async def trigger(self, hotkey: str) -> bool: ...
    async def move(self, *, rotation: float = 0.0, x: float = 0.0, y: float = 0.0, size: float = 0.0, seconds: float = 0.25, relative: bool = True) -> None: ...
    def health(self) -> Health: ...
class AvatarDriver(Protocol):
    def on_lip_track(self, track: LipTrack) -> None: ...
    def on_cut(self, utt_id: str, t: float) -> None: ...
    def set_state(self, state: AvatarState) -> None: ...
    def set_emotion(self, emotion: str | None) -> None: ...
    async def run(self) -> None: ...
```
### 3.8 contracts/chat.py
```python
class ChatSource(Protocol):            # TwitchAnonIrc | YouTubeListPoller | (M3) EventSub, streamList, alerts | Fake
    platform: Platform
    def messages(self) -> AsyncIterator[ChatMessage]: ...   # reconnects, dedupes, skips backlog
    def health(self) -> Health: ...
    async def aclose(self) -> None: ...
class ChannelActions(Protocol):        # M3
    platform: Platform; capabilities: frozenset[str]        # {"send","timeout","untimeout","poll","title"}
    async def send(self, text: str, reply_to: str | None = None) -> None: ...
    async def timeout(self, user: ChatUser, seconds: int, reason: str) -> None: ...
    async def untimeout(self, user: ChatUser) -> None: ...
    async def create_poll(self, title: str, choices: Sequence[str], seconds: int) -> str: ...
    async def set_title(self, title: str) -> None: ...
@dataclass(frozen=True, slots=True)
class ChatSelection: must_ack: tuple[ChatMessage, ...]; candidates: tuple[ChatMessage, ...]; ambient: tuple[ChatMessage, ...]
class ChatWindow(Protocol):
    def add(self, m: ChatMessage) -> Literal["window", "priority", "dropped"]: ...
    def select(self, now: float, k: int = 3) -> ChatSelection: ...  # consumes the window
    def pending(self) -> tuple[int, bool]: ...                      # (count, has_mention)
    def consume(self, message_id: str) -> ChatMessage | None: ...   # read-aloud dedupe
    def recent(self, seconds: float) -> tuple[ChatMessage, ...]: ...
    def snapshot(self) -> list[tuple[ChatMessage, float]]: ...     # with scores, for the panel
```
### 3.9 contracts/memory.py
```python
MemKind = Literal["core", "fact", "viewer", "episode"]; MemStatus = Literal["active", "quarantined", "deleted"]
MemSource = Literal["model", "operator", "consolidation", "import"]
@dataclass(frozen=True, slots=True)
class MemoryItem: id: int | None; kind: MemKind; text: str; slot: int | None = None; subject: str | None = None
    platform: str | None = None; user_id: str | None = None; importance: int = 3; source: MemSource = "model"
    origin: str = ""; status: MemStatus = "active"; pinned: bool = False; locked: bool = False
@dataclass(frozen=True, slots=True)
class Turn: role: Literal["user", "assistant", "tool", "note"]; text: str; source: str; speaker: str | None = None
    heard_text: str | None = None; interrupted: bool = False; filtered: bool = False; provider: str | None = None
    turn_ref: str = ""; tool_calls: str | None = None; provider_extra: str | None = None; ts: float = 0.0; id: int | None = None
@dataclass(frozen=True, slots=True)
class PrefixMemory: core: tuple[MemoryItem, ...]; episodes: tuple[str, ...]; rolling_summary: str; epoch: int; digest: str
class SlotsFull(Exception):
    slots: list[MemoryItem]
class MemoryStore(Protocol):
    async def start_session(self, title: str | None = None) -> int: ...
    async def resume_session(self, max_age_s: float) -> int | None: ...
    async def end_session(self, summary: str | None = None) -> None: ...
    async def append_turn(self, turn: Turn) -> int: ...
    async def recent_turns(self, epoch: int) -> list[Turn]: ...
    async def prefix_block(self) -> PrefixMemory: ...
    async def new_epoch(self, rolling_summary: str, upto_turn_id: int, prefix_hash: str) -> int: ...
    async def remember(self, item: MemoryItem, *, replace_slot: int | None = None) -> MemoryItem: ...   # may raise SlotsFull
    async def forget(self, memory_id: int, *, by: str, reason: str) -> None: ...          # locked items refuse
    async def set_status(self, memory_id: int, status: MemStatus, *, by: str) -> None: ...
    async def list_memories(self, *, kind: MemKind | None = None, status: MemStatus | None = None) -> list[MemoryItem]: ...
    async def pending_since_epoch(self) -> list[MemoryItem]: ...
    async def search(self, query: str, k: int = 3) -> list[MemoryItem]: ...               # FTS5 trigram (>=3 chars) else LIKE
    async def viewer_facts(self, users: Sequence[tuple[str, str]], limit: int = 6) -> list[MemoryItem]: ...
    async def upsert_viewer(self, platform: str, user_id: str, name: str) -> None: ...
    async def backup(self, dest_dir: Path, keep: int = 14) -> Path: ...
```
### 3.10 contracts/safety.py
```python
Direction = Literal["in", "out", "tool", "memory", "name", "game"]
class Verdict(StrEnum): PASS = "pass"; MASK = "mask"; REPLACE = "replace"; DROP = "drop"; BLOCK = "block"; REVIEW = "review"
@dataclass(frozen=True, slots=True)
class FilterContext: direction: Direction; character: str; platform: str | None = None; user_id: str | None = None; prev_tail: str = ""
@dataclass(frozen=True, slots=True)
class FilterResult: verdict: Verdict; text: str; tier: str; rule: str | None = None; category: str | None = None
                    score: float | None = None; fail_closed: bool = False
class TextFilter(Protocol):            # tier-0, sync, < 1 ms
    name: str
    def check(self, text: str, ctx: FilterContext) -> FilterResult: ...
    def reload(self) -> None: ...
class Classifier(Protocol):            # tier-1 (M2)
    name: str
    async def score(self, texts: Sequence[str], direction: Direction) -> list[float]: ...   # P(harmful)
class SafetyGate(Protocol):
    def check_input(self, msg: ChatMessage, *, character: str) -> tuple[FilterResult, str]: ...   # (text verdict, display-safe name)
    async def check_output(self, chunk: str, *, character: str, prev_tail: str) -> FilterResult: ...
    async def check_args(self, direction: Literal["tool", "memory", "game"], texts: Sequence[str], *, character: str) -> FilterResult: ...
    def strict_mode(self, character: str) -> bool: ...
    def reload(self) -> None: ...
```
### 3.11 contracts/tools.py
```python
Risk = Literal["safe", "moderate", "dangerous"]
@dataclass(frozen=True, slots=True)
class RateLimit: max_calls: int; per_s: float
@dataclass(frozen=True, slots=True)
class ToolPolicy: risk: Risk = "safe"; side_effect: bool = False; follow_up: bool = False; requires_approval: bool = False
                  rate_limit: RateLimit | None = None; timeout_s: float = 5.0; requires: frozenset[str] = frozenset()
@dataclass(frozen=True, slots=True)
class ToolResult: ok: bool; content: str; note: str | None = None   # content is fed back to the model
@dataclass(frozen=True, slots=True)
class ToolContext: character: str; turn_id: str; stimulus: Stimulus; memory: MemoryStore; speech: SpeechOutput
                   avatar: AvatarSink | None; channels: Mapping[Platform, ChannelActions]; bus: EventBus; clock: Clock
class Tool(Protocol):
    spec: ToolSpec; policy: ToolPolicy
    async def __call__(self, args: Mapping[str, Any], ctx: ToolContext) -> ToolResult: ...
class ToolRegistry(Protocol):
    def specs(self, character: str) -> tuple[ToolSpec, ...]: ...     # static per session (cache-stable order)
    async def execute(self, call: ToolCall, ctx: ToolContext) -> ToolResult: ...   # validate -> filter args -> policy -> run
    def set_enabled(self, name: str, enabled: bool) -> None: ...
    def set_mode(self, mode: Literal["live", "dry_run", "off"]) -> None: ...
```
### 3.12 contracts/control.py
```python
class OpKind(StrEnum): SKIP, MUTE, UNMUTE, FREEZE, RESUME, GO_LIVE, CHAT_INTAKE, MIC_MODE, PTT, SAY, DIRECT, FAKE_CHAT,
    INJECT_EVENT, LLM_USE, LLM_ROLLBACK, TTS_IDENTITY, TOOLS_MODE, TOOL_ENABLE, APPROVE, MEMORY_EDIT, MEMORY_STATUS,
    MUTE_USER, STRICT, FILTER_RELOAD, RESTART, END_STREAM          # values = lowercase names
@dataclass(frozen=True, slots=True)
class OpCommand: kind: OpKind; args: Mapping[str, Any] = field(default_factory=dict); character: str | None = None
                 operator: str = "local"; id: str = ""
@dataclass(frozen=True, slots=True)
class OpResult: ok: bool; detail: str = ""; latency_ms: float = 0.0
class ControlSurface(Protocol):
    async def execute(self, cmd: OpCommand) -> OpResult: ...   # FREEZE/SKIP/MUTE take a synchronous fast path
    def snapshot(self) -> Mapping[str, Any]: ...
```
### 3.13 contracts/games.py (frozen at M0, used from M4)
```python
@dataclass(frozen=True, slots=True)
class GameAction: game: str; name: str; description: str; schema: Mapping[str, Any] | None
@dataclass(frozen=True, slots=True)
class GameForce: id: str; game: str; query: str; state: str | None; action_names: tuple[str, ...]; priority: Priority; ephemeral: bool; attempt: int = 0
class GameServer(Protocol):
    character: str; port: int
    async def serve(self) -> None: ...
    def games(self) -> list[str]: ...
    def actions(self, game: str | None = None) -> list[GameAction]: ...
    def active_force(self) -> GameForce | None: ...
    async def execute(self, game: str, name: str, data: Mapping[str, Any] | None, force_id: str | None) -> str: ...
    async def speech_finished(self, is_final: bool, cancelled: bool = False, reason: str | None = None) -> None: ...
class GameBrain(Protocol):
    async def on_context(self, game: str, message: str, silent: bool) -> None: ...
    async def on_force(self, force: GameForce, actions: Sequence[GameAction]) -> None: ...
    async def on_force_dropped(self, force: GameForce, reason: str) -> None: ...
    async def on_action_result(self, game: str, name: str, success: bool, message: str | None, forced: bool) -> None: ...
    async def on_actions_changed(self, game: str) -> None: ...
```
### 3.14 contracts/ipc.py
```python
IPC_VERSION = 1
@dataclass(frozen=True, slots=True)
class Envelope: v: int; type: str; id: str; ts: float; data: Mapping[str, Any]; corr: str | None = None
class IpcError(Exception): ...
def encode(env: Envelope) -> str: ...
def decode(raw: str | bytes) -> Envelope: ...          # IpcError on malformed; unknown `type` is allowed (ignored + logged)
MESSAGE_SCHEMAS: Mapping[str, Mapping[str, Any]]       # JSON Schema per type (Appendix A)
def validate(env: Envelope) -> None: ...
```

## 4. The brain

There is one `Brain` per character. It runs only on the core's event loop. A **decision** is exactly one LLM generation, plus at most one tool follow-up round. Decisions are **serial**. Playback is not a decision, so under HIGH semantics a new decision may overlap the audio tail of the previous one.

### 4.1 Serial loop and DecisionSlot
```python
class Brain:
    async def run(self) -> None:                                   # critical supervised task
        while True:
            await self.gate.wait_open()                            # not PAUSED / PRE_SHOW-blocked; LLM ready
            stim = await self.arbiter.next(idle_deadline=self.idle.deadline(), user_speaking=self.user_speaking)
            stim = stim or self.idle.make_stimulus(self.character)
            ctx = self.arbiter.drain_context(stim)                 # everything queued since the last decision
            turn = self.turns.new(ctx)
            outcome = await self.slot.run(turn.id, self._decide(turn))   # the ONLY path to an LLM decision
            await self._after(turn, outcome)

    def submit(self, s: Stimulus) -> None:                         # adapters; sync, never blocks
        self.arbiter.push(s)
        act = decide_preemption(s, self.snapshot())                # pure table, §4.4
        if act.cancel_decision:
            self.slot.cancel(act.reason)
        if act.speech_stop is not None:
            self.tasks.track(self.speech.stop(self.utt_id, act.speech_stop, act.reason), name="preempt-stop")

class DecisionSlot:            # one per character; live, speculative (M2) and pipelined (M2) decisions all use it
    async def run(self, turn_id: str, coro: Coroutine[Any, Any, DecisionResult]) -> DecisionOutcome:
        async with self._lock:
            task = self._tasks.track(coro, name=f"decide:{turn_id}")
            self._current = (turn_id, task)
            try:
                await asyncio.wait({task})                         # does NOT swallow cancellation of the Brain task
            finally:
                self._current = None
                if not task.done():
                    task.cancel()                                  # Brain cancelled -> cancel child, then propagate
            if task.cancelled():
                return DecisionOutcome.aborted(turn_id, self._reasons.pop(turn_id, "cancelled"))
            if (exc := task.exception()) is not None:
                return DecisionOutcome.failed(turn_id, exc)
            return DecisionOutcome.ok(turn_id, task.result())
```

`_decide(turn)` runs these steps:
1. **Gather context.** FTS recall and viewer facts, with a 50 ms deadline.
2. **Build the request.** `PromptBuilder.build(epoch, history, ctx)`.
3. **Begin the utterance.** `speech.begin(utt, filler_after_s=1.2 if voice turn)`.
4. **Stream the reply.** `ReplyPipeline.run(router.stream(req))`.
5. **Run tools.** `ToolFlow`, with one optional follow-up (`/u2`).
6. **Record history.** Append the compact user line, then the assistant line.

The assistant's history text is **frozen when it is first rendered into a prompt**: whatever was heard up to that point, or the segments already started plus the one currently playing if a stop after that segment is guaranteed. Any later correction goes into a note in the next tail. This keeps the cached prefix byte-stable.

After a CRITICAL cut, the brain waits up to 150 ms for `UtteranceDone`, so the heard text is exact to the word.

### 4.2 Intake: inputs become stimuli

**Voice.**
- The voice worker concatenates the partial decodes produced by forced splits (every 15 s) until the real end of utterance (maximum 60 s). A 15 s split never ends the streamer's turn.
- `mic.mode`:
  - `open`: every utterance counts.
  - `ptt`: only speech while the key is held counts.
  - `deafened`: ignored ("talking to chat").
- `mic.addressing`:
  - `always` (co-host default): every utterance is addressed to her.
  - `name_or_question`: addressed only if it contains a name alias after the STT alias map, is shaped like a question, or falls within an 8 s follow-up window after she spoke. Utterances that are not addressed are appended to history as a `note` line and trigger no decision.
- **Read-aloud dedupe.** A transcript is compared with chat from the last 60 s (normalised; containment of at least 6 chars, or `SequenceMatcher` ratio ≥ 0.75). If it matches, that chat message is consumed from the window and the VOICE stimulus carries `payload.read_aloud_of`. The message is then answered once, not twice.

**Chat.**
- Tier-0 input check, then the display name goes through `check_name`.
- If the text mentions her, it becomes a MENTION stimulus. Donations, subs, raids and redeems become SUPPORT (must-acknowledge queue, TTL 600 s). Everything else goes to the `ChatWindow`.
- Donations whose text is blocked are still acknowledged by name and amount, and the text is read as "Filtered.".

**Operator.** SAY is CRITICAL and bypasses the LLM (the filter only warns). DIRECT is HIGH and is an instruction that is never read aloud.

### 4.3 Arbitration

When the brain is free, the eligible stimulus with the lowest `Rank` wins; ties go to the oldest. An aging bonus of −1 rank per 5 s of waiting prevents starvation. Expired stimuli are dropped.

| Rank | Source | Default priority | TTL |
|---|---|---|---|
| 0 | Operator SAY / DIRECT | CRITICAL / HIGH | 60 s |
| 10 | Streamer voice | HIGH (a confirmed barge-in has already cut the audio) | 20 s |
| 20 | Game force with priority high/critical (M4) | as sent | none |
| 30 | Support (donation/sub/raid/redeem) | MEDIUM | 600 s |
| 40 | Chat that mentions her name | LOW | 40 s |
| 50 | Game force low/medium (M4) | as sent | none |
| 60 | Chat window: k=3 softmax-sampled candidates; the LLM picks one or none | LOW | 40 s |
| 70 | Non-silent game context (M4) | LOW | 30 s |
| 75 | Twin's heard line (M5) | LOW | 15 s |
| 80 | Vision (M7) | LOW | 20 s |
| 90 | Idle timer | LOW | – |

Cadence rules:
- **While `user_speaking` is set, no decision starts except OPERATOR.** The flag is set by `vad.start` and cleared by `vad.end`, with a 20 s watchdog. (M4 exception: forced game actions may be decided and dispatched, but their speech is gated until the streamer stops talking.)
- Chat decisions are at least `chat_min_interval_s` apart: 4 s, or 8 s if there was a voice turn in the last 20 s. After the brain becomes free, chat gathers for 1 s before deciding.
- At most one SUPPORT acknowledgement per decision.
- Chat window score:

  `2·exp(−age/15s) + 3·mention + 0.8·question + 0.7·sub + 0.5·mod + 0.3·vip + 0.5·first_msg − 1.5·ln(dup) − 5·[picked < 60 s ago]`

  Constraints: one candidate per user, shared-chat messages dropped, the window is consumed after each decision. This scoring is our own design; Neuro's weighting is unknown.
- A stimulus arriving mid-decision is merged into the next decision (T1 semantics). Two exceptions:
  - (a) CRITICAL aborts the current decision.
  - (b) If the new stimulus has a strictly better rank, priority ≥ HIGH, and nothing is audible yet, the current decision is aborted and restarted with both merged. This costs at most one TTFT.

### 4.4 Priority and interrupt semantics (SDK meanings, T1)

| Priority | While SPEAKING | While DECIDING, nothing audible yet |
|---|---|---|
| LOW | Queue; decide after `UtteranceDone` | Merge into the next decision |
| MEDIUM | Cancel the LLM stream, drop segments not yet started, let the current segment finish (`after_segment`), then decide | Merge |
| HIGH | Same as MEDIUM, but start the next decision **immediately**, in parallel with the current segment. New audio queues behind it | Abort and restart merged (§4.3b) |
| CRITICAL | `stop(now)`: 30–60 ms fade, cancel LLM and TTS, decide at once | Abort and restart |

If she is not speaking, all four levels behave the same. Every cut emits `UtteranceDone(cancelled=True, reason)`, which maps to the SDK message `speech_finished{isFinal:true, cancelled:true, reason}` in M4.

### 4.5 State machine
```
BOOTING → PRE_SHOW ──GO_LIVE──► IDLE ◄─────────── PAUSED ◄── FREEZE / HARD KILL (any state)
                                 │  ▲                 │ RESUME (inbox cleared except SUPPORT)
                    stimulus     │  │ UtteranceDone & nothing eligible
                                 ▼  │
                              DECIDING ──first segment audible──► SPEAKING
                                 ▲  └─abort/merge─┘                  │ HIGH/CRITICAL/barge-in → DECIDING
Flags: user_speaking · muted · strict · mic_mode · degraded{voice,llm,tts,avatar,chat}
```
State changes are sent to the panel and to the avatar driver (idle, listening, thinking and speaking poses).

### 4.6 Reply pipeline and speech segmentation
```
LLM TextDelta ─► EmotionTagExtractor ([tag] only on change; unknown [..] stripped)
  ─► ThaiSpeechChunker (first 8–60 chars, later 40–160; 60–160 when Azure F0 is primary; stall 500 ms → cut at last space;
                        never before a combining mark or after เแโใไ; never inside "100 บาท"/"02:30"/Latin phrases)
  ─► is_speakable? (no → merge forward) ─► SafetyGate.check_output(chunk, prev_tail=last 40 emitted chars)
       BLOCK → Filtered path (§4.10) · PASS/MASK/REPLACE → normalize_cloud|local → Segment(seq, emotion) → SpeechOutput.segment
ToolCalls accumulate (key: index → id → position); JSON parsed after the stream ends
```

In the voice worker:
- **Prefetch.** At most 2 syntheses in flight: segment N+1 is synthesised while N plays.
- **Decoding.** A stateful PyAV MP3 decoder. Never decode edge's 720-byte messages independently.
- **Markers.** Two markers per segment (start and end), timed to when the audio is actually heard.
- **Lip tracks.** Sent as the PCM is decoded.
- **Filler.** Voice turns only: if there is no first audio 1.2 s after the decision started, a cached filler plays (`อืม…`, "umm…"). At most one per 30 s, and never written to history.
- **Heard text.** For a cut segment, the heard text is truncated at the last complete edge/Azure `WordBoundary`.

**Openers.** There is no forced opener rule. The persona asks her to vary them. The trace records each reply's first 6 characters. If one opener starts more than 30 % of the last 20 replies, the next tail adds a note saying "don't start with X".

### 4.7 Barge-in: the reflex is local, the policy is in the core

1. **Candidate.** VAD start while the player is active, using the barge threshold (0.6 with headphones, 0.65–0.7 with speakers + AEC). The worker **ducks locally by −12 dB** within ~80 ms and sends `barge.candidate`.
2. **Confirm.** After ≥ 500 ms of speech, a quick Typhoon RT decode runs on the last 0.8 s (20–60 ms). It confirms if the text has ≥ 3 non-space characters and is not a backchannel. Backchannels within 1.0 s of the start or end of her utterance are ignored.
3. **Act.** With `policy=interrupt` the worker **cuts locally** (60 ms fade, ≤ 100 ms to silence) and sends `barge.confirmed{cut_local:true}`. The core then cancels the LLM and treats the event as CRITICAL. `duck_only` keeps playing ducked; `off` is used for song mode.
4. **False alarm.** If nothing is confirmed within 2 s, or VAD ends first: un-duck and continue.
5. **The streamer's words.** The full utterance still goes through normal endpointing and becomes a VOICE stimulus.
6. **Echo handling.**
   - Headphones (default): no AEC.
   - Speakers: livekit AEC3, with the player's post-gain blocks as the reference. Barge-in is ignored for the first 3 s of AEC warm-up, and transcripts matching the last 10 s of TTS text are dropped (difflib > 0.6).
   - If livekit fails to import: `energy_dtd`. Bad rooms: `half_duplex`.

### 4.8 Prompt assembly and cache epochs

Typhoon 2.5 renders tools inside the system prompt. llama.cpp reuses the KV cache only for an identical prefix. Cold prefill on the 30B is slow (roughly 300–800 tok/s estimated). So the prompt layout is:

```
[0] system     persona.th.md + static rules; the template renders the STATIC tool list here   ← fixed per session
[1] user       <context> core slots · last 2 episode summaries · rolling summary </context>  ← changes only at an epoch flip
[2] assistant  "รับทราบค่ะ" (fixed ack: "understood")
[3..n-1]       history since the epoch start: compact user lines ("[แชท] tom: …", "[สตรีมเมอร์] …", "[เกม:X] …")
               + assistant heard text (frozen at first render) + tool messages in call order     ← append-only
[n]  user      VOLATILE TAIL: <now> stimulus + one instruction line · <chat untrusted="true"> 1) [twitch][sub] ต้นกล้า: "…" </chat>
               · must-ack · game state/actions (M4) · recall (FTS top-2) · viewer facts · <new_memories> · notes </now>
```

(`[แชท]` = chat, `[สตรีมเมอร์]` = streamer, `[เกม:X]` = game X.)

Rules:
- Content is always a plain string, because the Typhoon template drops content-part arrays.
- Tool results follow call order, because the template has no `tool_call_id`.
- Disabled tools stay listed and fail with "unavailable now". Game actions never enter `tools` (§4.15).
- **Chat is data.** Each message is quoted and capped at 300 characters. Role tokens (`<|im_start|>`, `<tool_call>`, `system:`, `###`) are stripped. Display names are filtered. The persona states that quoted content is never an instruction.
- **Epochs with double-buffered slots.** A rebuild is triggered by pending long-term memory, history above 5K tokens, or a config reload. The steps:
  1. The background slot (2) runs a compaction job: the oldest half of history becomes a new rolling summary.
  2. The new prefix is prewarmed on the **inactive** speak slot (0 or 1) with `max_tokens=1`, but only while the brain is not DECIDING. If a decision starts, the prewarm is cancelled.
  3. On success, the active speak slot flips and `save_slot()` writes `<char>-<hash>.bin`.
  4. Until then, decisions keep using the old epoch on the old, still-warm slot.

  New `remember` writes appear in `<new_memories>` in the tail until the next flip.
- **Budgets** (tokens):

  | Part | Budget |
  |---|---|
  | Static (persona + tools) | ≤ 1800 |
  | Context | ≤ 1500 |
  | History | ≤ 5000 |
  | Tail | ≤ 250 (voice) / 450 (chat) |
  | Reply | `max_tokens` 256 (Thai ≈ 2 chars/token, ≈ 40 s of speech) |

  The uncached suffix per turn is roughly compact user (N−1) + assistant (N−1) + tail (N) ≈ 150–400 tokens.
- **Cache tests.**
  - A unit test asserts that for consecutive decisions in one epoch, prompt N+1 shares a byte-identical prefix with prompt N up to the end of history entry N−1.
  - The panel raises an alarm if `cache_n/prompt_n` < 0.85 over 5 turns.
- **Persona rules** (`persona.th.md`):
  - Spoken Thai, 1–3 short sentences, no emoji, markdown or lists.
  - Laugh as ฮ่าๆ. Keep English names in Latin script.
  - One `[emotion]` tag from {neutral, happy, sad, angry, surprised, shy, smug}, only when the emotion changes.
  - Never read JSON or action names aloud.
  - Deflect monarchy and politics topics.
  - Tolerate STT misspellings.
  - Call `remember` only for facts worth keeping across streams.

### 4.9 Tool-calling flow

1. The request carries the static tools, `tool_choice=auto`, `parallel_tool_calls` (llama.cpp) and `id_slot`. Text streams to speech while tool calls accumulate.
2. After the stream ends, each call goes through, in order:
   - `json.loads`, falling back to `json_repair`;
   - jsonschema Draft 2020-12 validation;
   - `SafetyGate.check_args` on every free-text argument;
   - policy: enabled, rate limit, required capabilities, `requires_approval` (panel queue, default deny after 20 s), dry-run mode;
   - execution with the tool's timeout;
   - `tool_audit`.
3. Results are appended in call order. Side-effect tools never retry automatically; a failure becomes the result text.
4. If the reply had **no spoken content** (Typhoon sometimes emits only tool calls), or any tool has `follow_up`, run **one** follow-up completion (`/u2`). Maximum 2 rounds.
5. Tools by milestone:
   - **M1:** `remember(text, about?, importance?, replace_slot?)` and `forget(slot)`.
   - **M3:** `set_stream_title`, `create_poll` (hidden if the API returns 403, i.e. not Affiliate), `timeout_user` (≤ 600 s; never mods, VIPs, or the triggering chatter without corroboration; undo button in the panel), `spin_model` (4 × 90° relative moves), `set_talking_speed` (clamped −20..+30 %), `play_sound`.

### 4.10 "Filtered."

When chunk k is BLOCKed:
1. Close the LLM stream (llama frees the slot in ≤ 0.2 s) and drop every segment not yet started. The segment already playing was clean, so it finishes (`after_segment`).
2. `play_canned("filtered")`: a clip of "Filtered." in the character's voice, synthesised at setup. The captions overlay shows `Filtered.` and the avatar goes neutral.
3. History records `heard_text + " [Filtered.]"`. The blocked text **never** enters LLM history or memory. It goes (PII-masked, 30-day retention) to `moderation_log`. The next tail carries the note "(ประโยคก่อนหน้าถูกกรอง)" ("the previous sentence was filtered").
4. Games still receive `speech_finished{isFinal:true}`. The counters feed auto-strict (§2.11).

### 4.11 LLM router, hot-swap and fallback

**Chain.** `local-30b → local-4b` (cold standby, started on demand by the launcher, ~3–5 s from page cache) `→ typhoon-api → gemini → canned line`. Cloud entries are **skipped unless `privacy.cloud_llm_consent=true`**, which setup asks about. The consent notice states that the Typhoon API may train on inputs, and that Gemini free-tier data is used by Google, so a paid key is recommended. **No hedging by default** (`hedge_after_ms=0`); enabling it requires consent.

**Failure handling.**
- Each provider has a circuit breaker: `min(60, 2^fails)` s, with a half-open probe.
- **Fallback happens only before the first event is emitted.** If a provider dies mid-utterance, the utterance ends with "…" (`ProviderFailed(emitted=True)`) and the next decision uses the next provider. There are never two voices within one utterance.
- **`promote(name)`** takes effect at the next decision boundary, and the previous provider stays configured. **Auto-rollback**: within 600 s of a promotion, 3 failures or a TTFT p95 above 2× the previous baseline rolls back and publishes `ProviderSwitched(reason="auto_rollback")` (mirrors T1 July 2026).

**Adapters.**
- String content only.
- Strip `extra_content` (Gemini `thought_signature`) for non-Gemini providers; keep it verbatim for Gemini in `provider_extra`.
- Never call `models.list()` on the Typhoon API (it returns a bare array). Its error body is `{detail}`.
- Gemini runs with `reasoning_effort="minimal"`. Typhoon 2.5 is non-thinking.

**Placement of llama-server.**
- Default is `--fit on -c 24576 --fit-target 3584`: leave 3.5 GB free for OBS, games and a later 4B.
- `bench llm --tune` then writes `placement="pinned"` with `-fit off -ngl all --n-cpu-moe N`.
- **On a load failure** (process exits, or `/health` never reaches 200 within the load window), the launcher raises N by 2, up to `--cpu-moe`, retries, and saves the new N to `data/state/llama_tuning.json`. After 3 failures the server is marked FAILED and the chain serves from the 4B.
- `--slot-save-path data/kv` is always passed.

### 4.12 Timeouts

| Operation | Deadline | On failure |
|---|---|---|
| LLM connect | 2 s | Next provider |
| LLM first token | 30B 4 s, 4B 2 s, Typhoon 6 s, Gemini 8 s | Next provider (nothing emitted yet) |
| LLM inter-token stall | 3 s | End the utterance ("…") |
| Decision watchdog | 30 s | Abort + flight dump |
| STT decode | 3 s | Next STT backend |
| TTS first audio | 2.0 s first segment (0 retries), 4.0 s later segments (1 retry) | Same-voice backend, then captions |
| Tool | policy `timeout_s` (default 5 s) | Result `timeout` fed back |
| Tool approval | 20 s | Deny |
| Tier-1 verdict (M2) | 300 ms input, 250 ms playback hold | Input: drop; output: fail open, **except fail-closed categories** |
| VTS awaited request | 2 s | Reconnect loop |
| IPC heartbeat | 1 Hz, dead after 3 misses | Worker restart |
| Game action result (M4) | 20 s | Force retry (≤ 3) |

### 4.13 Idle

The idle timer starts when playback ends: 25 s ± 5 s. Any speech or decision cancels it. It backs off ×2 (up to 120 s) while chat is active, and it is suppressed while the streamer speaks. Idle lines **are** written to history to avoid repetition. The prompt asks for a new topic and gives the last 5 topics.

### 4.14 M2 latency pack (behind flags)

- **Speculative endpointing.**
  - At 300 ms of trailing silence, the worker sends `stt.speculative` (one decode). A gated decision starts through the same `DecisionSlot`: LLM, chunker, filter and synthesis of segment 1 all run, but the voice worker holds playback.
  - On commit, if the normalised final transcript equals the speculative one, `open_gate` plays immediately and the worker reuses the transcript, so the final decode costs nothing. Otherwise cancel and run a live decision.
  - If speech resumes before commit, cancel. The 1 s merge window joins utterances as long as nothing has been heard yet.
  - **Speculative synthesis is never sent to a quota-limited backend.**
- **Particle endpointing.** 420 ms after final particles (ค่ะ ครับ นะ …), 900 ms after continuation words (และ แต่ ว่า …). Enabled only after S7 calibration. Auto-tuned: +50 ms if the false-endpoint rate is above 5 %, −25 ms if below 1 %.
- **KV prefill** (only if S8 passes). On `vad.start`, freeze the context part of the tail and prefill up to the transcript position.
- **Pre-think.** When turn N's LLM is done, ≥ 0.8 s of audio remains and something is due, decide N+1 with its gate tied to N's outcome (open after a 300 ms breath). If N is cut or filtered, invalidate N+1.
- **Idle precompute.** Compute the idle line 6 s before it is due, gated.
- **Review mode.** Every utterance waits at the gate until the operator approves it. Used for the first streams.

### 4.15 M4: games inside the brain

- **Unforced actions** use one static tool, `game_action(game, action, data)`, so the cached prefix is never invalidated. Currently available actions (name, description, schema) are listed in the tail. The real schema is checked at execution time and errors are fed back.
- **Forced actions on llama.cpp ("grammar" strategy):**
  - Make a `purpose="game"` request on slot 3 with `response_schema = {action: anyOf[{name: const N, data: S_N'}], say: string ≤ 200}`, action first.
  - `S_N'` is the output of the **schema sanitiser**: drop `$ref`/`$defs`/`allOf`/`oneOf`/`not`/`if-then-else`/`patternProperties`; never mix `properties` with `anyOf` at one level; keep `minimum`/`maximum` only on integers; anchor patterns `^…$`.
  - If sanitising fails, use the "tools" strategy (only the forced actions, `tool_choice="required"`); cloud providers always use tools.
  - A brace-depth parser **dispatches the action as soon as the `action` object closes**, because the game is frozen until then. `say` then streams to speech.
  - The action is always validated with jsonschema before it is sent. On failure: re-prompt once with the errors, then send anyway and let the game's error message serve as feedback.
- **Retries.** Up to 3 (Neuro's real count is unpublished). Before each retry, prune action names that have been unregistered. Retries are silent or at most one short line; when they run out, one line of commentary.
- **Ephemeral context.** `ephemeral_context` state lives only while the force is active; afterwards history keeps a single trace line.
- **Wire rules:**
  - Accept any path; `/…/voice` gets `voice/unavailable` and is then closed.
  - Send `actions/reregister_all` on every new connection.
  - Startup ack: `{sessionId, characterId:"pailin", displayName:"Pailin"}`.
  - `action.data` is a JSON *string* holding an object, and is omitted when there is no schema. Never send `null`, never add extra keys.
  - Unknown or malformed input is silently ignored.
  - Result timeout 20 s.
  - A second force from the same game replaces the first; forces from other games queue FIFO.
  - Context that arrives during a pending result is held and delivered in order after the result.
  - State is keyed by the message's `game` field. On disconnect, that game's actions are dropped but memory is kept.
  - `speech_finished` is broadcast per segment and once final.

### 4.16 M5: the twin

- The `Stage` hosts N `CharacterRuntime`s. They share one router (slots: 0/1 Pailin speak, 4/5 twin speak, 2 background, 3 game; `parallel=6`, `ctx 40960`), one `AudioOut` (one channel per voice; the AEC reference is the mix), chat, the safety base and the panel.
- **Per character:** persona, voice (`th-TH-AcharaNeural` on Azure; Edge has only Premwadee as a female Thai voice), filter overlay (e.g. the twin's profanity exemption, like Evil's reported three words, T5), memory database, VTS instance (UDP discovery by `windowTitle`), and SDK port 8010.
- **`FloorManager`** gives the floor to one speaker at a time. CRITICAL can steal it. A character's heard line becomes a CHARACTER stimulus for the other (rank 40 if addressed, otherwise 75).
- **Anti-ping-pong:** at most 4 exchanges with no human input, then a 30 s cooldown.

## 5. Latency budget (this PC)

Voice turn, measured from the streamer's last phoneme to the first audible sample (milliseconds, p50):

| # | Stage | M1 30B | M1 4B | M2 30B | Basis |
|---|---|---|---|---|---|
| 1 | End-of-utterance silence | 600 | 600 | 600 commit (work starts at 300) | Silero endpointer |
| 2 | Resample + VAD lag | 30 | 30 | 30 | soxr HQ bursts |
| 3 | STT final (Typhoon RT int8, 2 threads, ~5 s utterance) | 100 | 100 | 0 (reused) | 140–240 ms measured on a slow Xeon; 60–150 estimated on the i7 |
| 4 | IPC + intake + arbitration + prompt | 10 | 10 | 10 | |
| 5 | LLM TTFT incl. uncached suffix ≤ 350 tokens | 450 | 90 | 250 (prefilled) | Estimate 0.3–0.8 s (S2) |
| 6 | First chunk: 8–14 Thai chars ≈ 4–7 tokens | 150 | 50 | 150 | 1.96 chars/token measured; 35–45 tok/s estimated |
| 7 | Filter + normalise | 2 | 2 | 2 | newmm warm 0.1 ms |
| 8 | TTS first audio | 300 | 300 | 300 | edge cached/healthy 0.23–0.32 s measured; novel Premwadee from a datacentre 1.6–5 s → S1/G1 |
| 9 | Decode + enqueue | 10 | 10 | 10 | PyAV ~8 ms |
| 10 | Device buffer | 40 | 40 | 40 | WASAPI, measured in S4 |
| | **Target p50 / p95** | **≤ 1.7 s / 3.0 s** | **≤ 1.3 s / 2.2 s** | **≤ 1.3 s / 2.5 s** | Filler at 1.2 s masks the tail |

Other paths:

| Path | Target |
|---|---|
| Chat turn, decision start to first audio | p50 ≤ 1.0 s |
| Barge-in: onset to duck | ≤ 300 ms |
| Barge-in: confirm to silence | ≤ 100 ms |
| FREEZE | ≤ 150 ms |
| HARD KILL | ≤ 300 ms |
| Mouth vs audio | within ±40 ms (tracks lead by 40 ms) |
| Emotion visible after segment start | ≤ 0.4 s |
| M4 force to action sent | p50 ≤ 2.0 s |

Neuro's reference points: about 700 ms average in 2024 (T5, relayed via wiki), and "seconds, not milliseconds" for the voice-chat loop (T1, 2026).

**How the budget is enforced:**
- TurnTrace marks on every turn, with badges in the panel.
- `bench e2e` replays committed clips through the real stack.
- CI asserts **structural invariants**: TTS for chunk 1 is requested before the LLM stream ends; chunk 2 is synthesised before chunk 1 finishes playing; with speculation on, the LLM request precedes the commit; our own pipeline overhead is ≤ 30 ms p95 with zero-delay fakes.
- Startup warms everything: slot restore or prewarm, cached phrases, pythainlp (330 ms cold), sherpa, Silero, VTS auth.

**VRAM (stream profile):**

| Consumer | Estimate |
|---|---|
| DWM + browser | 0.5–0.8 GB |
| OBS + NVENC | 0.5–1.0 GB |
| VTS (one model) | 0.3–0.6 GB |
| llama 30B via `--fit` (dense ~1 + KV q8 24K ~1.2 + compute ~0.5 + experts ~0.36 GB/layer) | ~6–7 GB, N ≈ 36–40 (S2) |
| STT / VAD / TTS / AEC / tier-1 | 0 |
| **Free at launch** | **≥ 3.5 GB** |

A GPU-heavy game uses the `gaming` profile (4B, or the 30B with `--cpu-moe`). CPU: llama `-t 8` on the P-cores, STT 2 threads, VAD/AEC 1; the core uses under 10 % of one core.

## 6. Memory

**Principles:**
- The model chooses what to remember (T1).
- Long-term memory is small and slot-like. Evil reportedly had 3 slots (T5); we use 16.
- Memory is cross-stream (T1) and separate per character (T4).
- The operator can see and edit everything.
- **Chat-sourced writes are quarantined by default.**

**Storage:**
- `data/memory/<char>.sqlite` holds memory; `data/ops.db` holds traces, moderation, audit and jobs.
- WAL mode, `synchronous=NORMAL`, `busy_timeout=5000`.
- A single writer thread. Numbered migrations tracked with `user_version`.
- FTS5 trigram needs SQLite ≥ 3.34; `doctor` checks it, and search falls back to `LIKE` if it is missing.
- Daily backups plus one per session end via `sqlite3.backup`, keeping 14. Tokens and `user.toml` are backed up too. This is the lesson of Evil's weights going about a year without a backup (T5).

```sql
-- memory/<char>.sqlite
CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE session(id INTEGER PRIMARY KEY, started_at REAL NOT NULL, ended_at REAL, title TEXT, platforms TEXT, summary TEXT);
CREATE TABLE epoch(id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL REFERENCES session(id), started_at REAL NOT NULL,
  rolling_summary TEXT NOT NULL DEFAULT '', upto_turn_id INTEGER, prefix_hash TEXT, slot_file TEXT);
CREATE TABLE turn(id INTEGER PRIMARY KEY, session_id INTEGER NOT NULL REFERENCES session(id), epoch_id INTEGER NOT NULL REFERENCES epoch(id),
  ts REAL NOT NULL, role TEXT NOT NULL CHECK(role IN ('user','assistant','tool','note')), source TEXT NOT NULL, speaker TEXT,
  text TEXT NOT NULL, heard_text TEXT, interrupted INTEGER NOT NULL DEFAULT 0, filtered INTEGER NOT NULL DEFAULT 0,
  provider TEXT, turn_ref TEXT, tool_calls TEXT, provider_extra TEXT, tokens INTEGER);
CREATE INDEX turn_epoch ON turn(epoch_id, id);
CREATE TABLE memory(id INTEGER PRIMARY KEY, kind TEXT NOT NULL CHECK(kind IN ('core','fact','viewer','episode')),
  slot INTEGER, subject TEXT, platform TEXT, user_id TEXT, text TEXT NOT NULL,
  importance INTEGER NOT NULL DEFAULT 3 CHECK(importance BETWEEN 1 AND 5),
  source TEXT NOT NULL CHECK(source IN ('model','operator','consolidation','import')), origin TEXT NOT NULL DEFAULT '',
  origin_turn_id INTEGER REFERENCES turn(id),
  status TEXT NOT NULL DEFAULT 'active' CHECK(status IN ('active','quarantined','deleted')),
  pinned INTEGER NOT NULL DEFAULT 0, locked INTEGER NOT NULL DEFAULT 0, epoch_seen INTEGER,
  created_at REAL NOT NULL, updated_at REAL NOT NULL, last_used_at REAL, uses INTEGER NOT NULL DEFAULT 0);
CREATE UNIQUE INDEX memory_core_slot ON memory(slot) WHERE kind='core' AND status='active';
CREATE INDEX memory_viewer ON memory(platform, user_id) WHERE kind='viewer' AND status='active';
CREATE VIRTUAL TABLE memory_fts USING fts5(text, subject, content='memory', content_rowid='id', tokenize='trigram');
CREATE TRIGGER memory_ai AFTER INSERT ON memory BEGIN INSERT INTO memory_fts(rowid,text,subject) VALUES(new.id,new.text,new.subject); END;
CREATE TRIGGER memory_ad AFTER DELETE ON memory BEGIN INSERT INTO memory_fts(memory_fts,rowid,text,subject) VALUES('delete',old.id,old.text,old.subject); END;
CREATE TRIGGER memory_au AFTER UPDATE ON memory BEGIN
  INSERT INTO memory_fts(memory_fts,rowid,text,subject) VALUES('delete',old.id,old.text,old.subject);
  INSERT INTO memory_fts(rowid,text,subject) VALUES(new.id,new.text,new.subject); END;
CREATE TABLE viewer(platform TEXT NOT NULL, user_id TEXT NOT NULL, name TEXT, first_seen REAL, last_seen REAL,
  messages INTEGER NOT NULL DEFAULT 0, picked INTEGER NOT NULL DEFAULT 0, opt_out INTEGER NOT NULL DEFAULT 0,
  muted_until REAL, PRIMARY KEY(platform, user_id));
CREATE TABLE runtime_state(key TEXT PRIMARY KEY, value TEXT NOT NULL);
-- data/ops.db
CREATE TABLE turn_trace(turn_id TEXT PRIMARY KEY, character TEXT, session_id INTEGER, kind TEXT, provider TEXT, tts_backend TEXT,
  tts_identity TEXT, stages TEXT NOT NULL, ttfa_ms REAL, tokens_out INTEGER, tok_s REAL, prompt_n INTEGER, cache_n INTEGER,
  speculative INTEGER, opener TEXT, outcome TEXT);
CREATE TABLE moderation_log(id INTEGER PRIMARY KEY, ts REAL, character TEXT, direction TEXT, source TEXT, tier TEXT,
  category TEXT, rule TEXT, verdict TEXT, text_masked TEXT, text_sha256 TEXT, author TEXT, turn_id TEXT);
CREATE TABLE tool_audit(id INTEGER PRIMARY KEY, ts REAL, character TEXT, turn_id TEXT, tool TEXT, args TEXT, verdict TEXT,
  result TEXT, approved_by TEXT, dry_run INTEGER);
CREATE TABLE op_audit(id INTEGER PRIMARY KEY, ts REAL, operator TEXT, command TEXT, args TEXT, result TEXT, latency_ms REAL);
CREATE TABLE job(id INTEGER PRIMARY KEY, kind TEXT NOT NULL, state TEXT NOT NULL, payload TEXT,
  attempts INTEGER NOT NULL DEFAULT 0, next_run_at REAL, last_error TEXT);
```

**Short-term memory.** Every turn is persisted. The prompt holds the turns since the start of the current epoch; compaction (§4.8) folds older turns into the rolling summary. On a core restart, a session younger than 6 h is resumed.

**Long-term slots.**
- `remember` writes `kind='core'` (pinned), up to 16 slots of ≤ 120 characters each.
- When all slots are full, the tool returns `SlotsFull` with the slot list, so the model must `forget` a slot or pass `replace_slot`.
- `about=<viewer name present in the chat block>` stores a `viewer` fact keyed by `(platform, user_id)`. Viewers with `opt_out` are refused.
- Every write passes tier-0 and the PII regexes (phone numbers, national ID, email).
- Rate limits: at most 1 write per 120 s and 8 per session.
- **Writes from a decision triggered by chat or support, and all viewer facts, are stored `quarantined`** until the operator approves them in the panel. Voice- and operator-sourced core writes are `active`.
- Locked items refuse `forget`.

**Episodes.** At session end (`END_STREAM`, shutdown, or 30 minutes idle), a job writes one `episode` summary of ≤ 150 tokens and proposes deduplication of facts, which the operator can revert. If the LLM is down, the job waits and runs at the next start.

**What goes into the prompt:**

| Where | What | Cap |
|---|---|---|
| Context block | Active core slots, last 2 episodes, rolling summary | ≤ 1500 tokens |
| Tail | FTS top-2 hits for the stimulus text (≥ 3 chars, not already in slots) | – |
| Tail | Viewer facts for authors in the chat block (max 3 per viewer, 5 viewers) | – |
| Tail | `<new_memories>` written since the epoch began | – |

## 7. Safety

A human moderator is expected: "there currently needs to be a human there to moderate" (T1).

**Normalisation (used for matching only; the original text is kept for display):**
1. NFKC.
2. Strip zero-width characters (U+200B/C/D, U+2060, U+FEFF, U+00AD).
3. `pythainlp.util.normalize`.
4. Casefold.
5. Collapse character runs longer than 2.
6. Convert Thai digits to Arabic.

Two extra forms are also built:
- a **despaced form**, where runs of single characters separated by spaces or `.-_*` are collapsed. This catches letter-by-letter spelling, a documented bypass (T5).
- a Latin **leet map** (0→o, 1→i, 3→e, 4→a, @→a, $→s).

**Tier-0 (sync, < 1 ms, M1):**
- **Token rules.** Thai words are matched on pythainlp newmm tokens. Raw substring matching over-blocks words such as หีบ ("box").
- **Substring rules.** Aho-Corasick (pyahocorasick) over the normal and despaced forms, used only for unambiguous entries (Latin slurs, invite links, scam domains).
- **Regex.** URLs, `@handles`, Thai phone numbers `0[689]\d{8}`, 13-digit national IDs, emails.
- **Replace rules.** Hard-coded speech patches, following Vedal "hard coding it out of her speech" (T1).
- Output checks run on `prev_tail(40) + chunk`, so a phrase split across two chunks is still caught.

| Category | Input | Output / tool / memory |
|---|---|---|
| slur, sexual, doxx, violent_extreme | DROP | BLOCK |
| self_harm | DROP + panel alert | BLOCK |
| **monarchy_112** (Thai lèse-majesté law) | DROP; persona deflects | BLOCK, **fail-closed even if tier-1 times out** |
| gambling_scam (สล็อต, เว็บตรง… — slots, direct-casino sites) | DROP | BLOCK |
| politics | REVIEW (configurable) | REVIEW (strict mode: BLOCK) |
| PII / URL | MASK (`[ลิงก์]`, "[link]") | BLOCK |
| role tokens / injection patterns | stripped | – |

**Lists and overlays:**
- `config/filters/base/` is committed and conservative.
- `config/filters/private/` is gitignored and curated by the streamer.
- Per-platform overlays (the Bilibili precedent, T4).
- Per-character `filters.toml` overlays (allow/deny/replace).
- Hot reload through `FILTER_RELOAD`.

**Tier-1 (M2):**
- Model: `typhoon-ai/typhoon2-safety-preview` (MIT, mDeBERTa-v3-base, trained on Thai sensitive topics). Exported once to ONNX int8 in CI and run with onnxruntime + tokenizers in the core's `guard` thread. No torch at runtime.
- What it scores: only the k selected chat candidates, must-ack texts, new display names, tool and memory arguments, and each output chunk. Output chunk k+1 is scored while chunk k plays, with a hold of at most 250 ms.
- Thresholds: above 0.8 BLOCK; 0.5–0.8 REVIEW, which means drop for chat, quarantine for memory, and BLOCK in strict mode.
- The threshold is calibrated on a labelled set of Thai stream slang before it is enforced.
- Alternative: Qwen3Guard-Gen-0.6B, if jailbreak detection matters, behind the same `Classifier` seam.

**Structural protections:**
- Chat, game strings and display names are always quoted data in the user role.
- Tool policy is enforced in code: allow-lists, caps, rate limits, approval, never bans.
- The LLM cannot change config, filters or tool flags.

**Logging.** Every BLOCK and REVIEW goes to `moderation_log` (masked text plus sha256, 30 days) and to the panel feed. The feed has one-click actions: mute user, add to blocklist, mark false positive.

## 8. Config and directory layout

**Layering** (later layers win):
1. `config/defaults.toml` (committed)
2. the active `[profiles.<name>]`, deep-merged
3. `characters/<id>/character.toml`
4. `config/user.toml` (written by `setup`, gitignored)
5. environment variables `AIVTUBE__SECTION__KEY`
6. CLI flags

Secrets live only in `.env`. The config is validated with pydantic v2, and errors come out in Thai and English with a fix hint. `aivtube config migrate` writes a `.bak` first. The full `config/defaults.toml` is given in the companion `config_example_toml`. Its structure is:
- `[ports]`, `[privacy]`, `[audio]`, `[mic]`, `[vad]`, `[barge_in]`;
- `[stt.backends.*]`, `[llm.servers.*]`, `[llm.providers.*]`;
- `[tts.identities.*]`, `[tts.backends.*]`, `[tts.chunker]`;
- `[brain]`, `[chat.*]`, `[memory]`, `[safety]`, `[avatar]`, `[tools]`, `[games]`;
- `[profiles.{light,gaming,text,offline,ci}]`.

```toml
# characters/pailin/character.toml
id = "pailin"
display_name = "Pailin"                      # ASCII: SDK displayName, game UIs
name_th = "ไพลิน"
persona = "persona.th.md"
aliases = ["ไพลิน", "pailin", "ไพ่ลิน", "น้องไพลิน"]
tts_identity_chain = ["premwadee"]
emotions = ["neutral", "happy", "sad", "angry", "surprised", "shy", "smug"]
cached_phrases = ["Filtered.", "อืม…", "เอ่อ…", "แป๊บนะ…", "เอ๊ะ สมองไพลินค้างแป๊บนึงนะ", "ขอบคุณมากนะคะ"]
[stt_aliases]
"ไพลิน" = ["ไทลิน", "ไทยลิน", "ไทลิล", "ไภลิน"]
[avatar]
vts_url = "ws://127.0.0.1:8001"
vts_window_title = ""                        # set for twins; UDP 47779 discovery picks the instance
plugin_name = "AI_Vtube Brain"
plugin_developer = "AI_Vtube"
token_file = "data/tokens/vts_pailin.txt"
[avatar.emotion_map]                         # verify exp_NN visually in VTS first (example: Live2D sample Mao)
neutral = { expressions = [], smile = 0.5, brows = 0.5 }
happy = { expressions = ["exp_03.exp3.json"], smile = 0.9, brows = 0.7 }
sad = { expressions = ["exp_05.exp3.json"], smile = 0.15, brows = 0.2 }
[memory]
db = "data/memory/pailin.sqlite"
[safety]
overlay = "filters.toml"
[games]
port = 8000
character_id = "pailin"
[tools]
enabled = ["remember", "forget"]             # M3 adds stream/avatar/voice/sound tools
```
```dotenv
# .env (never committed; .env.example is)
TYPHOON_API_KEY=
GEMINI_API_KEY=
AZURE_SPEECH_KEY=
AZURE_SPEECH_REGION=southeastasia
YOUTUBE_API_KEY=
TWITCH_CLIENT_ID=
```
```
AI_Vtube/
├─ pyproject.toml · uv.lock · README.md · LICENSE · NOTICE (Typhoon CC-BY-4.0, Live2D sample notice) · .env.example
├─ setup.ps1 · run.bat · config/{defaults.toml, user.toml.example, filters/{base/,private/(ignored)}}
├─ characters/{_template/, pailin/{character.toml, persona.th.md, filters.toml, lexicon.toml}}
├─ models/manifest.toml (url, sha256, size, licence) · models/** ignored · vendor/ (llama.cpp, ignored) · data/ logs/ (ignored)
├─ src/aivtube/
│  ├─ __main__.py cli.py plugins.py
│  ├─ contracts/  types events infra voice speech llm avatar chat memory safety tools control games ipc
│  ├─ config/     schema load migrate
│  ├─ infra/      bus tasks clock ticker logging trace flight metrics registry lag
│  ├─ ipc/        server client
│  ├─ launcher/   main supervisor jobobject emergency llama gpu console
│  ├─ text/       chunker normalize tags thai
│  ├─ voice/      worker audio_io aec vad endpointer frontend barge lipsync speech_queue stt/ tts/
│  ├─ speech/     output
│  ├─ llm/        openai_stream providers router llamacpp
│  ├─ brain/      loop decision intake arbiter preempt prompt reply tool_flow history background idle latency(M2) review(M2) floor(M5)
│  ├─ avatar/     vts_client vts_sink discovery driver idle_motion emotion browser/(M6)
│  ├─ chat/       window twitch_irc youtube_poll twitch_auth(M3) twitch_eventsub(M3) helix(M3) youtube_grpc(M3) alerts(M3) tiktok(M6)
│  ├─ memory/     store ops_db migrations/ backup
│  ├─ safety/     normalize keyword pii gate audit classifier(M2) lists/
│  ├─ tools/      registry builtin/{memory, stream(M3), avatar(M3), voice(M3), sound(M3), twin(M5), game(M4)}
│  ├─ games/      neuro_compat grammar bridge (M4)
│  ├─ panel/      server static/{index.html, overlay_captions.html}
│  ├─ console/    textmode
│  ├─ app/        core_main runtime
│  ├─ ops/        doctor bench setup_wizard models report db
│  └─ testing/    fakes/ contracts/ sim/
├─ workers/whisper (M2) · workers/voxcpm (M6, own pyproject)
├─ tools/ export_typhoon_rt_onnx.py · export_guard_onnx.py · convert_whisper_ct2.sh
├─ tests/ unit/ contract/ integration/ windows/ e2e/scenarios/*.yaml fixtures/{audio,sse,irc,vts,youtube,sdk}
├─ docs/ adr/ operator-guide.md obs-setup.md persona-guide.md
└─ .github/workflows/ ci.yml nightly.yml models-export.yml
```

## 9. Dependencies and install

**Python packages** (full block in `pyproject_dependencies`):
- **Core:** numpy, websockets 17.1, openai 3.19.2 (brings httpx2, which we also pin and use directly for Twitch and YouTube REST), aiohttp (panel), pydantic, pydantic-settings, python-dotenv, tomli-w, platformdirs, pythainlp 5.3.8, pyahocorasick 2.3.1, jsonschema 4.26.0, json-repair 0.63.5.
- **Extras:**
  - `voice`: sounddevice 0.5.6, soxr 1.1.0, onnxruntime 1.30.0, sherpa-onnx 1.13.8, edge-tts 7.2.8, av 18.1.0
  - `aec`: livekit 1.1.20
  - `tts-azure`: azure-cognitiveservices-speech 1.51.2
  - `tts-piper`
  - `stt-pythaiasr`
  - `stt-gpu` (imported only by the Whisper worker)
  - `guard`
  - `youtube-grpc`
  - `tiktok`
  - `dev`
  - `pc` = voice + aec + tts-azure + stt-pythaiasr
- **No torch.**

**External binaries and models** (pinned by sha256 in `models/manifest.toml`):
- llama.cpp b11177 `llama-…-bin-win-cuda-13.4-x64.zip` plus the matching `cudart` zip (use the 12.4 pair if the driver is older than R580).
- mradermacher `typhoon2.5-qwen3-30b-a3b.Q4_K_M.gguf` (18.56 GB) and `typhoon2.5-qwen3-4b.Q4_K_M.gguf` (2.50 GB).
- `silero_vad.onnx` (sha256 1a153a22…8788e3).
- Our own Typhoon RT int8 ONNX, about 138 MB, published on the project's HF repo with CC-BY-4.0 attribution. The third-party copy has no licence, so it is not used.
- Optional: Piper `th_TH-tsync2-medium` (non-commercial).

The user installs VTube Studio (Steam; the paid DLC is required if the stream is monetised) and OBS with the Spout2 plugin.

```powershell
# setup.ps1 — only bootstraps uv; all logic lives in `aivtube setup` (Python, testable, resumable)
$ErrorActionPreference = 'Stop'
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { irm https://astral.sh/uv/install.ps1 | iex; $env:Path = "$env:USERPROFILE\.local\bin;$env:Path" }
uv python install 3.12
uv sync --frozen --extra pc
uv run aivtube setup @args
```

`aivtube setup` steps:
1. Check the driver with `nvidia-smi`, download llama.cpp into `vendor/`, and verify that `--list-devices` shows the RTX 4070.
2. Download models in order: Silero, then STT (the Typhoon RT export, or PyThaiASR until it is published), then the 4B, then **the 30B as a resumable background download**. The profile stays `light` until the 30B hash verifies.
3. Audio wizard: list WASAPI devices by name, play a test tone, show the mic level, and ask "headphones?".
4. Chat platform choice and a read test.
5. VTS discovery and plugin authorisation (the streamer clicks Allow).
6. Privacy consent for cloud fallbacks (default: no).
7. `bench tts` → recommends edge or Azure.
8. Pre-synthesise the cached phrases, including "Filtered.".
9. Write `user.toml`, set `HF_HOME=models/hf`, and run `doctor`.

```bat
:: run.bat — one command to go live
@echo off
cd /d "%~dp0"
set HF_HOME=%~dp0models\hf
:loop
uv run --frozen aivtube run %*
if %errorlevel%==3 goto loop
pause
```

**Commands:**
- `aivtube run [--profile light|gaming] [--text [--speak]] [--safe] [--fake-llm]`
- `setup`, `doctor`, `bench {tts,llm [--tune],stt,audio,e2e}`, `report`
- `models {pull,verify}`, `sim <scenario.yaml>`, `replay <dump>`
- `config {show,migrate}`, `db {backup,restore}`, `llm serve` (standalone server, adopted later)
- `debug wedge-core` (test only)

**Text console:**
- A plain line is spoken by the streamer.
- `/chat name: text`, `/bits name n`, `/sub name months`
- `/op freeze|resume|skip|say …|direct …`
- `/mem`, `/state`

**Windows traps handled:**
- Never run in Session 0 (SSH or as a service).
- Devices are resolved by name through WASAPI, never via `sd.default` (which is MME).
- No exclusive audio mode.
- `127.0.0.1` everywhere, and `proxy=None`.
- edge-tts builds its own certifi SSL context; behind a TLS-intercepting proxy, `_SSL_CTX` must be overridden.
- A Defender exclusion for `models/` is suggested.

## 10. Test strategy (no GPU, no audio)

**Rules:**
- Every Protocol has a fake and a parametrised contract suite.
- Every adapter takes its I/O backend by injection.
- Time goes through `Clock`, and `FakeClock` makes timers deterministic.
- `sounddevice` is imported lazily, so ubuntu CI needs no libportaudio.

| Fake | Simulates |
|---|---|
| FakeSD | PortAudio host APIs and devices, `WasapiSettings`, streams. `.pump(n)` drives output callbacks, `.push(x)` input, `.die()` fires `finished_callback` |
| FakeVAD / FakeRecognizer | Scripted probabilities and texts with delays; marker tones in the mic audio map to Thai sentences |
| FakeTTS | Silence of `len/12.5` s plus word marks. Knobs for TTFA, failure rate and NoAudioReceived |
| FakeLLM + SseFixtureServer | Scripted TextDelta/ToolCall streams: Thai combining marks split across chunks, Gemini calls without an index, stalls, mid-stream death. The real provider is also exercised against `httpx2.MockTransport` using recorded llama-server SSE |
| FakeVTSServer | Answers once per 1/60 s. Auth, errors 453/454/202, events carrying the subscription's requestID |
| FakeIrcServer, FakeYouTube | Recorded IRC lines, PING, disconnects; list JSON with backlog |
| FakeSpeechOutput | In-process voice worker with an audible timeline |
| FakeLauncher | LocalServerManager + emergency endpoint |

**Test layers:**
1. **Unit.**
   - Chunker (hypothesis: split-invariant, lossless, never cuts before a combining mark, ≤ 160 chars).
   - Normaliser, tag extractor.
   - Tier-0: Thai false-positive list passes, red-team list blocks, despaced/leet/cross-chunk cases.
   - Window scoring, arbiter TTL and ranking, the preemption table, the state machine.
   - `DecisionSlot` cancellation semantics.
   - Prompt prefix byte-stability.
   - Memory: FTS on Thai, quarantine.
   - Router: fallback before/after first emission, breakers, auto-rollback, consent gating.
   - Tool policy. Config errors.
2. **Contract.** IPC message schemas validated at both ends; event JSON round-trip; each Protocol suite run against the real implementation and the fake.
3. **Integration** (real code, fake I/O).
   - Voice worker in-process with FakeSD, real soxr, the real committed Silero ONNX (2.3 MB, MIT), FakeRecognizer and FakeTTS. Covers barge-in timing, cancel-to-silence within one block, markers, heard text, the forced-split concatenation.
   - VTS client against FakeVTS. IRC against FakeIrc. The launcher supervisor with dummy children.
4. **Windows-only (windows-latest CI).**
   - **ORT coexistence:** the python onnxruntime runs Silero, and sherpa-onnx's own VAD runs the same file, in one process.
   - Job Object kill-on-close.
   - WASAPI resolution logic with FakeSD.
   - `perf_counter` values comparable across processes.
5. **Simulated end-to-end** (`aivtube sim`). The launcher starts core and voice (FakeSD, fake models), plus a fake SSE server, FakeVTS and FakeIrc over real localhost sockets.

   Scenarios:
   - `voice_turn`, `chat_pick`, `merge_mid_decision`, `priority_{low,medium,high,critical}`
   - `barge_in` (a backchannel and a 0.25 s blip must not cut), `long_monologue` (a 15 s split must not trigger a reply)
   - `read_aloud_dedupe`, `ptt_deafen`
   - `filtered` (a split phrase is caught), `freeze`, `hardkill_wedged_core`
   - `llm_kill_fallback`, `mid_utterance_death`, `promote_rollback`
   - `vts_restart`, `irc_drop`, `core_restart`
   - `memory_cross_session`, `memory_quarantine`, `epoch_flip`

   Assertions check the event trace (golden files), structural latency invariants, and overhead ≤ 30 ms. M2+ adds `speculation_hit/miss`, `game_force_retries`, `tool_abuse`, `twin_floor`.
6. **Nightly** (cached models):
   - Real Typhoon RT on committed Thai clips (CER bound; "ไพลิน" recovered via the alias map).
   - CPU `llama-server` with the 4B Q4_K_M: tool round trip, cancel frees the slot in ≤ 0.2 s, `cache_n > 0`, the S8 regression.
   - Piper; an edge smoke test (allowed to fail); a 1 h sim soak.
7. **Hardware** (on the PC only): `doctor`, `bench`, the M1 chaos drill, a 2 h soak.

**CI.** `ci.yml` runs a matrix of ubuntu-latest {3.11, 3.12} and windows-latest {3.12}:
- `uv sync --frozen --extra voice --extra aec --extra dev`
- `ruff`
- `mypy --strict` on `contracts/`, `brain/`, `ipc/`
- `pytest -m "not nightly and not hardware" --timeout 120`

`models-export.yml` is a manual job that exports Typhoon RT and the guard model to ONNX on Linux.

## 11. Risks

| # | Risk | Mitigation |
|---|---|---|
| R1 | edge Premwadee is slow or flaky on novel text (1.6–5 s, ~25 % failures from a datacentre). It is also an unofficial, grey-area endpoint | S1/G1 decides; Azure with the same voice; prefetch, filler, cached phrases; pinned version; captions |
| R2 | The 30B is too slow on this RAM, or VRAM is contended | S2/G2; `light`/`gaming` profiles; hot-swap; fit headroom of 3.5 GB; N stepping on OOM |
| R3 | Prompt-cache invalidation (4–10 s cold prefill) | Epochs with double-buffered slots; static tools; `game_action` in the tail; CI prefix test; cache alarm; slot save/restore |
| R4 | Two onnxruntime DLLs in the voice worker | S3 plus a Windows CI test; `silero_sherpa` fallback; STT in its own process |
| R5 | Typhoon RT garbles English words and her name | Alias map; tolerant prompt; Whisper tier (M2); evaluation set from the streamer's VODs |
| R6 | False barge-ins (game audio, laughter, echo) | Headphones by default; confirm step with backchannel filter; 2 s false-interruption resume; AEC3 plus warm-up; `half_duplex`; PTT/deafen |
| R7 | She answers when the streamer is not talking to her | PTT/deafen hotkeys; `name_or_question` addressing; read-aloud dedupe; no decision while the streamer is speaking |
| R8 | Harmful output; Thai §112 legal exposure; a Twitch ban (Neuro was banned for 2 weeks in Jan 2023, T1) | Tier-0 on every chunk plus the previous tail; monarchy fail-closed; tier-1 in M2; human moderator; kill ladder; review mode; audit |
| R9 | Prompt injection and memory poisoning through chat, names or game strings | Quoted untrusted blocks; role-token stripping; policy in code; quarantine; write caps |
| R10 | Tool abuse | Tools only in M3 after tier-1; caps; approval queue; no bans; mods/VIPs protected; undo; per-tool disable |
| R11 | Native crashes (PortAudio, livekit, sherpa) | Isolated in the voice worker; faulthandler; fast restart; captions/text degradation |
| R12 | GIL contention inside the voice worker (TTS network and decode next to the audio callbacks) | ABOVE_NORMAL priority; switch interval 1 ms; allocation-free callbacks; S4 soak; option to move TTS to its own process |
| R13 | Windows timer resolution (15.6 ms) breaking pacing and markers | `perf_counter` everywhere; `PrecisionTicker` thread; 30 Hz fallback |
| R14 | GGUF without a chat template silently loses tools | mradermacher GGUFs; startup assertion on `/props` |
| R15 | Azure F0 quota (20 transactions per minute, not adjustable) | Token bucket; `min_chars` 60; no hedging or speculative synthesis on F0 |
| R16 | Cloud privacy | Opt-in consent; a local-only default |
| R17 | Dependency and API churn (openai/httpx2, llama.cpp flags, websockets, Neuro SDK spec) | `uv.lock --frozen`; pinned binaries by hash; thin adapters; nightly tests; configurable retry counts |
| R18 | Licensing: Typhoon CC-BY attribution, non-commercial Piper voice, VTS DLC for monetised streams, Live2D samples, TikTokLive AGPL, the "Neuro" name | NOTICE; `doctor` flags non-commercial voices; no Cubism Core in the repo; describe M4 only as "compatible with the Neuro Game SDK protocol" |
| R19 | Parallel agents drifting apart | Contracts frozen at M0; ADRs; contract suites; sim scenarios that cover every seam |

## Appendix A: IPC protocol v1 (core ↔ voice worker)

**Transport and handshake:**
- WebSocket `ws://127.0.0.1:8771/bus`, JSON text frames, the core is the server.
- Handshake: `hello{role, pid, version, token, caps}`. A bad token or a different major version is rejected.
- 5 × `ping/pong` measure RTT and clock offset as a sanity check.
- Heartbeat at 1 Hz; a peer is dead after 3 misses.
- Envelope: `{"v":1,"type":…,"id":…,"corr":…,"ts":<perf_counter>,"data":{…}}`.

**Core → voice:**
- `voice.configure{audio, vad, barge_in, stt_chain, tts{identities, backends, chunk}, characters{id→{identity_chain, cached_phrases}}}`
- `voice.policy{VoicePolicy}`
- `speak.begin{utt, character, filler_after_s, gate_open}`
- `speak.segment{utt, seq, text, caption, emotion, last, kind}` → reply `ok|busy`
- `speak.gate{utt}`
- `speak.stop{utt|null, mode, reason, fade_ms}`
- `speak.duck{gain, ramp_ms}`
- `speak.canned{key, character}`
- `voice.mute{on}`
- `voice.rate{character, percent}`

**Voice → core:**
- `health{state, detail, stats}`
- `vad.start{t, barge}`, `vad.end{t, audio_s}`
- `stt.final{text, engine, latency_ms, t_end, audio_s, parts}`
- `stt.speculative{text, t, silence_ms}` (M2)
- `barge.candidate{t}`, `barge.confirmed{t, text, cut_local}`, `barge.rejected{t}`
- `speech.segment_started{utt, seq, t_audible, duration_s, backend, silent}`
- `lip.track{utt, seq, t0, fps, mouth[], form[], final}`
- `speech.segment_done{utt, seq, heard, heard_text}`
- `speech.utterance_done{utt, heard_text, cancelled, reason}`
- `tts.fallback{from, to, reason}`
- `tts.constraints{character, first_min_chars, min_chars, max_chars, backend, identity}`

**Backpressure:** at most 8 queued segments per utterance.

## Appendix B: work packages for M1

| WP | Modules | Depends on |
|---|---|---|
| WP0 (first, ~2 days) | contracts, events, contracts.*, config, infra, testing.fakes, testing.contracts, packaging | – |
| WP1 Voice I/O | voice.audio_io, voice.aec, voice.vad, voice.endpointer, voice.barge | WP0 |
| WP2 STT | voice.stt, export job | WP0 |
| WP3 TTS | voice.tts, voice.speech_queue, voice.lipsync | WP0, text |
| WP4 Worker + IPC | voice.worker, ipc, speech.output | WP0 |
| WP5 Text | text.chunker, text.normalize | WP0 |
| WP6 LLM | llm, llm.llamacpp | WP0 |
| WP7 Brain | brain.* | WP0 (fakes for the rest) |
| WP8 Safety | safety | WP0, text |
| WP9 Memory + tools | memory, tools | WP0 |
| WP10 Chat | chat.window, chat.twitch_irc, chat.youtube_poll | WP0 |
| WP11 Avatar | avatar | WP0 |
| WP12 Ops | launcher, panel, console, app, ops.*, cli | WP0, then integration |
| WP13 Sim | testing.sim, e2e scenarios | WP0, then all |

## Appendix C: runbook

| Symptom | First action | If it persists |
|---|---|---|
| She says something bad | **FREEZE** (F) | Check the moderation feed, add to blocklist, RESUME. Panel dead → launcher `K` or OBS mute |
| LLM red, canned line played | Wait 5 s for the fallback | Panel → LLM → `local-4b`; `aivtube report` afterwards |
| TTS fallbacks climbing | Panel → TTS identity order (Azure first) | Check network; captions keep the show going |
| Mic red | Check the headset (reopens every 2 s) | Restart voice; `aivtube setup --audio` |
| Avatar red | Restart VTS (auto-reconnect) | Re-approve the plugin in VTS |
| Latency badge amber | Read the waterfall | `gaming`/`light` profile during heavy games |
| Launcher shows a crash loop | `R` to retry | `aivtube report`; `run --safe` |

## Appendix D: must-fix ledger

| Must-fix | Resolution (section) |
|---|---|
| Twin SDK hub on 8001, which collides with VTS; panel port clashing with the browser renderer | Pailin 8000, twin 8010, panel 8770, renderer reserved 8765/8766, `doctor` port check (§2.2) |
| `time.monotonic` / `loop.time` are coarse on Windows 3.12 | `perf_counter` everywhere, `audio_io` patched, `PrecisionTicker` (§2.5) |
| onnxruntime + sherpa-onnx DLL coexistence | S3 + Windows CI test + `silero_sherpa` fallback (§1, §10) |
| llama placement: pinned N before measuring; thin fit-target; missing `--slot-save-path` | `--fit on`, fit-target 3.5 GB, pin after tuning, N step on OOM, slot path always passed, `/props` assertion (§4.11) |
| Brain concurrency: swallowed cancellation, unreferenced tasks, a single decision path | `DecisionSlot` with `asyncio.wait`, `TaskSupervisor.track`, every decision routed through the slot (§4.1) |
| Game grammar schemas | Sanitiser + tools fallback + jsonschema validation (§4.15) |
| Output filter must see the previous chunk's tail, plus tool/memory args, plus `speech_finished` | `prev_tail(40)`, `check_args`, blocked text never stored, games still notified (§4.10, §7) |
| Cloud privacy; Azure F0 quota | Consent flag, no hedging by default, token bucket, `min_chars` 60, no speculative synthesis to quota-limited backends (§4.11, §4.14) |
| Untested particle endpointing shipped as default | 600 ms in M1; particle heuristic behind a flag in M2 after S7 (§4.14) |
| TTS voice switching to Piper mid-utterance | Identity groups; Piper only for a whole utterance; captions for the rest of the utterance (§2.8) |
| Chat-sourced memory allowed by default | Quarantine by default, rate limits, tier-0 + PII checks (§6) |
| Hard kill must work when the core is wedged; audio in its own process | Launcher endpoint 8779 plus console key; voice worker from M1 (§2.11) |
| 15 s force-split starting a reply mid-monologue | Partials concatenated; no decision while `user_speaking` (§4.2, §4.3) |
| Co-host answering everything | PTT/deafen hotkeys, addressing modes, read-aloud dedupe (§4.2, §2.11) |
| Auto-freeze can be baited by trolls | Auto-strict with chatter mutes instead; auto-freeze opt-in (§2.11) |
| Split-phrase and letter-spelling bypass; monarchy fail-closed | Tail check, despaced/leet forms, fail-closed category (§7) |
| Chat platform and donations assumed | S9; YouTube polling in M1 if primary; `/api/event`; alert adapters in M3 (§1) |
| VRAM headroom | ≥ 3.5 GB free, `gaming` profile (§5) |
| Formulaic openers | No forced opener; opener repetition tracked (§4.6) |
