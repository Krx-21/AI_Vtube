# Thai STT + VAD — component brief

_Research snapshot: 2026-09-25. Verified facts carry a source; unverified items are marked._

## Recommendation

DEFAULT (ship first): VAD endpointing, then one STT pass per finished segment, running entirely on the CPU. The GPU is left for the LLM and TTS.
- VAD: Silero VAD v6.2.3 ONNX model (silero_vad.onnx, sha256 1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3). Driven by a ~40-line numpy+onnxruntime wrapper with no torch. Frames are 512 samples at 16 kHz. Settings: threshold 0.5, neg_threshold 0.35, min_speech 250 ms, end-of-utterance silence 600 ms (configurable 500-900), pre-roll 300 ms, force-split at 15 s.
- STT: typhoon-ai/typhoon-asr-realtime (114M FastConformer-RNNT, CC-BY-4.0). Export it once to ONNX int8 with the sherpa-onnx NeMo export recipe and run it in sherpa-onnx 1.13.8 OfflineRecognizer (model_type="nemo_transducer", provider="cpu", num_threads=2-4).
- Why this default:
  - (a) Measured here: 140-240 ms per 4-5.5 s utterance on a contended 2.1 GHz Xeon with 1-2 threads (RTF ~0.04), 0 VRAM, ~340 MB RAM. Expect roughly 60-150 ms on the i7 (not measured).
  - (b) Good Thai CER: 9.9-10.0 on TVSpeech and 6.8-6.9 on GigaSpeech2 (paper plus model card).
  - (c) No NeMo or torch needed at runtime, and native Windows wheels exist for sherpa-onnx and onnxruntime.
  - (d) It keeps the serial Neuro-style loop simple: VAD end, then text, then the queue.
- Measured weakness: English code-switch words come out as Thai phonetics ("Minecraft"→"เหมือนคราบ", "iPhone"→"ไอโฟน") and the character name is often misheard ("ไพลิน"→"ไทลิน/ไทยลิน/ไทลิล"). Mitigate with a name-alias post-correction map and an LLM prompt that tolerates misspellings.

ACCURACY TIER (pluggable, optional): typhoon-ai/typhoon-whisper-turbo (MIT), converted by us to CTranslate2 and run with faster-whisper 1.2.1 on CUDA, compute_type=int8_float16, beam_size=1, language="th".
- No public CT2 conversion exists on HF (searched). The conversion command was verified here.
- Measured: correct "Minecraft", "iPhone" and "ไพลิน" where the Typhoon realtime model failed.
- Published CER: 6.85 TVSpeech / 4.79 GS2.
- Cost (estimated): ~1.2-2.5 GB VRAM and ~250-400 ms per 5 s utterance on the 4070.
- CPU fallback is too slow for conversation: measured 4.3-5.5 s per clip on 4 vCPU. Expect ~1.2-2.5 s on the i7 (estimate).
- Use it when streaming English-heavy games, or run both engines (fast pass plus accurate pass) under a latency budget.

NOT RECOMMENDED for the Windows runtime:
- NeMo-based paths: the typhoon-asr pip package, native NeMo, and the cache-aware typhoon-asr-streaming models. The typhoon-asr repo says Windows is not officially supported, and the streaming models need a pinned NeMo commit.
- If true partial transcripts are needed later, run typhoon-asr-streaming-115m or nemotron-0.6b as a WSL2/Docker sidecar. It speaks the repo's OpenAI-Realtime-style WebSocket protocol.
- Cloud fallback: Typhoon API /v1/audio/transcriptions with model typhoon-asr-realtime.

BARGE-IN: keep mic VAD running while TTS plays.
1. On a VAD start: duck TTS by 12 dB immediately.
2. Confirm when speech has lasted ≥ 500 ms AND a quick Typhoon realtime decode of the first ~0.8 s gives ≥ 3 non-space characters that are not a backchannel (อืม/ครับ/ค่ะ/555...).
3. On confirmation: stop playback, cancel the LLM and TTS streams, and truncate the assistant turn to the segments actually played.
4. If not confirmed within 2 s: restore the volume and resume (LiveKit uses the same defaults: min_duration 0.5 s, false_interruption_timeout 2.0 s).
- Require headphones by default. If speakers are used, add WebRTC AEC (livekit.rtc.AudioProcessingModule(echo_cancellation=True), 10 ms frames, with the TTS PCM as the far-end reference) plus a filter that drops transcripts matching the TTS text.
- Priority policy is configurable, following Neuro: the streamer's voice interrupts at "high"; chat never interrupts speech unless "critical".

## Alternatives

### Typhoon ASR Realtime 114M → ONNX int8 → sherpa-onnx OfflineRecognizer (CPU)
- **Pros:** Verified working. ~0.2 s per utterance even on a weak CPU, 0 VRAM, ~340 MB RAM. Native Windows wheels, no torch or NeMo at runtime. CER 9.9 on TVSpeech, and no Whisper-style hallucination loops.
- **Cons:** Segment-level only, with no true partials. English words come out in Thai script. Proper names get misheard. Needs a one-time NeMo export (Linux/WSL/CI) or the unlicensed third-party ONNX.
- **When:** DEFAULT for all voice input and for barge-in confirmation.

### typhoon-whisper-turbo → CTranslate2 (self-converted) → faster-whisper 1.2.1 on CUDA int8_float16
- **Pros:** Better CER (6.85 TVSpeech, 4.79 GS2). Keeps Latin script for English terms and got the name right in testing. MIT license. The conversion command is verified.
- **Cons:** ~1.2-2.5 GB VRAM competing with the LLM. Estimated 250-400 ms on GPU and 1-2.5 s on CPU. Windows cuBLAS DLL setup. Whisper hallucination risk. Prompt and hotword biasing is unreliable.
- **When:** English-heavy content (game names), or as the accurate second pass in TwoPassRecognizer when VRAM allows.

### typhoon-whisper-large-v3 → CT2
- **Pros:** Best published Thai CER (6.32 TVSpeech, 4.69 GS2).
- **Cons:** 32-layer decoder: several times slower decode than turbo. VRAM ~2.9 GB int8 / 4.5 GB fp16.
- **When:** Offline transcription or VOD captioning, not the live loop.

### PyThaiASR 2.1.0 (Typhoon RT fp32 ONNX, pure onnxruntime + numpy)
- **Pros:** `pip install pythaiasr`, no NeMo or torch, auto-downloads the model, includes a mic streaming helper.
- **Cons:** fp32 435 MB and ~2-3x slower than int8 sherpa (measured). Brand-new packaging (Sep 2026). The ONNX files are named '*-quran-ar' (export-script artifact). The sliding-window streaming is unvalidated.
- **When:** Quick prototype, or a fallback if the sherpa-onnx NeMo path breaks.

### typhoon-asr-streaming-115m / typhoon-asr-streaming-nemotron-0.6b (NeMo cache-aware) as a WSL2/Docker sidecar over the OpenAI-Realtime-style WS
- **Pros:** True partial transcripts (first token ≈ 0.49 s at 480 ms look-ahead). Phrase boosting and n-gram fusion steer names. The 0.6B gives 14.1% CER streaming with better English code-switch.
- **Cons:** Needs NeMo pinned to 907edfd, fp32, Linux/WSL2, and GPU for comfortable speed. One stream per worker. The 0.6B is OpenMDW-licensed, and its n-gram is 2.4 GB. Streaming CER is worse than offline Typhoon RT (19.4% vs 9.9% for the 115M). The sherpa export does not work yet.
- **When:** Later phase, if live captions or early-intent partials matter more than accuracy.

### Typhoon hosted API (api.opentyphoon.ai/v1, model typhoon-asr-realtime)
- **Pros:** Zero local compute, OpenAI SDK compatible, free key, 100 req/min.
- **Cons:** Network latency and dependency. Audio leaves the machine. Rate limits. Same model accuracy as local.
- **When:** Fallback when local models fail to load, or for low-spec dev machines.

### typhoon-asr-qwen-0.6b-ctx / 1.7b-ctx (Qwen3-ASR LoRA, Apache-2.0, transformers ≥5.13)
- **Pros:** Contextual biasing with unlimited list size: pass the streamer, viewer and game names in the system prompt. CER 5.06-4.86 GS2 / 7.28-6.87 TVSpeech. Low false-alarm rate.
- **Cons:** 0.8-2.07B-parameter autoregressive decoder. Latency and VRAM on the 4070 unverified. Needs torch plus transformers in the voice process. No CT2 or ONNX path verified.
- **When:** If name and entity recognition becomes the main complaint. Evaluate on the target GPU first.

### Stock openai whisper-large-v3-turbo (mobiuslabsgmbh CT2)
- **Pros:** No conversion needed (faster-whisper alias 'turbo').
- **Cons:** Noticeably worse Thai in testing (name, 'Minecraft', 'แชท' errors).
- **When:** Not recommended; Typhoon's fine-tune is a drop-in replacement.

### Other Thai Whisper fine-tunes: Pathumma-whisper-th-large-v3, biodatlab whisper-th-* (CT2 conversions exist on HF)
- **Pros:** Ready-made CT2 repos. Pathumma is strong on FLEURS (6.29).
- **Cons:** Worse on noisy, in-the-wild TVSpeech (Pathumma 10.36, biodatlab 13.8-19.0). Biodatlab large-v3 shows repetition loops.
- **When:** Only as a comparison baseline.

### sherpa-onnx multilingual streaming zipformer (ar_en_id_ja_ru_th_vi_zh-2025-02-10)
- **Pros:** True streaming in sherpa, with native endpoint rules.
- **Cons:** Poor Thai in testing (garbage tokens, Thai digits).
- **When:** Not recommended.

### VAD alternatives: sherpa-onnx VoiceActivityDetector (Silero or TEN VAD), silero-vad pip VADIterator, webrtcvad
- **Pros:** sherpa VAD needs no extra dependency. The silero pip package is the official reference.
- **Cons:** sherpa VAD clipped an onset in testing and exposes no pre-roll control. The silero pip package drags in torch. webrtcvad is less accurate on noisy input (general knowledge, not tested).
- **When:** Use the custom numpy Silero endpointer by default; the others for comparison.

## Verified facts

- ✅ typhoon-ai/typhoon-asr-realtime: 114M FastConformer-Transducer, Thai, CC-BY-4.0, ~10k h training data, only file is typhoon-asr-realtime.nemo. Card says it is designed for streaming and runs efficiently on CPU. HF sha 2c58a30ba9a3bf92d095a5df91bec6996f04c3a1; scb10x/typhoon-asr-realtime redirects (HTTP 307) to typhoon-ai/.  
  Source: https://huggingface.co/typhoon-ai/typhoon-asr-realtime (README raw + HF API)
- ✅ Paper arXiv 2601.13044 (19 Jan 2026) Table 6 CER, TVSpeech/GigaSpeech2/FLEURS: Typhoon ASR Realtime 9.99/6.81/13.87; Typhoon Whisper Large-v3 6.32/4.69/9.98; Typhoon Whisper Turbo 6.85/4.79/10.52; Pathumma-Whisper Large-v3 10.36/5.84/6.29; Biodatlab Distil-Whisper Large 13.82/8.24/6.77; Biodatlab Whisper Large 18.96/13.22/16.50; Gemini 3 Pro 10.95/12.50/11.35. The paper states the realtime model forces phonetic mapping to Thai characters instead of Latin script under Thai-English code-switching.  
  Source: https://arxiv.org/html/2601.13044
- ✅ Model-card CER table (gigaspeech2 bench / TVSpeech): typhoon-asr-realtime 6.89/9.92; typhoon-asr-realtime-nemo-ctc 8.54/13.46; typhoon-whisper-medium 4.81/7.66; typhoon-whisper-large-v3 4.69/6.32; typhoon-asr-qwen-1.7b-ctx 4.86/6.87; typhoon-asr-qwen-0.6b-ctx 5.06/7.28; Qwen3-ASR-1.7B 6.09/10.63; Pathumma-whisper-th-large-v3 5.84/10.36; biodatlab/whisper-th-large-v3-combined 15.78 (repetition loops)/14.91.  
  Source: https://huggingface.co/typhoon-ai/typhoon-asr-realtime-nemo-ctc and https://huggingface.co/typhoon-ai/typhoon-asr-qwen-0.6b-ctx
- ✅ typhoon-ai/typhoon-whisper-turbo: MIT, fine-tune of openai/whisper-large-v3-turbo on ~11k h of Thai, transformers format only, sha 3c03fa84c26f172944422ceb8a4e88a2dbc08b10. The card's example uses model_id 'scb10x/typhoon-whisper-turbo', which returns HTTP 401 (stale id).  
  Source: https://huggingface.co/typhoon-ai/typhoon-whisper-turbo + HF API
- ✅ No CTranslate2 conversion of typhoon-whisper-turbo or typhoon-whisper-large-v3 exists on HF. Search hits were MLX, CoreML and GGML only (chayapats/typhoon-whisper-turbo-mlx, korakotlee/typhoon-whisper-turbo-coreml, JoaoZaokk/typhoon-whisper-turbo-ggml). CT2 conversions do exist for other Thai fine-tunes: Vinxscribe/biodatlab-whisper-th-large-v3-faster, Vinxscribe/biodatlab-whisper-th-medium-faster, s2p2/Pathumma-whisper-th-large-v3-ct2, CodeHardThailand/whisper-th-large-v3-combined-ct2, pariya47/distill-whisper-th-large-v3-ct2.  
  Source: https://huggingface.co/api/models?search=typhoon-whisper and ?search=whisper-th
- ✅ Conversion verified in this container (transformers 5.17.0, torch 2.14.0+cpu, ctranslate2 4.8.2): `ct2-transformers-converter --model typhoon-ai/typhoon-whisper-turbo --output_dir models/typhoon-whisper-turbo-ct2 --copy_files tokenizer.json preprocessor_config.json --quantization float16` exits 0. Output model.bin is 1.62 GB and loads in faster-whisper 1.2.1.  
  Source: local run; command from https://github.com/SYSTRAN/faster-whisper README 'Model conversion'
- ✅ faster-whisper latest is 1.2.1 (2025-10-31). Depends on ctranslate2>=4.0,<5, onnxruntime>=1.14,<2, av>=11. The 'large-v3-turbo' and 'turbo' aliases resolve to mobiuslabsgmbh/faster-whisper-large-v3-turbo. GPU needs cuBLAS for CUDA 12 and cuDNN 9. It bundles assets/silero_vad_v6.onnx. VadOptions defaults: threshold 0.5, min_silence_duration_ms 2000, speech_pad_ms 400.  
  Source: https://pypi.org/pypi/faster-whisper/json ; https://raw.githubusercontent.com/SYSTRAN/faster-whisper/master/README.md ; installed package inspection
- ✅ ctranslate2 4.8.2 released 2026-08-31 with win_amd64 wheels for cp39-cp314. The Windows CI builds against CUDA 12.8.1 and cuDNN 9.10.2. The Windows wheel bundles ctranslate2.dll, cudnn64_9.dll and libiomp5md.dll but not cuBLAS. Since 4.6.3, conv1d has a pure CUDA implementation and cuDNN is optional. 4.6.2 disabled INT8 on sm120 (Blackwell), which does not affect the 4070 (sm89).  
  Source: https://raw.githubusercontent.com/OpenNMT/CTranslate2/master/CHANGELOG.md ; python/tools/prepare_build_environment_windows.sh ; unpacked ctranslate2-4.8.2-cp312-win_amd64.whl
- ✅ Windows: 'Library cublas64_12.dll is not found' occurs at first inference. The reported fix (5 Sep 2026) is pip nvidia-cublas-cu12 (+ nvidia-cuda-runtime-cu12, nvidia-cudnn-cu12) and prepending site-packages/nvidia/*/bin to os.environ['PATH'] before building WhisperModel; os.add_dll_directory alone was reported insufficient. The nvidia-cublas-cu12 12.9.2.10 and nvidia-cudnn-cu12 9.26.0.51 win_amd64 wheels exist on PyPI.  
  Source: https://github.com/jaredrhod/backtalk/issues/28 ; https://github.com/SYSTRAN/faster-whisper/issues/1276 ; PyPI JSON
- ✅ Published faster-whisper large-v3-turbo figures: fp16, beam 5, 13 min of audio → 19.155 s and 2537 MB peak VRAM (large-v3: 52.0 s, 4521 MB). README large-v2 on RTX 3070 Ti: fp16 4525 MB, int8 2926 MB.  
  Source: https://github.com/SYSTRAN/faster-whisper/issues/1030 ; faster-whisper README
- ✅ silero-vad on PyPI is 6.2.3 (2026-09-23). It hard-requires torch>=1.12 (utils_vad.py does `import torch` at top level, even in ONNX mode). load_silero_vad(onnx=False, opset_version=16, sequence=False). VADIterator(model, threshold=0.5, sampling_rate=16000, min_silence_duration_ms=100, speech_pad_ms=30) exits speech at prob < threshold-0.15. get_speech_timestamps defaults: threshold 0.5, min_speech_duration_ms 250, min_silence_duration_ms 100, speech_pad_ms 30. The ONNX takes input float32[B,64+512] (64-sample context + 512-sample frame at 16 kHz), state float32[2,B,128] and sr int64. 6.2.2 added silero_vad_16k_sequence.onnx for batched offline use.  
  Source: https://pypi.org/pypi/silero-vad/json ; unpacked silero_vad-6.2.3 wheel (model.py, utils_vad.py, sequence_vad.py) ; https://github.com/snakers4/silero-vad/releases
- ✅ Silero release notes: v6.0 reports 16% fewer errors on noisy real-life data, with known issues on voice-like music instruments and very high-pitched voices. v6.2 improves unusual, child and cartoon voices and muted speech. The raw master silero_vad.onnx has the same sha256 as the 6.2.3 wheel copy (1a153a22…8788e3).  
  Source: https://github.com/snakers4/silero-vad/releases ; sha256 comparison
- ✅ The torch-free numpy Silero wrapper written here costs 0.156 ms of CPU per 32 ms frame, and the endpointer correctly segmented 3 utterances from a synthetic stream.  
  Source: local test (scratchpad/research/stt/silero_np.py, test_vad.py)
- ✅ Our own ONNX export of typhoon-asr-realtime works in sherpa-onnx 1.13.8 OfflineRecognizer.from_transducer(model_type='nemo_transducer'). The export (NeMo 3.0.0, adapted from sherpa-onnx scripts/nemo/fast-conformer-hybrid-transducer-ctc/export-onnx-transducer-non-streaming.py) produces encoder.int8.onnx 131 MB, decoder.int8.onnx 4.6 MB, joiner.int8.onnx 2.1 MB and tokens.txt (2049 lines). Metadata: normalize_type=per_feature, vocab 2048, pred_hidden 640. Its tokens.txt and int8 outputs are identical to the third-party edtzforai/typhoon-asr-int8-onnx repo, which has no README or license file.  
  Source: local export + test (export_typhoon_offline.py, exportrt/)
- ✅ Measured on the 3 synthetic edge-tts Thai clips (3.9/4.8/5.5 s): Typhoon RT int8 via sherpa, CPU, 1-2 threads: 175-237 ms median, RTF ≈0.04, with a concurrent llama-server on the same 4 vCPUs. 4 threads under that contention: 456-503 ms. Peak RSS ~340 MB. Output example: 'เมื่อวานผมไปเดินห้างสยามพารากรมาซื้อไอโฟนเครื่องใหม่ราคาสามหมื่นห้าพันบาท'.  
  Source: local benchmark (Xeon 2.1 GHz, 4 vCPU, shared)
- ✅ Measured typhoon-whisper-turbo CT2 int8 on CPU (4 threads): 4.3-5.5 s per clip, peak RSS ~2 GB. Transcripts contained 'Minecraft', 'iPhone' and 'ไพลิน' correctly; numbers came out as Thai words. Stock large-v3-turbo on the same clips: 4.0-4.5 s, and misrecognised 'ไภลิน์', 'มันครับ' (for Minecraft), 'ชัต' and 'พรากร', and wrote '35,000'.  
  Source: local benchmark
- ✅ Passing initial_prompt or hotwords='ไพลิน Minecraft สยามพารากอน' to typhoon-whisper-turbo fixed 'พารากอน' but broke 'Minecraft'→'เหมือนครับ'. The Typhoon cards state that typhoon-whisper-large-v3's prompt pathway collapses on bias lists (echo loops, CER 200+).  
  Source: local test ; https://huggingface.co/typhoon-ai/typhoon-asr-realtime-nemo-ctc
- ✅ typhoon-asr (PyPI 0.1.1, 2025-11-28) depends on nemo-toolkit[asr]>=1.21.0. Its transcribe() calls ASRModel.from_pretrained on every call and writes processed_<stem>.wav to the CWD. The GitHub README lists requirements 'Linux / Mac (Windows is not officially supported at the moment)' and Python 3.10.  
  Source: unpacked typhoon_asr-0.1.1 wheel ; https://raw.githubusercontent.com/scb-10x/typhoon-asr/main/README.md
- ✅ nemo-toolkit 3.0.0 (2026-08-07) ships as a py3-none-any wheel. PyPI says requires_python>=3.10; the README says Python 3.12+ and PyTorch 2.7+, and makes no Windows statement. Installs of nemo_toolkit[asr] on Windows have been reported to hit native-build failures (e.g. texterrors).  
  Source: https://pypi.org/pypi/nemo-toolkit/json ; https://raw.githubusercontent.com/NVIDIA/NeMo/main/README.md ; https://github.com/NVIDIA-NeMo/NeMo/discussions/15421
- ✅ typhoon-ai/typhoon-asr-streaming-115m (CC-BY-4.0, cache-aware, att_context_size [[70,13],[70,6],[70,1],[70,0]]): streaming CER 19.4% on TVSpeech at 1040 ms. typhoon-asr-streaming-nemotron-0.6b (OpenMDW-1.1, requires target_lang='th-TH'): 14.1% TVSpeech and 9.3% GigaSpeech2 at 1040 ms. The full-context realtime model forced to stream scores 62.8% TVSpeech. Streaming needs NeMo source commit 907edfd and compute_dtype float32. First token arrives after ~1.06 s at 1040 ms and ~0.49 s at 480 ms on an H100.  
  Source: https://huggingface.co/typhoon-ai/typhoon-asr-streaming-115m ; https://raw.githubusercontent.com/warit-s/typhoon-asr-streaming/main/README.md ; docs/HOSTING.md
- ✅ NeMo 3.0.0 cache-aware streaming of the 115m model works on CPU under Linux. Measured total compute 324-547 ms per 3.9-5.5 s clip, with growing partials (e.g. 'สวัสดี'→'สวัสดีครับ'→…).  
  Source: local test (test_nemo_stream.py)
- ✅ Exporting typhoon-asr-streaming-115m to a sherpa-onnx streaming transducer (cache_support=True, [70,6]) completes, but sherpa OnlineRecognizer returns EMPTY text for both fp32 and int8. The likely cause is per_feature normalization, which the streaming path cannot reproduce (not confirmed).  
  Source: local test (export_typhoon_streaming.py, test_sherpa_stream.py)
- ✅ The typhoon-asr-streaming server (FastAPI) exposes WS /realtime and /v1/realtime with subprotocol 'realtime'. Client events: session.update, input_audio_buffer.append {audio: base64 PCM16}, input_audio_buffer.commit, input_audio_buffer.clear. Server events: session.created/updated, conversation.item.created, conversation.item.input_audio_transcription.delta {delta, transcript}, conversation.item.input_audio_transcription.completed {transcript}, typhoon.metrics, error. Default input rate is 24000 Hz, settings live under session.typhoon {latency_ms, steering, terms, alpha}, and the server allows one active stream per worker.  
  Source: https://raw.githubusercontent.com/warit-s/typhoon-asr-streaming/main/server/realtime_asr_server.py
- ✅ PyThaiASR 2.1.0 (2026-09-14) needs only numpy, soundfile, onnxruntime>=1.16 and requests. Its default is Typhoon RT fp32 ONNX (wannaphong/typhoon-asr-realtime-onnx, 435 MB encoder) with its own numpy features and a sliding-window 'RealtimeStreamASR' (step 0.48 s, left context 0.64 s, right context 0.32 s). Measured on CPU: 407-707 ms per clip (RTF 0.09-0.13).  
  Source: https://pypi.org/pypi/pythaiasr/json ; unpacked wheel ; local test
- ✅ sherpa-onnx 1.13.8 (2026-09-10) has CPU win_amd64 wheels (sherpa-onnx-core bundles its own onnxruntime). CUDA wheels sherpa_onnx-1.13.8+cuda12.cudnn9-cp311..cp314-win_amd64 are listed on the k2-fsa CUDA index. It also includes VoiceActivityDetector (Silero or TEN VAD).  
  Source: https://pypi.org/pypi/sherpa-onnx-core/1.13.8/json ; https://k2-fsa.github.io/sherpa/onnx/cuda.html
- ✅ The sherpa-onnx VoiceActivityDetector (Silero v6, min_silence 0.6, min_speech 0.25) dropped the first word ('ไพลิน') of one utterance in testing. The custom endpointer with a 300 ms pre-roll kept it.  
  Source: local test (test_sherpa_vad.py vs test_pipeline.py)
- ✅ The multilingual sherpa zipformer csukuangfj/sherpa-onnx-streaming-zipformer-ar_en_id_ja_ru_th_vi_zh-2025-02-10 was poor on Thai: clip 2 gave 'hayn uỗnคอมเมนต์…' and clip 3 used Thai digits '๓๕๐๐'.  
  Source: local test
- ✅ nvidia/nemotron-3.5-asr-streaming-0.6b lists Thai (th-TH) only as 'adaptation-ready' (fine-tuning required), so the stock model is not usable for Thai. sherpa-onnx has an export script for it.  
  Source: https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b ; https://github.com/k2-fsa/sherpa-onnx/tree/master/scripts/nemo/nemotron-3.5-asr-streaming-0.6b
- ✅ Smart Turn v3.2 (Pipecat semantic end-of-turn, 8 MB ONNX) supports 23 languages and not Thai.  
  Source: https://raw.githubusercontent.com/pipecat-ai/smart-turn/main/README.md ; https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/
- ✅ Framework defaults for comparison. LiveKit Silero: min_speech 0.05 s, min_silence 0.55 s, prefix_padding 0.5 s, activation 0.5 (deactivation activation-0.15). LiveKit turn handling: interruption min_duration 0.5 s, min_words 0, resume_false_interruption True, false_interruption_timeout 2.0 s; endpointing min_delay 0.5 s, max_delay 3.0 s. Pipecat VADParams: confidence 0.7, start_secs 0.2, stop_secs 0.2, min_volume 0.6. Open-LLM-VTuber silero: prob_threshold 0.4, db_threshold 60, required_hits 3, required_misses 24.  
  Source: livekit/agents livekit-plugins-silero/vad.py and voice/turn.py ; pipecat src/pipecat/audio/vad/vad_analyzer.py ; Open-LLM-VTuber config_templates/conf.default.yaml
- ✅ The Typhoon hosted ASR API is OpenAI-compatible: base https://api.opentyphoon.ai/v1, client.audio.transcriptions.create(model='typhoon-asr-realtime', file=...), rate limit 100 req/min, free key.  
  Source: https://docs.opentyphoon.ai/en/asr/
- ✅ livekit 1.1.20 (2026-09-23) has a win_amd64 wheel. rtc.AudioProcessingModule(echo_cancellation, noise_suppression, high_pass_filter, auto_gain_control) provides process_stream (near end), process_reverse_stream (far end) and set_stream_delay_ms; frames must be exactly 10 ms. aec-audio-processing 1.0.1 and pyaec 1.0.1 also ship Windows wheels.  
  Source: https://raw.githubusercontent.com/livekit/python-sdks/main/livekit-rtc/livekit/rtc/apm.py ; PyPI JSON
- ✅ onnxruntime 1.30.0 and onnxruntime-gpu 1.30.0 (2026-09-10) require Python>=3.11. The onnxruntime-gpu 'cuda' extra pulls CUDA 13 wheels (nvidia-cuda-runtime~=13.0, nvidia-cudnn-cu13).  
  Source: https://pypi.org/pypi/onnxruntime-gpu/json
- ⚠️ unverified — RTX 4070 GPU latency and VRAM for typhoon-whisper-turbo int8_float16 (~250-400 ms per 5 s utterance, ~1.2-1.6 GB) and i7-14700KF CPU latency for Typhoon RT (~60-150 ms) are ESTIMATES. No GPU or target CPU was available.  
  Source: extrapolation from local CPU runs + published faster-whisper numbers
- ⚠️ unverified — Native-Windows install and runtime of nemo_toolkit[asr] 3.0.0 was not tested.  
  Source: n/a

## Install (Windows)

```
Python 3.11 or 3.12 x64. onnxruntime 1.30 requires ≥3.11; NeMo is not needed on Windows.

  py -3.12 -m venv .venv && .venv\Scripts\activate && python -m pip install -U pip

1) Default CPU path (VAD + Typhoon RT):
  pip install "sherpa-onnx==1.13.8" "onnxruntime==1.30.0" "numpy>=2" "soxr==1.1.0" "sounddevice==0.5.6" "soundfile==0.14.0" "huggingface_hub"
Models go under models\ and are fetched on first run with a sha256 check.
- silero_vad.onnx: https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx (sha256 1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3; identical to the silero-vad 6.2.3 wheel).
- Typhoon RT ONNX: encoder.int8.onnx, decoder.int8.onnx, joiner.int8.onnx, tokens.txt. Produce these with our export (step 3) and host them, e.g. on the project's own HF repo with CC-BY-4.0 attribution. As a stopgap, the third-party edtzforai/typhoon-asr-int8-onnx (rev ac0644a741e432bb17772e3b17621de6ea6a599d) gives byte-identical tokens and identical outputs but has no license file.

2) Optional GPU accuracy tier (faster-whisper):
  pip install "faster-whisper==1.2.1" "ctranslate2==4.8.2" "nvidia-cublas-cu12==12.9.2.10" "nvidia-cuda-runtime-cu12==12.9.*" "nvidia-cudnn-cu12==9.*"
Before creating WhisperModel, prepend PATH with site-packages\nvidia\cublas\bin, nvidia\cuda_runtime\bin and nvidia\cudnn\bin (snippet in api_notes). os.add_dll_directory is not enough. The alternative is to install the CUDA 12.x toolkit plus cuDNN 9 system-wide. Keep this in the same process only if every other GPU library there also uses CUDA 12. onnxruntime-gpu[cuda] 1.30 pulls CUDA 13, so either keep ORT on CPU or use the sherpa cuda12 wheel.

3) One-time model preparation. Linux, WSL2 or a CI job; CPU is fine; not needed on the streaming PC.
 a) Whisper CT2 conversion (verified here on Linux):
  pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install "transformers>=5" "ctranslate2==4.8.2"
  ct2-transformers-converter --model typhoon-ai/typhoon-whisper-turbo --revision 3c03fa84c26f172944422ceb8a4e88a2dbc08b10 --output_dir models/typhoon-whisper-turbo-ct2 --copy_files tokenizer.json preprocessor_config.json --quantization float16
  Output is ~1.62 GB. Pick int8_float16 (CUDA) or int8 (CPU) at load time via compute_type.
 b) Typhoon RT ONNX export (verified here with NeMo 3.0.0):
  pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install "nemo_toolkit[asr]==3.0.0" onnx onnxruntime huggingface_hub
  python export_typhoon_offline.py   # full script: /tmp/claude-0/-home-user-AI-Vtube/02efebb2-c032-5a2c-81bb-f71e8a5461a3/scratchpad/research/stt/export_typhoon_offline.py
  The script calls restore_from(hf_hub_download('typhoon-ai/typhoon-asr-realtime','typhoon-asr-realtime.nemo')), then encoder/decoder/joint.export(), writes tokens.txt as '<piece> <id>' plus '<blk> 2048', adds metadata {vocab_size, normalize_type, pred_rnn_layers, pred_hidden, subsampling_factor:8, model_type}, and runs onnxruntime quantize_dynamic(QUInt8).

4) Optional sherpa on GPU (not needed for 114M): pip install "sherpa-onnx==1.13.8+cuda12.cudnn9" -f https://k2-fsa.github.io/sherpa/onnx/cuda.html (wheel names verified on the index; install untested).

5) Optional AEC when the streamer uses speakers: pip install "livekit==1.1.20" (rtc.AudioProcessingModule).

6) Optional true-streaming sidecar (WSL2 Ubuntu + CUDA, not native Windows):
  git clone https://github.com/warit-s/typhoon-asr-streaming
  git clone https://github.com/NVIDIA/NeMo && cd NeMo && git checkout 907edfd && pip install -e .
  pip install -r requirements.txt
  uvicorn server.realtime_asr_server:app --host 127.0.0.1 --port 8000
  The Windows host streams PCM16 over ws://127.0.0.1:8000/v1/realtime (WSL2 localhost forwarding).

CI (GitHub Actions ubuntu + windows): install only numpy, onnxruntime and sherpa-onnx; unit-test with FakeRecognizer and a FakeVAD fed synthetic probability sequences. An optional nightly job caches the ~140 MB int8 Typhoon ONNX plus silero and runs a golden-clip test. The TTS-generated Thai clips are ~30 KB each and can be committed.
```

## API notes

All snippets below were run in this container except the Windows DLL block and the WS client. Scratch copies are in /tmp/claude-0/-home-user-AI-Vtube/02efebb2-c032-5a2c-81bb-f71e8a5461a3/scratchpad/research/stt/ (silero_np.py, stt_iface.py, test_iface.py, export_typhoon_offline.py, export_typhoon_streaming.py, test_nemo_stream.py).

(A) Torch-free Silero v6 wrapper and endpointer (silero_np.py, tested; 0.16 ms per 32 ms frame):
```python
import numpy as np, onnxruntime as ort
class SileroVAD:
    FRAME=512; CTX=64; SR=16000
    def __init__(self, path):
        so=ort.SessionOptions(); so.inter_op_num_threads=1; so.intra_op_num_threads=1
        self.sess=ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"]); self.reset()
    def reset(self):
        self.state=np.zeros((2,1,128),np.float32); self.ctx=np.zeros((1,self.CTX),np.float32)
    def __call__(self, frame):  # float32[512], 16 kHz, [-1,1]
        x=np.concatenate([self.ctx, frame.reshape(1,-1).astype(np.float32)],axis=1)
        out,self.state=self.sess.run(None,{"input":x,"state":self.state,"sr":np.array(16000,np.int64)})
        self.ctx=x[:,-self.CTX:]; return float(out[0,0])
class Endpointer:  # VADIterator semantics + min_speech + pre-roll + max length
    def __init__(self, vad, threshold=0.5, neg_threshold=None, min_speech_ms=250, min_silence_ms=600, pre_roll_ms=300, max_utt_s=15.0):
        self.vad,self.th=vad,threshold; self.neg=threshold-0.15 if neg_threshold is None else neg_threshold
        f=32.0; self.min_speech,self.min_sil=int(min_speech_ms/f),int(min_silence_ms/f)
        self.pre_roll,self.max_frames=int(pre_roll_ms/f),int(max_utt_s*1000/f); self.reset()
    def reset(self): self.vad.reset(); self.state="idle"; self.buf=[]; self.speech_run=0; self.sil_run=0
    def push(self, frame):
        p=self.vad(frame); ev=[]; self.buf.append(frame)
        if self.state=="idle":
            self.speech_run=self.speech_run+1 if p>=self.th else 0
            if self.speech_run>=self.min_speech:
                self.state="speech"; self.sil_run=0; ev.append(("start",p)); self.buf=self.buf[-(self.speech_run+self.pre_roll):]
            else: self.buf=self.buf[-(self.pre_roll+self.min_speech):]
        else:
            self.sil_run=self.sil_run+1 if p<self.neg else 0
            if self.sil_run>=self.min_sil or len(self.buf)>=self.max_frames:
                ev.append(("end",np.concatenate(self.buf))); self.state="idle"; self.buf=[]; self.speech_run=0
        return ev
```
Reference official API (pulls in torch):
```python
from silero_vad import load_silero_vad, VADIterator, get_speech_timestamps
m=load_silero_vad(onnx=True)
it=VADIterator(m, threshold=0.5, sampling_rate=16000, min_silence_duration_ms=600, speech_pad_ms=30)
# call it(chunk_512_float32_tensor) -> {'start': n} / {'end': n} / None
```
Tuning:
- threshold: 0.5; 0.6-0.7 in noisy rooms or during TTS playback.
- End of utterance (EoU): 500-800 ms. The VADIterator default of 100 ms is for splitting long audio, not for conversation.
- Always add a 250-400 ms pre-roll.

(B) Default STT: Typhoon RT via sherpa-onnx (tested):
```python
import sherpa_onnx, numpy as np
rec = sherpa_onnx.OfflineRecognizer.from_transducer(
    encoder="models/typhoon-rt/encoder.int8.onnx", decoder="models/typhoon-rt/decoder.int8.onnx",
    joiner="models/typhoon-rt/joiner.int8.onnx", tokens="models/typhoon-rt/tokens.txt",
    num_threads=2, sample_rate=16000, feature_dim=80, decoding_method="greedy_search",
    model_type="nemo_transducer", provider="cpu")
s = rec.create_stream(); s.accept_waveform(16000, utterance_f32); rec.decode_stream(s); text = s.result.text.strip()
```
- sherpa also takes hotwords_file/hotwords_score and lm/lm_scale; these were not tested for the NeMo transducer.
- Create one recognizer per process and reuse it. decode_stream releases the GIL; run it on a dedicated single-worker ThreadPoolExecutor.

(C) Accuracy tier: faster-whisper plus the Windows DLL fix:
```python
import os, sys, importlib.util
if sys.platform == "win32":
    for pkg in ("nvidia.cublas", "nvidia.cuda_runtime", "nvidia.cudnn"):
        spec = importlib.util.find_spec(pkg)
        if spec and spec.submodule_search_locations:
            b = os.path.join(list(spec.submodule_search_locations)[0], "bin")
            if os.path.isdir(b): os.environ["PATH"] = b + os.pathsep + os.environ["PATH"]; os.add_dll_directory(b)
from faster_whisper import WhisperModel
m = WhisperModel("models/typhoon-whisper-turbo-ct2", device="cuda", compute_type="int8_float16")  # CPU: device="cpu", compute_type="int8", cpu_threads=8
segs, info = m.transcribe(utt_f32_16k, language="th", task="transcribe", beam_size=1, best_of=1, temperature=0.0,
                          condition_on_previous_text=False, without_timestamps=True, vad_filter=False,
                          no_speech_threshold=0.6, log_prob_threshold=-1.0)
text = "".join(s.text for s in segs).strip()   # segs is a lazy generator: iterate to run
```
- Do not pass initial_prompt or hotwords by default: the Typhoon fine-tunes react unpredictably (measured).
- Pad or drop segments under 0.3 s to avoid hallucinations.

(D) Pluggable interface (stt_iface.py, tested):
```python
@dataclass(frozen=True)
class Transcript: text:str; is_final:bool; audio_s:float; latency_ms:float; engine:str
class SpeechRecognizer(Protocol):
    name:str
    def warmup(self)->None: ...
    def transcribe(self, pcm16k: np.ndarray)->Transcript: ...           # segment-level, required
class StreamingRecognizer(Protocol):                                     # optional capability
    def open(self)->"StreamingSession": ...
class StreamingSession(Protocol):
    def feed(self, pcm16k: np.ndarray)->Transcript|None: ...             # partials
    def finish(self)->Transcript: ...
class VoiceActivityDetector(Protocol):
    frame_samples:int                                                    # 512 @16k
    def reset(self)->None: ...
    def prob(self, frame: np.ndarray)->float: ...
```
- Implementations: SherpaTyphoonRT (default), FasterWhisperRecognizer, TyphoonApiRecognizer (OpenAI SDK, base_url https://api.opentyphoon.ai/v1, model 'typhoon-asr-realtime', send WAV bytes), RealtimeWsRecognizer (sidecar), FakeRecognizer(scripted texts, delay_s) for CI, and TwoPassRecognizer(fast, accurate, budget_ms): return the accurate result if it arrives within budget, else the fast one.
- Post-processing chain: strip, then an alias map (e.g. {'ไทลิน','ไทยลิน','ไทลิล','ไภลิน'}→'ไพลิน'), then drop empty text, then drop echoes of the assistant (difflib ratio > 0.6 against the last ~10 s of TTS text).
- Event-bus types: SpeechStarted(t), SpeechEnded(t, audio_s), UserTranscript(text, engine, latency_ms, is_final), BargeIn(confirmed: bool, text).

(E) Audio capture on Windows:
- sounddevice InputStream on the WASAPI device at its native 48 kHz, mono float32, blocksize=480 (10 ms). The callback only does queue.put_nowait(indata.copy()).
- A worker thread resamples with soxr.ResampleStream(48000, 16000, 1, dtype='float32').resample_chunk(x), re-blocks into 512-sample frames, feeds Endpointer.push, and on an 'end' event submits the segment to the STT executor.

(F) Barge-in, while state == SPEAKING:
- VAD runs with threshold 0.6 (configurable).
- On a 'start' event: tts.duck(-12 dB) and record the candidate time t0.
- Each frame: if speech_frames ≥ 16 (~500 ms), run the fast recognizer on the last 0.8 s (Typhoon RT: ~30-60 ms CPU). Confirm if the text without spaces has ≥ 3 characters and is not in BACKCHANNELS {'อืม','อือ','อ๋อ','เออ','ครับ','ค่ะ','คะ','จ้ะ','จ้า','ฮะ','โอเค','เหรอ','หรอ','555','ฮ่า','ฮ่าๆ'}.
- On confirm: tts.stop(fade_ms=80), then llm.cancel(), then mark the assistant turn truncated at the last fully played TTS segment. Keep capturing; the full user utterance is finalized by the normal EoU path.
- If no confirmation within 2.0 s, or VAD ends first: tts.unduck(), and continue.
- Tested: the confirm helper returns 'ไทยลิน' for real speech and None for a 0.25 s blip.
- AEC is needed only with speakers:
```python
from livekit import rtc
apm = rtc.AudioProcessingModule(echo_cancellation=True, noise_suppression=True, high_pass_filter=True)
# every 10 ms: apm.process_reverse_stream(rtc.AudioFrame(tts_pcm16_10ms, 48000, 1, 480)); then apm.process_stream(mic_frame)
# apm.set_stream_delay_ms(render_latency + capture_latency)
```

(G) Optional streaming sidecar protocol (OpenAI Realtime-style, from typhoon-asr-streaming server source). Connect ws://127.0.0.1:8000/v1/realtime?model=typhoon-asr-streaming-115m with subprotocol 'realtime'.
```json
{"type":"session.update","session":{"type":"transcription","audio":{"input":{"format":{"type":"audio/pcm","rate":16000},"transcription":{"model":"typhoon-asr-streaming-115m","language":"th"}}},"typhoon":{"latency_ms":480,"steering":true,"terms":["ไพลิน","Minecraft"],"alpha":1.0}}}
{"type":"input_audio_buffer.append","audio":"<base64 little-endian PCM16>"}
{"type":"input_audio_buffer.commit"}
```
- Server replies: session.created/updated, conversation.item.created, conversation.item.input_audio_transcription.delta {item_id, delta, transcript}, typhoon.metrics {first_token_seconds, rtf}, conversation.item.input_audio_transcription.completed {transcript}, error {code}.
- One active stream per worker; the latency must be one the server supports (e.g. 1040 or 480).

(H) NeMo streaming loop, sidecar only (tested on CPU, Linux):
```python
model.encoder.set_default_att_context_size([70, 6])   # 480-ms variant; [70,13] = 1040 ms
buf = CacheAwareStreamingAudioBuffer(model=model, online_normalization=False)
c1, c2, c3 = model.encoder.get_initial_cache_state(batch_size=1)
# then model.conformer_stream_step(..., drop_extra_pre_encoded=0 if step==0 else model.encoder.streaming_cfg.drop_extra_pre_encoded, return_transcription=True)
```

## Latency & resources

End-to-end speech-input budget (default path):
- From the user's last phoneme to the text being available ≈ EoU hangover (600 ms, configurable) + STT (Typhoon RT CPU ≈ 60-150 ms on the i7, estimate) ≈ 0.7-0.8 s.
- The LLM and TTS come after that. Neuro's historical average response delay was ~700 ms (wiki-relayed), so a 400-500 ms hangover plus an 'early decode' is needed to get close.

Per component. "Measured" means this container: 4 vCPU Xeon 2.1 GHz, shared with another agent's llama-server. "Est." means extrapolated.
1. Silero VAD v6 ONNX, CPU, 1 thread: 0.156 ms per 32 ms frame (measured), model 2.3 MB, 0 VRAM.
2. Typhoon RT int8 ONNX via sherpa-onnx, CPU:
   - Measured: 175-237 ms for 3.9-5.5 s clips with 1-2 threads; 140 ms for 4.6 s in another run; 456-503 ms with 4 threads under CPU contention.
   - Peak RSS 337 MB, 0 VRAM. Est. on i7-14700KF P-cores: 60-150 ms per 5 s utterance.
   - A quick 0.8 s barge-in confirm decode: est. 20-60 ms.
3. Typhoon RT fp32 ONNX (PyThaiASR/wannaphong): 407-707 ms per clip, RTF 0.09-0.13 (measured). Weights 435 MB.
4. Typhoon RT native NeMo, file API: 1.2-11 s per clip (dataloader overhead; measured). Not suitable for real time.
5. typhoon-whisper-turbo CT2:
   - CPU int8, 4 threads: 4.3-5.5 s per clip (measured), RSS ~2.0 GB. Est. on i7: 1.2-2.5 s, so CPU fallback only.
   - RTX 4070, int8_float16, beam 1: est. 250-400 ms per 5 s utterance. The encoder always processes the 30 s padded window (same 32-layer encoder as large-v3), and Thai costs ~0.7-0.9 Whisper tokens per character (measured: 54 chars→39 tokens, 78→57).
   - VRAM: fp16 ≈ 2.0-2.5 GB (published 2537 MB peak for stock turbo, beam 5, long audio); int8_float16 est. 1.2-1.6 GB, plus the CUDA context (~300-500 MB) if this is the process's first CUDA user.
6. typhoon-whisper-large-v3 CT2: best CER, but ~2-3x slower decoder than turbo. VRAM ≈ 2.9 GB int8 / 4.5 GB fp16 (published large-v2/v3 numbers). Does not fit comfortably next to a 12 GB-hungry LLM.
7. Stock large-v3-turbo CT2 int8 CPU: 4.0-4.5 s per clip (measured), with worse Thai.
8. typhoon-asr-streaming-115m, NeMo cache-aware:
   - CPU total compute 324-547 ms per 3.9-5.5 s clip (measured), with partials every chunk.
   - First partial latency ≈ look-ahead (0.48 or 1.04 s) + ~25 ms (H100, published).
   - GPU fp32 est. ~1 GB VRAM including the torch context. The 0.6B model est. ~3-3.5 GB VRAM, plus 2.4 GB if n-gram fusion is enabled.
9. Typhoon API (cloud): network RTT plus server time (not measured). 100 req/min.

CPU contention guidance:
- sherpa/onnxruntime at 4 threads was 2x slower than at 1-2 threads while llama.cpp shared the cores (measured). Use num_threads=2-4.
- Keep the LLM fully on GPU if possible. If the LLM (e.g. a 30B-A3B MoE) offloads experts to CPU, reserve cores for STT (process affinity to P-cores; hybrid-core scheduling on the 14700KF is untested).

VRAM plan on the 12 GB card:
- Default STT/VAD path: 0 GB.
- The optional Whisper tier costs ~1.5-2.5 GB: budget it only if LLM + TTS leave room, or load it lazily. Load and unload is ~3 s from disk for CT2 turbo (load 2.9 s measured on CPU).

## Pitfalls

- typhoon-asr-realtime is a full-context model. Forced cache-aware streaming collapses to 62.8% CER on TVSpeech (Typhoon README). Use it segment-by-segment after VAD. PyThaiASR's sliding-window 'streaming' is an unbenchmarked approximation.
- Thai-English code-switching: Typhoon RT writes English words in Thai script ('Minecraft'→'เหมือนคราบ', 'iPhone'→'ไอโฟน' or 'Iโฟน'; measured, and noted in the paper). Game and product names will be garbled. Use the Whisper tier or post-correction if this matters.
- The character name is misheard ('ไพลิน'→'ไทลิน/ไทยลิน/ไทลิล', and 'ไภลิน์' from stock turbo). Add an alias map and fuzzy name matching before wake-word or addressing logic.
- Typhoon models write numbers as spoken Thai words ('สามหมื่นห้าพันบาท'); stock Whisper writes digits ('35,000'). Downstream filters and tool-call parsing must accept both.
- Do not use initial_prompt or hotwords with typhoon-whisper-* by default. Measured: fixed one term and broke another. The card reports that large-v3's prompt pathway collapses (echo loops, CER 200+) on bias lists.
- Model-card code uses the stale id 'scb10x/typhoon-whisper-turbo' (HTTP 401). Use typhoon-ai/typhoon-whisper-turbo and pin the revision.
- Do not use the typhoon-asr pip package (0.1.1) in the loop: it reloads the NeMo model on every transcribe() call, writes processed_*.wav into the CWD, needs NeMo, and its README says Windows is not officially supported.
- The silero-vad pip package hard-requires torch, even for ONNX (import torch at module top). Use the numpy/onnxruntime wrapper or sherpa-onnx's VAD to keep torch out of the voice process.
- The VADIterator default min_silence_duration_ms=100 ends turns mid-sentence. The faster-whisper VadOptions default of 2000 ms is far too long for conversation. Use 500-800 ms for end of utterance.
- Clipped onsets: sherpa-onnx VoiceActivityDetector dropped the first word of an utterance in testing. Always keep a 250-400 ms pre-roll (the custom Endpointer does).
- Whisper hallucinates on silence, noise and very short segments. Gate with VAD, drop segments under 0.3 s, keep no_speech_threshold and log_prob_threshold, set condition_on_previous_text=False, and blacklist known outro phrases.
- Windows plus CTranslate2 GPU: cublas64_12.dll is not found unless the nvidia-*-cu12 bin dirs are on PATH before the model loads. It fails at first inference, not at import. ctranslate2 4.8.x Windows wheels are built for CUDA 12.8 and cuDNN 9.
- CUDA major version clashes: CT2 needs CUDA 12, while onnxruntime-gpu 1.30 [cuda] and torch 2.14 (Linux default) pull CUDA 13. Keep one CUDA major per process, or isolate GPU components in separate processes. The default STT path avoids this by running on CPU.
- onnxruntime 1.30 requires Python ≥ 3.11. Pin Python 3.11 or 3.12 for the project.
- CPU oversubscription: STT at 4 threads was 2x slower than at 1-2 threads while llama.cpp shared the CPU (measured). Cap num_threads, and avoid CPU-offloaded LLM layers competing with STT.
- Self-hearing: with speakers and no AEC, TTS audio triggers VAD, which causes self-interruption or transcribing Pailin's own voice. Default to headphones. Otherwise use WebRTC AEC (10 ms frames, far-end = TTS PCM) plus a filter against recent TTS text.
- Background game audio and music cause false VAD triggers (Silero v6 lists voice-like instruments as a known issue). Capture only the mic device, never desktop audio. Consider noise suppression and a threshold of 0.6+.
- No Thai-capable semantic end-of-turn model exists: Smart Turn v3.2 excludes Thai. Endpointing is silence-based. A Thai sentence-final-particle heuristic (ครับ/ค่ะ/นะ/ไหม/มั้ย/เหรอ → shorter hangover) is an untested idea.
- Exporting typhoon-asr-streaming-115m to a sherpa-onnx streaming model yields empty transcripts (tested). True streaming currently requires NeMo, pinned to commit 907edfd, with fp32 compute on Linux/WSL2.
- The Typhoon streaming reference server allows only one active stream per worker and defaults to 24 kHz PCM16. Set format.rate explicitly.
- Licenses: typhoon-asr-realtime and 115m are CC-BY-4.0 (attribution required, and the card asks users to accept the OpenTyphoon T&C). typhoon-whisper-* are MIT. The nemotron-0.6b Thai model is OpenMDW-1.1. The third-party edtzforai ONNX repo has no license or README, so self-export instead (verified identical).
- HF download sizes: the Typhoon RT .nemo is ~460 MB, the int8 ONNX set ~138 MB, and the turbo safetensors 1.6 GB. Do not download in unit-test CI; use fakes and cache models in the nightly integration job.
- The measured accuracy figures come from 3 synthetic edge-tts clips. They are indicative only, not a Thai CER benchmark. Real mic audio, laughter and shouting will be worse.

## Open questions

- Real latency and VRAM on the target machine are not measured: Typhoon RT int8 on i7-14700KF P-cores vs E-cores, faster-whisper turbo int8_float16 on the RTX 4070, and GPU contention with the chosen LLM backend.
- Accuracy on the streamer's real audio (mic, laughter, shouting, game audio bleed) is untested. Build a 50-100 utterance Thai eval set from the streamer's VODs, with the name 'ไพลิน', game terms and English, and compute CER per backend.
- Should the project host its own exported Typhoon RT ONNX (CC-BY-4.0 attribution, plus OpenTyphoon T&C acceptance) on HF, or export at install time (needs NeMo on Linux/WSL)?
- Does nemo_toolkit[asr] 3.0.0 install and run natively on Windows 11? Not tested. The recommendation avoids it.
- The sherpa-onnx streaming export of typhoon-asr-streaming-115m gives empty output. Is per_feature normalization the cause? Could a model fine-tuned with normalize=NA or a patched sherpa feature pipeline fix it?
- Are sherpa-onnx hotwords (hotwords_file) or LM shallow fusion usable with the offline NeMo transducer for names like ไพลิน? Not tested.
- End-of-turn tuning for Thai: the best hangover (500 vs 700 vs 900 ms), and whether a sentence-final-particle heuristic or a small LLM check beats pure silence. No Thai semantic turn detector exists (Smart Turn excludes Thai). LiveKit's multilingual turn detector's Thai support is unverified.
- Barge-in policy is a product decision: should the streamer's voice always interrupt Pailin (co-host style), or be queued like Neuro's serial loop unless marked critical? It is configurable, but a default is needed.
- Does the streamer wear headphones? This decides whether the AEC path (livekit rtc APM) must be in v1.
- Is the typhoon-asr-qwen-0.6b-ctx biasing model fast enough on the 4070 to replace the Whisper tier for names and code-switch?
