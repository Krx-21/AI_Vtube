# Thai TTS + text chunking — component brief

_Research snapshot: 2026-09-25. Verified facts carry a source; unverified items are marked._

## Recommendation

Design TTS as a pluggable backend interface: async synth(text, voice, rate, pitch) yields (audio int16 PCM 24 kHz | word timing). Ship 3 backends plus a FakeTTS for CI. The chunker and normaliser are pure Python and unit-testable on Linux.

1) PRIMARY, v1: edge-tts==7.2.8 with voice th-TH-PremwadeeNeural (female, the only female Thai voice in the Edge list; fits Pailin). Starting prosody for a young cheerful voice: pitch="+20Hz", rate="+8%". These values are subjective and not listened to; tune by ear. Treat it as BEST-EFFORT. Measured from this container (datacenter IP, TLS-intercepting proxy):
- Novel Premwadee text: time-to-first-audio (TTFA) 1.6–5.1 s, about 3 s typical. About 25% of requests ended in NoAudioReceived after about 4.6 s.
- Identical/cached text: about 0.23–0.32 s.
- The male voice (Niwat) and en-US voices on novel text: 0.38–0.96 s.
- So the slowness is specific to Premwadee (backend capacity), not the client.
The user MUST re-run the bench script (api_notes §1b) from their Thai residential connection before committing. Required mitigations:
- (a) Per-chunk first-audio timeout: 2.0 s for the first chunk, 4 s for prefetched chunks. One retry, then fall back.
- (b) Prefetch pipeline: synthesize chunk N+1 (max 2 in flight) while chunk N plays. 3 parallel requests worked.
- (c) Local disk cache keyed by (voice, rate, pitch, text) for stock phrases: greetings, donation thanks, fillers such as "อืม…", "เอ่อ…". Pre-warm at startup and play a cached filler when TTFA exceeds about 1.2 s.
- (d) Decode the MP3 stream with PyAV (av==18.1.0) CodecContext for bit-exact incremental PCM. For whole-chunk decode, miniaudio==1.71 miniaudio.decode() (MIT, tiny) is fine.
Raw PCM is NOT obtainable from the Edge endpoint. Verified live: only audio-24khz-48kbitrate-mono-mp3 (default), audio-24khz-96kbitrate-mono-mp3 and webm-24khz-16bit-mono-opus return audio. raw-24khz-16bit-mono-pcm, riff-*, ogg-opus and 48 kHz mp3 all return NoAudioReceived.

2) COMMERCIAL-SAFE UPGRADE, same voice identity: Azure AI Speech (azure-cognitiveservices-speech==1.51.2) with the SAME th-TH-PremwadeeNeural voice and the same prosody SSML.
- Official API with raw PCM output: Raw24Khz16BitMonoPcm.
- F0 free tier: 20 transactions / 60 s (not adjustable) and about 0.5 M chars/month. About 12.5 Thai chars/s means roughly 11 h of speech per month.
- S0: 30 TPS default, about $15–16 per 1 M chars (≈$0.70 per hour of continuous speech).
Moving from edge-tts to Azure keeps Pailin's voice unchanged. Recommend it once the stream is monetised, or if edge latency or failures are bad from Thailand. F0's 20 req/min matches roughly one 3 s chunk per 3 s, so use larger chunks (min_chars 60) on F0.

3) LOCAL GPU option, Apache-2.0 and commercially clean: VoxCPM2 (openbmb/VoxCPM2, pip voxcpm==2.0.3).
- 2B params, native Thai, 48 kHz.
- Voice design from a text prompt, e.g. "(young cheerful Thai girl)…", or cloning from a WAV.
- True streaming: generate_streaming() yields one 160 ms patch per LM step.
- Vendor figures: ~8 GB VRAM bf16, RTF 0.30 on an RTX 4090 (0.13 with Nano-vLLM).
- RTX 4070 estimate (unverified): RTF ~0.5–0.8, worse on Windows where torch.compile/triton is unavailable.
- It only fits in 12 GB if the LLM is in the cloud and STT runs on CPU (Typhoon ASR realtime). Do NOT co-host it with a local LLM.
- Thai CER: 2.96 on the Minimax multilingual set (vendor) and 4.98/3.37 short/long on the JaiTTS benchmark.

4) OFFLINE/CI fallback, CPU, 0 VRAM: Piper piper-tts[th]==1.8.0 with rhasspy voice th_TH-tsync2-medium (added 2026-09-04).
- RTF 0.15 on a 4-vCPU 2.1 GHz Xeon (measured). Expect several times faster on an i7-14700KF.
- Formal news-reader voice, not "cute".
- It DROPS all Latin letters, so English words need a Thai-script lexicon.
- The dataset is CC BY-NC-SA 3.0, i.e. NON-commercial. The engine is GPL-3.0.
Use it for dev, offline and emergency only.

Do NOT build on JaiTTS (weights not public; only a demo and benchmark code). Also rule out:
- ThonburianTTS / F5-TTS-THAI / MMS-TTS-tha / OmniVoice: non-commercial licences, or an NC base model.
- Qwen3-TTS: no official Thai; qwen-tts 0.1.1 has no true streaming.
- Kokoro and Chatterbox: no Thai.

TEXT PIPELINE: LLM delta stream → ThaiSpeechChunker (code in api_notes §4) → normalize_cloud() for edge/Azure, or normalize_local() for Piper/VITS → TTS queue.
- Chunk rules: first chunk ≥8 chars, cut at the first clause/particle/punct boundary, ≤60 chars. Later chunks 40–160 chars.
- Candidate boundaries: newline, !?… and '.' followed by a space (not "3.14", not ค.ศ./น.), ", ", and Thai inter-clause spaces. Spaces after sentence-final particles (ค่ะ/คะ/ครับ/นะ/เลย/จ้า…) are strong boundaries.
- Never split: inside or next to numbers ("100 บาท", "เวลา 02:30"), between two Latin words, before attach-left particles (กันนะคะ/ค่ะ/เลย…), before a combining mark, or after a leading vowel เแโใไ. The laughter "555+" attaches to the previous clause.
- Forced cut at max length uses pythainlp newmm word boundaries (optional) with 12 chars of lookahead.
- Normalise AFTER chunking: strip emojis (Edge SPEAKS emoji names), *actions*, markdown and URLs. Map 555+ to ฮ่าฮ่าฮ่า (Edge reads 555 as a number). Collapse elongations (ว้าวววว→ว้าว) and !!! runs. Map Thai digits to Arabic. Edge reads 1,250.50 / 50% / 2026 / phone numbers natively, so leave them for cloud voices. Expand them with pythainlp num_to_thaiword for local VITS/Piper.

## Alternatives

### edge-tts 7.2.8 + th-TH-PremwadeeNeural
- **Pros:** Free, good natural female Thai, reads digits/%/English words natively, word timings for lip-sync/subtitles, 0 VRAM, tiny install, ~0.25–0.3 s TTFA when cached or healthy
- **Cons:** Unofficial API with no SLA or terms. Premwadee measured 2–5 s TTFA and ~25–30% failures on novel text from a datacenter IP. MP3 only; no SSML beyond prosody
- **When:** Default v1 / hobby streaming, behind timeout + cache + fallback. Re-benchmark from the user's Thai residential IP first.

### Azure AI Speech (azure-cognitiveservices-speech 1.51.2) + th-TH-PremwadeeNeural / AcharaNeural
- **Pros:** Same Premwadee voice identity, official SLA, raw PCM 24 kHz, word boundaries, visemes, pre-connect, 0.5 M chars/month free (F0)
- **Cons:** Needs an Azure account/key. F0 is limited to 20 req/min; S0 ≈ $15–16 per 1 M chars (≈$0.7 per hour of speech). Thai HD MAI-Voice-2 voices are male only
- **When:** Monetised streams, or when edge is slow or flaky from Thailand. Drop-in replacement that keeps Pailin's voice.

### Gemini 3.8 Flash TTS (gemini-3.8-flash-tts)
- **Pros:** Thai supported; streaming raw L16 PCM 24 kHz; style prompts; the user already has a Google key
- **Cons:** Latency and pricing not measured or verified. Uses the new Interactions API. Voice identity differs from Premwadee
- **When:** Expressive or emotional lines, if latency proves acceptable; evaluate later.

### VoxCPM2 local (openbmb/VoxCPM2, voxcpm 2.0.3)
- **Pros:** Apache-2.0 (commercial OK), native Thai, 48 kHz, voice design and cloning for a unique Pailin voice, true streaming (~160 ms chunks), no network dependency
- **Cons:** ~8 GB VRAM, EST RTF 0.5–0.8 on a 4070, heavy deps, Windows torch.compile/triton issues, occasional instability on long or expressive input, and it competes with a local LLM for VRAM
- **When:** When the LLM runs in the cloud and a unique, owned voice is wanted; run it as a separate local TTS server process.

### Piper 1.8.0 + th_TH-tsync2-medium (CPU)
- **Pros:** 0 VRAM, RTF ~0.15 even on a weak CPU, small ONNX, works offline, simple API, Windows wheel
- **Cons:** Non-commercial voice licence (CC BY-NC-SA 3.0), GPL engine, formal newsreader timbre, drops Latin text, whole-sentence synthesis (no intra-sentence streaming)
- **When:** Offline dev, emergency fallback, and demos. Not for the final persona voice.

### ThonburianTTS / VIZINTZOR F5-TTS-THAI (F5 flow matching)
- **Pros:** Good Thai quality with voice cloning from a 2–8 s reference; RTF ~0.1
- **Cons:** Non-commercial (CC BY-NC-SA 4.0 / F5 base CC-BY-NC); non-streaming; skips or repeats words on long text; needs ref audio + text per call; pins old numpy/torch
- **When:** Private or non-monetised experiments only.

### MMS-TTS-tha / VIZINTZOR MMS fine-tunes (VITS)
- **Pros:** Tiny, CPU real-time, transformers or Piper-format ONNX
- **Cons:** CC-BY-NC-4.0, robotic quality, vocab lacks most digits and all Latin
- **When:** Not recommended except as a toy fallback.

### Qwen3-TTS (0.6B/1.7B)
- **Pros:** Apache-2.0; Thai CER 2.56 in the JaiTTS benchmark despite no official Thai support
- **Cons:** No official Thai, no true streaming in qwen-tts 0.1.1, RTF ~1.5 in the JaiTTS measurement, pinned transformers
- **When:** Skip for now; revisit if vLLM-Omni streaming plus an official Thai release appears.

### JaiTTS-v1.0
- **Pros:** Best reported Thai CER (1.94%); handles numerals and code-switching without normalisation
- **Cons:** Weights not public as of 2026-09-25 (demo only)
- **When:** Watch HF org JTS-AI for a release.

## Verified facts

- ✅ edge-tts latest PyPI version is 7.2.8, uploaded 2026-03-22; requires_python >=3.7; deps aiohttp>=3.8,<4, certifi, tabulate, typing-extensions; license classifier LGPLv3. Earlier releases: 7.2.4–7.2.7 in Dec 2025, 7.2.0–7.2.3 in Aug 2025.  
  Source: https://pypi.org/pypi/edge-tts/json
- ✅ edge-tts works as of 2026-09-25 from this container: live Thai synthesis succeeded with th-TH-PremwadeeNeural and th-TH-NiwatNeural. The CLI --list-voices returned 322 voices.  
  Source: live test, scratchpad/research/tts/ttfb*.py logs
- ✅ Edge voice list has exactly two Thai voices: th-TH-NiwatNeural (Male) and th-TH-PremwadeeNeural (Female). th-TH-AcharaNeural exists on Azure but not in the Edge list.  
  Source: live `edge-tts --list-voices`; https://raw.githubusercontent.com/MicrosoftDocs/azure-ai-docs/main/articles/ai-services/speech-service/includes/language-support/tts.md
- ✅ Azure th-TH voices: PremwadeeNeural (F), NiwatNeural (M), AcharaNeural (F), plus public-preview HD voices th-TH-Krit:MAI-Voice-2(-Flash) and th-TH-Nattapong:MAI-Voice-2(-Flash). Both HD voices are male and support styles such as friendlycheerful and excited.  
  Source: https://raw.githubusercontent.com/MicrosoftDocs/azure-ai-docs/main/articles/ai-services/speech-service/includes/language-support/tts.md
- ✅ edge_tts.Communicate(text, voice, *, rate='+0%', volume='+0%', pitch='+0Hz', boundary='SentenceBoundary', connector=None, proxy=None, connect_timeout=10, receive_timeout=60). Validation regexes: rate/volume ^[+-]\d+%$, pitch ^[+-]\d+Hz$. stream() is an async generator of TypedDict {type:'audio', data:bytes} or {type:'WordBoundary'|'SentenceBoundary', offset, duration (100-ns ticks), text}. stream_sync(), save() and save_sync() also exist, and stream() can only be called once per object.  
  Source: edge_tts 7.2.8 source (communicate.py, typing.py, data_classes.py)
- ✅ The default boundary became SentenceBoundary in 7.2.0. WordBoundary gives per-word Thai segmentation with timings (e.g. สวัสดี 0.10–0.66 s, ค่ะ, ทุก, คน, …, Minecraft). Most word metadata arrives BEFORE the first audio frame (observed order MMMMMMMMMMMMAAAMMAAAA…). Boundary mode made no TTFA difference on cached text (226–236 ms vs 251–274 ms).  
  Source: live test + https://github.com/rany2/edge-tts/releases
- ✅ edge-tts hardcodes outputFormat audio-24khz-48kbitrate-mono-mp3. Each binary WS message observed was 720 bytes = 5 MPEG-2 Layer III frames (144 B each) = 120 ms of audio.  
  Source: communicate.py; live chunk sizes
- ✅ The Edge endpoint accepted audio-24khz-48kbitrate-mono-mp3, audio-24khz-96kbitrate-mono-mp3 (TTFA 238 ms) and webm-24khz-16bit-mono-opus (414 ms). It returned NoAudioReceived for raw-24khz-16bit-mono-pcm, riff-24khz-16bit-mono-pcm, ogg-24khz-16bit-mono-opus and audio-48khz-192kbitrate-mono-mp3, so raw PCM is not available via Edge. Probed once each with Niwat on 2026-09-25 by source-patching edge-tts.  
  Source: live probe scratchpad/research/tts/fmt_test.py
- ✅ Custom SSML is unsupported: the service only permits a single <voice> with a single <prosody> (rate/pitch/volume). No express-as styles.  
  Source: https://raw.githubusercontent.com/rany2/edge-tts/master/README.md
- ✅ edge-tts auth: hardcoded TrustedClientToken plus a Sec-MS-GEC token (SHA-256 of Windows-epoch ticks rounded down to 5 min + token), a random MUID cookie and an Edge 143 User-Agent. A 403 triggers clock-skew correction from the server Date header and one retry. There is no connection reuse: one new WebSocket per Communicate.  
  Source: edge_tts/drm.py, constants.py, communicate.py (7.2.8)
- ✅ Open edge-tts issues in 2026: #473 intermittent 'No audio was received' with 90–150 s responses (Apr 2026, open); #481 NoAudioReceived for some languages (Jul 2026); #482 WSServerHandshakeError 503 (Jul 2026, open). No maintainer response on #473/#482.  
  Source: https://github.com/rany2/edge-tts/issues, /issues/473, /issues/482
- ✅ Measured TTFA from this container (2026-09-25 06:27–06:53 UTC, via proxy):
- Premwadee identical/cached text: 226–316 ms (n≈20); total 340–510 ms for 4.34 s of audio.
- Premwadee novel text: 1.6–5.1 s (typically 2.3–3.5 s, n≈20). 8 of ~28 novel Premwadee requests failed with NoAudioReceived after ~4.5–4.7 s.
- Niwat novel: 394/479/603 ms. en-US-AvaNeural novel: 384/580/957 ms.
- 3 parallel cached requests: 255–262 ms each, 392 ms wall.
- The very first 3 Premwadee texts at 06:27 were 255–290 ms; unclear whether cached or a lower-load window.  
  Source: live tests ttfb.py…ttfb6.py
- ✅ The Premwadee slowness is voice-specific and persisted after a 3.5-min cool-down, so it is not a client-side burst limit. Microsoft docs state that most TTS 429s are due to per-voice backend capacity in a region.  
  Source: live tests; https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-services-quotas-and-limits
- ✅ Emoji are spoken by the Edge Thai voice: WordBoundary for 'ว้าวววว!!! 😂' contained a '😂' token and produced 3.0 s of audio. Numbers are read natively, judged from WordBoundary durations: '1,250.50' 1.65 s, '50%' 0.70 s, '555' 0.84 s. 555 was therefore likely read as ห้าร้อยห้าสิบห้า, not laughter; inferred from duration, not listened to.  
  Source: live WordBoundary probes ttfb5/ttfb6
- ⚠️ unverified — Claim that Microsoft tightened filtering of cloud/datacenter IPs for Read Aloud (accepted WS, no audio frames).  
  Source: https://huggingface.co/datasets/John6666/forum2/blob/main/edge_tts_hf_colab_1.md (secondary)
- ⚠️ unverified — No official Microsoft terms authorise third-party use of the Edge Read Aloud endpoint; it is an undocumented browser API with no SLA and no published rate limits. Treat it as grey-area for commercial streaming.  
  Source: absence of any official doc; edge-tts README
- ✅ Azure real-time TTS limits: F0 20 transactions per 60 s (not adjustable); S0 30 TPS default, adjustable to 1,000; max 10 min audio per request; 64 KB SSML per WS turn. Doc updated 2026-09-24.  
  Source: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/speech-services-quotas-and-limits
- ⚠️ unverified — Azure F0 free tier = 0.5 M neural chars/month; paid ≈ $16 per 1 M chars (MS quota doc example uses $15/M).  
  Source: https://texttolab.com/blog/azure-text-to-speech-pricing (secondary) + MS quota doc
- ✅ Azure REST/SDK output formats include raw-24khz-16bit-mono-pcm, raw-16khz-16bit-mono-pcm, audio-24khz-48kbitrate-mono-mp3, ogg-24khz-16bit-mono-opus and webm-24khz-16bit-mono-opus.  
  Source: https://raw.githubusercontent.com/MicrosoftDocs/azure-ai-docs/main/articles/ai-services/speech-service/rest-text-to-speech.md
- ✅ azure-cognitiveservices-speech latest is 1.51.2 (2026-08-20), with a win_amd64 wheel.  
  Source: https://pypi.org/pypi/azure-cognitiveservices-speech/json
- ✅ Gemini TTS (docs fetched 2026-09-25):
- Models: gemini-3.8-flash-tts and gemini-3.8-flash-lite-tts. Thai is supported only by gemini-3.8-flash-tts.
- Streaming returns headerless audio/l16 PCM, 24 kHz mono s16le; unary returns WAV.
- Uses client.interactions.create(model=...).
- Latency and pricing were not measured.  
  Source: https://ai.google.dev/gemini-api/docs/speech-generation
- ✅ miniaudio 1.71 (2026-04-29, MIT) has win_amd64 wheels for cp310–cp314.
- miniaudio.decode(bytes, output_format=SampleFormat.SIGNED16, nchannels=1, sample_rate=24000) decoded 4.34 s of Edge MP3 in 2.1 ms.
- Decoding each 720-byte chunk independently is BROKEN: errors and 25% of samples lost, because of the bit reservoir.
- stream_any() first requests 64 KiB. With a partial-read source it gave first PCM after 2.9 KB but lost 1152 samples.  
  Source: https://pypi.org/pypi/miniaudio/json; live dec.py/mastream.py
- ✅ PyAV av 18.1.0 (2026-08-12, BSD-3-Clause, requires Python >=3.11): abi3 win_amd64 wheel of 27 MB bundles the FFmpeg libs, so no ffmpeg.exe is needed. av.CodecContext.create('mp3','r') + parse()/decode() streamed Edge chunks to s16p 24 kHz mono. It is bit-exact against miniaudio (104256 samples, mean abs diff 0.21 LSB), gave first PCM from the first chunk, and took 8 ms total.  
  Source: https://pypi.org/pypi/av/json; live dec.py
- ✅ soundfile 0.14.0 (2026-06-06) with bundled libsndfile 1.2.2 decodes in-memory MP3 via sf.read(io.BytesIO(b), dtype='int16'): 2.7 ms, identical to miniaudio. It has win_amd64 wheels.  
  Source: live test; https://pypi.org/pypi/soundfile/json
- ✅ pythainlp 5.3.8 (uploaded 2026-09-24, Apache-2.0, py>=3.9):
- No mandatory deps except tzdata on win32. Wheel 19.9 MB, installed 63 MB.
- import 69 ms. First newmm call 350 ms (dictionary load), then about 0.06 ms per sentence.
- num_to_thaiword(21)='ยี่สิบเอ็ด', (101)='หนึ่งร้อยเอ็ด', (-5)='ลบห้า'. bahttext(1250.50) works. time_to_thaiword('10:30')='สิบนาฬิกาสามสิบนาที'; with fmt='6h' it gives 'สี่โมงเช้าครึ่ง'.
- sent_tokenize engine='crfcut' needs python-crfsuite (the [compact] extra); 'whitespace+newline' works out of the box.  
  Source: https://pypi.org/pypi/pythainlp/json; live tests
- ✅ VoxCPM2 (openbmb/VoxCPM2):
- Apache-2.0 weights and code, 2B params (MiniCPM-4 backbone), 30 languages including Thai, 48 kHz output.
- ~8 GB VRAM; RTF ~0.30 on RTX 4090 (~0.13 with Nano-vLLM).
- Voice design via a '(description)text' prefix; cloning via reference_wav_path; generate_streaming().
- Thai CER 2.961 vs Minimax 2.701 and ElevenLabs 73.936 on the vendor's Minimax multilingual table.
- VoxCPM1.5 and 0.5B are zh/en only.
- pip voxcpm 2.0.3 (2026-05-11); README requires Python >=3.10,<3.13, torch>=2.5, CUDA>=12.0.  
  Source: https://huggingface.co/openbmb/VoxCPM2 ; https://raw.githubusercontent.com/OpenBMB/VoxCPM/main/README.md ; https://pypi.org/pypi/voxcpm/json
- ✅ VoxCPM2 streaming internals:
- Streaming yields one decoded chunk per LM step. LM token rate is 6.25 Hz with patch_size 4, i.e. about 160 ms of audio per yield.
- retry_badcase is disabled in streaming mode.
- optimize=True uses torch.compile(mode='reduce-overhead') and needs triton; it catches the failure and prints 'torch.compile disabled'.
- Reference audio is loaded with librosa (WAV works without FFmpeg). torchcodec is still a declared dependency.  
  Source: voxcpm 2.0.3 wheel source (core.py, model/voxcpm2.py)
- ✅ JaiTTS-v1.0 (JTS/Jasmine, arXiv 2604.27607):
- Adapted from VoxCPM; about 10k h of Thai; reads numerals and Thai-English code-switch without normalisation.
- Short CER 1.94%, long CER 2.55%, RTF 0.1136 (hardware undisclosed).
- The GitHub repo has only benchmark code (Apache-2.0) and a demo link.
- No weights are published: HF org JTS-AI lists only OpenJAI-v1.0-14B. CC-BY-4.0 is the arXiv paper licence, not a model licence.  
  Source: https://raw.githubusercontent.com/JTS-AI-Team/JaiTTS/main/README.md ; https://arxiv.org/abs/2604.27607 ; https://huggingface.co/api/models?author=JTS-AI
- ✅ The JaiTTS paper's Thai benchmark lists:
- Qwen3-TTS-1.7B: CER 2.56/3.64, RTF 1.54.
- ThonburianTTS: CER 6.26, RTF 0.115.
- VoxCPM2: CER 4.98/3.37, SIM 0.68/0.80.
- OmniVoice: CER 2.73/6.28.  
  Source: https://raw.githubusercontent.com/JTS-AI-Team/JaiTTS/main/README.md
- ✅ ThonburianTTS (biodatlab) is F5-TTS-based: megaF5 Thai-script and megaIPA checkpoints. Models are CC BY-NC-SA 4.0 (non-commercial). It needs reference audio + text and uses the flowtts pipeline from github biodatlab/thonburian-tts.  
  Source: https://huggingface.co/biodatlab/ThonburianTTS
- ✅ VIZINTZOR/F5-TTS-THAI is tagged cc-by-4.0 but its base_model is SWivid/F5-TTS, which is cc-by-nc-4.0. pip f5-tts-th 1.0.9 (2025-11-17) pins numpy<=1.26.4. The card warns it skips or repeats words on long text; ref audio should be 2–8 s.  
  Source: https://huggingface.co/VIZINTZOR/F5-TTS-THAI ; https://huggingface.co/api/models/SWivid/F5-TTS ; https://pypi.org/pypi/f5-tts-th/json
- ✅ facebook/mms-tts-tha is VITS under CC-BY-NC-4.0. The MMS Thai vocab has only digits 0,1,2,4; other digits are silently dropped.  
  Source: https://huggingface.co/facebook/mms-tts-tha ; https://huggingface.co/tranhuyluyen/piper-voices-thai
- ✅ Piper:
- piper-tts 1.8.0 (2026-09-04, GPL-3.0-or-later, abi3 win_amd64 wheel) adds a [th] extra (tltk, pandas, unicode-rbnf).
- rhasspy/piper-voices added th/th_TH/tsync2/medium on 2026-09-04: 63 MB ONNX, 22.05 kHz, 1 speaker.
- Its dataset (NECTEC TSync2 via dubbing-ai/vaja-thai) is CC BY-NC-SA 3.0, non-commercial.  
  Source: https://pypi.org/pypi/piper-tts/json ; https://huggingface.co/rhasspy/piper-voices/tree/main/th ; MODEL_CARD
- ✅ Piper Thai measured on a 4-vCPU Xeon 2.1 GHz:
- tsync2-medium: load 0.8 s, 875 ms for 5.91 s of audio (RTF 0.148).
- MMS-female (piper-format): RTF 0.17.
- Output is one chunk per Thai sentence.
- tltk phonemiser reads '100' as หนึ่งร้อย but drops Latin letters ('Dropping non-Thai character').
- tltk also needs `requests`, which piper-tts[th] does not pull in.  
  Source: live piper_test.py
- ✅ Qwen3-TTS officially supports 10 languages without Thai. qwen-tts 0.1.1 (2026-02-06) pins transformers==4.57.3. Its non_streaming_mode doc says it 'only simulates streaming text input … rather than enabling true streaming input or streaming generation'. A community Thai SFT exists (3scale/qwen-tts-thai-sft, apache-2.0 tag) with an upstream README and no data/eval info.  
  Source: https://raw.githubusercontent.com/QwenLM/Qwen3-TTS/main/README.md ; qwen-tts 0.1.1 wheel source
- ✅ Kokoro-82M has no Thai (languages a/b/j/z/e/f/h/i/p). ResembleAI/chatterbox's 23 languages include no Thai. k2-fsa/OmniVoice covers Thai but its weights are CC-BY-NC (Emilia data), with RTF 0.025. fishaudio/s2-pro lists Thai under licence 'other'.  
  Source: https://huggingface.co/hexgrad/Kokoro-82M/raw/main/VOICES.md ; HF API metadata ; https://huggingface.co/k2-fsa/OmniVoice
- ✅ Typhoon (SCB10X) has no public TTS weights. HF typhoon-ai lists ASR/LLM/OCR models only; Typhoon-TTS is a research preview collecting voices at voice.opentyphoon.ai.  
  Source: https://huggingface.co/api/models?author=typhoon-ai ; https://voice.opentyphoon.ai/
- ✅ Neuro-sama context:
- Neuro's actual TTS engine is unknown. The 'Azure Ashley +25% pitch' claim is uncited fan-wiki lore (T5).
- Vedal's T1 statements show an unshipped 'V3 voice'.
- Latency figures: ~700 ms average (Jul 2024, wiki-relayed); +400 ms accepted for an LLM upgrade (Apr 2024, T1).
- Do not cite any engine as Neuro's.  
  Source: research corpus bcd41d17-C04-tts.md

## Install (Windows)

```
Python: use 3.11 or 3.12. PyAV 18.1 needs >=3.11; voxcpm needs <3.13. Coordinate with the STT/NeMo choice.

py -3.12 -m venv .venv && .venv\Scripts\activate
python -m pip install -U pip

# Core, always installed (pure wheels; no ffmpeg.exe, no MSVC)
pip install "edge-tts==7.2.8" "av==18.1.0" "miniaudio==1.71" "numpy>=1.26" "pythainlp==5.3.8"
#   optional: soundfile==0.14.0 (bundled libsndfile 1.2.2, also decodes MP3); sounddevice for playback
#   pythainlp: do NOT install [full]/[compact] unless needed; the core has newmm, num_to_thaiword, time_to_thaiword

# Official Azure fallback (same Premwadee voice, raw PCM)
pip install "azure-cognitiveservices-speech==1.51.2"
#   needs a Speech resource key + region; F0 is free

# Offline CPU fallback (non-commercial voice)
pip install "piper-tts[th]==1.8.0" requests
#   requests: tltk imports it but the [th] extra doesn't pull it (verified)
curl -L -o th_TH-tsync2-medium.onnx https://huggingface.co/rhasspy/piper-voices/resolve/main/th/th_TH/tsync2/medium/th_TH-tsync2-medium.onnx
curl -L -o th_TH-tsync2-medium.onnx.json https://huggingface.co/rhasspy/piper-voices/resolve/main/th/th_TH/tsync2/medium/th_TH-tsync2-medium.onnx.json

# Local GPU TTS (optional; ~8 GB VRAM): VoxCPM2
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
#   CUDA 12.x build; the RTX 4070 driver must support it
pip install "voxcpm==2.0.3"
#   pulls gradio, funasr, modelscope, datasets, torchcodec… Heavy: keep it in a separate venv/process and talk to it over localhost
#   optional speed-up: pip install triton-windows (UNVERIFIED with voxcpm; without triton, voxcpm prints 'torch.compile disabled' and runs eager)
#   weights are pulled on first VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False); pre-download with huggingface-cli download openbmb/VoxCPM2

# Sanity checks
edge-tts --list-voices | findstr th-TH
edge-tts --voice th-TH-PremwadeeNeural --pitch=+20Hz --rate=+8% --text "สวัสดีค่ะ ไพลินเองค่ะ" --write-media test.mp3 --write-subtitles test.srt
#   negative values need = syntax, e.g. --rate=-10%
edge-playback --voice th-TH-PremwadeeNeural --text "ทดสอบค่ะ"
#   edge-playback needs no mpv on Windows

# Corporate proxy / TLS interception: edge-tts builds its SSL context from certifi (communicate._SSL_CTX), NOT the Windows store. If behind an intercepting proxy, override edge_tts.communicate._SSL_CTX = ssl.create_default_context(cafile=...). It honours HTTPS_PROXY (aiohttp trust_env=True) or Communicate(proxy=...).
```

## API notes

§0 FILES (runnable, tested in this container):
/tmp/claude-0/-home-user-AI-Vtube/02efebb2-c032-5a2c-81bb-f71e8a5461a3/scratchpad/research/tts/
- thai_chunker.py: incremental chunker
- test_chunker.py: property tests
- thai_tts_normalize.py: normaliser
- edge_stream.py: edge-tts → PCM streaming wrapper
- ttfb*.py: latency benchmarks
- fmt_test.py: output-format probe
- piper_test.py
- dec.py / mastream.py: decoder comparisons
Treat them as reference code to port into the repo.

§1 edge-tts WIRE FORMAT (7.2.8)
wss://speech.platform.bing.com/consumer/speech/synthesize/readaloud/edge/v1?TrustedClientToken=6A5AA1D4EAFF4E9FB37E23D68491D6F4&ConnectionId=<uuid-hex>&Sec-MS-GEC=<SHA256 upper>&Sec-MS-GEC-Version=1-143.0.3650.75

Client sends two text frames:
(1) speech.config frame:
X-Timestamp:<js date>\r\nContent-Type:application/json; charset=utf-8\r\nPath:speech.config\r\n\r\n{"context":{"synthesis":{"audio":{"metadataoptions":{"sentenceBoundaryEnabled":"false","wordBoundaryEnabled":"true"},"outputFormat":"audio-24khz-48kbitrate-mono-mp3"}}}}
(2) SSML frame:
X-RequestId:<hex>\r\nContent-Type:application/ssml+xml\r\nX-Timestamp:<date>Z\r\nPath:ssml\r\n\r\n<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='en-US'><voice name='Microsoft Server Speech Text to Speech Voice (th-TH, PremwadeeNeural)'><prosody pitch='+20Hz' rate='+8%' volume='+0%'>ESCAPED TEXT</prosody></voice></speak>

Server sends:
- text frames Path:turn.start, response, audio.metadata. Metadata JSON is {"Metadata":[{"Type":"WordBoundary","Data":{"Offset":ticks,"Duration":ticks,"text":{"Text":"..."}}}]}.
- binary frames: 2-byte big-endian header length, headers (Path:audio, Content-Type:audio/mpeg), then MP3 bytes.
- text frame Path:turn.end.
Text over 4096 bytes is split into several SSML turns.

§1b BENCH (run on the user's PC from Thailand before deciding):
python scratchpad/.../ttfb4.py
It alternates NOVEL (random) vs REPEAT text every 10 s and prints ttfa/total. Remove the 3 lines that override _SSL_CTX; they are only for this sandbox's proxy.

§2 STREAMING edge-tts → PCM (edge_stream.py, verified live: first PCM 284 ms cached, 104256 samples exact):
```python
import asyncio, av, numpy as np, edge_tts
class TTSUnavailable(Exception): pass
async def edge_pcm_stream(text, voice="th-TH-PremwadeeNeural", rate="+0%", pitch="+0Hz", volume="+0%",
                          first_audio_timeout=6.0, idle_timeout=5.0, retries=1):
    for attempt in range(retries + 1):
        got = False
        codec = av.CodecContext.create("mp3", "r")          # stateful decoder (MP3 bit reservoir!)
        agen = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch, volume=volume,
                                    boundary="WordBoundary", receive_timeout=int(idle_timeout)).stream().__aiter__()
        try:
            while True:
                try: ch = await asyncio.wait_for(agen.__anext__(), idle_timeout if got else first_audio_timeout)
                except StopAsyncIteration: break
                if ch["type"] == "audio":
                    got = True
                    for pkt in codec.parse(ch["data"]):
                        for fr in codec.decode(pkt):
                            yield "audio", fr.to_ndarray().reshape(-1)       # int16 mono 24 kHz (s16p)
                else:
                    yield "word", {"text": ch["text"], "offset_s": ch["offset"]/1e7, "duration_s": ch["duration"]/1e7}
            for fr in codec.decode(None): yield "audio", fr.to_ndarray().reshape(-1)
            return
        except (asyncio.TimeoutError, edge_tts.exceptions.NoAudioReceived, edge_tts.exceptions.WebSocketError, OSError) as e:
            await agen.aclose()
            if got or attempt == retries: raise TTSUnavailable(type(e).__name__) from e
```
Recommended timeouts: first chunk first_audio_timeout=2.0 with retries=0, then fall back (cached filler, Azure or Piper). Prefetched chunks: 4.0 s with 1 retry. Failures observed arrive at ~4.6 s, so waiting longer is pointless.

§3 DECODE ALTERNATIVES (no ffmpeg.exe):
(a) Per-chunk/utterance, simplest. Collect all 'audio' bytes of ONE Communicate, then:
```python
import miniaudio, numpy as np
d = miniaudio.decode(mp3_bytes, output_format=miniaudio.SampleFormat.SIGNED16, nchannels=1, sample_rate=24000)
pcm = np.frombuffer(d.samples, dtype=np.int16)
```
It takes 2 ms per 4 s of audio. Cost: it adds (total − TTFA) ≈ 110–160 ms per short chunk (cached case) versus streaming.
(b) soundfile: sf.read(io.BytesIO(mp3_bytes), dtype='int16'), 24 kHz.
(c) NEVER call miniaudio.decode on each 720-byte WS chunk: it fails or loses samples.
(d) Streaming decode: PyAV as in §2. For miniaudio.stream_any, the StreamableSource.read(n) must return available bytes immediately (not block for n). Even then it lost 2 frames, so prefer PyAV.

§4 INCREMENTAL CHUNKER (pure stdlib; pythainlp optional). Tested: split-invariant over 30 random delta splittings; lossless; never emits a chunk starting with a combining mark or ending with เแโใไ; ≤160 chars; 1.4 ms per 168 chars.
```python
import re, unicodedata
from dataclasses import dataclass
try: from pythainlp.tokenize import word_tokenize as _wt
except Exception: _wt = None
_LAUGH = re.compile(r"5{3,}\+*"); _LOOKAHEAD = 12
_THAI = re.compile(r"[฀-๿]"); _LATIN = re.compile(r"[A-Za-z]")
_LEAD = set("เแโใไ"); _HARD = set("!?！？。…\n")
_FINAL = ("นะคะ","นะคับ","นะครับ","ค่ะ","คะ","ครับ","คับ","จ้า","จ้ะ","จ๊ะ","นะ","น้า","เลย","ล่ะ","หรอ","เหรอ","มั้ย","ไหม","สิ","ซิ","ฮะ","เนอะ","แหละ","ด้วย","กัน","ๆ")
_ATTACH = ("กันนะ","กัน ","นะคะ","นะครับ","ค่ะ","คะ","ครับ","นะ ","เลย","ด้วย","จ้า","ๆ")
_ABBR = re.compile(r"(?:^|[\s(])(?:[ก-ฮ]\.)+[ก-ฮ]?$")
def _comb(c): return unicodedata.category(c) in ("Mn","Mc","Me")
@dataclass
class ChunkerConfig:
    first_min_chars:int=8; first_max_chars:int=60; min_chars:int=40; max_chars:int=160; strong_min_chars:int=20
class ThaiSpeechChunker:
    def __init__(s, cfg=None): s.cfg=cfg or ChunkerConfig(); s.buf=""; s.emitted=0
    def feed(s, delta):
        s.buf += delta; out=[]
        while (cut := s._find(False)) is not None: out += s._emit(cut)
        return out
    def flush(s):
        out=[]
        while s.buf.strip():
            cut = s._find(True); out += s._emit(cut if cut is not None else len(s.buf))
        s.buf=""; return out
    def reset(s): s.buf, s.emitted = "", 0              # call on interruption
    def _lim(s):
        c=s.cfg
        return (c.first_min_chars,c.first_max_chars,c.first_min_chars) if s.emitted==0 else (c.min_chars,c.max_chars,c.strong_min_chars)
    def _emit(s, cut):
        p, s.buf = s.buf[:cut].strip(), s.buf[cut:].lstrip()
        if not p: return []
        s.emitted += 1; return [p]
    def _kind(s, i, final):
        b=s.buf; ch=b[i]; j=i+1
        while j<len(b) and b[j] in " \t": j+=1
        right = b[j] if j<len(b) else None
        if right is None and not final and ((ch in _HARD and ch!=".") or ch==" "): return "wait"
        left=b[:i+1].rstrip()
        if ch=="\n": return "hard"
        if ch in _HARD: return None if (right is not None and right in _HARD) else "hard"
        if ch==".":
            if right is None: return "hard" if final else "wait"
            if j==i+1 or _ABBR.search(left): return None     # 3.14 / ค.ศ. / น.
            return "hard"
        if ch==",": return None if j==i+1 else "soft"      # 1,250
        if ch in " \t":
            if (i>0 and b[i-1] in " \t") or not left: return None
            l=left[-1]
            if right is None: return "soft" if final else "wait"
            ltok=left.split()[-1]; rtok=(b[j:].split() or [""])[0]
            if _LAUGH.fullmatch(ltok): return "strong"
            if _LAUGH.match(rtok): return None                 # "…เลย 5555" stays together
            if not final and rtok and set(rtok)<={"5"} and j+len(rtok)==len(b): return "wait"
            if l.isdigit() or right.isdigit(): return None     # "100 บาท", "เวลา 02:30"
            rest=b[j:]
            if any(rest.startswith(p) for p in _ATTACH): return None
            if not final and len(rest)<6 and any(p.startswith(rest) for p in _ATTACH): return "wait"
            if _LATIN.match(l) and _LATIN.match(right): return None  # English phrase
            if _comb(right) or l in _LEAD: return None
            if any(left.endswith(p) for p in _FINAL): return "strong"
            if _THAI.match(l) or _THAI.match(right): return "soft"
            return "soft" if len(left)>20 else None
        return None
    def _find(s, final):
        mn,mx,smin=s._lim(); b=s.buf; bs=bst=None
        for i in range(min(len(b),mx)):
            k=s._kind(i,final)
            if k=="wait": break
            if k is None: continue
            n=len(b[:i+1].strip())
            if k=="hard": return i+1
            if k=="strong":
                if n>=smin: return i+1
                bst=i+1
            if k=="soft":
                if n>=mn: return i+1
                bs=i+1
        if len(b)>=mx:
            for c in (bst,bs):
                if c and c>=mx//3: return c
            if not final and len(b)<mx+_LOOKAHEAD: return None
            return s._forced(mx)
        return len(b) if (final and b.strip()) else None
    def _safe(s,k):
        b=s.buf
        while k>1 and ((k<len(b) and _comb(b[k])) or b[k-1] in _LEAD): k-=1
        return k
    def _forced(s,mx):
        if _wt:
            win=s.buf[:mx+20]; ws=_wt(win, keep_whitespace=True)
            if len(win)==len(s.buf): ws=ws[:-1]              # last token may be partial
            pos=last=0
            for w in ws:
                if pos+len(w)>mx: break
                pos+=len(w); last=pos
            if last>mx//3: return s._safe(last)
        return s._safe(mx)
```
Example output (3-char deltas):
['สวัสดีค่ะทุกคน!!', 'วันนี้ไพลินจะมาเล่นเกม **Minecraft** กันนะคะ', '😂 ขอบคุณคุณ John สำหรับ 1,250.50 บาท ใจดีมากเลย 5555', 'เมื่อวานนอนดึกมากเพราะดูหนังผี ใครเคยเป็นแบบนี้บ้างคะ']
Stall handling: if the LLM produces no delta for >500 ms and the buffer is ≥ first_min, emit up to the last whitespace, not flush(). flush() may cut a Thai word that an LLM token split. Warm up pythainlp at startup: word_tokenize('ทดสอบ') costs 350 ms the first time.

§5 NORMALISER (thai_tts_normalize.py). Apply per chunk after chunking.
normalize_cloud(), used for edge/Azure:
- **x** → x; remove *action*; URL → ' ลิงก์ '
- strip emoji [\U0001F000-\U0001FAFF☀-➿️‍]
- strip markdown [*_`#>~|]
- (?<!\d)5{3,}\+*(?!\d) → ' ฮ่าฮ่าฮ่า '
- collapse ≥3 repeated chars (ว้าวววว→ว้าว), [!?]{2,} → single
- Thai digits ๐-๙ → 0-9
normalize_local(text, lexicon), used for Piper/MMS:
- everything in cloud, plus lexicon EN→Thai script (e.g. {"Minecraft":"มายคราฟต์","VTuber":"วีทูบเบอร์"})
- phones 0\d{8,9} → digit words; HH:MM → NนาฬิกาMนาที; $29.99 → 29.99 ดอลลาร์; N% → N เปอร์เซ็นต์
- numbers → pythainlp.util.num_to_thaiword(int) + 'จุด' + digit words for decimals
- X ๆ → XX; finally strip remaining [A-Za-z]
Output example:
'ว้าว! ฮ่าฮ่าฮ่า ขอบคุณ จอห์น ที่ให้ ยี่สิบเก้าจุดเก้าเก้า ดอลลาร์ และ หนึ่งพันสองร้อยห้าสิบจุดห้าศูนย์ บาท เวลา สองนาฬิกาสามสิบนาที ลด ห้าสิบ เปอร์เซ็นต์ …'
Better long-term: tell the LLM in its system prompt to write laughter as ฮ่าๆ, avoid emoji/markdown, and put stage directions in [tags] that the pipeline strips and routes to the avatar.

§6 PIPELINE SHAPE
LLM stream → chunker → normaliser → asyncio.Queue(maxsize=2) → TTS worker(s), with ≤2 synth in flight so chunk N+1 synthesises while N plays → PCM ring buffer → sounddevice.OutputStream(24000, int16, mono) → RMS/visemes to the avatar.
- Interrupt: cancel synth tasks (aclose the generators), clear queues, stop and flush the output stream, chunker.reset().
- Speech segment events (start/word/end markers) come from WordBoundary offsets (edge/Azure) or from PCM sample counts (local).
- CI: FakeTTS yields silence of len(text)/12.5 s plus fake word marks; no network.

§7 AZURE (same voice, raw PCM). Not run here (no key); standard SDK API:
```python
import azure.cognitiveservices.speech as sdk
cfg = sdk.SpeechConfig(subscription=KEY, region=REGION)   # e.g. southeastasia (voice availability per region: check)
cfg.set_speech_synthesis_output_format(sdk.SpeechSynthesisOutputFormat.Raw24Khz16BitMonoPcm)
syn = sdk.SpeechSynthesizer(speech_config=cfg, audio_config=None)
syn.synthesizing.connect(lambda e: pcm_q.put_nowait(e.result.audio_data))   # incremental PCM bytes
ssml = ("<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='th-TH'>"
        "<voice name='th-TH-PremwadeeNeural'><prosody pitch='+20Hz' rate='+8%'>สวัสดีค่ะ</prosody></voice></speak>")
syn.speak_ssml_async(ssml).get()
```
Pre-connect with sdk.Connection.from_speech_synthesizer(syn).open(True) to cut the handshake.

§8 PIPER (CPU):
```python
from piper.voice import PiperVoice
v = PiperVoice.load("th_TH-tsync2-medium.onnx")
for ch in v.synthesize(text):      # AudioChunk
    pcm16 = ch.audio_int16_bytes    # ch.sample_rate == 22050 → resample to 24k or open the device at 22050
```

§9 VoxCPM2 (GPU, run in its own process):
```python
from voxcpm import VoxCPM
m = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False)   # optimize=True default → torch.compile if triton
for chunk in m.generate_streaming(text="(เสียงเด็กผู้หญิงสดใส ร่าเริง)สวัสดีค่ะทุกคน", cfg_value=2.0, inference_timesteps=10):
    play(chunk)                    # float32 numpy, 48 kHz, ~160 ms per yield
```
For a fixed identity, use reference_wav_path='pailin_ref.wav' or prompt_wav_path + prompt_text. Voice design output varies between runs; after designing Pailin once, save a reference WAV and clone it thereafter.

## Latency & resources

All numbers were MEASURED in this Linux container unless marked EST/VENDOR.

edge-tts, Premwadee (as of 2026-09-25):
- Cached/repeated text: TTFA 0.23–0.32 s; full 4.3 s utterance arrives in 0.34–0.51 s.
- Novel text: TTFA 1.6–5.1 s (typically 2.3–3.5 s). About 25–30% NoAudioReceived after ~4.6 s.
- Niwat and English voices on novel text: TTFA 0.38–0.96 s.
- Audio arrives about 8–10× faster than real time once started.
- 3 concurrent requests OK.
- Thai speaking rate ≈12.5 chars/s: 54 chars → 4.34 s.
- MP3 is 48 kbps (6 KB/s of audio); 120 ms per WS message.

Expected end-to-end first-sound budget when edge behaves well: LLM first clause (~8–20 Thai chars) + TTS TTFA ~0.3 s + PyAV decode <5 ms + output buffer 20–50 ms.

Local:
- Piper tsync2: RTF 0.148 on 4 vCPU Xeon 2.1 GHz. Whole sentence at once: 0.9 s for 5.9 s of audio. EST 2–3× faster on an i7-14700KF P-cores. 0 VRAM, ~150 MB RAM, 63 MB model. Load 0.8 s.
- VoxCPM2: VENDOR ~8 GB VRAM bf16, RTF 0.30 on 4090 (0.13 with Nano-vLLM). EST RTF 0.5–0.8 on 4070 in eager mode on Windows; first 160 ms chunk after prefill plus one step, EST 0.2–0.5 s. Unverified: no GPU here.
- ThonburianTTS/F5: RTF 0.115 on undisclosed hardware (JaiTTS paper); non-streaming; EST 1–3 GB VRAM.
- MMS-VITS: CPU real-time, tiny.

12 GB VRAM plan:
- VoxCPM2 (8 GB) + CPU STT + cloud LLM fits.
- VoxCPM2 + any local LLM does not fit comfortably.
- edge/Azure/Piper use 0 VRAM, leaving the whole GPU for a local LLM/STT.

Decoders: miniaudio 2.1 ms, soundfile 2.7 ms per 4.3 s MP3; PyAV streaming 8 ms total.
pythainlp: import 69 ms; first tokenize 350 ms; then ~0.06 ms per sentence; 63 MB on disk.
Chunker: 1.4 ms per 168 chars.
Azure F0: 20 requests per 60 s. At ~3–6 s chunks that is sustainable only with min_chars ≥ 60.

## Pitfalls

- th-TH-PremwadeeNeural (the only female Thai Edge voice) was the slow and flaky one in testing: novel text had 2–5 s TTFA and ~25–30% NoAudioReceived, while Niwat and English voices were 0.4–0.9 s. Cached identical text is fast, so a quick demo with a repeated test sentence will mislead; always benchmark with random text.
- NoAudioReceived arrives only after ~4.5–4.7 s. Set a first-audio timeout of about 2 s rather than relying on the library's 60 s receive_timeout.
- edge-tts builds its own SSL context from certifi (communicate._SSL_CTX), so it ignores OS/corporate CA stores. Behind TLS interception you must override _SSL_CTX; SSL_CERT_FILE does not help.
- Raw PCM is impossible via the Edge endpoint (raw-/riff-/ogg-/48 kHz formats → NoAudioReceived). Decode MP3 client-side. If you source-patch 96 kbps MP3 in, edge-tts's CBR offset maths (hardcoded 48 kbps) breaks for texts over 4096 bytes.
- Never decode each 720-byte MP3 WS message independently: the bit reservoir makes miniaudio fail or lose ~25% of samples. Use a stateful PyAV CodecContext, or decode per whole Communicate.
- miniaudio.stream_any requests 64 KiB up front; a blocking source stalls until the whole utterance arrives.
- Communicate.stream() can be called only once per object; build a new Communicate per chunk. There is also no WS connection reuse (a new TLS+WS handshake each time).
- Custom SSML is rejected by the Edge service: no <break>, <phoneme>, styles or multiple voices. Only rate '+N%', pitch '+NHz' and volume '+N%' strings are allowed; negative CLI values need --rate=-10%.
- Edge Thai voices SPEAK emoji names and read '555' as a number. Strip emoji and markdown and map 5{3,} to laughter before TTS.
- Piper th_TH-tsync2 silently drops every Latin character ('Minecraft' vanishes). MMS/VIZINTZOR Thai VITS vocab lacks digits 3,5,6,7,8,9. Local VITS engines need full number expansion plus an EN→Thai-script lexicon.
- piper-tts[th]==1.8.0 is missing the `requests` dependency required by tltk (ModuleNotFoundError at import).
- Licences: ThonburianTTS (CC BY-NC-SA 4.0), MMS-TTS (CC-BY-NC-4.0), Piper tsync2 voice (CC BY-NC-SA 3.0), OmniVoice (CC-BY-NC) and F5-TTS base (CC-BY-NC-4.0, which VIZINTZOR/F5-TTS-THAI builds on despite its cc-by-4.0 tag) are non-commercial. A monetised Twitch/YouTube stream is arguably commercial.
- edge-tts is an unofficial client of Edge's Read Aloud endpoint (hardcoded TrustedClientToken, Sec-MS-GEC, spoofed Edge 143 UA). Microsoft has broken it before (endpoint change Aug 2025 → 7.2.2; NoAudioReceived fix Dec 2025 → 7.2.4). Open 2026 issues #473/#481/#482 have no maintainer response. Pin the version, keep a second backend, and plan for Azure.
- JaiTTS has no downloadable weights (demo plus benchmark code only). Do not plan on it. The earlier 'CC-BY-4.0' note refers to the arXiv paper, not the model.
- Qwen3-TTS: no official Thai. The qwen-tts 0.1.1 package does not implement true streaming generation despite the '97 ms' marketing, and pins transformers==4.57.3, which will clash with other stacks.
- VoxCPM2 on Windows: torch.compile needs triton; without it VoxCPM prints 'torch.compile disabled' and runs slower. It pulls gradio/funasr/modelscope/datasets<4 and needs Python <3.13, so isolate it in its own venv/process. Streaming mode disables bad-case retry, so rare runaway/garbled chunks are possible; cap max_len.
- PyAV 18.1 requires Python ≥3.11, while voxcpm requires <3.13. Pin 3.11 or 3.12 for the main app.
- Thai chunking: LLM deltas can split a base consonant from its tone/vowel marks. Never cut before a combining mark (Mn) or after a leading vowel เแโใไ. Defer space decisions until right context arrives. Keep 'Edition กันนะคะ'-style particles attached; don't split '100 บาท' or 'เวลา 02:30'.
- pythainlp newmm's first call costs ~350 ms (dictionary load); warm it up at startup. Its crfcut sentence splitter needs python-crfsuite, which is not installed by default.
- Azure F0's 20 transactions/min limit is easy to hit with small chunks. Use larger later-chunks (min_chars ≥ 60) or S0.

## Open questions

- Is Premwadee's novel-text TTFA (2–5 s) and ~25–30% NoAudioReceived specific to this US datacenter egress/proxy, or does it also happen from the user's Thai residential connection? The user must run the bench script (ttfb4.py) on the target PC.
- Which Azure region serves th-TH-PremwadeeNeural with the lowest latency from Thailand (southeastasia?), and what is the real TTFA? Not measured: no key.
- What is VoxCPM2's real RTF, first-chunk latency and VRAM on an RTX 4070 under Windows, with and without triton-windows? Is its Thai 'young cheerful girl' voice-design quality acceptable?
- Exact Azure TTS price per 1 M chars as of Sept 2026 (secondary sources say $16; MS quota doc example uses $15).
- Gemini 3.8 Flash TTS latency and pricing for Thai streaming.
- Does en-US-AvaMultilingualNeural (or another multilingual voice) speak acceptable Thai? It emitted Thai word boundaries, but MS docs don't list Thai for multilingual voices, and quality wasn't heard.
- Subjective tuning of prosody (pitch/rate) for Pailin: needs listening, not possible here.
- Whether Microsoft's terms explicitly prohibit third-party use of the Edge Read Aloud endpoint. No explicit document was found either way.
