# LLM serving + clients — component brief

_Research snapshot: 2026-09-25. Verified facts carry a source; unverified items are marked._

## Recommendation

Run everything through the OpenAI Chat Completions wire format (openai 3.19.2 AsyncOpenAI), with a provider registry and fallback chain. Servers: prefer llama.cpp llama-server. Treat Ollama and LM Studio as optional "easy mode" backends only.

DEFAULT PROFILE "pailin-30b" (quality):
- Model: Typhoon2.5-Qwen3-30B-A3B (Apache-2.0, non-thinking, Hermes-style tools), Q4_K_M, 18.56 GB.
- Use the mradermacher GGUF, which embeds the chat template. If you use the official typhoon-ai GGUF you must also pass --chat-template-file.
- Serve with llama-server CUDA. Attention/dense weights stay on the GPU. Keep only as many MoE expert layers on the GPU as the VRAM budget allows; the rest live in the 64 GB DDR5.
- Context: -c 16384, q8_0 KV, flash-attn on.
- Slots: 2 unified slots. Pin slot 0 to the serial "speak" loop so its prompt cache survives; slot 1 is for background work (memory summarisation, the chat-pick classifier).
- Why: with 3B active params, decode is limited by RAM bandwidth, not VRAM. Community data on 12 GB cards with similar A3B MoE models shows 39–53 tok/s. Thai text costs about 2 chars per token, so generation runs far faster than speech.
- Latency: time to first token (TTFT) stays low only if the prompt prefix is cached, so order the prompt stable→volatile and keep the tools list static.

LOW-LATENCY PROFILE "pailin-4b":
- Model: Typhoon2.5-Qwen3-4B, fully on the GPU (-ngl all). Quant: Q4_K_M (2.50 GB) or Q6_K (3.31 GB); total VRAM about 4–6 GB with a 16K context.
- Expected (estimate): about 120–150 tok/s, warm TTFT under 100 ms.
- Use it when a GPU-heavy game is running, when TTS/STT need the VRAM, or as the instant local fallback. It has the same template and tool format as the 30B, so swapping needs no prompt changes.

CLOUD FALLBACKS (in order):
1. Typhoon API, model typhoon-v2.5-30b-a3b-instruct, base https://api.opentyphoon.ai/v1.
   - Same model family, so persona and tool behaviour stay consistent. Free: 5 req/s, 200 req/min.
   - Caveats: a research showcase with no SLA; your inputs may be used for training.
2. Gemini via the OpenAI-compatible endpoint https://generativelanguage.googleapis.com/v1beta/openai/.
   - Model gemini-3.5-flash-lite with reasoning_effort="minimal" (thinking cannot be fully disabled on Gemini 3).
   - Use gemini-3.8-flash (lowest level "low") for a "smart" mode.
   - Use a paid-tier key: free-tier data is used to improve Google products, and reported free quotas (about 500 requests/day for Flash-Lite, about 20/day for Flash) are too small for a stream.

HOT-SWAP WITH FALLBACK (mirrors Neuro's T1 July 2026 "switch model live, keep old ready to roll back"):
- Keep conversation state in a provider-neutral OpenAI message list.
- A FallbackChain tries providers in order. Each has a per-provider circuit breaker (exponential backoff up to 60 s) and a first-token timeout (4 s local, 6–8 s cloud).
- Fall back only if nothing has been emitted to TTS yet. If a provider dies mid-utterance, end the utterance and let the next decision-loop tick continue.
- An operator command chain.use(name) moves a provider to the front. Every provider stays warm: two llama-server processes on ports 8080/8081 (VRAM permitting), or router mode with a presets INI.
- Health-check /health, which returns 503 while loading and 200 {"status":"ok"} when ready.
- Per-provider adapters:
  - Strip extra_content/thought_signature for non-Gemini providers and keep it for Gemini.
  - Always send string content, because the Typhoon template turns content-part arrays into an empty string.
  - Map thinking controls: none for Typhoon, reasoning_effort for Gemini, chat_template_kwargs {"enable_thinking": false} for Qwen3.5/3.6.
- Interruption (critical priority): cancel the asyncio task and await stream.close() in finally. I verified that llama-server logs "cancel task" and frees the slot within about 0.1–0.2 s.

## Alternatives

### Ollama (v0.34.4) with scb10x/typhoon2.5-qwen3-4b or -30b-a3b
- **Pros:** Single installer with a tray app; official scb10x tags ship a working template (Tools badge); OpenAI-compatible endpoint at :11434/v1; keep_alive and multiple model management.
- **Cons:** No --n-cpu-moe equivalent (issue #11772 open), so the 30B-A3B cannot be split experts-only; 4096-token context by default; no tool_choice; less control over slots, prompt cache and KV types; lags llama.cpp releases.
- **When:** Quick demo or the 4B low-latency profile on a machine where the user will not manage command-line flags.

### LM Studio 0.4.25 (GUI or headless llmster)
- **Pros:** GUI for trying quants; OpenAI-compatible endpoint at :1234/v1; continuous batching; n_cpu_moe slider; lms CLI for scripting.
- **Cons:** Closed source; the MoE slider reportedly offloads whole layers (issue #1421); tool parsing is native only for known templates, otherwise the [TOOL_REQUEST] fallback; harder to supervise from Python.
- **When:** User-facing tuning and experimentation; not recommended as the production backend.

### Qwen3.6-35B-A3B / Qwen3.5-9B / Gemma-4-26B-A4B or 12B via the same llama-server
- **Pros:** Newer (2026) and likely stronger general reasoning; Apache-2.0; can think when wanted; strong community support for MoE offload on 12 GB GPUs (about 40–53 tok/s reported).
- **Cons:** Thai persona fluency vs Typhoon 2.5 not evaluated; thinking is on by default (Qwen3.5/3.6) and must be disabled per request with chat_template_kwargs {enable_thinking:false} or -rea off; the Qwen3.5 tool format is Qwen3-Coder XML (different parser, still handled by llama.cpp); Qwen3.6 tool calls reportedly drop in long agent loops.
- **When:** A/B against Typhoon 2.5 with a Thai persona eval set. Also a candidate 'smart' or reasoning hot-swap profile, mirroring Neuro's 2026 reasoning-model swap.

### Typhoon2.1-Gemma3-12B (Q4 GGUF ~7 GB)
- **Pros:** Thai-tuned dense model; optional thinking mode; fits entirely in VRAM with a small context.
- **Cons:** Gemma license; older (May 2025); dense 12B means about 35–40 tok/s at best on a 4070 and leaves little VRAM for TTS; pythonic tool format.
- **When:** Only if 30B-A3B quality is needed but CPU/RAM bandwidth turns out poor.

### Cloud-primary (Gemini 3.8 Flash or Typhoon API as main brain)
- **Pros:** Zero VRAM, frees the GPU for TTS and the game; strongest model (3.8 Flash).
- **Cons:** Network latency and jitter; Gemini thinking cannot be fully disabled; cost (about $0.75/$3.75 per 1M tokens); data policies; no Typhoon SLA; outages kill the stream.
- **When:** Game-heavy streams where the GPU is saturated, or as the 'smart mode' target for hot-swap.

### vLLM
- **Pros:** Model cards give ready-made commands (--tool-call-parser hermes); high throughput.
- **Cons:** No native Windows (WSL2 only); needs the full model in VRAM (the 30B does not fit in 12 GB unquantised); not needed for a serial single-user loop.
- **When:** Not recommended for this PC.

## Verified facts

- ✅ Typhoon 2.5 open LLMs on HF: typhoon-ai/typhoon2.5-qwen3-4b (Qwen3ForCausalLM, 36 layers, 8 KV heads, head_dim 128, tied embeddings, bf16 safetensors 2 shards ~8.0 GB) and typhoon-ai/typhoon2.5-qwen3-30b-a3b (Qwen3MoeForCausalLM, 48 layers, 128 experts / 8 active, moe_intermediate 768, 4 KV heads, 13 shards ~61 GB). Both created 2025-09-23, Apache-2.0 (license_link = Qwen3-*-Instruct-2507 LICENSE), max_position_embeddings 262144 (card: 256K context), model cards still reference scb10x/... ids. No Typhoon 3 LLM exists on HF as of 2026-09-25 (newest typhoon-ai uploads are ASR models).  
  Source: https://huggingface.co/api/models?author=typhoon-ai ; https://huggingface.co/typhoon-ai/typhoon2.5-qwen3-30b-a3b/raw/main/config.json ; .../typhoon2.5-qwen3-4b/raw/main/config.json
- ✅ Official GGUF repos contain exactly one quant each: typhoon-ai/typhoon2.5-qwen3-4b-gguf -> typhoon2.5-qwen3-4b-q4_k_m.gguf (2,497,276,128 B); typhoon-ai/typhoon2.5-qwen3-30b-a3b-gguf -> typhoon2.5-qwen3-30b-a3b-q4_k_m.gguf (18,556,681,696 B). Made with GGUF-my-repo.  
  Source: https://huggingface.co/api/models/typhoon-ai/typhoon2.5-qwen3-30b-a3b-gguf/tree/main
- ✅ The official typhoon-ai GGUFs have NO tokenizer.chat_template key in GGUF metadata (parsed the header by HTTP range request; 27/29 KV pairs, none is chat_template). mradermacher GGUFs DO embed the 2630-char Typhoon template.  
  Source: GGUF header parse of https://huggingface.co/typhoon-ai/typhoon2.5-qwen3-4b-gguf/resolve/main/typhoon2.5-qwen3-4b-q4_k_m.gguf and mradermacher/typhoon2.5-qwen3-4b-GGUF Q8_0 / -30b-a3b-GGUF Q4_K_M
- ✅ Local test (llama.cpp master 1ab7e5a, 2026-09-25, CPU build): official 4B GGUF with no template override falls back to a plain ChatML template with no tool support. The tools field was silently ignored; the model replied 'title changed' (a hallucination) and then degenerated into repeated 'า'. With --chat-template-file <repo chat_template.jinja>, /props showed chat_template_caps supports_tool_calls=true and supports_parallel_tool_calls=true. The response had finish_reason 'tool_calls' and parsed arguments {"title": "ไพลินเล่นเกม Minecraft"}.  
  Source: local run, scratchpad/research/srv_notmpl.log & srv_tmpl.log
- ✅ mradermacher quant sizes. 30B-A3B: Q3_K_M 14.71 GB, IQ4_XS 16.56, Q4_K_S 17.46, Q4_K_M 18.56, Q5_K_M 21.73, Q6_K 25.09, Q8_0 32.48. i1 variants include IQ3_XXS 11.85, i1-IQ4_XS 16.37. 4B: Q4_K_M 2.50 GB, Q5_K_M 2.89, Q6_K 3.31, Q8_0 4.28, f16 8.05. File names follow the pattern typhoon2.5-qwen3-30b-a3b.Q4_K_M.gguf.  
  Source: https://huggingface.co/api/models/mradermacher/typhoon2.5-qwen3-30b-a3b-GGUF/tree/main ; .../typhoon2.5-qwen3-4b-GGUF
- ✅ Typhoon 2.5 is NOT a reasoning model. It is built on Qwen3 Instruct 2507 (non-thinking); the blog says 'It isn't designed for long-horizon planning or complex logical chains'. Its chat template has no <think> handling and no enable_thinking kwarg, so no /no_think or chat_template_kwargs are needed.  
  Source: https://opentyphoon.ai/blog/en/typhoon2-5-release ; https://huggingface.co/typhoon-ai/typhoon2.5-qwen3-4b/raw/main/chat_template.jinja
- ✅ Typhoon 2.5 tool format is Hermes/Qwen2.5 style. Tools are rendered into the system prompt as JSON inside <tools></tools>. Calls are emitted as <tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>. Tool results are rendered as <tool_response>...</tool_response> inside a *user* turn with no tool_call_id, so result order must match call order. Non-string message content is replaced by ''. vLLM card uses --tool-call-parser hermes --enable-auto-tool-choice.  
  Source: chat_template.jinja (identical for 4B and 30B) + /apply-template output from local llama-server + HF model card
- ✅ Parallel tool calls work with Typhoon2.5-4B on llama.cpp when the request includes "parallel_tool_calls": true. Tested: one Thai chat message produced 2 calls (set_stream_title + create_poll with array args). The follow-up with role:'tool' results produced a Thai spoken summary that included emoji.  
  Source: local run scratchpad/research/roundtrip.py
- ✅ Typhoon recommended sampling: temperature 0.6, top_p 0.95, repetition_penalty 1.05 (raise to 1.1–1.2 if you see repetition), max_tokens 512.  
  Source: https://docs.opentyphoon.ai/en/quickstart/ ; HF model cards
- ✅ Other Typhoon LLMs: typhoon-ai/typhoon2.1-gemma3-12b (Gemma license, 128K context, hybrid thinking with enable_thinking default False, vLLM --tool-call-parser pythonic, GGUF repo typhoon2.1-gemma3-12b-gguf). typhoon-ai/typhoon-s-thaillm-8b-instruct-research-preview (Qwen3 arch on ThaiLLM-8B base, 32K context, Apache-2.0, Dec 2025).  
  Source: https://huggingface.co/typhoon-ai/typhoon2.1-gemma3-12b ; https://huggingface.co/typhoon-ai/typhoon-s-thaillm-8b-instruct-research-preview
- ✅ Typhoon API: base URL https://api.opentyphoon.ai/v1, header 'Authorization: Bearer <KEY>' (key from playground.opentyphoon.ai/api-key), POST /v1/chat/completions, OpenAI-compatible SSE streaming. Tool calling is shown in the docs with the OpenAI SDK. Documented models: typhoon-v2.5-30b-a3b-instruct (128K context, 5 req/s, 200 req/min) and typhoon-v2.1-12b-instruct (56K, same limits). Over the limit it returns 429 {"error":{"type":"rate_limit_error","code":"rate_limit_exceeded"}}. Extra parameter repetition_penalty (1.0–2.0).  
  Source: https://docs.opentyphoon.ai/en/api-reference/ ; /en/models/ ; /en/rate-limits/ ; /en/tool/
- ✅ Live probe 2026-09-25: GET https://api.opentyphoon.ai/v1/models (no auth needed) lists only typhoon-v2.5-30b-a3b-instruct as a chat LLM, plus typhoon-ocr*, typhoon-asr-realtime and typhoon-isan-asr-realtime; typhoon-v2.1-12b-instruct is NOT listed. It returns a bare JSON array, so openai SDK client.models.list() raises AttributeError. A bad key returns 401 {"detail":"Invalid API Key"} (not OpenAI error shape); no auth header returns 403.  
  Source: curl + openai 3.19.2 against api.opentyphoon.ai
- ⚠️ unverified — The Typhoon API reference says max_tokens 'Default is 150, maximum is 8192 tokens shared between prompt and completion'. This contradicts the 128K context listed on the models page; actual enforcement is unknown.  
  Source: https://docs.opentyphoon.ai/en/api-reference/
- ✅ Typhoon API terms: a free research showcase with no formal support. SCB 10X collects usage data, and the TaC allows using Input to train its models. No availability guarantee; the service can change or be terminated without notice. Prohibited: deepfakes, hate or harassment, porn, and using Output to build competing models. For production the docs say to use 'API Pro via Together AI', but the Together serverless model list currently shows no Typhoon/scb10x models.  
  Source: https://docs.opentyphoon.ai/en/faq/ ; https://opentyphoon.ai/tac ; https://docs.together.ai/docs/serverless/models
- ✅ llama.cpp latest release b11177 (2026-09-25). Windows CUDA assets: llama-b11177-bin-win-cuda-12.4-x64.zip (243 MB) + cudart-llama-bin-win-cuda-12.4-x64.zip (373 MB), or llama-b11177-bin-win-cuda-13.4-x64.zip (143 MB) + cudart-llama-bin-win-cuda-13.4-x64.zip (404 MB). Also win-vulkan, win-cpu and others. Also installable with 'winget install llama.cpp' (which backend the winget build uses was not verified).  
  Source: https://github.com/ggml-org/llama.cpp/releases/expanded_assets/b11177 ; https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/install.md
- ✅ CUDA 13.x builds need NVIDIA driver >= 580 (minor-version compatibility).  
  Source: https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html
- ✅ Current llama-server defaults/flags (master 1ab7e5a): --jinja enabled by default; -ngl default 'auto' (accepts N|auto|all); -fa default auto; --fit on by default (--fit-target MiB margin default 1024, --fit-ctx min 4096); -cmoe/--cpu-moe; -ncmoe/--n-cpu-moe N; -ot/--override-tensor; -ctk/-ctv; --cache-prompt enabled by default; --cache-reuse N default 0; -cram/--cache-ram default 8192 MiB; --slot-save-path; -np default -1 (auto = 4 slots + kv_unified); -kvu/--kv-unified; -rea/--reasoning on|off|auto; --reasoning-format; --chat-template-kwargs; --chat-template-file; --models-dir/--models-preset/--models-max (router mode); --sleep-idle-seconds; -a/--alias; --api-key; --metrics.  
  Source: https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md
- ✅ --no-mmap and --mlock were removed. The built binary rejects them with 'error: invalid argument: --no-mmap'. Use -lm/--load-mode auto|none|mmap|mlock|mmap+mlock|dio instead.  
  Source: local build llama-server 0.5.0-dev commit 1ab7e5a; common/arg.cpp
- ✅ --fit is MoE-aware. It first reduces unset context size, then fills layers back-to-front with dense weights while forcing all ffn_*_exps expert tensors to system RAM, then promotes whole layers to GPU. It throws (logs 'failed to fit params', continues with user params) if the user already set tensor overrides; --n-cpu-moe/-ot are implemented as overrides. So use EITHER --fit (auto) OR explicit -ngl/--n-cpu-moe with -fit off.  
  Source: llama.cpp common/fit.cpp (steps 1-4) + common/common.cpp
- ✅ Explicit -np 2 gives kv_unified=false and n_ctx_slot = c/np (log: 'n_slots = 2, n_ctx_slot = 2048' for -c 4096). Auto mode gives 'n_slots = 4, n_ctx_slot = 4096, kv_unified = true'. Add -kvu when setting -np explicitly.  
  Source: local llama-server logs
- ✅ Cancellation works. An asyncio task cancel plus 'await stream.close()' on the openai AsyncStream closes the HTTP response; llama-server logs 'stop: cancel task' within about 0.1 s and slot.is_processing becomes false within ≤0.2 s. Mechanism: req.should_stop = httplib is_connection_closed → server_response_reader::stop() posts cancel tasks.  
  Source: local test cancel_probe.py; tools/server/server-http.cpp, server-queue.cpp
- ✅ Prompt caching works. On the repeated tool request the server reported timings cache_n=288, prompt_n=1. id_slot is accepted in /v1/chat/completions (request routed to slot 0). Streaming responses include a 'timings' object in the final chunk.  
  Source: local llama-server test
- ✅ llama-server streamed tool calls (Typhoon template): the first delta carries {index:0,id,type:'function',function:{name,arguments:'{'}}; later deltas carry only index plus argument fragments (Thai split mid-word); the final chunk has finish_reason 'tool_calls'. Content deltas also split Thai combining marks into separate chunks (e.g. 'โอ' then '้').  
  Source: raw SSE captured from local llama-server
- ✅ Thai token cost with the Typhoon/Qwen3 tokenizer: a 100-character Thai sentence = 51 tokens (1.96 chars/token). The English equivalent = 27 tokens.  
  Source: local /tokenize on typhoon2.5-qwen3-4b
- ✅ llama-server /health returns 503 {"error":{"code":503,"message":"Loading model"}} while loading and 200 {"status":"ok"} when ready. It is public (no API key). /v1/chat/completions/control with action 'reasoning_end' can end the thinking phase early (needs reasoning_control:true).  
  Source: tools/server/README.md
- ✅ openai-python latest is 3.19.2 (2026-09-24), requires Python>=3.10. 3.0.0 (2026-08-12) switched to HTTPX2: httpx is no longer installed, certifi is dropped and the OS trust store is used (SSL_CERT_FILE honoured), respx cannot intercept, and the replacement is httpx2.MockTransport. Default timeout is 10 min. Stream consumption is not auto-retried. Stream/AsyncStream have close().  
  Source: https://pypi.org/pypi/openai/json ; https://raw.githubusercontent.com/openai/openai-python/main/CHANGELOG.md ; .../httpx2.md ; README.md
- ✅ openai 3.19.2 does not validate streamed chunks. A tool_call delta with no 'index' yields tc.index == None (no exception). Unknown fields such as extra_content are available via tc.model_extra.  
  Source: local test with httpx2.MockTransport (test_stream.py)
- ✅ Gemini OpenAI-compat base_url https://generativelanguage.googleapis.com/v1beta/openai/ (API key as Bearer). Current stable text models: gemini-3.8-flash (2026-09-02), gemini-3.7-flash, gemini-3.6-flash, gemini-3.5-flash, gemini-3.5-flash-lite (2026-07-21), gemini-3.1-flash-lite (shutdown ≥2027-05-07). Preview: gemini-3.1-pro-preview, gemini-3-flash-preview. Alias gemini-flash-latest. gemini-2.0-flash and gemini-2.0-flash-001 were shut down 2026-06-01; the recommended replacement is gemini-3.6-flash. 2.5 models are limited to prior users.  
  Source: https://ai.google.dev/gemini-api/docs/models.md.txt ; /deprecations.md.txt
- ✅ Gemini thinking defaults: 3.8-flash On (medium), levels low|medium|high; 3.7-flash same; 3.6-flash and 3.5-flash On (medium), minimal..high; 3.5-flash-lite On (minimal), minimal..high. Per the OpenAI-compat docs, 'Reasoning cannot be turned off for Gemini 2.5 Pro or 3 models'. reasoning_effort minimal|low|medium|high maps to thinking_level; extra_body {'extra_body':{'google':{'thinking_config':{...}}}} is also accepted.  
  Source: https://ai.google.dev/gemini-api/docs/thinking.md.txt ; /openai.md.txt
- ✅ Gemini pricing, paid tier per 1M tokens: 3.8/3.7/3.6-flash $0.75 in / $3.75 out through 2026-12-31 (doubling 2027-01-01); 3.5-flash-lite $0.30/$2.50; 3.5-flash $1.50/$9.00. The free tier is 'Free of charge', but 'Used to improve our products: Yes'. Free-tier RPM/RPD are only shown in AI Studio.  
  Source: https://ai.google.dev/gemini-api/docs/pricing.md.txt ; /rate-limits.md.txt
- ⚠️ unverified — Free-tier daily quotas as of Sept 2026 (secondary source): 3.8/3.7/3.6/3.5 Flash about 20 requests/day; 3.5/3.1 Flash-Lite about 500 requests/day.  
  Source: https://www.scriptbyai.com/gemini-api-free-tier-limits/
- ⚠️ unverified — Gemini OpenAI-compat streamed tool calls arrive as a complete call in one chunk, with NO tool_calls[].index and with tool_calls[].extra_content.google.thought_signature. The signature must be echoed back verbatim in the next request's assistant tool_calls, or multi-turn tool use fails with HTTP 400.  
  Source: https://github.com/THU-MAIC/OpenMAIC/issues/1556 ; https://github.com/vellum-ai/vellum-assistant/issues/42421 ; ai.google.dev openai.md (mentions thought signature support)
- ✅ Ollama: latest stable v0.34.4 (2026-09-23; v0.40.0 pre-release 2026-09-25). OpenAI-compatible endpoint http://localhost:11434/v1 supports streaming, tools and reasoning_effort, but NOT tool_choice. Default context is 4096 (OLLAMA_CONTEXT_LENGTH or Modelfile num_ctx). No MoE expert-offload control: issue #11772 is still open. Windows needs NVIDIA driver >= 551.61. Official tags scb10x/typhoon2.5-qwen3-30b-a3b (19 GB, Tools badge) and scb10x/typhoon2.5-qwen3-4b (2.5 GB, Tools).  
  Source: https://raw.githubusercontent.com/ollama/ollama/main/docs/api/openai-compatibility.mdx ; docs/faq.mdx ; docs/windows.mdx ; https://github.com/ollama/ollama/releases ; https://ollama.com/scb10x ; https://github.com/ollama/ollama/issues/11772
- ✅ LM Studio: latest 0.4.25 (2026-09-19). OpenAI-compatible endpoint http://localhost:1234/v1 (chat/completions, responses, embeddings, models). Tool calls stream in chunks via delta.tool_calls; the non-native format is [TOOL_REQUEST]. An n_cpu_moe slider has existed since 0.4.0, but issue #1421 (open) reports it offloads whole layers rather than only experts. Headless daemon 'llmster': Windows install 'irm https://lmstudio.ai/install.ps1 | iex', then 'lms daemon up', 'lms server start'.  
  Source: https://lmstudio.ai/changelog/lmstudio ; https://lmstudio.ai/blog/0.4.0 ; https://lmstudio.ai/docs/developer/openai-compat/tools ; https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1421
- ✅ Measured RTX 4070 (12 GB) llama.cpp results: Llama-3.1-8B Q4_K_M 3192 tok/s prompt, 76.3 tok/s gen, TTFT 415 ms; Qwen2.5-14B Q4_K_M 1692 / 37.9 / 779 ms; Llama-3.2-1B 12717 / 283 / 112 ms.  
  Source: https://www.localscore.ai/accelerator/147
- ⚠️ unverified — Community A3B-MoE-on-12GB numbers (no Typhoon-specific 4070 benchmark found): (a) RTX 3060 12GB + DDR4-2133, Qwen3.6-35B-A3B UD-Q4_K_XL, n-cpu-moe 24, q8_0 KV, 64K ctx: 38.9 tok/s gen, 413 tok/s prompt. (b) RTX 3060 12GB + 32GB DDR5, Qwen3.6-35B IQ4_NL, '-ngl all --n-cpu-moe 25 --flash-attn on -lm none -c 65536': about 51–53 tok/s. (c) RTX 4070 SUPER + DDR5-6000: 79.8–97 tok/s with MTP in llama.cpp, 110 tok/s in the ik_llama.cpp fork.  
  Source: https://insiderllm.com/guides/best-way-run-qwen-3-6-35b-moe-locally/ ; https://openclawdc.com/blog/llama-cpp-moe-offload-flags-explained/ ; https://startupfortune.com/110-toks-on-rtx-4070-super-with-qwen36-35b/
- ✅ Newer general open models exist as alternatives, with Thai quality vs Typhoon 2.5 NOT evaluated. Qwen3.5 (Feb 2026: 0.8B/2B/4B/9B/27B/35B-A3B, Apache-2.0, 201 languages): thinks by default and does not support /think or /nothink; disable with chat_template_kwargs {"enable_thinking": false}. Qwen3.6 (Apr 2026: 27B, 35B-A3B). Gemma 4 (E2B/E4B/12B/26B-A4B/31B, Apache-2.0, 140 languages, native function calling, thinking via <|think|> / enable_thinking).  
  Source: https://huggingface.co/Qwen/Qwen3.5-4B ; https://huggingface.co/Qwen/Qwen3.6-35B-A3B ; https://huggingface.co/google/gemma-4-E4B-it
- ✅ huggingface-hub 2.0.0 (2026-09-24) provides the 'hf download REPO FILE --local-dir DIR' CLI. google-genai 2.25.0, ollama-python 0.6.2, llama-cpp-python 0.3.35 are the current PyPI versions.  
  Source: https://pypi.org/pypi/<pkg>/json ; local hf --help

## Install (Windows)

```
1) NVIDIA driver: install the current Game Ready/Studio driver, R580 or newer. That lets you use the CUDA 13.4 build; the CUDA 12.4 build also runs on older R55x+ drivers.

2) llama.cpp prebuilt binaries (PowerShell). Pin the tag and bump it deliberately; b11177 was the latest on 2026-09-25.
  $tag = "b11177"; $dst = "C:\ai\llama.cpp"; New-Item -ItemType Directory -Force $dst | Out-Null
  Invoke-WebRequest "https://github.com/ggml-org/llama.cpp/releases/download/$tag/llama-$tag-bin-win-cuda-13.4-x64.zip" -OutFile "$env:TEMP\llama.zip"
  Invoke-WebRequest "https://github.com/ggml-org/llama.cpp/releases/download/$tag/cudart-llama-bin-win-cuda-13.4-x64.zip" -OutFile "$env:TEMP\cudart.zip"
  Expand-Archive "$env:TEMP\llama.zip" $dst -Force; Expand-Archive "$env:TEMP\cudart.zip" $dst -Force   # the cudart DLLs must sit next to llama-server.exe
  & "$dst\llama-server.exe" --version; & "$dst\llama-server.exe" --list-devices   # should list CUDA0: NVIDIA GeForce RTX 4070
  For an older driver, use the same two zips with "cuda-12.4" in place of "cuda-13.4".
  Alternative: winget install llama.cpp (auto-updated; the GPU backend of the winget package was not verified, and pinning versions is harder).

3) Models (pip install -U "huggingface_hub>=2.0"):
  hf download mradermacher/typhoon2.5-qwen3-30b-a3b-GGUF typhoon2.5-qwen3-30b-a3b.Q4_K_M.gguf --local-dir C:\ai\models
  hf download mradermacher/typhoon2.5-qwen3-4b-GGUF typhoon2.5-qwen3-4b.Q6_K.gguf --local-dir C:\ai\models
  hf download typhoon-ai/typhoon2.5-qwen3-4b chat_template.jinja --local-dir C:\ai\models\typhoon25   # needed only if you use the official typhoon-ai GGUFs
  The official typhoon-ai/*-gguf files have no embedded template; add --chat-template-file C:\ai\models\typhoon25\chat_template.jinja when using them.

4) DEFAULT server, port 8080 (30B-A3B, automatic MoE placement that leaves VRAM for OBS/VTube Studio/TTS/STT). Start it AFTER the TTS/STT GPU models are loaded, because --fit measures free VRAM at launch.
  C:\ai\llama.cpp\llama-server.exe -m C:\ai\models\typhoon2.5-qwen3-30b-a3b.Q4_K_M.gguf -a pailin-30b --host 127.0.0.1 --port 8080 -c 16384 -np 2 -kvu -fa on -ctk q8_0 -ctv q8_0 --fit on --fit-target 4096 -lm none -t 8 --temp 0.6 --top-p 0.95 --repeat-penalty 1.05 --cache-reuse 256 --slot-save-path C:\ai\kv --metrics
  Pinned/reproducible variant (after tuning): replace '--fit on --fit-target 4096' with '-fit off -ngl all --n-cpu-moe 34'.
  Tuning --n-cpu-moe: lower N gives more experts on the GPU (faster, more VRAM). Each step is about 0.36 GB of VRAM for Q4_K_M. --cpu-moe (all experts on CPU) needs only about 2.5–3 GB of VRAM.
  -t 8 matches the 8 P-cores of the i7-14700KF. This is a heuristic; sweep with llama-bench.exe -m <gguf> -ngl 99 -ncmoe 30,34,38 -t 6,8,12,16 -fa 1 -p 512 -n 128.

5) LOW-LATENCY server, port 8081 (4B fully on the GPU):
  C:\ai\llama.cpp\llama-server.exe -m C:\ai\models\typhoon2.5-qwen3-4b.Q6_K.gguf -a pailin-4b --host 127.0.0.1 --port 8081 -c 16384 -np 2 -kvu -ngl all -fa on --temp 0.6 --top-p 0.95 --repeat-penalty 1.05 --cache-reuse 256

6) Optional router mode (one process, hot-load by 'model' name): llama-server.exe --models-preset C:\ai\presets.ini --models-max 2 --port 8080. INI sections [pailin-30b] and [pailin-4b], each with model = C:\ai\models\...gguf and the flags above as key = value, plus load-on-startup = true.

7) Python on Windows (3.11 or 3.12 recommended):
  py -3.12 -m venv .venv; .venv\Scripts\pip install "openai==3.19.2" "huggingface_hub>=2.0" pytest pytest-asyncio   # httpx2 comes in as an openai dependency
  Set cloud keys: setx TYPHOON_API_KEY "..."; setx GEMINI_API_KEY "...".

8) Easy-mode alternatives:
  Ollama: install OllamaSetup.exe; setx OLLAMA_CONTEXT_LENGTH 16384; ollama pull scb10x/typhoon2.5-qwen3-4b; endpoint http://localhost:11434/v1.
  LM Studio 0.4.25: irm https://lmstudio.ai/install.ps1 | iex ; lms daemon up ; lms get <model> ; lms server start ; endpoint http://localhost:1234/v1.

9) Supervision: run each llama-server as a child process of the core (or with NSSM), poll /health, and restart on exit. Bind to 127.0.0.1, or set --api-key if another machine must reach it.
```

## API notes

== Endpoints ==
- Local llama-server: base_url http://127.0.0.1:8080/v1, any api_key (e.g. "sk-local"). The model field is ignored in single-model mode and used for routing in router mode.
- Typhoon: base_url https://api.opentyphoon.ai/v1, env TYPHOON_API_KEY, model "typhoon-v2.5-30b-a3b-instruct". Send extra_body {"repetition_penalty": 1.05}. Do NOT call client.models.list(): the endpoint returns a bare array and the SDK crashes.
- Gemini: base_url https://generativelanguage.googleapis.com/v1beta/openai/, env GEMINI_API_KEY, model "gemini-3.5-flash-lite" with reasoning_effort="minimal" (or "gemini-3.8-flash" with "low").

== llama-server request (Typhoon, verified shape) ==
POST /v1/chat/completions
{"model":"pailin-30b","stream":true,"max_tokens":300,
 "messages":[{"role":"system","content":"<persona + static rules>"},{"role":"user","content":"[chat] tom123: ..."}],
 "tools":[{"type":"function","function":{"name":"set_stream_title","description":"Change the live stream title","parameters":{"type":"object","properties":{"title":{"type":"string"}},"required":["title"]}}}],
 "parallel_tool_calls":true, "id_slot":0, "cache_prompt":true}
Optional llama.cpp-only fields: "return_progress":true (prompt-processing progress), "timings_per_token":true, "chat_template_kwargs":{"enable_thinking":false} (Qwen3.5/3.6/Gemma4 only), "response_format":{"type":"json_schema","schema":{...}} (grammar-constrained, useful for a "which chat message to answer" classifier).

Captured SSE (tool call):
data: {"choices":[{"index":0,"delta":{"role":"assistant","content":null}}],...}
data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"HC6M...","type":"function","function":{"name":"set_stream_title","arguments":"{"}}]}}],...}
data: {"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"title\": \"ไพ"}}]}}],...}
data: {"choices":[{"finish_reason":"tool_calls","index":0,"delta":{}}],...,"timings":{"cache_n":288,"prompt_n":1,...}}
data: [DONE]

Tool round trip: append the assistant message as returned (m.model_dump(exclude_none=True)), then one {"role":"tool","tool_call_id":id,"content":"<json string>"} per call, in call order. Typhoon drops tool_call_id, so order is everything.

Gemini-shaped chunk: {"delta":{"role":"assistant","tool_calls":[{"id":"call_1","type":"function","function":{"name":"x","arguments":"{...}"},"extra_content":{"google":{"thought_signature":"..."}}}]}}. There is no index. Echo extra_content back in the assistant tool_calls on the next turn.

== Python (tested with openai 3.19.2 + httpx2 2.13.1; 3 pytest cases pass with httpx2.MockTransport: connect-error fallback, Gemini no-index + signature, first-token-timeout fallback) ==
Full file: /tmp/claude-0/-home-user-AI-Vtube/02efebb2-c032-5a2c-81bb-f71e8a5461a3/scratchpad/research/brain/llm_client.py (+ test_llm_client.py). Core:

import asyncio, json, os, time
from dataclasses import dataclass, field
from typing import Any
import httpx2
from openai import AsyncOpenAI, APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

@dataclass
class Provider:
    name: str; base_url: str; model: str
    api_key_env: str | None = None; extra_body: dict = field(default_factory=dict)
    first_token_timeout: float = 4.0; keeps_thought_signatures: bool = False
    http_client: Any = None; _client: AsyncOpenAI | None = None; down_until: float = 0.0; fails: int = 0
    def client(self):
        if self._client is None:
            key = os.environ.get(self.api_key_env, "") if self.api_key_env else "sk-local"
            self._client = AsyncOpenAI(api_key=key or "missing", base_url=self.base_url, max_retries=0,
                                       timeout=httpx2.Timeout(60.0, connect=2.0), http_client=self.http_client)
        return self._client

@dataclass
class TextDelta: text: str
@dataclass
class ToolCall: id: str; name: str; arguments: dict; extra: dict | None
@dataclass
class Done: provider: str; finish_reason: str | None; ttft_ms: float
class ProviderFailed(Exception): ...

async def stream_turn(p, messages, tools=None, max_tokens=300):
    msgs = messages if p.keeps_thought_signatures else [_strip_extra(m) for m in messages]
    kw = dict(model=p.model, messages=msgs, stream=True, max_tokens=max_tokens, extra_body=p.extra_body or None)
    if tools: kw["tools"] = tools
    t0 = time.perf_counter(); ttft = None; calls = {}; order = []; finish = None
    stream = await p.client().chat.completions.create(**kw)
    it = stream.__aiter__()
    try:
        while True:
            try:
                chunk = await (asyncio.wait_for(it.__anext__(), p.first_token_timeout) if ttft is None else it.__anext__())
            except StopAsyncIteration:
                break
            if not chunk.choices: continue
            ch = chunk.choices[0]; d = ch.delta; finish = ch.finish_reason or finish
            if d.content:
                ttft = ttft or (time.perf_counter() - t0) * 1e3; yield TextDelta(d.content)
            for pos, tc in enumerate(d.tool_calls or []):
                ttft = ttft or (time.perf_counter() - t0) * 1e3
                idx = getattr(tc, "index", None)                      # Gemini omits index
                key = idx if idx is not None else (tc.id or f"pos{pos}")
                if key not in calls:
                    calls[key] = {"id": tc.id or f"call_{len(calls)}", "name": "", "args": "", "extra": None}; order.append(key)
                c = calls[key]
                if tc.function and tc.function.name and not c["name"]: c["name"] = tc.function.name
                if tc.function and tc.function.arguments: c["args"] += tc.function.arguments
                if (tc.model_extra or {}).get("extra_content"): c["extra"] = tc.model_extra["extra_content"]
    finally:
        await stream.close()      # interruption path: server sees disconnect -> cancels task (verified)
    for k in order:
        c = calls[k]
        try: args = json.loads(c["args"] or "{}")
        except json.JSONDecodeError: args = {"_raw": c["args"]}      # validate against the schema before executing
        yield ToolCall(c["id"], c["name"], args, c["extra"])
    yield Done(p.name, finish, ttft or -1.0)

def _strip_extra(m):
    if m.get("tool_calls"):
        m = dict(m); m["tool_calls"] = [{k: v for k, v in tc.items() if k != "extra_content"} for tc in m["tool_calls"]]
    return m

class FallbackChain:
    def __init__(self, providers): self.providers = providers
    def use(self, name): self.providers.sort(key=lambda p: p.name != name)   # live hot-swap, others stay warm
    async def run(self, messages, tools=None, **kw):
        last = None
        for p in self.providers:
            if time.monotonic() < p.down_until: continue
            emitted = False
            try:
                async for ev in stream_turn(p, messages, tools, **kw):
                    emitted = True; yield ev
                p.fails = 0; return
            except (APIConnectionError, APITimeoutError, RateLimitError, asyncio.TimeoutError) as e: last = e
            except APIStatusError as e:
                if e.status_code < 500 and e.status_code not in (401, 403, 404, 408, 409): raise
                last = e
            p.fails += 1; p.down_until = time.monotonic() + min(60.0, 2.0 ** p.fails)
            if emitted: raise ProviderFailed(f"{p.name} died mid-utterance") from last
        raise ProviderFailed("all providers down") from last

chain = FallbackChain([
    Provider("pailin-30b", "http://127.0.0.1:8080/v1", "pailin-30b", extra_body={"id_slot": 0, "parallel_tool_calls": True}),
    Provider("pailin-4b",  "http://127.0.0.1:8081/v1", "pailin-4b",  extra_body={"id_slot": 0, "parallel_tool_calls": True}, first_token_timeout=2.0),
    Provider("typhoon-api", "https://api.opentyphoon.ai/v1", "typhoon-v2.5-30b-a3b-instruct", "TYPHOON_API_KEY", {"repetition_penalty": 1.05}, 6.0),
    Provider("gemini", "https://generativelanguage.googleapis.com/v1beta/openai/", "gemini-3.5-flash-lite", "GEMINI_API_KEY", {"reasoning_effort": "minimal"}, 8.0, keeps_thought_signatures=True),
])

# Interruption from the speech arbiter (critical priority):
#   task = asyncio.create_task(consume(chain.run(msgs, tools)));  ...;  task.cancel()   -> finally: stream.close()
# Health: httpx2.get("http://127.0.0.1:8080/health") -> 200 {"status":"ok"} | 503 loading.
# Prompt-cache persistence across restarts: POST /slots/0?action=save {"filename":"pailin.bin"} and ?action=restore (needs --slot-save-path).

Unit testing (Linux CI, no GPU): inject http_client=httpx2.AsyncClient(transport=httpx2.MockTransport(handler)) and return text/event-stream bodies. respx does not intercept openai>=3.

Integration smoke test on CI without a GPU (optional; about 2.5 GB download): build llama-server for CPU (cmake -B build -DLLAMA_CURL=OFF && cmake --build build --target llama-server) and run typhoon2.5-qwen3-4b Q4_K_M with --chat-template-file. The 4-core container gave 88 tok/s prompt processing and 7.8 tok/s generation, which is enough for tool-call parsing tests.

## Latency & resources

Hardware facts: RTX 4070 has 12 GB GDDR6X at about 504 GB/s. DDR5 dual-channel is about 60–90 GB/s in practice. PCIe 4.0 x16.

Typhoon2.5-Qwen3-4B (fully on GPU):
- Weights: Q4_K_M 2.50 GB / Q6_K 3.31 GB.
- KV cache: 147,456 B/token at f16 (36 layers × 8 KV heads × 128 × 2 × 2 B). 16K context = 2.25 GiB f16 or about 1.2 GiB q8_0. Total VRAM about 4–6 GB including compute buffers.
- Estimate (scaled from localscore RTX 4070 measurements, 8B Q4_K_M = 3192 tok/s prompt / 76 tok/s gen): about 120–150 tok/s generation, about 5–6k tok/s prompt processing.
- Time to first token: warm cache (≤200 new tokens) under 100 ms; cold 3K-token prompt about 0.5–0.7 s.

Typhoon2.5-Qwen3-30B-A3B Q4_K_M (18.56 GB), MoE offload:
- Rough VRAM model: about 1 GB dense (attention, output head, norms, routers) + KV + about 0.5 GB compute + (48 − N_cpu_moe) × about 0.36 GB of experts.
- KV cache: 98,304 B/token f16 (48 × 4 × 128 × 2 × 2), so 16K = 1.5 GiB f16 or about 0.8 GiB q8_0.
- --cpu-moe (all experts in RAM): about 2.5–3 GB VRAM, about 17 GB RAM.
- 7 GB LLM budget: N ≈ 34–36. 9 GB budget: N ≈ 29–31.
- Decode per token reads about 8 experts × 2.8 MB × N layers from RAM: about 0.85 GB/token at N = 36, which puts the RAM-bandwidth ceiling around 70–90 tok/s.
- Estimate for 4070 + DDR5: 30–50 tok/s generation and 300–800 tok/s prompt processing. Community A3B data: 38.9 tok/s on a 3060 with DDR4-2133; 51–53 tok/s on a 3060 with DDR5; 80–97 tok/s on a 4070 SUPER with MTP; all Qwen3.6-35B-A3B, not Typhoon.
- Time to first token: warm cache with 100–300 new tokens about 0.3–0.8 s. A cold 3–4K-token prompt takes about 4–10 s, which is why prompt caching and pinned slot 0 are mandatory.
- Model load: about 18.6 GB read from NVMe on first start, faster from the OS cache with 64 GB RAM. Use -lm none to avoid mmap page-fault stalls on Windows.

Thai throughput:
- Measured 1.96 Thai chars/token, about 1.9× the tokens of equivalent English.
- A 20-token first phrase is about 40 characters, so first-audio time ≈ TTFT + 20/gen_rate: about 0.15 s on the 4B and about 0.5–1.2 s on the 30B, before TTS.

Cloud:
- Typhoon API: network plus a shared GPU. Latency was not measured (no key); expect roughly 0.5–2 s TTFT.
- Gemini 3.5 Flash-Lite with minimal thinking: typically sub-second to about 1 s TTFT (not measured).

Coexistence:
- STT (Typhoon ASR realtime is CPU) and TTS (GPU models such as JaiTTS/F5 need about 2–4 GB), plus OBS/VTube Studio (about 1 GB), must fit beside the LLM.
- Set --fit-target to the VRAM you want left free, or pin --n-cpu-moe. During GPU-heavy games switch to pailin-4b, or run the 30B with --cpu-moe.

CPU:
- Expert matmuls run on the P-cores (-t 8). The E-cores stay free for STT, OBS x264 (if used), and Python.

## Pitfalls

- The official typhoon-ai GGUFs have no embedded chat template. llama-server then silently falls back to a tool-less ChatML template: tools are ignored, the model hallucinates having acted, and output degenerates (verified). Always use --chat-template-file with the repo's chat_template.jinja, or use the mradermacher GGUFs, and assert /props chat_template_caps.supports_tool_calls == true at startup.
- The Typhoon template renders any non-string message content (OpenAI content-part arrays) as an empty string. Always send plain-string content to Typhoon providers.
- Tool results carry no tool_call_id in the Typhoon template (they become <tool_response> blocks in a user turn). Keep results in exactly the call order.
- The model may call tools without speaking first even when told to speak (content '' + tool_calls). Design the loop as tool call → execute → second completion for speech, or synthesize a short acknowledgement yourself.
- The model emits emoji and markdown (e.g. '🏡✨'). Strip them before TTS and forbid them in the system prompt.
- Streamed Thai text deltas split grapheme clusters: combining vowel/tone marks arrive as separate chunks ('โอ' + '้'). Never send raw deltas to TTS. Buffer and segment on Thai phrase boundaries (space, punctuation, or a word segmenter such as PyThaiNLP) and never cut before a combining mark (U+0E31, U+0E34–0E3A, U+0E47–0E4E).
- Streamed tool-call arguments arrive fragmented (llama.cpp) or complete in one chunk with no index (Gemini). Key the accumulator on index, else id, else position. Parse JSON only after the stream ends, and validate args against the tool schema before executing (model output is untrusted, e.g. timeout_user).
- Gemini 3 needs tool_calls[].extra_content.google.thought_signature echoed back verbatim in multi-turn tool use, otherwise HTTP 400. Strip it for every other provider.
- Gemini: reasoning cannot be turned off for 3.x models. Use reasoning_effort 'minimal' on 3.5-flash-lite/3.6-flash/3.5-flash; 3.8-flash and 3.7-flash only go down to 'low'. gemini-2.0-flash (the old code's model) has been shut down since 2026-06-01, and 2.5 models are gated to prior users.
- The free-tier Gemini quotas (about 20 requests/day for Flash, about 500 for Flash-Lite, secondary source) cannot sustain a stream, and free-tier data is used by Google. Use a billing-enabled key for the fallback.
- Typhoon API: /v1/models returns a bare array, so openai client.models.list() crashes (verified). Error bodies are {'detail': ...}, not OpenAI-shaped. The docs list typhoon-v2.1-12b-instruct but it is absent from the live model list. The docs also mention a max_tokens ceiling of 8192 'shared between prompt and completion'. Inputs may be used for training, and there is no SLA.
- llama.cpp flags changed in 2026: --no-mmap/--mlock were removed (use -lm none|mlock|mmap+mlock); --jinja is on by default; -ngl defaults to auto; --fit is on by default. Old guides and scripts break or misbehave.
- --fit (default on) measures free VRAM at launch and will shrink the context if -c is unset. It also refuses to place weights if you pass --n-cpu-moe/-ot (warning only). Use either fit mode (set -c and --fit-target) or manual mode (-fit off -ngl all --n-cpu-moe N), not both, and launch after the other GPU models are loaded.
- Setting -np N explicitly disables the unified KV and gives each slot c/N context (verified 4096/2 → 2048). Add -kvu or size -c accordingly.
- Prompt-cache hygiene: any change early in the prompt (system prompt, tool list, memory block) invalidates the cache and costs seconds on the 30B. Keep persona and tools byte-stable. Put long-term memory next, then append-only history, then the volatile chat window and game state last. Pin the speaking loop to id_slot 0 and send background jobs to id_slot 1.
- Extreme KV quantisation (q4_0) degrades tool calling (llama.cpp docs). Stay at f16 or q8_0 KV.
- Do not fall back to another provider mid-utterance. The persona or voice can change and the listener hears a restart. Only retry if nothing has been emitted to TTS.
- openai>=3.0 uses httpx2: httpx is not installed, respx will not intercept, and the OS trust store replaces certifi. Corporate proxies or odd Windows setups may need SSL_CERT_FILE. Use httpx2.MockTransport in unit tests. Python must be >= 3.10.
- Always close streams in a finally block (await stream.close()). Otherwise a cancelled asyncio task can leave the server generating into a dead socket until the next write; closing triggers the server-side cancel in ≤0.2 s (verified).
- Ollama defaults to a 4096-token context, does not support tool_choice, and has no expert-only MoE offload, so the 30B-A3B either spills whole layers or needs more VRAM. LM Studio's n_cpu_moe slider reportedly offloads whole layers (issue #1421).
- Windows: bind llama-server to 127.0.0.1, or add --api-key, because the default CORS allows all origins (the server warns about this). Windows Defender may slow first loads of 18 GB GGUFs; add an exclusion for the models folder.
- Thai costs about 2 chars/token, so max_tokens budgets for speech should be set about 2× what you would use for English.

## Open questions

- No measured Typhoon2.5-30B-A3B numbers exist for an RTX 4070 + DDR5. Run llama-bench on the user's PC (sweep -ncmoe 28..40, -t 6/8/12) to pin N and confirm the 30–50 tok/s estimate. Record the RAM speed (DDR5-5600 vs 6000+) and whether XMP is on.
- How much VRAM will the chosen TTS (JaiTTS/VoxCPM or F5-based ThonburianTTS) and STT (Typhoon ASR CPU vs typhoon-whisper-turbo GPU) actually use? This sets --fit-target / --n-cpu-moe. Also: will games run on the same GPU during streams?
- Thai persona quality: Typhoon2.5-30B-A3B vs Typhoon2.5-4B vs Qwen3.6-35B-A3B vs Gemma-4 (no public head-to-head found). Build a small Thai VTuber eval covering chat banter, code-switching with English names, tool-call accuracy and safety refusals.
- Typhoon API behaviour for stream=true + tools (fragmented or complete deltas? index present?), the real max context, and the actual max_tokens cap. These need a real API key to verify.
- The user's actual Gemini free-tier quota for their project (shown only in AI Studio), and whether they will enable billing for the fallback.
- Whether --cache-reuse (KV shifting) gives measurable gains with this prompt layout, versus relying only on prefix caching plus --cache-ram host cache; measure cache_n in timings.
- Which backend the winget llama.cpp package ships (CUDA vs Vulkan), and the exact Windows driver minimum for the cuda-12.4 zip (not verified).
- Whether to keep both local models resident, which needs VRAM for the 4B (about 4–5 GB) plus the 30B dense part and KV (about 3–4 GB), or use router mode with swap-on-demand. Swap latency from the RAM page cache was not measured.
