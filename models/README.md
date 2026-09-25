# models/

Downloaded models live here. Everything in this folder is git-ignored except this README and
`manifest.toml`. Nothing here is needed to run the tests: CI uses fakes plus the small Silero
file committed under `tests/fixtures/`.

## How files get here

`setup.ps1` runs `aivtube setup`, which reads `manifest.toml` and downloads, in this order:

1. `silero_vad` (2.3 MB)
2. the Typhoon ASR Realtime int8 ONNX files (about 138 MB). Until they are published, the STT
   chain falls back to PyThaiASR.
3. the 4B GGUF (2.5 GB), so the `light` profile works quickly
4. the 30B GGUF (18.6 GB) as a resumable background download. The profile stays `light`
   until its sha256 verifies.

The llama.cpp zips go to `vendor/llama.cpp/` (also git-ignored). Setup picks the CUDA 13.4 pair
for NVIDIA driver R580 or newer and the 12.4 pair otherwise.

To download or check files by hand, run `aivtube models pull [name ...]` or
`aivtube models verify`. Setting `HF_HOME=models/hf` (which `run.bat` does) keeps the Hugging
Face cache inside this folder too.

## Layout

```
models/
├─ vad/silero_vad.onnx
├─ stt/typhoon-asr-rt-int8/{encoder,decoder,joiner}.int8.onnx, tokens.txt, README.md
├─ llm/typhoon2.5-qwen3-4b.Q4_K_M.gguf
├─ llm/typhoon2.5-qwen3-30b-a3b.Q4_K_M.gguf
└─ hf/                      (Hugging Face cache, when used)
```

## The manifest

Every artifact has a `[models.<name>]` table with `url`, `sha256`, `size` (bytes), `licence`,
`required_by` and `dest`. Optional keys are `variant`, `unpack`, `optional` and `notes`.
URLs are pinned to a tag or a commit, never to a moving branch.

An empty `sha256` (or a `size` of 0) means "not pinned yet". `verify` reports such a file as
unpinned rather than corrupt. Setup prints the digest it computed, so it can be pinned in a
follow-up commit. This applies to the Typhoon RT ONNX files until the first run of the
`models-export` workflow publishes its `SHA256SUMS`, because ONNX export is not
byte-reproducible across environments.

## Publishing the Typhoon RT export

The manual GitHub Actions workflow **models-export** (`.github/workflows/models-export.yml`):

1. installs torch (CPU) and `nemo_toolkit[asr]==3.0.0` into a throwaway venv;
2. runs `tools/export_typhoon_rt_onnx.py` against a pinned revision of
   `typhoon-ai/typhoon-asr-realtime`;
3. verifies the result in the project's own runtime (sherpa-onnx 1.13.8) on a generated Thai
   sample (`tools/verify_typhoon_rt_onnx.py`);
4. uploads `encoder.int8.onnx`, `decoder.int8.onnx`, `joiner.int8.onnx`, `tokens.txt`,
   `SHA256SUMS` and a `README.md` with the CC-BY-4.0 attribution to the GitHub Release
   `models-typhoon-rt-v1`.

After it runs, copy the digests and sizes from the release's `SHA256SUMS` into
`manifest.toml`.

## Licences

| Artifact | Licence | Notes |
|---|---|---|
| Silero VAD | MIT | © Silero Team |
| Typhoon ASR Realtime (our int8 export) | CC-BY-4.0 | Attribution to the Typhoon team at SCB 10X is required; see `NOTICE` |
| Typhoon2.5-Qwen3 4B / 30B-A3B GGUF | Apache-2.0 | Quantised by mradermacher; OpenTyphoon terms apply |
| llama.cpp | MIT | ggml-org |
| CUDA runtime (cudart zip) | NVIDIA CUDA EULA | Redistributable runtime components |
| Piper `th_TH-tsync2-medium` (optional, M1 fallback) | Non-commercial | `doctor` warns when `monetised = true` |

Do not commit model files, and do not use the third-party `edtzforai/typhoon-asr-int8-onnx`
copy: it ships without a licence or attribution.
