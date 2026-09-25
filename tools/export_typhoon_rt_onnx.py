#!/usr/bin/env python3
"""Export Typhoon ASR Realtime (NeMo FastConformer-RNNT) to int8 ONNX for sherpa-onnx.

Source model: https://huggingface.co/typhoon-ai/typhoon-asr-realtime (CC-BY-4.0, SCB 10X /
Typhoon team). Output, loadable with ``sherpa_onnx.OfflineRecognizer.from_transducer(...,
model_type="nemo_transducer")``: ``encoder.int8.onnx``, ``decoder.int8.onnx``,
``joiner.int8.onnx``, ``tokens.txt``, plus ``SHA256SUMS`` and a ``README.md`` with the
CC-BY-4.0 attribution.

This needs torch + ``nemo_toolkit[asr]==3.0.0`` and runs in a throwaway venv (the
``models-export.yml`` workflow), never in the aivtube venv, which has no torch. Adapted from
k2-fsa/sherpa-onnx ``scripts/nemo/fast-conformer-hybrid-transducer-ctc/
export-onnx-transducer-non-streaming.py`` (Apache-2.0, Xiaomi Corp., Fangjun Kuang).

Usage: python tools/export_typhoon_rt_onnx.py --out dist/typhoon-rt [--revision SHA]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

REPO_ID = "typhoon-ai/typhoon-asr-realtime"
NEMO_FILE = "typhoon-asr-realtime.nemo"
DEFAULT_REVISION = "2c58a30ba9a3bf92d095a5df91bec6996f04c3a1"  # HF main on 2026-06-11
PARTS = ("encoder", "decoder", "joiner")
OUTPUTS = (*(f"{p}.int8.onnx" for p in PARTS), "tokens.txt")

README = """\
# Typhoon ASR Realtime, int8 ONNX for sherpa-onnx

These files are an ONNX export (dynamic int8 quantisation) of
[typhoon-ai/typhoon-asr-realtime]({url}) at revision `{revision}`, made by the AI_Vtube
project (https://github.com/Krx-21/AI_Vtube) for local, torch-free inference with
sherpa-onnx 1.13.8 (`OfflineRecognizer.from_transducer(..., model_type="nemo_transducer",
feature_dim=80, sample_rate=16000)`).

## Attribution and licence

- **Model:** Typhoon ASR Realtime, by the Typhoon team at SCB 10X (typhoon-ai).
- **Licence:** Creative Commons Attribution 4.0 International (CC-BY-4.0),
  https://creativecommons.org/licenses/by/4.0/ . The model card also asks users to accept the
  OpenTyphoon terms: https://opentyphoon.ai/tac
- **Changes made:** the NeMo checkpoint was exported to ONNX (encoder, decoder and joiner
  exported separately; ONNX metadata added to the encoder) and the weights were quantised to
  8-bit integers with onnxruntime `quantize_dynamic` (QUInt8). `tokens.txt` lists the
  SentencePiece vocabulary plus `<blk>`. No retraining or fine-tuning was done. Quantisation can
  change accuracy slightly compared with the original model.
- **No endorsement:** this export is not made or endorsed by SCB 10X or the Typhoon team.

Please cite the technical report if you use the model:

    @misc{{warit2026typhoonasr,
      title={{Typhoon ASR Real-time: FastConformer-Transducer for Thai Automatic Speech Recognition}},
      author={{Warit Sirichotedumrong and Adisai Na-Thalang and Potsawee Manakul and
              Pittawat Taveekitworachai and Sittipong Sripaisarnmongkol and Kunat Pipatanakul}},
      year={{2026}}, eprint={{2601.13044}}, archivePrefix={{arXiv}}, primaryClass={{cs.CL}},
      url={{https://arxiv.org/abs/2601.13044}}
    }}

## Files

{files}

Export recipe adapted from k2-fsa/sherpa-onnx
`scripts/nemo/fast-conformer-hybrid-transducer-ctc/export-onnx-transducer-non-streaming.py`
(Apache-2.0).
"""


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def add_meta_data(filename: Path, meta: dict[str, Any]) -> None:
    import onnx

    model = onnx.load(str(filename))
    while len(model.metadata_props):
        model.metadata_props.pop()
    for key, value in meta.items():
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = str(value)
    onnx.save(model, str(filename))


def export(out: Path, revision: str, keep_fp32: bool) -> None:
    import nemo.collections.asr as nemo_asr
    import torch
    from huggingface_hub import hf_hub_download
    from onnxruntime.quantization import QuantType, quantize_dynamic

    out.mkdir(parents=True, exist_ok=True)
    nemo_path = hf_hub_download(REPO_ID, NEMO_FILE, revision=revision)
    with torch.no_grad():
        asr = nemo_asr.models.ASRModel.restore_from(nemo_path, map_location="cpu")
        asr.eval()
        vocab = list(asr.joint.vocabulary)
        with (out / "tokens.txt").open("w", encoding="utf-8", newline="\n") as f:
            for i, piece in enumerate(vocab):
                f.write(f"{piece} {i}\n")
            f.write(f"<blk> {len(vocab)}\n")
        asr.encoder.export(str(out / "encoder.onnx"))
        asr.decoder.export(str(out / "decoder.onnx"))
        asr.joint.export(str(out / "joiner.onnx"))

    normalize = asr.cfg.preprocessor.normalize
    meta = {
        "vocab_size": asr.decoder.vocab_size,
        "normalize_type": "" if normalize == "NA" else normalize,
        "pred_rnn_layers": asr.decoder.pred_rnn_layers,
        "pred_hidden": asr.decoder.pred_hidden,
        "subsampling_factor": int(getattr(asr.cfg.encoder, "subsampling_factor", 8) or 8),
        "model_type": "EncDecRNNTBPEModel",
        "version": "1",
        "model_author": "typhoon-ai",
        "url": f"https://huggingface.co/{REPO_ID}",
        "license": "CC-BY-4.0",
        "comment": f"exported by AI_Vtube from revision {revision}",
    }
    print("metadata:", meta, flush=True)
    add_meta_data(out / "encoder.onnx", meta)
    for part in PARTS:
        quantize_dynamic(
            model_input=str(out / f"{part}.onnx"),
            model_output=str(out / f"{part}.int8.onnx"),
            weight_type=QuantType.QUInt8,
        )
        if not keep_fp32:
            (out / f"{part}.onnx").unlink()


def write_sums_and_readme(out: Path, revision: str) -> None:
    lines = []
    rows = []
    for name in OUTPUTS:
        path = out / name
        digest, size = sha256(path), path.stat().st_size
        lines.append(f"{digest}  {name}\n")
        rows.append(f"| `{name}` | {size} | `{digest}` |")
    (out / "SHA256SUMS").write_text("".join(lines), encoding="utf-8", newline="\n")
    table = "| File | Bytes | SHA-256 |\n|---|---|---|\n" + "\n".join(rows)
    text = README.format(url=f"https://huggingface.co/{REPO_ID}", revision=revision, files=table)
    (out / "README.md").write_text(text, encoding="utf-8", newline="\n")
    print("".join(lines), end="", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("dist/typhoon-rt"))
    ap.add_argument("--revision", default=DEFAULT_REVISION, help="Hugging Face commit to export")
    ap.add_argument("--keep-fp32", action="store_true", help="keep the fp32 ONNX files too")
    ap.add_argument(
        "--readme-only", action="store_true", help="only (re)write SHA256SUMS and README.md"
    )
    args = ap.parse_args(argv)
    if not args.readme_only:
        export(args.out, args.revision, args.keep_fp32)
    missing = [n for n in OUTPUTS if not (args.out / n).is_file()]
    if missing:
        print(f"missing outputs: {missing}", file=sys.stderr)
        return 1
    write_sums_and_readme(args.out, args.revision)
    return 0


if __name__ == "__main__":
    sys.exit(main())
