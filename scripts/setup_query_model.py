"""Rebuild the optional Apple Silicon local model runtime and a zero-shot manifest."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys
import venv

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"experiments/query_quality"))
from prompts import MODEL_ID,MODEL_REVISION,PROMPTS
from runtime import sha,write_json,MLX_PYTHON


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",type=Path,default=ROOT/"outputs/query_quality_standalone/frozen_candidates.json")
    parser.add_argument("--variant",choices=["closed_choice","taxonomy","evidence_first"],default="closed_choice")
    args=parser.parse_args()
    if sys.version_info < (3,11): raise RuntimeError("Use Python 3.11 or later for the reproducibility helpers")
    if args.manifest.exists(): raise RuntimeError("Manifest already exists; preserve the existing model configuration")
    if platform.system()!="Darwin" or platform.machine()!="arm64":
        raise RuntimeError("This optional MLX runtime requires Apple Silicon; retrieval without labels remains available")
    if not MLX_PYTHON.exists(): venv.EnvBuilder(with_pip=True).create(MLX_PYTHON.parents[1])
    subprocess.run([str(MLX_PYTHON),"-m","pip","install","mlx-lm==0.31.2"],check=True)
    from huggingface_hub import snapshot_download
    path=Path(snapshot_download(MODEL_ID,revision=MODEL_REVISION,
        allow_patterns=["*.json","*.safetensors","*.txt","*.model","*.jinja","README.md"]))
    files=[*ROOT.joinpath("experiments/query_quality").glob("*.py"),ROOT/"src/quality.py"]
    manifest={"selected_without_gold":"standalone_qwen_"+args.variant, "selected_spec":{"mode":"direct","source":args.variant},
        "source_hashes":{str(p):sha(p) for p in files},"input_hashes":{},
        "model_weight_hashes":{str(p):sha(p) for p in path.glob("*.safetensors")},
        "model_id":MODEL_ID,"model_revision":MODEL_REVISION,
        "project_gold_used_for_training":False,"project_gold_used_for_selection":False,
        "production_ready":False,"selection_policy":"fixed prompt variant; no optimization or quality claim"}
    write_json(args.manifest,manifest)
    print(str(args.manifest))


if __name__=="__main__":main()
