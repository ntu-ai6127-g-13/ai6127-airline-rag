"""Download the three models needed for Linux RAG; record their exact revisions."""
from pathlib import Path
import json

from huggingface_hub import HfApi, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "embedding": "sentence-transformers/all-MiniLM-L6-v2",
    "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "generator": "Qwen/Qwen3-4B-Instruct-2507",
}


def main():
    manifest_path = ROOT / "outputs/minimal_rag_models.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    api = HfApi()
    for role, model_id in MODELS.items():
        previous = manifest.get(role, {})
        revision = previous.get("revision") or api.model_info(model_id).sha
        print(f"Downloading {role}: {model_id}@{revision}", flush=True)
        path = snapshot_download(
            model_id, revision=revision, cache_dir=ROOT / "models/huggingface",
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"],
        )
        manifest[role] = {"model_id": model_id, "revision": revision, "path": path}
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
        print(f"Ready: {role}", flush=True)


if __name__ == "__main__":
    main()
