"""Download and cache sentence-transformers/all-MiniLM-L6-v2 locally."""

import os
import sys
from pathlib import Path

def main():
    print("Initializing model downloader for local embeddings...")
    project_root = Path(__file__).resolve().parent.parent
    target_dir = project_root / "local_models" / "all-MiniLM-L6-v2"

    target_dir.parent.mkdir(parents=True, exist_ok=True)

    if (target_dir / "modules.json").exists():
        print(f"Model already downloaded and cached at: {target_dir}")
        return

    print("Downloading 'sentence-transformers/all-MiniLM-L6-v2' (Apache 2.0)...")
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        model.save(str(target_dir))
        print(f"Successfully saved local model to: {target_dir}")
    except Exception as e:
        print(f"Error downloading model: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
