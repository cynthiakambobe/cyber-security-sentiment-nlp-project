"""
Import the DistilBERT artifact ZIP downloaded from the Colab notebook.

Usage from the repository root:
    python src/05_import_distilbert_colab.py distilbert_colab_output.zip

The archive is extracted to models/distilbert. Existing contents are preserved;
the command stops if the destination is non-empty.
"""

import argparse
import json
import shutil
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DESTINATION = PROJECT_ROOT / "models" / "distilbert"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Import DistilBERT artifacts downloaded from Google Colab."
    )
    parser.add_argument("archive", type=Path, help="Path to distilbert_colab_output.zip")
    return parser.parse_args()


def main():
    args = parse_args()
    archive = args.archive.expanduser().resolve()

    if not archive.exists():
        raise FileNotFoundError(archive)
    if not zipfile.is_zipfile(archive):
        raise ValueError(f"Not a valid ZIP archive: {archive}")

    if DESTINATION.exists() and any(DESTINATION.iterdir()):
        raise FileExistsError(
            f"{DESTINATION} is not empty. Move or remove it before importing."
        )

    DESTINATION.mkdir(parents=True, exist_ok=True)
    shutil.unpack_archive(str(archive), str(DESTINATION), "zip")

    required = [
        DESTINATION / "model" / "config.json",
        DESTINATION / "test_predictions.csv",
        DESTINATION / "test_metrics.json",
        DESTINATION / "confusion_matrix.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            "Archive extracted but required files are missing:\n"
            + "\n".join(missing)
        )

    with open(DESTINATION / "test_metrics.json", encoding="utf-8") as handle:
        metrics = json.load(handle)

    print(f"Imported DistilBERT artifacts to {DESTINATION}")
    print(f"Test accuracy: {metrics['test_accuracy']:.3f}")
    print(f"Test macro-F1: {metrics['test_macro_f1']:.3f}")


if __name__ == "__main__":
    main()
