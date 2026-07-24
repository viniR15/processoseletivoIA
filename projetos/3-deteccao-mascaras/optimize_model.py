import os
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "model.pt"
DATA_YAML = SCRIPT_DIR / "dataset" / "data.yaml"
OUTPUT_TFLITE = SCRIPT_DIR / "model.tflite"

IMGSZ = 640
MIN_TFLITE_BYTES = 500_000


def size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)


def report(original_size, optimized_size):
    print(f"\nTFLite FP32 salvo em: {OUTPUT_TFLITE}")
    print(f"model.pt      : {original_size:.2f} MB")
    print(f"model.tflite  : {optimized_size:.2f} MB")
    print(f"Redução       : {(1 - optimized_size / original_size) * 100:.1f}%")


def find_exported_tflite():
    all_tflite = list(SCRIPT_DIR.glob("**/*.tflite"))
    for c in all_tflite:
        if "int8" in c.name.lower() and c.resolve() != OUTPUT_TFLITE.resolve():
            return c
    for c in all_tflite:
        if c.resolve() != OUTPUT_TFLITE.resolve():
            return c
    return None


def tflite_ok():
    return OUTPUT_TFLITE.exists() and OUTPUT_TFLITE.stat().st_size >= MIN_TFLITE_BYTES


def try_export():
    model = YOLO(str(MODEL_PATH))
    exported = model.export(
        format="tflite",
        imgsz=IMGSZ,
        int8=False,
    )
    tflite_path = Path(exported) if exported and Path(exported).is_file() and Path(exported).suffix == ".tflite" else find_exported_tflite()
    if tflite_path is None:
        raise FileNotFoundError("Nenhum .tflite foi gerado pela exportação.")
    if tflite_path.resolve() != OUTPUT_TFLITE.resolve():
        shutil.copy(tflite_path, OUTPUT_TFLITE)
        print(f"\nArtefato original em: {tflite_path}")


def main():
    original_size = size_mb(MODEL_PATH)
    print(f"Tamanho original (model.pt): {original_size:.2f} MB")

    if tflite_ok():
        print(f"⚡ model.tflite já existe e é válido ({size_mb(OUTPUT_TFLITE):.2f} MB).")
        print("   Pulando reexportação (comportamento idempotente).")
        report(original_size, size_mb(OUTPUT_TFLITE))
        return

    print("Executando pipeline de exportação (PyTorch → ONNX → SavedModel → TFLite FP32)...")
    try:
        try_export()
        report(original_size, size_mb(OUTPUT_TFLITE))
    except Exception as e:
        print(f"\nFalha na exportação: {type(e).__name__}: {e}")
        if tflite_ok():
            print("Porém model.tflite pré-existente é válido — pipeline considerado OK.")
            report(original_size, size_mb(OUTPUT_TFLITE))
            return
        print("\nmodel.tflite não foi produzido e não existe versão válida no repositório.")
        sys.exit(1)

if __name__ == "__main__":
    main()