import os
import shutil
from pathlib import Path
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
MODEL_PATH = SCRIPT_DIR / "model.pt"
DATA_YAML = SCRIPT_DIR / "dataset" / "data.yaml"
OUTPUT_TFLITE = SCRIPT_DIR / "model.tflite"

IMGSZ = 416

def size_mb(path):
    return os.path.getsize(path) / (1024 * 1024)

def find_exported_tflite(export_result):
    p = Path(export_result)
    if p.is_file() and p.suffix == ".tflite":
        return p
        
    all_tflite = list(SCRIPT_DIR.glob("**/*.tflite"))
    for c in all_tflite:
        if "int8" in c.name.lower() and c.resolve() != OUTPUT_TFLITE.resolve():
            return c
    for c in all_tflite:
        if c.resolve() != OUTPUT_TFLITE.resolve():
            return c
    raise FileNotFoundError("Nenhum artefato .tflite encontrado após o export.")


def main():
    original_size = size_mb(MODEL_PATH)
    print(f"Tamanho original (model.pt): {original_size:.2f} MB\n")

    model = YOLO(str(MODEL_PATH))
    
    exported = model.export(
        format="tflite",
        imgsz=IMGSZ,
        int8=True,
        data=str(DATA_YAML),
    )

    tflite_path = find_exported_tflite(exported)

    if tflite_path.resolve() != OUTPUT_TFLITE.resolve():
        shutil.copy(tflite_path, OUTPUT_TFLITE)
        print(f"\nArtefato original em: {tflite_path}")

    optimized_size = size_mb(OUTPUT_TFLITE)
    print(f"\nTFLite INT8 salvo em: {OUTPUT_TFLITE}")
    print(f"model.pt      : {original_size:.2f} MB")
    print(f"model.tflite  : {optimized_size:.2f} MB")
    print(f"Redução       : {(1 - optimized_size / original_size) * 100:.1f}%")

if __name__ == "__main__":
    main()