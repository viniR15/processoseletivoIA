import shutil
from pathlib import Path
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_YAML = SCRIPT_DIR / "dataset" / "data.yaml"
OUTPUT_MODEL = SCRIPT_DIR / "model.pt"

EPOCHS = 15
IMGSZ = 640
BATCH = 16

def main():
    model = YOLO("yolo11n.pt")

    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        device="cpu",
        project=str(SCRIPT_DIR / "runs" / "detect"),
        name="train",
        exist_ok=True,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    shutil.copy(best_weights, OUTPUT_MODEL)
    print(f"\nPesos copiados para: {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()