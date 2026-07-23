import os
from ultralytics import YOLO

N_SAMPLES = 5
CONF_THRESHOLD = 0.25
CLASS_NAMES = ["with_mask", "without_mask", "mask_weared_incorrect"]


def main():
    print("=" * 60)
    print("Projeto 3 — Inferência com model.tflite (Edge AI)")
    print("=" * 60)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.tflite")
    model = YOLO(model_path, task="detect")

    val_dir = os.path.join(script_dir, "dataset", "images", "val")
    all_images = sorted(
        [f for f in os.listdir(val_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    )
    sample_paths = [os.path.join(val_dir, f) for f in all_images[:N_SAMPLES]]

    print(f"\nRodando inferência em {len(sample_paths)} amostras usando model.tflite:\n")
    print(f"{'Imagem':<35} {'Detecções':>10}  Detalhes")
    print("-" * 70)

    total_detections = 0

    for path in sample_paths:
        result = model.predict(
            source=path,
            imgsz=640,
            conf=CONF_THRESHOLD,
            iou=0.45,
            save=True,
            project=os.path.join(script_dir, "runs", "detect"),
            name="inferencia_exemplos/predicoes",
            exist_ok=True,
            verbose=False,
        )[0]

        confs = result.boxes.conf.tolist()
        cls_ids = result.boxes.cls.tolist()
        kept = [(c, k) for c, k in zip(confs, cls_ids) if c >= CONF_THRESHOLD]
        n_det = len(kept)
        total_detections += n_det

        if n_det > 0:
            class_counts = {}
            for _, cid in kept:
                cname = CLASS_NAMES[int(cid)] if int(cid) < len(CLASS_NAMES) else str(int(cid))
                class_counts[cname] = class_counts.get(cname, 0) + 1
            details = ", ".join(f"{v}x {k}" for k, v in class_counts.items())
        else:
            details = "nenhuma detecção"

        print(f"{os.path.basename(path):<35} {n_det:>10}  [{details}]")

    print("-" * 70)
    print(f"{'TOTAL':<35} {total_detections:>10}")
    print(f"\nImagens anotadas salvas em: runs/detect/inferencia_exemplos/predicoes/")


if __name__ == "__main__":
    main()