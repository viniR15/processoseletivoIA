import os

from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Projeto 3 — Inferência com o Modelo Otimizado (model.tflite)
#
# Requisitos (veja README.md desta pasta para detalhes completos):
#   1. Carregar especificamente o "model.tflite" (o artefato de edge, não o
#      model.pt) usando YOLO(..., task="detect")
#   2. Rodar inferência em pelo menos 5 imagens do conjunto de validação
#      (dataset/images/val/), UMA DE CADA VEZ — o model.tflite exportado
#      aceita apenas 1 imagem por chamada (batch=1), que é aliás o cenário
#      real de uso em edge (uma foto por vez, não em lote)
#   3. Imprimir no terminal, para cada amostra, o número de detecções
#   4. As imagens anotadas com as caixas preditas são salvas automaticamente
#      pelo Ultralytics em runs/detect/... — abra essa pasta localmente para
#      conferir visualmente as predições (não precisa ser commitada)
# ---------------------------------------------------------------------------

N_SAMPLES = 5
CLASS_NAMES = ["with_mask", "without_mask", "mask_weared_incorrect"]


def main():
    print("=" * 60)
    print("Projeto 3 — Inferência com model.tflite (Edge AI)")
    print("=" * 60)

    # 1. Carregar o modelo TFLite (artefato de edge)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.tflite")
    model = YOLO(model_path, task="detect")

    # 2. Selecionar imagens do conjunto de validação
    val_dir = os.path.join(script_dir, "dataset", "images", "val")
    all_images = sorted(
        [f for f in os.listdir(val_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    )
    sample_images = all_images[:N_SAMPLES]
    sample_paths = [os.path.join(val_dir, f) for f in sample_images]

    print(f"\nRodando inferência em {len(sample_paths)} amostras usando model.tflite:\n")
    print(f"{'Imagem':<35} {'Detecções':>10}  Detalhes")
    print("-" * 70)

    total_detections = 0

    # 3. Inferência uma imagem por vez (batch=1 — requisito do tflite em edge)
    for path in sample_paths:
        result = model.predict(
            source=path,
            imgsz=416,
            conf=0.25,
            save=True,
            project=os.path.join(script_dir, "runs", "detect"),
            name="inferencia_exemplos/predicoes",
            exist_ok=True,
            verbose=False,
        )[0]

        n_det = len(result.boxes)
        total_detections += n_det

        # Detalhar classes detectadas
        if n_det > 0:
            class_ids = result.boxes.cls.tolist()
            class_counts = {}
            for cid in class_ids:
                cname = CLASS_NAMES[int(cid)] if int(cid) < len(CLASS_NAMES) else str(int(cid))
                class_counts[cname] = class_counts.get(cname, 0) + 1
            details = ", ".join(f"{v}x {k}" for k, v in class_counts.items())
        else:
            details = "nenhuma detecção"

        print(f"{os.path.basename(path):<35} {n_det:>10}  [{details}]")

    print("-" * 70)
    print(f"{'TOTAL':<35} {total_detections:>10}")
    print(f"\n✅ Imagens anotadas salvas em: runs/detect/inferencia_exemplos/predicoes/")
    print("   (Abra essa pasta para verificar visualmente as bounding boxes preditas)")


if __name__ == "__main__":
    main()
