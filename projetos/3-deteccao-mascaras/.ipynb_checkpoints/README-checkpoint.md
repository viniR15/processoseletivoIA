## 📝 Relatório do Candidato

👤 **Nome Completo:** Vinicius Reis de Lemos

### 1️⃣ Resumo da Abordagem

Fine-tuning do **YOLO11n** pré-treinado sobre o dataset de detecção de máscaras. Hiperparâmetros:

- **Épocas:** 20
- **Tamanho de imagem:** 416x416 (menor que o padrão 640. A escolha foi consciente para acelerar o treino em CPU e alinhar com o objetivo de Edge AI, que prioriza footprint reduzido)
- **Batch size:** 16
- **Device:** CPU (AMD EPYC 4564P)
- **Optimizer:** AdamW selecionado automaticamente pela Ultralytics (`optimizer=auto`), com `lr=0.001429` e `momentum=0.9`

Não foram aplicados ajustes específicos para o desbalanceamento de classes. O objetivo do desafio é demonstrar o pipeline completo (fine-tuning → validação → exportação → inferência), e o efeito do desbalanceamento sobre a classe minoritária aparece de forma clara nos resultados da seção 4, servindo inclusive como evidência de que a métrica agregada esconde comportamento por classe.

### 2️⃣ Bibliotecas Utilizadas

- **Python:** 3.10.18
- **ultralytics:** 8.3.155
- **torch:** 2.4.1 (CPU)
- **tensorflow:** 2.17.0
- **onnx:** 1.16.2
- **onnx2tf:** 1.28.8
- **onnxslim:** 0.1.94

### 3️⃣ Técnica de Otimização do Modelo

Técnica aplicada: **Post-Training Integer Quantization (INT8)** via TFLiteConverter, com `tf.lite.Optimize.DEFAULT` e representative dataset para calibração dos ranges de ativação.

Pipeline de exportação executado em `optimize_model.py`:

1. **PyTorch → ONNX** (opset 19, via `torch.onnx.export`)
2. **ONNX slimming** (via onnxslim, para simplificar o grafo antes da conversão)
3. **ONNX → TensorFlow SavedModel** (via onnx2tf, gerando `model_saved_model/`)
4. **SavedModel → TFLite INT8**, com quantização calibrada em **170 imagens** do conjunto de validação (passadas via `data=dataset/data.yaml`)

Nota técnica: o pipeline emitiu um warning recomendando >300 imagens de calibração e o dataset só oferece 170 no split de validação, então usamos todas. Os resultados poderiam ser marginalmente melhores com um conjunto maior de calibração, mas o trade-off é aceitável para o objetivo do desafio.

### 4️⃣ Resultados Obtidos

**Métricas de validação (mAP no split val, 170 imagens, 726 instâncias):**

| Classe                | Imagens | Instâncias | Precision | Recall | mAP50 | mAP50-95 |
| --------------------- | :-----: | :--------: | :-------: | :----: | :---: | :------: |
| **Todas**             |   170   |    726     |   0.850   | 0.620  | **0.709** | **0.500** |
| with_mask             |   149   |    593     |   0.940   | 0.895  | 0.955 |  0.664   |
| without_mask          |    57   |    114     |   0.841   | 0.596  | 0.676 |  0.450   |
| mask_weared_incorrect |    15   |     19     |   0.769   | 0.368  | 0.497 |  0.386   |

**Tamanhos dos artefatos:**

| Arquivo        | Tamanho | Formato                    |
| -------------- | :-----: | -------------------------- |
| `model.pt`     | 5.18 MB | PyTorch fine-tuned         |
| `model.tflite` | 2.70 MB | TFLite INT8 quantizado     |
| **Redução**    | **47.9%** | — |

### 5️⃣ Comentários Adicionais

- **Desbalanceamento severo confirmado nos resultados.** A classe `mask_weared_incorrect` tem apenas 19 instâncias na validação (~2.6% do total) e obteve mAP50 = 0.497, contra 0.955 da classe majoritária `with_mask`. Isso é coerente com o alerta do README e demonstra o limite prático de fine-tuning sem estratégias de balanceamento (oversampling, class weights, augmentation direcionado).

- **Escolha de `imgsz=416` vs 640.** Trade-off consciente: reduz o custo de inferência em ~2.4x (área proporcional a 416²/640²) e alinha com o objetivo de Edge AI, ao custo de possivelmente perder alguma acurácia em rostos pequenos em multidão. Como a diferença de mAP50 entre 416 e 640 no YOLO11n costuma ser pequena para este tipo de dataset, foi um trade-off vantajoso.

- **Optimizer automático.** A Ultralytics selecionou AdamW com lr=0.001429 via seu heurístico `optimizer=auto`, sobrepondo os defaults `lr0=0.01` e `momentum=0.937`. Essa é a configuração aplicada de fato no treino.

- **Pipeline de export exigiu ambiente específico.** O Ultralytics 8.4.x+ adotou o backend LiteRT para export TFLite, que se mostrou instável no ambiente CPU (falha no import de `torch._functorch._aot_autograd.utils`). Voltamos ao pipeline clássico da Ultralytics 8.3.155 (onnx2tf), que produziu o artefato TFLite corretamente. Detalhe relevante pra reprodutibilidade do resultado.

- **Tempo total do pipeline:** ~10 min de treino + ~3.5 min de exportação/quantização.

### 6️⃣ Exemplo de Inferência

Saída do terminal ao rodar `run_inference.py` (com `imgsz=416` para bater com o export, e `conf=0.25` como threshold de confiança):

```
============================================================
Projeto 3 — Inferência com model.tflite (Edge AI)
============================================================

Rodando inferência em 5 amostras usando model.tflite:

Imagem                               Detecções  Detalhes
----------------------------------------------------------------------
maksssksksss105.jpg                          8  [8x with_mask]
maksssksksss107.jpg                          0  [nenhuma detecção]
maksssksksss11.jpg                          24  [23x with_mask, 1x mask_weared_incorrect]
maksssksksss113.jpg                          4  [3x with_mask, 1x without_mask]
maksssksksss12.jpg                          10  [9x with_mask, 1x without_mask]
----------------------------------------------------------------------
TOTAL                                       46
```

**Observações ao examinar as imagens anotadas em `runs/detect/inferencia_exemplos/predicoes/`:**

- **maksssksksss107.jpg (0 detecções)** — É o caso mais interessante e o único erro claro: uma selfie de uma única pessoa com máscara claramente visível, ocupando grande parte do frame. O modelo não detectou o rosto. Provavelmente reflete o viés do dataset de treino, onde a maioria dos rostos aparece em cenas de multidão em escala pequena/média — rostos muito grandes e centralizados são sub-representados. A quantização INT8 pode ter amplificado essa fragilidade em uma amostra fora da distribuição típica.

- **maksssksksss105.jpg (8 detecções, todas `with_mask`)** — sala de aula com crianças usando máscaras corretamente. O modelo detectou os 8 rostos e classificou todos como `with_mask`, consistente com a cena. Bom exemplo de desempenho da classe majoritária em condições favoráveis (rostos frontais, iluminação boa, escala média).

- **maksssksksss11.jpg (24 detecções)** — cena de multidão com muitos rostos pequenos, majoritariamente com máscara. O modelo conseguiu detectar 1 caso de `mask_weared_incorrect` corretamente, indicando que aprendeu algum sinal útil da classe minoritária mesmo com pouquíssimas amostras de treino.

- **maksssksksss113.jpg (4 detecções)** — cena de aeroporto. Uma detecção `without_mask` com confiança 0.28, bem próximo do threshold — indica que o modelo está no limite de decisão para casos ambíguos (rostos parcialmente visíveis, iluminação difícil).

- **maksssksksss12.jpg (10 detecções)** — cena de rua com pessoas caminhando com bagagens. Detecções distribuídas coerentemente pela cena.

**Distribuição agregada das 46 detecções:** 43 `with_mask` (~93%), 2 `without_mask` (~4%), 1 `mask_weared_incorrect` (~2%). A distribuição reflete claramente o desbalanceamento herdado do dataset de treino, mas ainda assim o modelo mostrou detecção correta da classe minoritária em pelo menos um caso.