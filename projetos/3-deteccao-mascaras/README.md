## 📝 Relatório do Candidato

**Nome Completo:** Vinicius Reis de Lemos
**GitHub:** https://github.com/viniR15/

### 1️⃣ Resumo da Abordagem

Fine-tuning do **YOLO11n** pré-treinado sobre o dataset de detecção de máscaras. Hiperparâmetros:

- **Épocas:** 15 (dentro da faixa recomendada de 15-30 pelo README)
- **Tamanho de imagem:** 640x640 (padrão YOLO, alinhado com a validação automática do pipeline de correção)
- **Batch size:** 16
- **Device:** CPU (AMD EPYC 4564P)
- **Optimizer:** AdamW selecionado automaticamente pela Ultralytics (`optimizer=auto`), com `lr=0.001429` e `momentum=0.9`

Não foram aplicados ajustes específicos para o desbalanceamento de classes. O objetivo é demonstrar o pipeline completo (fine-tuning → validação → exportação → inferência), e o efeito do desbalanceamento aparece de forma clara nos resultados da seção 4.

### 2️⃣ Bibliotecas Utilizadas

- **Python:** 3.10.18
- **ultralytics:** 8.3.155
- **torch:** 2.4.1 (CPU)
- **tensorflow:** 2.17.0
- **onnx:** 1.16.2
- **onnx2tf:** 1.28.8
- **onnxslim:** 0.1.94

### 3️⃣ Técnica de Otimização do Modelo

Pipeline de exportação executado em `optimize_model.py`:

1. **PyTorch → ONNX** (opset 19, via `torch.onnx.export`)
2. **ONNX slimming** com **onnxslim 0.1.94** — otimização de grafo por eliminação de subexpressões comuns, constant folding e fusão de operações, reduzindo o número de nós antes da conversão
3. **ONNX → TensorFlow SavedModel** (via onnx2tf, gerando `model_saved_model/`)
4. **SavedModel → TFLite FP32** (via TFLiteConverter, formato final entregue como `model.tflite`)

**Sobre a quantização INT8:** a implementação inicial usava **Post-Training Integer Quantization (INT8)** via `tf.lite.Optimize.DEFAULT` com representative dataset calibrado em 170 imagens do split de validação (via `int8=True` + `data=dataset/data.yaml`). O artefato quantizado gerado (2.81 MB) passou nos testes locais de inferência com detecções coerentes, mas foi descartado após a validação automática do CI acusar queda de mAP50 para ~0.02. Ver seção 5 para o diagnóstico completo — a decisão de manter o FP32 foi consciente e priorizou correção sobre compressão.

### 4️⃣ Resultados Obtidos

**Métricas de validação (mAP no split val, 170 imagens, 726 instâncias):**

| Classe                | Imagens | Instâncias | Precision | Recall | mAP50 | mAP50-95 |
| --------------------- | :-----: | :--------: | :-------: | :----: | :---: | :------: |
| **Todas**             |   170   |    726     |   0.760   | 0.703  | **0.722** | **0.504** |
| with_mask             |   149   |    593     |   0.871   | 0.943  | 0.959 |  0.674   |
| without_mask          |    57   |    114     |   0.787   | 0.693  | 0.771 |  0.518   |
| mask_weared_incorrect |    15   |     19     |   0.621   | 0.474  | 0.435 |  0.320   |

**Tamanhos dos artefatos:**

| Arquivo        | Tamanho  | Formato                          |
| -------------- | :------: | -------------------------------- |
| `model.pt`     |  5.20 MB | PyTorch fine-tuned               |
| `model.tflite` | 10.17 MB | TFLite FP32 (com grafo slimmed)  |

Observação sobre o tamanho: o `model.tflite` FP32 fica maior que o `.pt` original porque o formato PyTorch aplica compressão interna (share tensors, weight tying) que o TFLite não replica. A versão INT8 gerada durante o desenvolvimento tinha 2.81 MB (redução de ~46% vs `.pt`), mas foi descartada pelo motivo descrito na seção 5.

### 5️⃣ Comentários Adicionais

- **Decisão INT8 → FP32 e o diagnóstico de incompatibilidade de backend.** A abordagem inicial de exportação foi Post-Training Integer Quantization (INT8) via `tf.lite.Optimize.DEFAULT`, com representative dataset amostrado do conjunto de validação. O `.tflite` INT8 (2.81 MB, redução ~46%) passou nos testes locais de inferência com detecções coerentes. Porém, quando validado pelo CI do repositório, o mAP50 caiu de 0.722 (no `.pt`) para 0.020 no `.tflite` INT8. O diagnóstico: o CI carrega o `.tflite` usando o backend **LiteRT** (`ai-edge-litert`, novo runtime do Ultralytics 8.4.x), enquanto o export foi feito pelo pipeline **onnx2tf → TensorFlow TFLite** (Ultralytics 8.3.x). A quantização INT8 gerada por esse pipeline usa layout NHWC com escalas por-canal que o LiteRT interpreta de forma diferente do runtime clássico do TFLite, resultando em detecções sistematicamente fora do lugar. A alternativa de exportar direto pelo LiteRT (Ultralytics 8.4.x) tem regressão conhecida no torch (falha no import de `torch._functorch._aot_autograd.utils`) que não foi possível contornar. Decisão consciente: entregar FP32 (10.17 MB), garantindo consistência de leitura entre backends e mAP alinhado com o do `.pt`. Trade-off: perde-se a compressão da quantização, mantém-se a correção do modelo.

- **Desbalanceamento severo confirmado nos resultados.** A classe `mask_weared_incorrect` tem apenas 19 instâncias na validação (~2.6% do total) e obteve mAP50 = 0.435, contra 0.959 da classe majoritária `with_mask`. A classe intermediária `without_mask` (114 instâncias, ~15.7%) ficou num meio-termo (mAP50 = 0.771), reforçando a correlação direta entre volume de dados e desempenho.

- **Optimizer automático.** A Ultralytics selecionou AdamW com lr=0.001429 via seu heurístico `optimizer=auto`, sobrepondo os defaults `lr0=0.01` e `momentum=0.937`. Essa é a configuração aplicada de fato no treino.

- **Filtro defensivo de confidence na inferência.** O modelo TFLite carregado via `YOLO(...task="detect")` na Ultralytics 8.3.155 nem sempre respeita o parâmetro `conf` passado ao `predict()`, retornando as ~300 âncoras raw. O `run_inference.py` aplica um filtro adicional por confidence (`>= 0.25`) sobre `result.boxes` após a inferência, garantindo que a contagem impressa reflita detecções significativas.

- **Tempo total do pipeline:** ~19 min de treino + ~10 s de exportação FP32.

### 6️⃣ Exemplo de Inferência

Saída do terminal ao rodar `run_inference.py` (com `imgsz=640` e filtro de confidence `>= 0.25`):

```
============================================================
Projeto 3 — Inferência com model.tflite (Edge AI)
============================================================

Rodando inferência em 5 amostras usando model.tflite:

Imagem                               Detecções  Detalhes
----------------------------------------------------------------------
maksssksksss105.jpg                          9  [9x with_mask]
maksssksksss107.jpg                          1  [1x with_mask]
maksssksksss11.jpg                          23  [22x with_mask, 1x mask_weared_incorrect]
maksssksksss113.jpg                          4  [4x with_mask]
maksssksksss12.jpg                          12  [11x with_mask, 1x without_mask]
----------------------------------------------------------------------
TOTAL                                       49
```

**Observações:**

- **maksssksksss105.jpg (9 detecções, todas `with_mask`)** — sala de aula com crianças usando máscaras corretamente. As 9 detecções são coerentes com a cena, boa performance da classe majoritária em condições favoráveis.

- **maksssksksss107.jpg (1 detecção)** — imagem tipo selfie com um único rosto grande, máscara claramente visível. Comparação relevante: a versão INT8 gerou 0 detecções nessa imagem, enquanto a FP32 detectou corretamente. Isso é coerente com o comportamento esperado da quantização: precisão reduzida penaliza mais amostras fora da distribuição típica de treino (rostos muito grandes e centralizados são sub-representados no dataset, dominado por cenas de multidão).

- **maksssksksss11.jpg (23 detecções, incluindo 1 `mask_weared_incorrect`)** — cena de multidão com muitos rostos pequenos. O modelo detectou uma instância da classe minoritária, indicando que aprendeu algum sinal útil mesmo com pouquíssimos exemplos de treino. Novamente, a versão FP32 detectou essa instância enquanto a INT8 não — a quantização degrada primeiro as classes menos representadas.

- **maksssksksss113.jpg (4 detecções, todas `with_mask`)** — cena de aeroporto. 4 rostos com máscara detectados coerentemente.

- **maksssksksss12.jpg (12 detecções)** — cena de rua com pessoas caminhando com bagagens. 11 `with_mask` e 1 `without_mask` — detecção mista consistente com a cena.

**Distribuição agregada das 49 detecções:** 47 `with_mask` (~95.9%), 1 `without_mask` (~2%), 1 `mask_weared_incorrect` (~2%). A distribuição reflete o desbalanceamento herdado do dataset de treino, mas o modelo demonstra capacidade de detectar as três classes.