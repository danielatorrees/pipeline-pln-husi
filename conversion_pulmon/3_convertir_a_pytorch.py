"""Step 3 (main environment). Convert the deepLungCancer BERT (the model weights of the TensorFlow checkpoint) to a
transformers model in PyTorch, without TensorFlow, and check the result against the reference sha256.

- The 199 encoder variables (trainable_variables/0..198) go to BertForTokenClassification in the order TensorFlow
  gives them; Dense kernels [in, out] are transposed to Linear weights [out, in].
- The author's classifier (768 -> 40) is followed by Dense(40 -> 40, softmax). Two linear layers in a row are one:
  h K1 + c1, then (.) K2 + c2 = h (K1 K2) + (c1 K2 + c2). So the probabilities are those of the original model.
- Outputs 37, 38 and 39 ([CLS], [SEP] and [PAD] of the author's training) are labelled "O".
- Tokenizer: the one of bert-base-multilingual-cased (mbert/), saved next to the model.

The destination must be an empty or new folder. Then copy its 6 files to modelos/deepLungCancer-BERT, next to the
notebooks: that is the model pipeline_pulmon.ipynb uses (see README.md).
Usage: python 3_convertir_a_pytorch.py <destination folder>"""
import hashlib
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from transformers import AutoTokenizer, BertConfig, BertForTokenClassification

from indice_tf import BASE, INDICE, PESOS, leer_indice

SHA256_REFERENCIA = "fadf9bf51399823ec12d8f9111d63f0b38d4a31571e1b27bf3e012861ee51d76"   # model.safetensors of modelos/deepLungCancer-BERT
ETIQUETAS = ["B-BIOMARKER", "B-BIOMARKER_STATUS", "B-CANCER_CONCEPT", "B-CHEMOTHERAPY", "B-CLINICAL_SERVICE",
             "B-COMORBIDITY", "B-DRUG", "B-EXPLICIT_DATE", "B-FAMILY", "B-FREQ", "B-IMPLICIT_DATE", "B-METRIC",
             "B-OCURRENCE_EVENT", "B-QUANTITY", "B-RADIOTHERAPY", "B-STAGE", "B-SURGERY", "B-TNM", "B-TOXIC_HABIT",
             "I-BIOMARKER", "I-BIOMARKER_STATUS", "I-CANCER_CONCEPT", "I-CLINICAL_SERVICE", "I-COMORBIDITY", "I-DRUG",
             "I-EXPLICIT_DATE", "I-FAMILY", "I-FREQ", "I-IMPLICIT_DATE", "I-METRIC", "I-OCURRENCE_EVENT",
             "I-QUANTITY", "I-STAGE", "I-SURGERY", "I-TNM", "I-TOXIC_HABIT", "O",
             "O", "O", "O"]                       # 37 [CLS], 38 [SEP] (rest of a word) and 39 [PAD] -> O

if len(sys.argv) != 2:
    sys.exit(__doc__)
destino = Path(sys.argv[1])
if destino.exists() and any(destino.iterdir()):
    sys.exit("La carpeta de destino no está vacía: usa una carpeta nueva (no se sobrescribe ningún modelo)")

indice = leer_indice(INDICE)
pesos_modelo = np.memmap(PESOS, dtype=np.uint8, mode="r")      # bytes 0.. of the author's data file


def tensor(clave):
    t = indice[clave]
    if t["dtype"] != "float32":
        raise ValueError(f"{clave}: {t['dtype']}")
    datos = np.asarray(pesos_modelo[t["offset"]:t["offset"] + t["bytes"]])
    if datos.size != t["bytes"]:
        raise ValueError(f"{clave}: faltan bytes (descarga incompleta)")
    return datos.view("<f4").reshape(t["forma"]).copy()


def tv(n):
    return tensor(f"trainable_variables/{n}/.ATTRIBUTES/VARIABLE_VALUE")


def lineal(n):
    """TF Dense kernel [in, out] -> torch Linear weight [out, in]."""
    return tv(n).T, tv(n + 1)


pesos = {"bert.embeddings.word_embeddings.weight": tv(0),
         "bert.embeddings.token_type_embeddings.weight": tv(1),
         "bert.embeddings.position_embeddings.weight": tv(2),
         "bert.embeddings.LayerNorm.weight": tv(3), "bert.embeddings.LayerNorm.bias": tv(4)}
for capa in range(12):
    b, p = 5 + 16 * capa, f"bert.encoder.layer.{capa}."
    nombres = ["attention.self.query", "attention.self.key", "attention.self.value", "attention.output.dense",
               "attention.output.LayerNorm", "intermediate.dense", "output.dense", "output.LayerNorm"]
    for k, nombre in enumerate(nombres):
        if nombre.endswith("LayerNorm"):
            pesos[p + nombre + ".weight"], pesos[p + nombre + ".bias"] = tv(b + 2 * k), tv(b + 2 * k + 1)
        else:
            pesos[p + nombre + ".weight"], pesos[p + nombre + ".bias"] = lineal(b + 2 * k)

# classifier (768 -> 40) followed by Dense(40 -> 40): h K1 + c1, then (.) K2 + c2 = h (K1 K2) + (c1 K2 + c2)
k1, c1 = tv(197), tv(198)
k2 = tensor("layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE")
c2 = tensor("layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE")
pesos["classifier.weight"] = (k1 @ k2).T
pesos["classifier.bias"] = c1 @ k2 + c2

config = BertConfig.from_pretrained(BASE / "mbert", num_labels=len(ETIQUETAS),
                                    id2label=dict(enumerate(ETIQUETAS)),
                                    label2id={e: i for i, e in enumerate(ETIQUETAS[:37])})
config.architectures = ["BertForTokenClassification"]
modelo = BertForTokenClassification(config)
carga = modelo.load_state_dict({k: torch.from_numpy(np.ascontiguousarray(v)) for k, v in pesos.items()}, strict=False)
if carga.missing_keys or carga.unexpected_keys:
    sys.exit(f"no encajan: faltan {carga.missing_keys} | sobran {carga.unexpected_keys}")

modelo.save_pretrained(destino)
AutoTokenizer.from_pretrained(BASE / "mbert").save_pretrained(destino)
h = hashlib.sha256()
with open(destino / "model.safetensors", "rb") as f:
    for trozo in iter(lambda: f.read(1 << 20), b""):
        h.update(trozo)
print(f"modelo convertido: {sum(v.size for v in pesos.values()):,} parámetros en {len(pesos)} tensores, "
      f"{len(ETIQUETAS)} salidas, {len({e[2:] for e in ETIQUETAS if e != 'O'})} entidades")
print(f"sha256 de model.safetensors: {h.hexdigest()} | igual a la referencia (fadf9bf5…): "
      f"{'Sí' if h.hexdigest() == SHA256_REFERENCIA else 'No'}")
