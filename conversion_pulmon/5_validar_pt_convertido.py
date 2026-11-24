"""Step 5 (main environment: Python 3.12, transformers 4.49, PyTorch). Run the CONVERTED model on exactly the same ids
as step 4 and compare with TensorFlow: largest probability difference, most likely label per piece and per word.
Also compares the pieces of the fast tokenizer used by the pipeline with those of the author's tokenizer. No internet.
Usage: python 5_validar_pt_convertido.py <converted model folder>   (the folder given to step 3)"""
import json
import os
import sys
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

AQUI = Path(__file__).parent
if len(sys.argv) != 2:
    sys.exit(__doc__)
HF = Path(sys.argv[1])
FRASES = json.loads((AQUI / "frases_validacion.json").read_text(encoding="utf-8"))
if not (AQUI / "tf_salidas.npz").exists():
    sys.exit("Falta tf_salidas.npz: corre antes 4_validar_tf_original.py con el entorno tf24-pulmon")
tf_out = np.load(AQUI / "tf_salidas.npz")
tokenizador = AutoTokenizer.from_pretrained(HF)
modelo = AutoModelForTokenClassification.from_pretrained(HF).eval()
id2label = modelo.config.id2label

dif_max, piezas_total, piezas_iguales, palabras_total, palabras_iguales, tokens_iguales = 0.0, 0, 0, 0, 0, 0
for k, frase in enumerate(FRASES):
    ids = tf_out[f"ids_{k}"]
    with torch.no_grad():
        p_pt = torch.softmax(modelo(input_ids=torch.tensor([ids], dtype=torch.long)).logits[0], dim=-1).numpy()
    p_tf = tf_out[f"probs_{k}"]
    dif_max = max(dif_max, float(np.abs(p_pt - p_tf).max()))
    a_pt, a_tf = p_pt.argmax(-1), p_tf.argmax(-1)
    piezas_total += len(ids)
    piezas_iguales += int((a_pt == a_tf).sum())
    primera = tf_out[f"primera_{k}"]
    palabras_total += len(primera)
    palabras_iguales += int(sum(id2label[int(a_pt[i])] == id2label[int(a_tf[i])] for i in primera))
    # the fast tokenizer used by the pipeline, with words split by spaces, against the author's pieces
    rapido = tokenizador(frase.split(), is_split_into_words=True)["input_ids"]
    tokens_iguales += int(list(rapido) == list(ids))

print(f"frases: {len(FRASES)} | piezas: {piezas_total} | palabras: {palabras_total}")
print(f"diferencia máxima de probabilidad entre TensorFlow (original) y PyTorch (convertido): {dif_max:.2e}")
print(f"etiqueta más probable igual en {piezas_iguales} de {piezas_total} piezas "
      f"y en {palabras_iguales} de {palabras_total} palabras")
print(f"tokenización del pipeline igual a la del autor en {tokens_iguales} de {len(FRASES)} frases")
