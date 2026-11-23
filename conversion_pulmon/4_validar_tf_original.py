"""Step 4 (environment tf24-pulmon: Python 3.8.10, TensorFlow 2.4.1, transformers 4.13.0, the author's versions).
Build the deepLungCancer BERT as in BERT-training.ipynb of the repository (TFBertForTokenClassification + Dropout +
Dense(40, softmax)), give it the ORIGINAL weights read with TensorFlow's own checkpoint reader, and run it on the
sentences of frases_validacion.json (14 from the repository, 6 invented), tokenized as in loadBERT.ipynb (words split
by spaces, tokenizer.tokenize per word, encode_plus). Saves pieces, ids and probabilities in tf_salidas.npz, for
step 5. No internet. Usage: <tf24-pulmon python> 4_validar_tf_original.py"""
import json
import os
import sys
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout, Input
from transformers import BertConfig, BertTokenizer, TFBertForTokenClassification

AQUI = Path(__file__).parent
BASE = Path.home() / ".cache" / "huggingface" / "modelos" / "deepLungCancer-BERT"
FRASES = json.loads((AQUI / "frases_validacion.json").read_text(encoding="utf-8"))

# the original TensorFlow checkpoint: the index and the part of the data file with the model weights, under the
# names TensorFlow expects (hard links, outside OneDrive: no copy of 709 MB)
ckpt = BASE / "ckpt_tf"
ckpt.mkdir(exist_ok=True)
for origen, destino in ((BASE / "variables" / "variables.index", ckpt / "variables.index"),
                        (BASE / "variables" / "pesos_modelo.bin", ckpt / "variables.data-00000-of-00001")):
    if not destino.exists():
        os.link(origen, destino)
lector = tf.train.load_checkpoint(str(ckpt / "variables"))

# the author's model (BERT-training.ipynb, create_model), with the multilingual BERT configuration and 40 outputs
config = BertConfig.from_json_file(str(BASE / "mbert" / "config.json"))
config.num_labels = 40
encoder = TFBertForTokenClassification(config)
entradas = [Input(shape=(None,), dtype=tf.int32) for _ in range(3)]     # author: shape=(596,); here any length
embedding = encoder(input_ids=entradas[0], token_type_ids=entradas[1], attention_mask=entradas[2])[0]
salida = Dense(40, activation="softmax")(Dropout(0.4)(embedding))
modelo = keras.Model(inputs=entradas, outputs=[salida])

# weights: trainable_variables/N of the checkpoint, in the order TensorFlow itself gives to the encoder's variables
variables = encoder.trainable_variables
if len(variables) != 199:
    sys.exit(f"el encoder tiene {len(variables)} variables, se esperaban 199")
for n, v in enumerate(variables):
    t = lector.get_tensor(f"trainable_variables/{n}/.ATTRIBUTES/VARIABLE_VALUE")
    if tuple(t.shape) != tuple(v.shape):
        sys.exit(f"variable {n} ({v.name}): forma {t.shape} en el checkpoint y {v.shape} en el modelo")
    v.assign(t)
densa = modelo.layers[-1]
densa.kernel.assign(lector.get_tensor("layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE"))
densa.bias.assign(lector.get_tensor("layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE"))
print("pesos originales cargados: 199 del encoder + la capa Dense(40) | primeras variables:",
      [v.name for v in variables[:3]], "| capa 0:", [v.name.split("/")[-2] for v in variables[5:11]])

# the author's tokenization (loadBERT.ipynb / BERT-training.ipynb: convert_to_input)
tokenizador = BertTokenizer(str(BASE / "mbert" / "vocab.txt"), do_lower_case=False)
guardar = {}
for k, frase in enumerate(FRASES):
    piezas, primera = [], []
    for palabra in frase.split():
        p = tokenizador.tokenize(palabra)
        primera.append(len(piezas) + 1)               # +1: [CLS] goes first
        piezas.extend(p)
    enc = tokenizador.encode_plus(piezas, add_special_tokens=True, max_length=596, truncation=True)
    ids = np.array([enc["input_ids"]], dtype=np.int32)
    probs = modelo.predict([ids, np.array([enc["token_type_ids"]], dtype=np.int32),
                            np.array([enc["attention_mask"]], dtype=np.int32)])[0]
    guardar[f"ids_{k}"], guardar[f"probs_{k}"], guardar[f"primera_{k}"] = ids[0], probs, np.array(primera)
    guardar[f"piezas_{k}"] = np.array(["[CLS]"] + piezas + ["[SEP]"])
np.savez(AQUI / "tf_salidas.npz", **guardar)
print(f"frases corridas en TensorFlow {tf.__version__} (transformers 4.13.0): {len(FRASES)} | salida: tf_salidas.npz")
