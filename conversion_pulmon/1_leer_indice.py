"""Step 1 (main environment). Get variables.index of the author's checkpoint (Google Drive, 41 KB; only if it is not
already on disk) and read it without TensorFlow, to see which part of the data file holds the model weights: the
201 model tensors are bytes 0 to 709,181,016 (with 24 bytes of optimizer hyperparameters among them); the rest of
the 2.13 GB file is the Adam moments, the object graph and training metrics, which are not needed to use the model. Writes indice_pulmon.json next to variables.index.
Usage: python 1_leer_indice.py"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from indice_tf import INDICE, leer_indice, tensores_modelo

# BERT/variables/variables.index in the Drive folder linked from the README of github.com/solarte7/deepLungCancer
URL_INDICE = "https://drive.google.com/uc?export=download&id=1yJTUbtHGWst8gQioCv1wxwjV-4BObpWA"

if not INDICE.exists():
    INDICE.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["curl", "-s", "-L", "--retry", "3", "--max-time", "120", "-o", str(INDICE), URL_INDICE], check=True)
    print(f"variables.index bajado: {INDICE.stat().st_size:,} bytes")
tensores = leer_indice(INDICE)
modelo = tensores_modelo(tensores)
desde = min(t["offset"] for t in modelo.values())
hasta = max(t["offset"] + t["bytes"] for t in modelo.values())
otros = {k: t for k, t in tensores.items() if k not in modelo and t["bytes"]}
dentro = {k: t for k, t in otros.items() if t["offset"] < hasta}          # other tensors inside the model's range
(INDICE.parent / "indice_pulmon.json").write_text(json.dumps(tensores, indent=1), encoding="utf-8")

print(f"tensores en el checkpoint: {len(tensores)} | shards: {sorted({t['shard'] for t in tensores.values()})} | "
      f"bytes del archivo de datos: {sum(t['bytes'] for t in tensores.values()):,}")
print(f"tensores del modelo: {len(modelo)} (199 del encoder BERT + kernel y bias de la capa Dense(40)) | tipos: "
      f"{sorted({t['dtype'] for t in modelo.values()})} | todos con CRC: {all(t['crc'] is not None for t in modelo.values())}")
print(f"el modelo está entre los bytes {desde:,} y {hasta:,} del archivo de datos: {sum(t['bytes'] for t in modelo.values()):,}"
      f" bytes del modelo y {sum(t['bytes'] for t in dentro.values()):,} de {len(dentro)} tensores pequeños del optimizador "
      f"{sorted(dentro)[:4]}")
despues = {k: t for k, t in otros.items() if k not in dentro}
slots = [k for k in despues if "OPTIMIZER_SLOT" in k]
print(f"después del byte {hasta:,}: {len(despues)} tensores, {sum(t['bytes'] for t in despues.values()):,} bytes; "
      f"{len(slots)} son los momentos de Adam ({sum(despues[k]['bytes'] for k in slots):,} bytes) y el resto "
      f"{sorted(set(despues) - set(slots))[:3]}…, nada del modelo")
print(f"índice guardado en {INDICE.parent.name}/indice_pulmon.json")
