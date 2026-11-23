"""Step 2 (main environment). Download only the model weights from the author's data file in Google Drive (HTTP
ranges, bytes 0 to 709,181,016, in parallel pieces with curl that resume where they stopped) into pesos_modelo.bin,
and the configuration and tokenizer of bert-base-multilingual-cased from Hugging Face (no weights, pinned revision).
Then it checks the 201 model tensors against the CRC32C stored in variables.index (pure Python, about 1.5 min).
Files that are already complete are not downloaded again.
Usage: python 2_descarga_parcial.py            (download what is missing, then check)
       python 2_descarga_parcial.py --verificar (no network: only check the files on disk)"""
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from indice_tf import BASE, INDICE, PESOS, crc32c, crc_enmascarado, leer_indice, tensores_modelo

# BERT/variables/variables.data-00000-of-00001 (2.13 GB) in the author's Drive folder
URL_DATOS = "https://drive.usercontent.google.com/download?id=171zVpGextjIRF5l_gt6iCnkAsPClAUw5&export=download&confirm=t"
MBERT, REVISION_MBERT = "google-bert/bert-base-multilingual-cased", "3f076fdb1ab68d5b2880cb87a0886f315b8146f8"
ARCHIVOS_MBERT = ("config.json", "vocab.txt", "tokenizer.json", "tokenizer_config.json")
TROZOS = 12

solo_verificar = "--verificar" in sys.argv
if not INDICE.exists():
    sys.exit("Falta variables.index: corre antes 1_leer_indice.py")
modelo = tensores_modelo(leer_indice(INDICE))
TOTAL = max(t["offset"] + t["bytes"] for t in modelo.values())      # 709,181,016: the model weights start at byte 0


def tamano(ruta):
    return ruta.stat().st_size if ruta.exists() else 0


def bajar_pesos():
    """Bytes 0..TOTAL-1 of the Drive file in TROZOS pieces; each short piece is asked again from where it stopped."""
    hecho = tamano(PESOS)
    if hecho >= TOTAL:
        print(f"pesos_modelo.bin ya está completo ({hecho:,} bytes): no se baja")
        return
    paso = -(-(TOTAL - hecho) // TROZOS)
    trozos = [(hecho + i * paso, min(hecho + (i + 1) * paso, TOTAL)) for i in range(TROZOS) if hecho + i * paso < TOTAL]
    print(f"pesos_modelo.bin: ya bajados {hecho:,} | faltan {TOTAL - hecho:,} bytes en {len(trozos)} trozos", flush=True)
    reloj = time.time()
    for intento in range(20):
        procesos = []
        for k, (desde, hasta) in enumerate(trozos):
            archivo = PESOS.parent / f"trozo_{k}.bin"
            tiene = tamano(archivo)
            if tiene < hasta - desde:                  # ask only for what is missing, appended to the piece
                salida = open(archivo, "ab")
                procesos.append((subprocess.Popen(["curl", "-s", "-L", "--retry", "5", "-r",
                                                   f"{desde + tiene}-{hasta - 1}", URL_DATOS], stdout=salida), salida))
        if not procesos:
            break
        for p, salida in procesos:
            p.wait()
            salida.close()
        bajado = hecho + sum(tamano(PESOS.parent / f"trozo_{k}.bin") for k in range(len(trozos)))
        print(f"vuelta {intento + 1}: {bajado:,} de {TOTAL:,} bytes, {(time.time() - reloj) / 60:.1f} min", flush=True)
    for k, (desde, hasta) in enumerate(trozos):
        if tamano(PESOS.parent / f"trozo_{k}.bin") != hasta - desde:
            sys.exit(f"el trozo {k} quedó incompleto: no se pegó nada; vuelve a correr el script")
    with open(PESOS, "ab") as salida:
        for k in range(len(trozos)):
            archivo = PESOS.parent / f"trozo_{k}.bin"
            salida.write(archivo.read_bytes())
            archivo.unlink()


def bajar_mbert():
    destino = BASE / "mbert"
    faltan = [a for a in ARCHIVOS_MBERT if not (destino / a).exists()]
    if not faltan:
        print(f"tokenizador de mBERT ya está en {destino.name}/: no se baja")
        return
    from pip._vendor import truststore              # the network intercepts HTTPS: use the Windows certificate store
    truststore.inject_into_ssl()
    from huggingface_hub import hf_hub_download
    for a in faltan:
        hf_hub_download(MBERT, a, revision=REVISION_MBERT, local_dir=destino)
        print(f"{a}: {(destino / a).stat().st_size:,} bytes")


if not solo_verificar:
    bajar_pesos()
    bajar_mbert()

if tamano(PESOS) != TOTAL:
    sys.exit(f"pesos_modelo.bin tiene {tamano(PESOS):,} bytes y deben ser {TOTAL:,}")
print(f"mBERT ({REVISION_MBERT[:8]}): {sum((BASE / 'mbert' / a).exists() for a in ARCHIVOS_MBERT)} de "
      f"{len(ARCHIVOS_MBERT)} archivos")
datos = PESOS.read_bytes()
malos = [k for k, t in modelo.items() if crc_enmascarado(crc32c(datos[t["offset"]:t["offset"] + t["bytes"]])) != t["crc"]]
print(f"pesos_modelo.bin: {TOTAL:,} bytes | tensores del modelo revisados con el CRC32C del índice: {len(modelo)} | "
      f"no coinciden: {len(malos)}" + (f" {malos[:5]}" if malos else ""))
