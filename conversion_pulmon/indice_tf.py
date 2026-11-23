"""Read a TensorFlow checkpoint index (variables.index) without TensorFlow, and check tensors with its CRC32C.

The index is a LevelDB-style table with uncompressed blocks. Each entry is a BundleEntryProto: dtype (1), shape (2),
shard (3), offset (4) and size (5) of the tensor inside the data file, and its masked CRC32C (6).
Used by 1_leer_indice.py, 2_descarga_parcial.py and 3_convertir_a_pytorch.py."""
import re
import struct
from pathlib import Path

BASE = Path.home() / ".cache" / "huggingface" / "modelos" / "deepLungCancer-BERT"
INDICE = BASE / "variables" / "variables.index"
PESOS = BASE / "variables" / "pesos_modelo.bin"      # bytes 0.. of the author's variables.data-00000-of-00001

DTYPES = {1: "float32", 2: "float64", 3: "int32", 7: "string", 9: "int64", 10: "bool", 14: "bfloat16", 19: "float16"}
# the model weights: the 199 variables of the BERT encoder and the final Dense(40) layer. The Adam slots also start
# with trainable_variables/N/ but go on with /.OPTIMIZER_SLOT/..., so they do not match
CLAVE_MODELO = re.compile(r"(trainable_variables/\d+|layer_with_weights-1/(kernel|bias))/\.ATTRIBUTES/VARIABLE_VALUE")


def varint(buf, i):
    resultado, desplazamiento = 0, 0
    while True:
        b = buf[i]
        i += 1
        resultado |= (b & 0x7F) << desplazamiento
        if b < 0x80:
            return resultado, i
        desplazamiento += 7


def bloque(datos, desde, tamano):
    """(key, value) entries of one table block."""
    if datos[desde + tamano] != 0:
        raise ValueError("bloque comprimido: este lector solo lee bloques sin compresión")
    contenido = datos[desde:desde + tamano]
    reinicios = struct.unpack_from("<I", contenido, len(contenido) - 4)[0]
    fin = len(contenido) - 4 - 4 * reinicios
    i, clave, entradas = 0, b"", []
    while i < fin:
        compartido, i = varint(contenido, i)
        propio, i = varint(contenido, i)
        largo, i = varint(contenido, i)
        clave = clave[:compartido] + contenido[i:i + propio]
        i += propio
        entradas.append((clave, contenido[i:i + largo]))
        i += largo
    return entradas


def campos(buf):
    """Fields of a protobuf message: [(number, value)]."""
    i, salida = 0, []
    while i < len(buf):
        etiqueta, i = varint(buf, i)
        numero, tipo = etiqueta >> 3, etiqueta & 7
        if tipo == 0:
            valor, i = varint(buf, i)
        elif tipo == 2:
            largo, i = varint(buf, i)
            valor, i = buf[i:i + largo], i + largo
        elif tipo == 5:
            valor, i = struct.unpack_from("<I", buf, i)[0], i + 4
        elif tipo == 1:
            valor, i = buf[i:i + 8], i + 8
        else:
            raise ValueError(f"tipo protobuf {tipo}")
        salida.append((numero, valor))
    return salida


def entrada(valor):
    info = {"dtype": "", "forma": [], "shard": 0, "offset": 0, "bytes": 0, "crc": None}
    for numero, v in campos(valor):
        if numero == 1:
            info["dtype"] = DTYPES.get(v, str(v))
        elif numero == 2:
            info["forma"] = [dict(campos(d)).get(1, 0) for n, d in campos(v) if n == 2]
        elif numero == 3:
            info["shard"] = v
        elif numero == 4:
            info["offset"] = v
        elif numero == 5:
            info["bytes"] = v
        elif numero == 6:
            info["crc"] = v
    return info


def leer_indice(ruta=INDICE):
    """{tensor name: {dtype, forma, shard, offset, bytes, crc}} for every tensor of the checkpoint."""
    datos = Path(ruta).read_bytes()
    pie = datos[-48:]
    if struct.unpack("<Q", pie[-8:])[0] != 0xDB4775248B80FB57:
        raise ValueError("no es una tabla de TensorFlow")
    _, i = varint(pie, 0)                              # metaindex handle (not used)
    _, i = varint(pie, i)
    desde, i = varint(pie, i)                          # index block handle
    tamano, _ = varint(pie, i)
    tensores = {}
    for _, manejador in bloque(datos, desde, tamano):
        d, j = varint(manejador, 0)
        t, _ = varint(manejador, j)
        for clave, valor in bloque(datos, d, t):
            if clave:                                  # the empty key is the bundle header
                tensores[clave.decode("utf-8")] = entrada(valor)
    return tensores


def tensores_modelo(tensores):
    """Only the model weights (201 tensors), not the optimizer state."""
    return {k: v for k, v in tensores.items() if CLAVE_MODELO.fullmatch(k)}


# CRC32C (Castagnoli), table version
_TABLA = []
for _i in range(256):
    _c = _i
    for _ in range(8):
        _c = (_c >> 1) ^ 0x82F63B78 if _c & 1 else _c >> 1
    _TABLA.append(_c)


def crc32c(datos):
    c = 0xFFFFFFFF
    for b in datos:
        c = _TABLA[(c ^ b) & 0xFF] ^ (c >> 8)
    return c ^ 0xFFFFFFFF


def crc_enmascarado(c):
    """TensorFlow stores the masked CRC32C."""
    return ((((c >> 15) | (c << 17)) & 0xFFFFFFFF) + 0xA282EAD8) & 0xFFFFFFFF
