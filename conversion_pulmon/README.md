# Reconstrucción del modelo de pulmón

Scripts que reconstruyen el BERT de solarte7/deepLungCancer a partir de sus pesos de TensorFlow en Google Drive, lo convierten a PyTorch y lo validan contra el modelo original en TensorFlow 2.4.1. El resultado es la carpeta del modelo que usa pipeline_pulmon.ipynb: modelos/deepLungCancer-BERT, junto a los notebooks.

**Por qué hace falta.** El repositorio del autor da instrucciones para cargar el modelo con tf.saved_model.load, pero la carpeta de Google Drive solo tiene los pesos (variables/), no el archivo con la estructura de la red (saved_model.pb), así que esas instrucciones no funcionan con lo publicado. Por eso la red se arma con la función create_model() del autor (BERT/BERT-training.ipynb) y se le cargan los pesos originales.

Se usan los pesos del autor y frases de ejemplo (frases_validacion.json: 14 del repositorio y 6 inventadas).

## Entornos

- **Principal:** Python 3.12.10, transformers 4.49.0, torch 2.11.0 (CPU), numpy 2.4.4, huggingface_hub y el curl de Windows.
- **tf24-pulmon** (un venv aparte), solo para el paso 4: Python 3.8.10, tensorflow 2.4.1, transformers 4.13.0, tokenizers 0.10.3, numpy 1.19.5 y protobuf 3.20.3 (las versiones del Requeriments.txt del autor).

## Orden

| Paso | Script | Entorno | Descarga | Qué hace | Deja |
|---|---|---|---|---|---|
| 1 | 1_leer_indice.py | principal | solo si falta el índice (41 KB) | Baja variables.index de Drive y lo lee sin TensorFlow (indice_tf.py): el modelo son 201 tensores en los bytes 0 a 709 181 016; el resto de los 2,13 GB es el estado de Adam. | variables/variables.index, variables/indice_pulmon.json |
| 2 | 2_descarga_parcial.py | principal | sí (709 MB de Drive y 4 archivos de Hugging Face) | Baja solo esos bytes, por rangos HTTP en 12 trozos que se reanudan, y el tokenizador de bert-base-multilingual-cased. Comprueba los 201 tensores con el CRC32C del índice. | variables/pesos_modelo.bin, mbert/ |
| 3 | 3_convertir_a_pytorch.py, con la carpeta donde se guarda el modelo | principal | no | Pone cada tensor en un BertForTokenClassification, junta la capa Dense(40, softmax) con el clasificador y guarda modelo y tokenizador. Imprime el sha256 y si es igual a la referencia. | model.safetensors, config.json y el tokenizador |
| 4 | 4_validar_tf_original.py | tf24-pulmon | no | Arma la red del autor (TFBertForTokenClassification + Dropout + Dense(40, softmax)) con sus pesos originales leídos por TensorFlow y corre las 20 frases tokenizadas como en su código. | tf_salidas.npz |
| 5 | 5_validar_pt_convertido.py, con la carpeta del modelo convertido | principal | no | Corre el modelo convertido sobre los mismos ids y compara con TensorFlow. | solo imprime |

Los pasos 1 y 2 guardan lo que descargan en la carpeta de trabajo ~/.cache/huggingface/modelos/deepLungCancer-BERT, fuera de OneDrive.

## Para correr el pipeline de pulmón

**Basta con los pasos 1 a 3**, en el entorno principal y con conexión a internet:

1. python 1_leer_indice.py
2. python 2_descarga_parcial.py
3. python 3_convertir_a_pytorch.py modelo_convertido (modelo_convertido es una carpeta nueva y vacía, donde se guarda el modelo). Al terminar, debe imprimir que el sha256 es igual a la referencia.

Después, copiar los 6 archivos de esa carpeta a modelos/deepLungCancer-BERT, junto a los notebooks, y correr pipeline_pulmon.ipynb. El notebook vuelve a comprobar que model.safetensors tenga el sha256 de referencia antes de usarlo.

**Los pasos 4 y 5 no hacen falta para correr el pipeline:** solo comprueban que la conversión es correcta, comparando el modelo convertido con el original en TensorFlow. El paso 5 se corre con la carpeta del paso 3 (python 5_validar_pt_convertido.py modelo_convertido) y usa tf_salidas.npz, que ya trae las salidas del original. El paso 4 solo se necesita para volver a generar ese archivo, en el entorno tf24-pulmon.

## Resultado esperado

- Paso 1: 613 tensores en el checkpoint; 201 del modelo, en los bytes 0 a 709 181 016.
- Paso 2: pesos_modelo.bin de 709 181 016 bytes; 201 de 201 tensores coinciden con el CRC32C del índice (0 errores); los 4 archivos de mBERT están.
- Paso 3: sha256 de model.safetensors = fadf9bf51399823ec12d8f9111d63f0b38d4a31571e1b27bf3e012861ee51d76 (el que exige pipeline_pulmon.ipynb), y los otros 5 archivos idénticos a los de modelos/deepLungCancer-BERT.
- Pasos 4 y 5: etiqueta igual en 616 de 616 piezas y 250 de 250 palabras; diferencia máxima de probabilidad 5,3 × 10⁻⁶; tokenización igual en 20 de 20 frases.
