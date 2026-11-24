# PLN pipelines

Tres notebooks de Jupyter que extraen entidades clínicas de notas oncológicas en español con modelos preentrenados, validan el TNM y guardan un CSV y dos gráficas por paciente.

| Notebook | Modelo |
|---|---|
| pipeline_mama.ipynb | Alvaro8gb/es_breast_cancer_ehr_ner (spaCy) |
| pipeline_pulmon.ipynb | solarte7/deepLungCancer (BERT) |
| pipeline_prostata.ipynb | ralzate/beto-prostata-ner (BETO) |

## Entornos

Se necesitan dos entornos:

- **Mama:** Python 3.11 con los paquetes de requirements_mama.txt. Su modelo pide spaCy 3.5 (>=3.5.0,<3.6.0), que no funciona con Python 3.12.
- **Pulmón y próstata:** Python 3.12 con los paquetes de requirements_pulmon_and_prostata.txt.

Cada entorno se crea con python -m venv y se instala con pip install -r y el archivo de requirements que le corresponde. Cada notebook se abre con el kernel de su entorno.

## Modelos

Los modelos no están en este repositorio. Van en una carpeta modelos, junto a los notebooks, con una subcarpeta por modelo:

| Subcarpeta | De dónde sale |
|---|---|
| modelos/es_breast_cancer_ehr_ner | Hugging Face, Alvaro8gb/es_breast_cancer_ehr_ner |
| modelos/deepLungCancer-BERT | Pesos de TensorFlow del autor (Google Drive enlazado en el repositorio de GitHub solarte7/deepLungCancer), convertidos a PyTorch con los scripts de conversion_pulmon |
| modelos/beto-prostata-ner | Hugging Face, ralzate/beto-prostata-ner |

El modelo de pulmón no se puede descargar listo: se arma con los pasos 1 a 3 de conversion_pulmon y luego se copian sus 6 archivos a modelos/deepLungCancer-BERT (ver el README de esa carpeta).

Cada notebook comprueba la huella sha256 de su modelo antes de usarlo.

## Uso

En la celda de configuración de cada notebook, llenar:

- **ENTRADA:** la carpeta de los pacientes de ese cáncer (una subcarpeta por paciente), la carpeta de un paciente o una nota .txt.
- **CARPETA_SALIDA_BASE:** una carpeta local donde se guardan las salidas.
- **CARPETA_MODELOS:** solo si los modelos no están en la carpeta modelos junto a los notebooks. En ese caso, la ruta completa de la carpeta modelos, sin el nombre del modelo.

Luego, correr todas las celdas en orden.

## Salidas

Cada corrida guarda en la carpeta de salida:

- **El CSV de extracciones:** una fila por entidad, con el paciente, la nota, su tipo y su fecha, la etiqueta, el texto extraído, su posición en la nota, el texto que la rodea y la validación del TNM.
- **Entidades resaltadas:** una página HTML por paciente, con todas sus notas y las entidades resaltadas, un color por etiqueta.
- **Línea de tiempo:** una imagen PNG por paciente, con una fila por etiqueta y los valores de cada nota en el tiempo.
- **Registros:** registro_corridas.csv, con los conteos de cada corrida, y registro_versiones.csv, con la huella del modelo y las versiones usadas.

El CSV y las gráficas traen texto de las notas: deben quedarse en el equipo o servidor donde se corre el pipeline.

## Contenido

- **pipeline_mama.ipynb, pipeline_pulmon.ipynb, pipeline_prostata.ipynb:** los tres pipelines.
- **requirements_mama.txt, requirements_pulmon_and_prostata.txt:** los paquetes de cada entorno.
- **rush_rules_es.tsv:** las reglas en español de PyRuSH, que usan los tres notebooks para dividir las notas en oraciones. Es una copia sin cambios de resources/es/rush_rules.tsv de medSpaCy 1.3.1 (© 2016 Department of Biomedical Informatics, University of Utah, licencia Apache 2.0).
- **conversion_pulmon:** los scripts que arman el modelo de pulmón a partir de los pesos del autor y lo validan.
