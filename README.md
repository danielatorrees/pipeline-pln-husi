# NLP pipelines

Three Jupyter notebooks that extract clinical entities from oncology notes in Spanish with pretrained models.

| Notebook | Model 
|---|---|---|---|
| `pipeline_mama.ipynb` | `Alvaro8gb/es_breast_cancer_ehr_ner` (spaCy) 
| `pipeline_pulmon.ipynb` | `solarte7/deepLungCancer`, BERT 
| `pipeline_prostata.ipynb` | `ralzate/beto-prostata-ner` 

## Setup

The breast pipeline needs its own environment, because its model asks for spaCy `>=3.5.0,<3.6.0`:

```
python3.11 -m venv venv_mama
venv_mama/bin/pip install -r requirements_mama.txt
python3.12 -m venv venv_pulmon_prostata
venv_pulmon_prostata/bin/pip install -r requirements_pulmon_and_prostata.txt
```

Open each notebook with the kernel of its environment.

## Models

The models are not in this repository. Put them in a `modelos/` folder next to the notebooks, one subfolder each:

| Subfolder | Source |
|---|---|
| `modelos/es_breast_cancer_ehr_ner` | Hugging Face `Alvaro8gb/es_breast_cancer_ehr_ner`|
| `modelos/deepLungCancer-BERT` | The author's TensorFlow weights (Google Drive linked from the `solarte7/deepLungCancer` GitHub repository), converted to PyTorch |
| `modelos/beto-prostata-ner` | Hugging Face `ralzate/beto-prostata-ner`|

If they are in another folder, write its full path in `CARPETA_MODELOS`.

## Use

In the configuration cell of each notebook, fill in:

- `ENTRADA`: the folder of the patients of that cancer, one patient's folder, or one note.
- `CARPETA_SALIDA_BASE`: a folder for the outputs.
- `CARPETA_MODELOS`: only if the models are not in `modelos/` next to the notebooks.

Then run all the cells in order.

## Contents

- `pipeline_mama.ipynb`, `pipeline_pulmon.ipynb`, `pipeline_prostata.ipynb`: the three pipelines.
- `requirements_mama.txt`, `requirements_pulmon_and_prostata.txt`: the packages of each environment.
- `rush_rules_es.tsv`: Spanish sentence rules of PyRuSH (from medSpaCy 1.3.1), used by the three
  notebooks to split the notes into sentences.

## Cómo correrlo (en español)

1. Crear los dos entornos de la sección *Setup*: uno con Python 3.11 para mama (spaCy 3.5) y otro con Python 3.12
   para pulmón y próstata.
2. Abrir cada notebook con el kernel de su entorno.
3. Poner los modelos en una carpeta `modelos/` junto a los notebooks, con las subcarpetas de la sección *Models*.
   Si están en otra carpeta, escribir su ruta completa en `CARPETA_MODELOS`.
4. En la celda de configuración, llenar `ENTRADA` (la carpeta de los pacientes de ese cáncer, la de un paciente o
   una nota `.txt`) y `CARPETA_SALIDA_BASE` (la carpeta donde quedan el CSV y las gráficas).
5. Correr todas las celdas en orden. La salida es un CSV con las entidades y una gráfica por paciente.
