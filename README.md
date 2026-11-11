# Cancer NLP pipelines (HUSI)

Three Jupyter notebooks that extract clinical entities from oncology notes in Spanish with pretrained models,
validate the TNM (AJCC 8th edition) and save a CSV and one timeline per patient. Everything runs offline.

| Notebook | Model | Python | Requirements |
|---|---|---|---|
| `pipeline_mama.ipynb` | `Alvaro8gb/es_breast_cancer_ehr_ner` (spaCy) | 3.11 | `requirements_mama.txt` (spaCy 3.5, as the model asks) |
| `pipeline_pulmon.ipynb` | `solarte7/deepLungCancer`, BERT (converted to PyTorch) | 3.12 | `requirements_pulmon_and_prostata.txt` |
| `pipeline_prostata.ipynb` | `ralzate/beto-prostata-ner` | 3.12 | `requirements_pulmon_and_prostata.txt` |

## Setup

The breast pipeline needs its own environment, because its model asks for spaCy `>=3.5.0,<3.6.0`:

```
python3.11 -m venv venv_mama
venv_mama/bin/pip install -r requirements_mama.txt
python3.12 -m venv venv_pulmon_prostata
venv_pulmon_prostata/bin/pip install -r requirements_pulmon_and_prostata.txt
```

Open each notebook with the kernel of its environment.

## Use

In the configuration cell of each notebook, fill in:

- `ENTRADA`: the folder of the patients of that cancer (one subfolder per patient, notes as `.txt`), one
  patient's folder, or one note;
- `CARPETA_SALIDA_BASE`: a local folder for the outputs;
- `CARPETA_MODELOS`: the folder with the models (by default `modelos`, next to the notebooks).

Then run all the cells in order. Only counts are printed. The CSV and the charts contain note text: keep them on
the server.

## Contents

- `modelos/`: the three models. Breast and prostate are as published on Hugging Face. Lung is the author's
  TensorFlow weights converted to PyTorch and validated against the original model.
- `rush_rules_es.tsv`: Spanish sentence rules of PyRuSH (from medSpaCy 1.3.1), used by the three
  notebooks to split the notes into sentences (stage 4).

Values that come from the data or from the specialists stay in Spanish, because the pipelines depend on them: the
labels of the prostate model (`CIRUGIA`, `FECHA`…), the note types (`Apoyo Dx Patología`…) and the reasons saved in
the CSV (`falta_T_N_o_M`, `no_es_TNM_AJCC`).

## Cómo correrlo (en español)

1. Crear los dos entornos de la sección *Setup*: uno con Python 3.11 para mama (spaCy 3.5) y otro con Python 3.12
   para pulmón y próstata.
2. Abrir cada notebook con el kernel de su entorno.
3. En la celda de configuración, llenar `ENTRADA` (la carpeta de los pacientes de ese cáncer, la de un paciente o
   una nota `.txt`), `CARPETA_SALIDA_BASE` (carpeta donde quedan el CSV y las gráficas) y, si los modelos no están
   en `modelos/` junto a los notebooks, `CARPETA_MODELOS`.
4. Correr todas las celdas en orden. Solo se imprimen conteos.
5. El CSV y las gráficas traen texto de las notas: deben quedarse en el servidor.
