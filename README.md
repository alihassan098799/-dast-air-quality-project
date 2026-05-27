# Machine Learning Prediction of Urban Air Quality

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20378637.svg)](https://doi.org/10.5281/zenodo.20378637)

## Abstract
This project predicts urban air quality classes in Austrian cities using pollutant and weather features. The workflow includes data preparation, machine learning model training, evaluation, DBRepo documentation, and FAIR metadata publication.

## Repository structure
```text
data/raw/                 Input CSV dataset
data/processed/           Processed ML-ready dataset
src/                      Python source code
outputs/figures/          Generated plots and confusion matrix
outputs/predictions/      Predictions and metrics
models/                   Trained model artefacts
dbrepo/                   SQL schema, view, and subset definitions
docs/metadata/            FAIR4ML and Croissant metadata
docs/model-card.md        Model card
ro-crate-metadata.json    RO-Crate metadata
codemeta.json             CodeMeta metadata
CITATION.cff              Citation metadata
```

## Installation
```bash
pip install -r requirements.txt
```

## Run the experiment
```bash
python src/train_model.py
```

## Inputs
The input data contains pollutant and weather features:
PM10, NO2, O3, temperature, humidity, wind speed, and air quality class.

## Outputs
The script creates trained models, predictions, evaluation metrics, confusion matrix, feature importance chart, and PM10 histogram.

## Related identifiers
- GitHub: https://github.com/alihassan098799/-dast-air-quality-project
- Zenodo DOI: https://doi.org/10.5281/zenodo.20378637
- DBRepo database: https://test.dbrepo.tuwien.ac.at/database/ed511834-2154-4cee-8676-0ec57829e465/info
- DBRepo view PID: https://handle.test.datacite.org/10.82556/037y-wk92
- DBRepo subset PID: https://handle.test.datacite.org/10.82556/8xrb-a603
- Initial DMP DOI: https://doi.org/10.70124/rnr36-1rn91

## Licences
- Code: MIT License
- Generated data and metadata: CC BY 4.0
- Input data: follows the original open-data source licence.
