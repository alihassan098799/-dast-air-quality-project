# Model Card: Air Quality Prediction Models

## Model description
This project trains machine learning models for classifying urban air quality levels into Good, Moderate, and Poor. The main model is a Random Forest classifier, and Logistic Regression is used as a comparison baseline. The models use pollutant and weather features such as PM10, NO2, O3, temperature, humidity, and wind speed.

## Intended use
The models are intended for a FAIR Data Science course experiment. They demonstrate how environmental and weather measurements can be used in a reproducible machine learning workflow. They are suitable for educational demonstration, metadata documentation, and repository publication.

## Out-of-scope uses
The models should not be used as an official public health warning system. They should not replace certified air quality monitoring services. They should not be used for regulatory, legal, or medical decisions.

## Training data
The training data contains urban pollutant and weather measurements. The workflow uses a training/test split and produces model artefacts, predictions, and evaluation metrics. DBRepo is used to document the input data infrastructure and provide citable access.

## Evaluation results
The evaluation results are stored in `outputs/predictions/evaluation_metrics.csv` after running the training script.

| Metric | Description |
|---|---|
| Accuracy | Overall correct classifications |
| Precision | Weighted class precision |
| Recall | Weighted class recall |
| F1 | Weighted F1 score |

## Limitations
The sample data in this repository is prepared for course demonstration and should be replaced with the final authoritative dataset for real research. The model performance depends on the quality and representativeness of environmental measurements. The air quality labels are simplified.

## Ethical considerations
The project does not use personal data. The main risk is over-interpreting outputs as authoritative predictions. Documentation and licensing are included to make reuse transparent.

## Licence
Generated model artefacts and metadata are shared under CC BY 4.0.
