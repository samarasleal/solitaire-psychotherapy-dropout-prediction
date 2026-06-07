# GRU-based longitudinal multimodal signals processing for early dropout prediction in psychotherapy

## Overview

This repository contains the Python scripts developed for longitudinal multimodal signals processing for early dropout prediction in psychotherapy using machine learning models. The data used here was collected during the SOLITAIRE project's clinical trial and can be shared upon reasonable request to the authors. The package supports reproducible analysis of acoustic features and engagement markers and their association with therapy disengagement.

## How to cite

If you use this resource, please cite the associated publication bellow.

### Associated publication

⁠Leal, S. S., Ntalampiras, S., Trabacca, A., Bellani, M., & Sassi, R. (2026). GRU-based temporal modeling of longitudinal multimodal signals for early dropout in psychotherapy. In Proceedings of the 39th IEEE International Symposium on Computer-Based Medical Systems (IEEE CBMS). [DOI to be added]

## Data contents

```text
code/
  DataPreparation.ipynb
  Exp1_GRU.ipynb
  Exp1_MLP.ipynb
  Exp2_GRU.ipynb
  Exp3_Eng.ipynb
  Run_RNN_dropout.ipynb
  Run_RNN_dropout.py
```

## Dataset relationship

This repository is associated with the main clinical project repository: 10.5281/zenodo.20526320


## Expected data format

The data files are organized in wide format, with one row per anonymized patient. Acoustic features are stored as session- and segment-specific columns.

| Column pattern | Description |
|---|---|
| `Patient_ID` | Anonymized patient identifier |
| `MFCC{coeff}_mean` | Mean MFCC value for coefficient `{coeff}` extracted from a 5-second speech segment |
| `Embedding_npy` | Vav2vec 2.0 embedding "npy" tensor file per session. Each session is composed by a set 5-second speech segments |

 MFCC coefficients range from `coeff1` to `coeff13`.

## Code summary

The code supports the following steps:

1. loading the dataset "df_all_Horizon2.xlsx" containing the extracted feature [The data used here was collected during the SOLITAIRE clinical trial and can be shared upon reasonable request to the authors];
2. restructuring feature tables into model-ready formats;
3. preparing MFCCs and wav2vec 2.0 feature representations;
4. running patient-independent Leave-One-Out cross-validation;
5. training and evaluating;
6. summarizing metrics.


## Ethical and access notes

This repository does not contain raw clinical audio or directly identifiable patient information. 

## License

- Dataset/features: CC BY 4.0;
- Code: MIT License or Apache 2.0.

## Versioning

If scripts or features are updated later, a new Zenodo version can be released while preserving the same concept DOI.
