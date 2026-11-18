# PCDS Brain-Age Classification

Python scripts for ILAS classification, NILS-LSA external validation, and SHAP analysis using regional brain age gap (rBAG) features.

## 1. Installation

Create and activate a Conda environment, then install the required packages:

```bash
conda create -n pcds-revise python=3.11 -y
conda activate pcds-revise
python -m pip install -r requirements.txt
```

Run the following commands from the directory containing the Python scripts and input TXT files.

## 2. Input data

The TXT files contain the numerical data extracted from the supplementary Excel files. The Excel sheets include subject identifiers and headers, which are excluded from the TXT feature matrices.

| TXT file | Supplementary file | Worksheet | Contents |
|---|---|---|---|
| `ILAS_rBAG_53.txt` | `Supplementary Data 3.xlsx` | `Classifier_Input_53` | ILAS, 125 × 53 ROI features |
| `ILAS_Group.txt` | `Supplementary Data 3.xlsx` | `Classifier_Input_53` | ILAS, 125 group labels |
| `ILAS_rBAG_442.txt` | `Supplementary Data 4.xlsx` | `Classifier_Input_442` | ILAS, 125 × 442 ROI features |
| `NILS_rBAG_53.txt` | `Supplementary Data 5.xlsx` | `Classifier_Input_validation` | NILS-LSA, 211 × 53 ROI features |
| `NILS_Group.txt` | `Supplementary Data 5.xlsx` | `Classifier_Input_validation` | NILS-LSA, 211 group labels |

Feature files are whitespace-delimited, with one participant per row and one ROI per column. Group labels are `0 = Robust` and `1 = PCDS`. Preserve participant order between each feature file and its group-label file, and preserve the 53-ROI column order across cohorts.

## 3. Run the analyses

### ILAS: 53-ROI model training and cross-validation

```bash
python run_roi53_ILAS.py \
  --ilas-features ILAS_rBAG_53.txt \
  --ilas-labels ILAS_Group.txt \
  --output-dir outputs/roi53_ilas
```

### NILS-LSA: external validation

Run this after the ILAS analysis; it uses the saved ILAS model without refitting.

```bash
python run_roi53_NILS.py \
  --model outputs/roi53_ilas/roi53_final_model.joblib \
  --nils-features NILS_rBAG_53.txt \
  --nils-labels NILS_Group.txt \
  --output-dir outputs/roi53_nils
```

### SHAP: explain NILS-LSA predictions

Requires `roi53_labels.txt`, a text file with one ROI name per line in the same order as the 53 feature columns.

```bash
python run_roi53_SHAP.py \
  --model outputs/roi53_ilas/roi53_final_model.joblib \
  --ilas-features ILAS_rBAG_53.txt \
  --nils-features NILS_rBAG_53.txt \
  --roi-labels roi53_labels.txt \
  --output-dir outputs/roi53_shap
```

### ILAS: 442-ROI model training and cross-validation

```bash
python run_roi442_ILAS.py \
  --ilas-features ILAS_rBAG_442.txt \
  --ilas-labels ILAS_Group.txt \
  --output-dir outputs/roi442_ilas
```
