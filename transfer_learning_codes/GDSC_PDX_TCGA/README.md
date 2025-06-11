# Transfer Learning for Pan-Drug Response Prediction using Multi-Omics Integration and Deep Neural Networks

This repository contains deep learning models and analysis pipelines used for predicting drug response using transfer learning (TL) and multi-omics integration, inspired by the MOLI framework (Sharifi-Noghabi et al., 2019). It expands the scope of MOLI by incorporating drugs targeting four major pathways and testing across PDX and TCGA datasets.

---

## Overview

- Models are trained on **GDSC** and tested on **PDX**, **TCGA**, or both.
- Uses gene expression, somatic mutation, and copy number alteration (CNA) datasets.
- Implements **transfer learning** (TL) using a **deep neural network (DNN)** based architecture.
- Applies **four training strategies** to address potential biases in shared drugs and cancer types across datasets.

---

## Training & Testing Setup

### Datasets

- **GDSC_PDX**: Trained on GDSC, tested on PDX.
- **GDSC_TCGA**: Trained on GDSC, tested on TCGA.
- **GDSC_PDX_TCGA**: Trained on GDSC, tested on both PDX and TCGA (includes four-strategy setup).

---

## Model Architecture

Feed-forward subnetworks were independently constructed for:
- **Gene Expression**
- **Somatic Mutation**
- **Copy Number Alteration (CNA)**

Each omics layer follows a common architecture and hyper-parameter ranges from the MOLI TL framework to ensure fair comparison.

After separate encoding, outputs were:
- Concatenated to form three integration strategies:
  - **EM** (Expression + Mutation)
  - **EC** (Expression + CNA)
  - **EMC** (Expression + Mutation + CNA)

> Only genes common across datasets were retained.

### Final Classification Layer
- Fine-tuned using frozen omics layers
- Trained using **binary cross-entropy loss**
- Optimized using **Adagrad**

---

## Hyper-Parameter Search

- 50 iterations of **random search**
- **5-fold cross-validation** for model selection and generalizability

| Hyper-Parameter | Values |
|------------------|--------|
| Mini-batch size  | 8, 16, 32, 64 |
| Neuron sizes     | 1024, 512, 256, 128, 64, 32, 16 |
| Learning rate    | 0.5, 0.1, 0.05, 0.01, 0.001, 0.005, 0.0005, 0.0001, 0.00005, 0.00001 |
| Epochs           | 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100 |
| Dropout rate     | 0.3, 0.4, 0.5, 0.6, 0.7, 0.8 |
| Weight decay     | 0.01, 0.001, 0.1, 0.0001 |

---

## Improvements Over MOLI

- **Original MOLI**:
  - Used all genes with >5% variance
  - No fine-tuning step
  - TL limited to EGFR inhibitors (erlotinib, cetuximab)

- **This Study**:
  - Expanded to 4 pathway classes:
    - Mitotic
    - Cytoskeleton
    - DNA replication
    - EGFR signaling
  - Tested 5 drugs in PDX and 8 drugs in TCGA
  - Included cross-dataset drugs: **Paclitaxel, 5-FU, Gemcitabine, Cetuximab**
  - Gene features = union of DEGs per drug class → reduced noise and improved interpretability

---

## Transfer Learning Strategies (Only for GDSC_PDX_TCGA)

1. **Strategy 1**:  
   - All drugs in the same pathway included  
   - All cancer types retained  

2. **Strategy 2**:  
   - Removed tested drug's response and unique features (DEGs)  
   - All cancer types retained  

3. **Strategy 3**:  
   - Excluded overlapping cancer types between training and test datasets  
   - All pathway drugs retained  

4. **Strategy 4**:  
   - Removed both drug-specific response and cancer type overlap  

> These strategies were designed to avoid performance overestimation caused by drug/cancer type leakage between training and test sets.

---

## Evaluation

- **Primary Metric**: AUCPR (Area Under the Precision-Recall Curve)
- **Secondary Metric**: AUC (ROC)

---

## Citation

Sharifi-Noghabi H, et al. (2019). MOLI: Multi-Omics Late Integration with Deep Neural Networks for drug response prediction. *Bioinformatics*.

---

## Contact

**Semih Alpsoy**  
Türkisch-Deutsche Universität, Department of Molecular Biotechnology  
Acıbadem University, Department of Biostatistics and Bioinformatics  
📧 *your-email@example.com*

---

This repository provides a scalable framework for pathway-based transfer learning in drug response prediction using deep multi-omics integration.
