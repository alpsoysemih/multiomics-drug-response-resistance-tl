# Multi-Omics Preprocessing Workflow (MOLI-Based)

This repository contains preprocessing scripts and descriptions used to standardize gene expression, somatic mutation, copy number alteration (CNA), and drug response datasets from GDSC, PDX, and TCGA. All preprocessing steps follow the **Multi-Omics Late Integration (MOLI)** pipeline (Sharifi-Noghabi et al., 2019), ensuring compatibility across datasets and facilitating downstream drug response prediction.

## Datasets Processed

- **GDSC** (Genomics of Drug Sensitivity in Cancer)
- **PDX** (Patient-Derived Xenografts)
- **TCGA** (The Cancer Genome Atlas)

---

## 1. Gene Expression Data

### PDX
- Converted from **FPKM** to **TPM**
- Applied **log2(TPM + 1)** transformation

### TCGA
- Already normalized to **TPM**
- Applied **log2(TPM + 1)** transformation

### GDSC
- Used **RMA-normalized** data
- Applied **log2 transformation**

➡️ All datasets were made comparable through consistent **log2 TPM-style transformations**.

---

## 2. Batch Effect Removal

- **Pairwise homogenization** performed between datasets
- Applied **`ComBat`** function from the `sva` R package (Leek et al., 2012)
- Ensures consistent scale and removes platform- or dataset-specific biases

---

## 3. Somatic Mutation Data

- Only **non-synonymous mutations** affecting protein structure were retained
- Mutation data was **binarized**:
  - `1` = gene contains a relevant mutation
  - `0` = gene is wild-type or silent

---

## 4. Copy Number Alteration (CNA) Data

### TCGA
- CNA values calculated by dividing gene-level copy number by **sample-specific ploidy**
- Resulting values were **log2-transformed**
- Ensures comparability with GDSC and PDX CNA profiles

### GDSC & PDX
- Used **gene-level total copy number estimates**
- **Insertions or deletions** were binarized:
  - `1` = gene has copy number alteration (gain/loss)
  - `0` = copy-neutral region

---

## 5. Drug Response Data

### GDSC
- Used in its **binarized** form as described in Iorio et al. (2016)

### PDX & TCGA
- Drug response based on **RECIST criteria** (Schwartz et al., 2016)
  - **Sensitive**: Complete Response (CR) or Partial Response (PR)
  - **Resistant**: Progressive Disease (PD) or Stable Disease (SD)

---

## References

- Sharifi-Noghabi H, et al. (2019). MOLI: Multi-Omics Late Integration with Deep Neural Networks for drug response prediction. *Bioinformatics*.
- Leek JT, et al. (2012). The sva package for removing batch effects and other unwanted variation in high-throughput experiments. *Bioinformatics*.
- Iorio F, et al. (2016). A landscape of pharmacogenomic interactions in cancer. *Cell*.
- Schwartz LH, et al. (2016). RECIST 1.1—Update and clarification: From the RECIST committee. *Eur J Cancer*.

---


This preprocessing pipeline ensures high-quality, standardized multi-omics input for predictive modeling and pathway analysis.
