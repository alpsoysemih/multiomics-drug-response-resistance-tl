# Pathway Enrichment Analysis Using pathfindR on GDSC Data

These codes perform pathway enrichment analysis using the [`pathfindR`](https://cran.r-project.org/web/packages/pathfindR/index.html) R package on the Genomics of Drug Sensitivity in Cancer (GDSC) dataset. The aim is to identify enriched biological pathways associated with drug resistance mechanisms in cancer cell lines for key drugs across major inhibitor classes.

## Objective

To explore biologically relevant mechanisms of resistance to:
- **Paclitaxel**
- **5-Fluorouracil (5-FU)**
- **Gemcitabine**
- **Cetuximab**

## Drug Classes

Each drug belongs to one of the following inhibitor classes:
- **Mitotic inhibitors** (Paclitaxel)
- **Cytoskeleton inhibitors** (Paclitaxel)
- **DNA replication inhibitors** (5-FU, Gemcitabine)
- **EGFR signaling inhibitors** (Cetuximab)

## Gene Selection Criteria

Differentially expressed genes (DEGs) were selected using the following thresholds:
- **|log₂FC| > 1**
- **Adjusted p-value < 0.05**

DEGs used in enrichment were derived from the best-performing **pan-drug models** trained using multi-omics data.

## Strategies Used

Pathway enrichment was performed under two distinct training strategies:

### Strategy 1
- **Training:** Included *all* drugs in the same inhibitor class.
- **Cancer types:** Included *all* cancer types from GDSC.
- **Test drugs:** Retained in training data.

### Strategy 2
- **Training:** Excluded *response profiles* of the test drug within the inhibitor class.
- **Drug-specific DEGs** and **all cancer types** were retained.
- Better simulates drug de novo prediction and removes potential data leakage.

## Methodology

- Gene features from top-performing pan-drug models were used for each strategy.
- Enrichment analysis was performed using `pathfindR` based on KEGG pathways.
- Each analysis was repeated per drug-strategy pair, resulting in **two enrichment outputs per drug**.

