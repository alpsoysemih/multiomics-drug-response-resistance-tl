# Differential Expression Analysis of GDSC Drugs by Inhibitor Class

This code performs differential gene expression analyses with the Genomics of Drug Sensitivity in Cancer (GDSC) dataset. The aim is to identify drug-specific differentially expressed genes (DEGs) by comparing sensitive and resistant cancer cell lines for each drug within major drug classes.

## Drug Classes Analyzed

Drugs were grouped into the following inhibitor classes based on their mechanisms of action:

- **Mitotic inhibitors**
- **Cytoskeleton inhibitors**
- **DNA replication inhibitors**
- **EGFR signaling inhibitors**

## Methodology

- Cell lines in the GDSC dataset were labeled as **sensitive** or **resistant**.
- For each drug, sensitive and resistant groups were compared to identify differentially expressed genes (DEGs).
- DEGs were identified independently for each drug using the **limma** R package.
- Unionized DEGs across all drugs within each class were recorded.

