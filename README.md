# Code Repository for "Untargeted screening of drug-like compounds via high-throughput metabolomics reveals inhibitors of clinically-relevant pathways"

It contains the code developed for preprocessing metabolomics data, downstream analysis, and training the CNN–GNN model described in the paper. The repository is provided as a reference for readers who wish to examine the implementation details and follow along with the methodological description.  

---

## Repository Structure  

- **`data_analysis/`**  
  MATLAB scripts used for preprocessing the raw metabolomics data into response matrices suitable for downstream analysis.  

- **`machine_learning/`**  
  Python scripts and modules for downstream data processing, integration with pathway information, and model utilities.  

- **`machine_learning/notebooks/runs.ipynb/`**  
  Jupyter notebook that defines model configurations, runs training of the CNN–GNN model, and saves results.  

---

## Notes  

- The repository reflects the code used in the published work and is provided for transparency and reference.  
- Readers are encouraged to explore the scripts and notebooks to understand the workflow and the implementation of the described methods.  
- Preprocessing and training depend on large-scale datasets and environment-specific configurations, which are not included here.  

---
