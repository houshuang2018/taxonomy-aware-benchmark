This repository contains the datasets and scripts needed to reproduce the analysis of the manuscript: Hou, Shuang, Hexin Shen, and Yong Zhang. "Taxonomy-aware, disorder-matched benchmarking of phase-separating protein predictors." *bioRxiv* (2026): 2026-02.

# Data sources

The following public databases were used for the construction of benchmark datasets:

- [PhaSePro](https://phasepro.elte.hu/)
- [PhaSepDB](https://db.phasep.pro/)
- [LLPSDB](http://bio-comp.org.cn/llpsdbv2/home.html)
- [DrLLPS](https://llps.biocuckoo.cn/)
- [Uniprot](https://www.uniprot.org/)
- [CD-CODE](https://cd-code.org/)
- [BioGRID](https://thebiogrid.org/)

# Dataset construction

The scripts used to construct the positive and negative datasets are provided in the `XX` directory. The dataset construction pipelines are summarized below.

## **Positive dataset construction**

Phase separating positive proteins were collected by integrating four curated LLPS-focused databases, including [PhaSePro](https://phasepro.elte.hu/), [PhaSepDB](https://db.phasep.pro/), [LLPSDB](http://bio-comp.org.cn/llpsdbv2/home.html), [DrLLPS](https://llps.biocuckoo.cn/). Proteins shorter than 50 amino acids or longer than 3,000 amino acids were excluded. Proteins containing uncommon amino acids such as “BJOUXZ” were also excluded.

The combined positive records were subjected to systematic screening and deduplication. For LLPSDB, proteins from “Ambiguous” subset were excluded. For DrLLPS, only proteins with condensate information and tissue/cell annotations were included. Proteins reported in publications covering more than 10 entries from DrLLPS were excluded to avoid possible high-throughput annotations. Redundant records across databases were then collapsed to a non-redundant protein set, and the resulting taxonomic composition of the set was quantified. Protists were excluded from the set due to insufficient number of high-confidence phase separating positive proteins. A final set of 2,216 high-confidence PSPs was obtained and used as the positive benchmark set. 

## **Negative dataset construction**

Candidate negative proteins were derived from the Swiss-Prot database with sequence length from 50 to 3,000 amino acids. To minimize contamination by proteins with reported condensate association, candidates showing sequence similarity to any protein included in the LLPS-focused databases used for positive set construction, as well as [CD-CODE v1](https://cd-code.org/), were removed based on BLASTP alignments using thresholds of sequence identity ≥40% and alignment coverage ≥70%. To reduce potential contamination by proteins used as positives in existing predictors, we removed the available training positive sets of all evaluated predictors from the benchmark negative set. This exclusion could not be applied to Droppler because its training sets were unavailable. In addition, to minimize the likelihood of retaining proteins with potential LLPS-related functional association, we further removed proteins annotated in [BioGRID v4.4](https://thebiogrid.org/) as first-degree interactors of the positive proteins. Candidates containing uncommon amino acids such as “BJOUXZ” were further excluded. The remaining proteins were classified into taxonomic groups. To reduce sequence redundancy while preserving taxon-specific diversity, CD-HIT clustering was performed independently within each taxonomic group using a sequence identity cutoff of 0.3.

Final negative set construction was performed by stratified sampling, with target counts guided by the taxonomic group and IDP/non-IDP composition of the positive set. Within each stratum, specific proteins were selected using an embedding-based diversity sampling strategy. Mean-pooled ESM-2 embeddings were used to represent protein sequences, and a cosine-distance-based k-center greedy algorithm was applied to select proteins that maximally covered the sequence embedding space. The final negative benchmark set consisted of 12,071 proteins.

# Files description

- File name: `benchmark_datasets.csv`
  - Description: List of proteins in benchmark positive and negative sets.
- Directory name: `./scripts`
  - `positive_dataset_construction.ipynb`: jupyter file to construct positive benchmark dataset.
  - `negative_dataset_construction.ipynb`: jupyter file to construct negative benchmark dataset.
  - `calculate_sequence_biophysical_features.py`: python script to calculate sequence and biophysical features.
  - `predictor_evaluation.py`: python script to evaluate predictors with ten negative subsampling repeats.
- Directory name: `./figures`
  - Description: Figures generated for the manuscript.
- Directory name: `./tables`
  - Description: Tables generated for the manuscript.
- File name: `./features/benchmark_sequence_biophysical_features.csv`
  - Description: sequence and biophysical features for proteins in benchmark dataset.  

