# GNN-TCMcomp

## Project Description
This project proposes a **heterogeneous graph neural network (Hetero-GNN)** for **herb–herb compatibility prediction** by integrating multi-level biological information, including **herbs, ingredients, and targets**.  
Different molecular fingerprints and protein representations can be flexibly selected to evaluate their impact on prediction performance.

---

## Usage
The project supports flexible feature selection via command-line arguments.  
Different scripts correspond to different stages of the experimental pipeline:

```bash
# Run a preliminary experiment to explore appropriate training epochs under difficult learning settings
python shiyunxing_open.py --ingredient_feats rdkit --target_feats esm

# Hyperparameter search (hidden dimensions and learning rate)
python base_open.py --ingredient_feats rdkit --target_feats esm

# Epoch search with early stopping
python zaotingzhi_open.py --ingredient_feats rdkit --target_feats esm

# Final training and evaluation on the test set using optimal parameters
python ceshi_open.py --ingredient_feats rdkit --target_feats esm
```

## Arguments
--ingredient_feats: Ingredient feature type (maccs, morgan, or rdkit)
--target_feats: Target feature type (esm or probert)

## Required Data Files
Necessary node information, relational data, and feature representations should be prepared by the user in advance.
