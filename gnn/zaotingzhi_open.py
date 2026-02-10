import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score
import json
import copy
import os
import argparse

def run_epoch_search(ingredient_feat_type, target_feat_type):
    """
    Main function to run the epoch search experiment for a given feature combination.
    """
    # --- 1. Setup and Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dynamic output directory based on selected features
    output_dir = f'{ingredient_feat_type}_{target_feat_type}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved in: '{output_dir}/'")

    # --- 2. Data Loading ---
    # Static files
    herb_ingredient_file = 'ingredient_shuru.xlsx'
    ingr_tgt_pred_file = 'ingredients_target_predict_final.xlsx'
    ingr_tgt_known_file = 'ingredient_target_known_final.xlsx'
    compatibility_file = 'com_shuru.xlsx'
    herb_feature_file = 'herb_features.csv'

    # Dynamic feature file selection
    feature_files = {
        'ingredient': {
            'maccs': 'ingredient_features_maccs.csv',
            'morgan': 'ingredient_features_morgan.csv',
            'rdkit': 'ingredient_features_rdkit.csv'
        },
        'target': {
            'esm': 'target_features_esm.csv',
            'probert': 'target_features_probert.csv'
        }
    }
    
    ingredient_feature_file = feature_files['ingredient'][ingredient_feat_type]
    target_feature_file = feature_files['target'][target_feat_type]
    
    print(f"Loading ingredient features: {ingredient_feature_file}")
    print(f"Loading target features: {target_feature_file}")

    # Load dataframes
    df_hi = pd.read_excel(herb_ingredient_file)
    df_ip = pd.read_excel(ingr_tgt_pred_file)
    df_ik = pd.read_excel(ingr_tgt_known_file)
    df_ingredient_target = pd.concat([df_ip, df_ik], ignore_index=True)
    df_com = pd.read_excel(compatibility_file)
    df_herb_features = pd.read_csv(herb_feature_file)
    df_ingredient_features = pd.read_csv(ingredient_feature_file)
    df_target_features = pd.read_csv(target_feature_file)

    # Standardize identifier column names for easier processing
    df_ingredient_features.rename(columns={df_ingredient_features.columns[0]: 'ingredient'}, inplace=True)
    df_target_features.rename(columns={df_target_features.columns[0]: 'target'}, inplace=True)

    # --- 3. Graph Construction ---
    data = HeteroData()
    herbs = pd.unique(df_hi['herb'])
    ingredients = pd.unique(df_hi['ingredient'])
    targets = pd.unique(df_ingredient_target['target'])

    herb2id = {name: i for i, name in enumerate(herbs)}
    ingr2id = {name: i for i, name in enumerate(ingredients)}
    tgt2id = {name: i for i, name in enumerate(targets)}

    # Get feature dimensions dynamically
    herb_dim = df_herb_features.shape[1] - 1
    ingredient_dim = df_ingredient_features.shape[1] - 1
    target_dim = df_target_features.shape[1] - 1

    # Load node features
    herb_features = np.zeros((len(herbs), herb_dim), dtype=np.float32)
    for _, row in df_herb_features.iterrows():
        herb_name = row['herb']
        if herb_name in herb2id:
            herb_features[herb2id[herb_name]] = row.drop('herb').values
    data['herb'].x = torch.tensor(herb_features, dtype=torch.float)

    ingredient_features = np.zeros((len(ingredients), ingredient_dim), dtype=np.float32)
    for _, row in df_ingredient_features.iterrows():
        ingr_name = row['ingredient']
        if ingr_name in ingr2id:
            ingredient_features[ingr2id[ingr_name]] = row.drop('ingredient').values
    data['ingredient'].x = torch.tensor(ingredient_features, dtype=torch.float)

    target_features = np.zeros((len(targets), target_dim), dtype=np.float32)
    for _, row in df_target_features.iterrows():
        tgt_name = row['target']
        if tgt_name in tgt2id:
            target_features[tgt2id[tgt_name]] = row.drop('target').values
    data['target'].x = torch.tensor(target_features, dtype=torch.float)

    # Verify feature dimensions
    print(f"Herb feature dimension: {data['herb'].x.shape}")
    print(f"Ingredient feature dimension: {data['ingredient'].x.shape}")
    print(f"Target feature dimension: {data['target'].x.shape}")
    assert data['herb'].x.shape == (len(herbs), herb_dim), "Herb feature dimension mismatch"
    assert data['ingredient'].x.shape == (len(ingredients), ingredient_dim), "Ingredient feature dimension mismatch"
    assert data['target'].x.shape == (len(targets), target_dim), "Target feature dimension mismatch"

    # Edge indices
    src = df_hi['herb'].map(herb2id).values
    dst = df_hi['ingredient'].map(ingr2id).values
    data['herb', 'has', 'ingredient'].edge_index = torch.tensor([src, dst], dtype=torch.long)
    data['ingredient', 'rev_has', 'herb'].edge_index = torch.tensor([dst, src], dtype=torch.long)

    src = df_ingredient_target['ingredient'].map(ingr2id).dropna().astype(int)
    dst = df_ingredient_target['target'].map(tgt2id).dropna().astype(int)
    data['ingredient', 'hits', 'target'].edge_index = torch.tensor([src.values, dst.values], dtype=torch.long)
    data['target', 'rev_hits', 'ingredient'].edge_index = torch.tensor([dst.values, src.values], dtype=torch.long)

    # Herb pair labels
    herb_pairs, compat_labels = [], []
    for _, row in df_com.iterrows():
        h1, h2, cij = row['Herb1'], row['Herb2'], row['Cij']
        if h1 in herb2id and h2 in herb2id:
            herb_pairs.append((herb2id[h1], herb2id[h2]))
            compat_labels.append(cij)

    herb_pairs = torch.tensor(herb_pairs, dtype=torch.long).T
    compat_labels = torch.tensor(compat_labels, dtype=torch.float)

    # Move to GPU
    data = data.to(device)
    herb_pairs, compat_labels = herb_pairs.to(device), compat_labels.to(device)

    # --- 4. Data Splitting ---
    all_indices = np.arange(herb_pairs.shape[1])
    trainval_idx, test_idx = train_test_split(all_indices, test_size=0.1, random_state=66)
    trainval_pairs = herb_pairs[:, trainval_idx]
    trainval_labels = compat_labels[trainval_idx]

    # --- 5. Model Definition (Now with dynamic input dimensions) ---
    class HerbEncoder(torch.nn.Module):
        def __init__(self, hidden_channels, herb_dim, ingredient_dim, target_dim):
            super().__init__()
            self.herb_lin = Linear(herb_dim, hidden_channels)
            self.ingredient_lin = Linear(ingredient_dim, hidden_channels)
            self.target_lin = Linear(target_dim, hidden_channels)
            self.conv1 = HeteroConv({
                ('herb', 'has', 'ingredient'): SAGEConv((-1, -1), hidden_channels),
                ('ingredient', 'hits', 'target'): SAGEConv((-1, -1), hidden_channels),
                ('ingredient', 'rev_has', 'herb'): SAGEConv((-1, -1), hidden_channels),
                ('target', 'rev_hits', 'ingredient'): SAGEConv((-1, -1), hidden_channels),
            }, aggr='mean')
            self.conv2 = HeteroConv({
                ('herb', 'has', 'ingredient'): SAGEConv(hidden_channels, hidden_channels),
                ('ingredient', 'hits', 'target'): SAGEConv(hidden_channels, hidden_channels),
                ('ingredient', 'rev_has', 'herb'): SAGEConv(hidden_channels, hidden_channels),
                ('target', 'rev_hits', 'ingredient'): SAGEConv(hidden_channels, hidden_channels),
            }, aggr='mean')
            self.lin = Linear(hidden_channels, hidden_channels)

        def forward(self, x_dict, edge_index_dict):
            x_dict = {
                'herb': self.herb_lin(x_dict['herb']),
                'ingredient': self.ingredient_lin(x_dict['ingredient']),
                'target': self.target_lin(x_dict['target']),
            }
            x_dict = self.conv1(x_dict, edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
            x_dict = self.conv2(x_dict, edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
            x_dict['herb'] = self.lin(x_dict['herb'])
            return x_dict

    class CompatibilityPredictor(nn.Module):
        def __init__(self, hidden_channels, herb_dim, ingredient_dim, target_dim):
            super().__init__()
            self.encoder = HerbEncoder(hidden_channels, herb_dim, ingredient_dim, target_dim)
            self.predictor = nn.Sequential(
                nn.Linear(2 * hidden_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_channels, hidden_channels // 2),
                nn.ReLU(),
                nn.Linear(hidden_channels // 2, 1)
            )

        def forward(self, data, herb_pairs):
            x_dict = self.encoder(data.x_dict, data.edge_index_dict)
            h1 = x_dict['herb'][herb_pairs[0]]
            h2 = x_dict['herb'][herb_pairs[1]]
            pair = torch.cat([h1, h2], dim=1)
            return self.predictor(pair).squeeze()

    # --- 6. Training Function with Early Stopping ---
    def train_with_early_stopping(hidden_channels, lr, max_epochs, patience, min_epochs, train_idx, val_idx, data, herb_pairs, compat_labels):
        model = CompatibilityPredictor(hidden_channels, herb_dim, ingredient_dim, target_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        train_pairs, train_labels = herb_pairs[:, train_idx], compat_labels[train_idx]
        val_pairs, val_labels = herb_pairs[:, val_idx], compat_labels[val_idx]

        best_val_mse = float('inf')
        best_epoch = 0
        patience_counter = 0
        best_model_state = None

        for epoch in range(1, max_epochs + 1):
            model.train()
            optimizer.zero_grad()
            pred = model(data, train_pairs)
            train_loss = criterion(pred, train_labels)
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            model.eval()
            with torch.no_grad():
                val_pred = model(data, val_pairs)
                val_mse = criterion(val_pred, val_labels).item()

            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_epoch = epoch
                best_model_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            if epoch >= min_epochs and patience_counter >= patience:
                print(f"    Early stopping at epoch {epoch}. Best epoch was {best_epoch} with Val MSE: {best_val_mse:.4f}")
                break
        
        # Load the best model state before final evaluation
        model.load_state_dict(best_model_state)
        model.eval()
        with torch.no_grad():
            val_pred = model(data, val_pairs)
            # Re-calculate final mse and r2 on the best model state
            final_val_mse = criterion(val_pred, val_labels).item()
            final_val_r2 = r2_score(val_labels.cpu().numpy(), val_pred.cpu().numpy())

        return best_epoch, final_val_mse, final_val_r2

    # --- 7. 5-Fold Cross-Validation for Epoch Search --- Need to change to the best hyper-parameters!!!
    hidden_channels = 512
    lr = 0.001
    max_epochs = 800
    patience = 15
    min_epochs = 100
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    best_epochs, fold_mse, fold_r2 = [], [], []
    print("\n--- Starting 5-Fold CV for Epoch Search ---")
    for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(trainval_pairs.shape[1]))):
        print(f"--- Fold {fold+1}/5 ---")
        epoch, mse, r2 = train_with_early_stopping(
            hidden_channels, lr, max_epochs, patience, min_epochs, train_idx, val_idx, data, trainval_pairs, trainval_labels
        )
        best_epochs.append(epoch)
        fold_mse.append(mse)
        fold_r2.append(r2)
        print(f"Fold {fold+1} Finished. Best Epoch: {epoch}, Final Val MSE: {mse:.4f}, Final Val R2: {r2:.4f}")

    # --- 8. Aggregate and Save Results ---
    avg_best_epoch = int(np.mean(best_epochs))
    avg_mse = np.mean(fold_mse)
    avg_r2 = np.mean(fold_r2)
    print("\n--- CV Finished ---")
    print(f"Average Best Epoch: {avg_best_epoch}")
    print(f"Average Val MSE: {avg_mse:.4f}")
    print(f"Average Val R2: {avg_r2:.4f}")

    results = {
        'Hidden_Channels': hidden_channels,
        'Learning_Rate': lr,
        'Average_Best_Epoch': avg_best_epoch,
        'Average_Val_MSE': avg_mse,
        'Average_Val_R2': avg_r2,
        'Minimum_Epochs': min_epochs,
        'Patience': patience,
        'Max_Epochs': max_epochs,
        'Fold_Results': [
            {'Fold': i+1, 'Best_Epoch': e, 'Val_MSE': m, 'Val_R2': r}
            for i, (e, m, r) in enumerate(zip(best_epochs, fold_mse, fold_r2))
        ]
    }

    results_path = os.path.join(output_dir, "epoch_search_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nEpoch search results saved to '{results_path}'")
    print("Experiment completed.")

if __name__ == '__main__':
    # --- Argument Parser for Feature Selection ---
    parser = argparse.ArgumentParser(description="Run Epoch Search for Herb Compatibility Prediction with selectable features.")
    parser.add_argument(
        '--ingredient_feats', 
        type=str, 
        default='maccs', 
        choices=['maccs', 'morgan', 'rdkit'],
        help='Type of fingerprint to use for ingredients.'
    )
    parser.add_argument(
        '--target_feats', 
        type=str, 
        default='esm', 
        choices=['esm', 'probert'],
        help='Type of features to use for targets.'
    )
    args = parser.parse_args()

    # Run the experiment with the specified features
    run_epoch_search(args.ingredient_feats, args.target_feats)