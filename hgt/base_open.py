import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import Linear, HGTConv
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score
from itertools import product
import json
import os
import argparse

def run_hyperparameter_search(ingredient_feat_type, target_feat_type):
    """
    Main function to run the hyperparameter search experiment for a given feature combination.
    """
    # --- 1. Setup and Configuration ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create dynamic output directory based on selected features
    output_dir = f'{ingredient_feat_type}_{target_feat_type}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"Results will be saved in: '{output_dir}/'")

    # --- 2. Data Loading ---
    herb_ingredient_file = 'ingredient_shuru.xlsx'
    ingr_tgt_pred_file = 'ingredients_target_predict_final.xlsx'
    ingr_tgt_known_file = 'ingredient_target_known_final.xlsx'
    compatibility_file = 'com_shuru.xlsx'
    herb_feature_file = 'herb_features.csv'

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

    df_hi = pd.read_excel(herb_ingredient_file)
    df_ip = pd.read_excel(ingr_tgt_pred_file)
    df_ik = pd.read_excel(ingr_tgt_known_file)
    df_ingredient_target = pd.concat([df_ip, df_ik], ignore_index=True)
    df_com = pd.read_excel(compatibility_file)
    df_herb_features = pd.read_csv(herb_feature_file)
    df_ingredient_features = pd.read_csv(ingredient_feature_file)
    df_target_features = pd.read_csv(target_feature_file)

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

    herb_dim = df_herb_features.shape[1] - 1
    ingredient_dim = df_ingredient_features.shape[1] - 1
    target_dim = df_target_features.shape[1] - 1
    
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

    print(f"Herb feature dimension: {data['herb'].x.shape}")
    print(f"Ingredient feature dimension: {data['ingredient'].x.shape}")
    print(f"Target feature dimension: {data['target'].x.shape}")

    src = df_hi['herb'].map(herb2id).values
    dst = df_hi['ingredient'].map(ingr2id).values
    data['herb', 'has', 'ingredient'].edge_index = torch.tensor([src, dst], dtype=torch.long)
    data['ingredient', 'rev_has', 'herb'].edge_index = torch.tensor([dst, src], dtype=torch.long)

    src = df_ingredient_target['ingredient'].map(ingr2id).dropna().astype(int)
    dst = df_ingredient_target['target'].map(tgt2id).dropna().astype(int)
    data['ingredient', 'hits', 'target'].edge_index = torch.tensor([src.values, dst.values], dtype=torch.long)
    data['target', 'rev_hits', 'ingredient'].edge_index = torch.tensor([dst.values, src.values], dtype=torch.long)

    herb_pairs, compat_labels = [], []
    for _, row in df_com.iterrows():
        h1, h2, cij = row['Herb1'], row['Herb2'], row['Cij']
        if h1 in herb2id and h2 in herb2id:
            herb_pairs.append((herb2id[h1], herb2id[h2]))
            compat_labels.append(cij)

    herb_pairs = torch.tensor(herb_pairs, dtype=torch.long).T
    compat_labels = torch.tensor(compat_labels, dtype=torch.float)
    
    data = data.to(device)
    herb_pairs, compat_labels = herb_pairs.to(device), compat_labels.to(device)

    # --- 4. Data Splitting ---
    all_indices = np.arange(herb_pairs.shape[1])
    trainval_idx, test_idx = train_test_split(all_indices, test_size=0.1, random_state=66)
    trainval_pairs = herb_pairs[:, trainval_idx]
    trainval_labels = compat_labels[trainval_idx]
    test_pairs = herb_pairs[:, test_idx]
    test_labels = compat_labels[test_idx]

    # --- 5. Model Definition (Use HGTConv) ---
    class HerbEncoder(torch.nn.Module):
        def __init__(self, hidden_channels, herb_dim, ingredient_dim, target_dim, num_heads):
            super().__init__()
            self.herb_lin = Linear(herb_dim, hidden_channels)
            self.ingredient_lin = Linear(ingredient_dim, hidden_channels)
            self.target_lin = Linear(target_dim, hidden_channels)

            self.conv = HGTConv(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                metadata=data.metadata(),
                heads=num_heads
            )
            self.lin = Linear(hidden_channels, hidden_channels)

        def forward(self, x_dict, edge_index_dict):
            x_dict = {
                'herb': self.herb_lin(x_dict['herb']),
                'ingredient': self.ingredient_lin(x_dict['ingredient']),
                'target': self.target_lin(x_dict['target']),
            }
            x_dict = self.conv(x_dict, edge_index_dict)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
            x_dict['herb'] = self.lin(x_dict['herb'])
            return x_dict

    class CompatibilityPredictor(nn.Module):
        def __init__(self, hidden_channels, herb_dim, ingredient_dim, target_dim, num_heads):
            super().__init__()
            self.encoder = HerbEncoder(hidden_channels, herb_dim, ingredient_dim, target_dim, num_heads)
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

    # --- 6. Cross-Validation Training ---
    def train_and_evaluate(hidden_channels, lr, epochs, num_heads, train_idx, val_idx, data, herb_pairs, compat_labels):
        model = CompatibilityPredictor(hidden_channels, herb_dim, ingredient_dim, target_dim, num_heads).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        train_pairs_fold, train_labels_fold = herb_pairs[:, train_idx], compat_labels[train_idx]
        val_pairs_fold, val_labels_fold = herb_pairs[:, val_idx], compat_labels[val_idx]

        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            pred = model(data, train_pairs_fold)
            loss = criterion(pred, train_labels_fold)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(data, val_pairs_fold)
            mse = criterion(val_pred, val_labels_fold).item()
            r2 = r2_score(val_labels_fold.cpu().numpy(), val_pred.cpu().numpy())
        return mse, r2

    # --- 7. Hyperparameter Search ---
    learning_rates = [0.001, 0.0001]
    hidden_channels_list = [64, 128, 256, 512]
    epochs_list = [600]
    num_heads_list = [4, 8]

    results = []
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    print("\n--- Starting Hyperparameter Search ---")
    for lr, hidden_channels, epochs, num_heads in product(learning_rates, hidden_channels_list, epochs_list, num_heads_list):
        fold_mse, fold_r2 = [], []
        for fold, (train_idx, val_idx) in enumerate(kf.split(np.arange(trainval_pairs.shape[1]))):
            mse, r2 = train_and_evaluate(hidden_channels, lr, epochs, num_heads, train_idx, val_idx, data, trainval_pairs, trainval_labels)
            fold_mse.append(mse)
            fold_r2.append(r2)
            print(f"Fold {fold+1}, LR={lr}, Hidden={hidden_channels}, Heads={num_heads}, Epochs={epochs}, MSE={mse:.4f}, R2={r2:.4f}")
            
        avg_mse, avg_r2 = np.mean(fold_mse), np.mean(fold_r2)
        results.append({'LR': lr, 'Hidden': hidden_channels, 'Heads': num_heads, 'Epochs': epochs, 'Avg_MSE': avg_mse, 'Avg_R2': avg_r2})
        print(f"--- Avg for LR={lr}, Hidden={hidden_channels}, Heads={num_heads} -> Avg MSE={avg_mse:.4f}, Avg R2={avg_r2:.4f} ---\n")

    # --- 8. Save Results ---
    results_df = pd.DataFrame(results)
    results_path = os.path.join(output_dir, 'hyperparameter_results.csv')
    results_df.to_csv(results_path, index=False)
    print(f"Hyperparameter search results saved to '{results_path}'")

    best_params = results_df.loc[results_df['Avg_R2'].idxmax()].to_dict()
    for key, value in best_params.items():
        if isinstance(value, np.generic):
            best_params[key] = value.item()
            
    print("\nBest parameters found:", best_params)
    
    best_params_path = os.path.join(output_dir, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=4)
    print(f"Best parameters saved to '{best_params_path}'")
    print("\nExperiment completed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Hyperparameter Search for Herb Compatibility Prediction with selectable features.")
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

    run_hyperparameter_search(args.ingredient_feats, args.target_feats)
