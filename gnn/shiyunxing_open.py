import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os
import argparse

def run_experiment(ingredient_feat_type, target_feat_type):
    """
    Main function to run the training and evaluation experiment.
    """
    # --- 1. Setup and Configuration ---
    # Set device
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

    # Dynamic feature file selection based on arguments
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
    # The first column is assumed to be the identifier.
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
    
    # --- 4. Data Splitting ---
    all_indices = np.arange(herb_pairs.shape[1])
    trainval_idx, test_idx = train_test_split(all_indices, test_size=0.1, random_state=40)
    trainval_pairs = herb_pairs[:, trainval_idx]
    trainval_labels = compat_labels[trainval_idx]

    train_idx, val_idx = train_test_split(np.arange(len(trainval_idx)), test_size=0.2, random_state=42)
    train_pairs = trainval_pairs[:, train_idx]
    train_labels = trainval_labels[train_idx]
    val_pairs = trainval_pairs[:, val_idx]
    val_labels = trainval_labels[val_idx]

    # Move all data to the selected device
    data = data.to(device)
    train_pairs, train_labels = train_pairs.to(device), train_labels.to(device)
    val_pairs, val_labels = val_pairs.to(device), val_labels.to(device)


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

    # --- 6. Training and Evaluation ---
    def train_long_trial(hidden_channels=512, lr=0.0001, epochs=1000):
        # Pass dynamic feature dimensions to the model
        model = CompatibilityPredictor(hidden_channels, herb_dim, ingredient_dim, target_dim).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        train_losses = []
        val_losses = []

        print("\n--- Starting Training ---")
        for epoch in range(1, epochs + 1):
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
                val_loss = criterion(val_pred, val_labels)
                val_r2 = r2_score(val_labels.cpu().numpy(), val_pred.cpu().numpy())

            train_losses.append(train_loss.item())
            val_losses.append(val_loss.item())

            if epoch % 100 == 0:
                print(f"Epoch {epoch:4d}, Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val R2: {val_r2:.4f}")

        # Plotting
        plt.rcParams.update({
            'font.family': 'Arial', 'font.size': 16, 'axes.titlesize': 18,
            'axes.labelsize': 16, 'xtick.labelsize': 14, 'ytick.labelsize': 14,
            'legend.fontsize': 14
        })
        
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
        plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Training and Validation Loss over Epochs')
        plt.legend()
        plt.grid(True)
        # Save plot to the specific results directory
        plot_path = os.path.join(output_dir, 'loss_plot.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"\nLoss plot saved to '{plot_path}'")

        # Save losses to CSV in the specific results directory
        loss_df = pd.DataFrame({
            'Epoch': range(1, epochs + 1),
            'Train_Loss': train_losses,
            'Val_Loss': val_losses
        })
        csv_path = os.path.join(output_dir, 'long_trial_losses.csv')
        loss_df.to_csv(csv_path, index=False)
        print(f"Loss data saved to '{csv_path}'")

    # Run the training process
    train_long_trial()
    print("\nExperiment completed.")


if __name__ == '__main__':
    # --- Argument Parser for Feature Selection ---
    parser = argparse.ArgumentParser(description="Run Herb Compatibility Prediction with selectable features.")
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
    run_experiment(args.ingredient_feats, args.target_feats)