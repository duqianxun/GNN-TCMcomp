import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HGTConv, Linear
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import json
import os
import argparse

# ===============================
# Argparse for command line input
# ===============================
parser = argparse.ArgumentParser(description="Herb compatibility prediction with selectable features.")
parser.add_argument("--ingredient_feats", type=str, choices=["maccs", "morgan", "rdkit"], default="maccs",
                    help="Choose ingredient features: maccs | morgan | rdkit")
parser.add_argument("--target_feats", type=str, choices=["esm", "probert"], default="esm",
                    help="Choose target features: esm | probert")
args = parser.parse_args()

ingredient_feature_type = args.ingredient_feats
target_feature_type = args.target_feats

# ===============================
# File Paths
# ===============================
herb_ingredient_file = 'ingredient_shuru.xlsx'
ingr_tgt_pred_file = 'ingredients_target_predict_final.xlsx'
ingr_tgt_known_file = 'ingredient_target_known_final.xlsx'
compatibility_file = 'com_shuru.xlsx'
herb_feature_file = 'herb_features.csv'

ingredient_feature_files = {
    "maccs": "ingredient_features_maccs.csv",
    "morgan": "ingredient_features_morgan.csv",
    "rdkit": "ingredient_features_rdkit.csv"
}
target_feature_files = {
    "esm": "target_features_esm.csv",
    "probert": "target_features_probert.csv"
}

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
print(f"Ingredient features: {ingredient_feature_type}, Target features: {target_feature_type}")

# Load base data
df_hi = pd.read_excel(herb_ingredient_file)
df_ip = pd.read_excel(ingr_tgt_pred_file)
df_ik = pd.read_excel(ingr_tgt_known_file)
df_ingredient_target = pd.concat([df_ip, df_ik], ignore_index=True)
df_com = pd.read_excel(compatibility_file)
df_herb_features = pd.read_csv(herb_feature_file)

# Load selected features
df_ingredient_features = pd.read_csv(ingredient_feature_files[ingredient_feature_type])
df_target_features = pd.read_csv(target_feature_files[target_feature_type])

# Create output directory
output_dir = f"{ingredient_feature_type}_{target_feature_type}"
os.makedirs(output_dir, exist_ok=True)

# ===============================
# Graph Construction
# ===============================
data = HeteroData()
herbs = pd.unique(df_hi['herb'])
ingredients = pd.unique(df_hi['ingredient'])
targets = pd.unique(df_ingredient_target['target'])

herb2id = {name: i for i, name in enumerate(herbs)}
ingr2id = {name: i for i, name in enumerate(ingredients)}
tgt2id = {name: i for i, name in enumerate(targets)}

# Herb features
herb_features = np.zeros((len(herbs), df_herb_features.shape[1] - 1), dtype=np.float32)
for _, row in df_herb_features.iterrows():
    if row['herb'] in herb2id:
        herb_features[herb2id[row['herb']]] = row.drop('herb').values
data['herb'].x = torch.tensor(herb_features, dtype=torch.float).to(device)

# Ingredient features
ingredient_features = np.zeros((len(ingredients), df_ingredient_features.shape[1] - 1), dtype=np.float32)
for _, row in df_ingredient_features.iterrows():
    if row['ingredient'] in ingr2id:
        ingredient_features[ingr2id[row['ingredient']]] = row.drop('ingredient').values
data['ingredient'].x = torch.tensor(ingredient_features, dtype=torch.float).to(device)

# Target features
target_features = np.zeros((len(targets), df_target_features.shape[1] - 1), dtype=np.float32)
for _, row in df_target_features.iterrows():
    if row['target'] in tgt2id:
        target_features[tgt2id[row['target']]] = row.drop('target').values
data['target'].x = torch.tensor(target_features, dtype=torch.float).to(device)

# Build edges
src = df_hi['herb'].map(herb2id).values
dst = df_hi['ingredient'].map(ingr2id).values
data['herb', 'has', 'ingredient'].edge_index = torch.tensor([src, dst], dtype=torch.long)
data['ingredient', 'rev_has', 'herb'].edge_index = torch.tensor([dst, src], dtype=torch.long)

src = df_ingredient_target['ingredient'].map(ingr2id).dropna().astype(int)
dst = df_ingredient_target['target'].map(tgt2id).dropna().astype(int)
data['ingredient', 'hits', 'target'].edge_index = torch.tensor([src.values, dst.values], dtype=torch.long)
data['target', 'rev_hits', 'ingredient'].edge_index = torch.tensor([dst.values, src.values], dtype=torch.long)

# Herb pairs & labels
herb_pairs, compat_labels = [], []
for _, row in df_com.iterrows():
    h1, h2, cij = row['Herb1'], row['Herb2'], row['Cij']
    if h1 in herb2id and h2 in herb2id:
        herb_pairs.append((herb2id[h1], herb2id[h2]))
        compat_labels.append(cij)

herb_pairs = torch.tensor(herb_pairs, dtype=torch.long).T.to(device)
compat_labels = torch.tensor(compat_labels, dtype=torch.float).to(device)

data = data.to(device)

# ===============================
# Train/Test Split
# ===============================
all_indices = np.arange(herb_pairs.shape[1])
train_idx, test_idx = train_test_split(all_indices, test_size=0.1, random_state=66)

train_pairs, train_labels = herb_pairs[:, train_idx], compat_labels[train_idx]
test_pairs, test_labels = herb_pairs[:, test_idx], compat_labels[test_idx]

# ===============================
# Model Definition (Single-layer HGTConv)
# ===============================
class HerbEncoder(torch.nn.Module):
    def __init__(self, hidden_channels, herb_dim, ingredient_dim, target_dim, num_heads):
        super().__init__()
        self.herb_lin = Linear(herb_dim, hidden_channels)
        self.ingredient_lin = Linear(ingredient_dim, hidden_channels)
        self.target_lin = Linear(target_dim, hidden_channels)
        self.hgt = HGTConv(hidden_channels, hidden_channels, data.metadata(), heads=num_heads)
        self.lin = Linear(hidden_channels, hidden_channels)

    def forward(self, x_dict, edge_index_dict):
        x_dict['herb'] = self.herb_lin(x_dict['herb'])
        x_dict['ingredient'] = self.ingredient_lin(x_dict['ingredient'])
        x_dict['target'] = self.target_lin(x_dict['target'])
        x_dict = self.hgt(x_dict, edge_index_dict)
        x_dict = {k: F.relu(v) for k, v in x_dict.items()}
        x_dict['herb'] = self.lin(x_dict['herb'])
        return x_dict

class CompatibilityPredictor(nn.Module):
    def __init__(self, hidden_channels, herb_dim, ingredient_dim, target_dim, num_heads):
        super().__init__()
        self.encoder = HerbEncoder(hidden_channels, herb_dim, ingredient_dim, target_dim, num_heads=num_heads)
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

# ===============================
# Train & Evaluate
# ===============================
def train_model(hidden_channels, lr, epochs, data, train_pairs, train_labels, herb_dim, ingredient_dim, target_dim):
    model = CompatibilityPredictor(hidden_channels, herb_dim, ingredient_dim, target_dim, num_heads).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        pred = model(data, train_pairs)
        loss = criterion(pred, train_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch}, Training Loss: {loss.item():.4f}")
    return model

def evaluate_model(model, data, test_pairs, test_labels):
    model.eval()
    with torch.no_grad():
        test_pred = model(data, test_pairs)
        mse = mean_squared_error(test_labels.cpu().numpy(), test_pred.cpu().numpy())
        r2 = r2_score(test_labels.cpu().numpy(), test_pred.cpu().numpy())
    return mse, r2

# Load hyperparameters from JSON
results_dir = f"{ingredient_feature_type}_{target_feature_type}"
json_path = os.path.join(results_dir, "epoch_search_results.json")

if not os.path.exists(json_path):
    raise FileNotFoundError(f"Hyperparameter file not found: {json_path}")

with open(json_path, "r") as f:
    search_results = json.load(f)

hidden_channels = search_results.get("Hidden_Channels")
learning_rate = search_results.get("Learning_Rate")
epochs = search_results.get("Average_Best_Epoch")
num_heads = search_results.get("Heads")

print(f"Loaded hyperparameters from {json_path}:")
print(f"  Hidden_Channels = {hidden_channels}")
print(f"  Learning_Rate   = {learning_rate}")
print(f"  Epochs          = {epochs}")
print(f"  Heads           = {num_heads}")

herb_dim = data['herb'].x.shape[1]
ingredient_dim = data['ingredient'].x.shape[1]
target_dim = data['target'].x.shape[1]

model = train_model(hidden_channels, learning_rate, epochs, data,
                    train_pairs, train_labels,
                    herb_dim, ingredient_dim, target_dim)

test_mse, test_r2 = evaluate_model(model, data, test_pairs, test_labels)
print(f"Test MSE: {test_mse:.4f}, Test R2: {test_r2:.4f}")

results = {
    'Ingredient_Feature': ingredient_feature_type,
    'Target_Feature': target_feature_type,
    'Hidden_Channels': hidden_channels,
    'Learning_Rate': learning_rate,
    'Epochs': epochs,
    'Test_MSE': test_mse,
    'Test_R2': test_r2
}
with open(f"{output_dir}/test_results.json", "w") as f:
    json.dump(results, f, indent=4)
