import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
from scipy.ndimage import gaussian_filter1d
import os
import json

from dual_transformer import DualTransformerWinPredictor, HoKDataset, collate_fn, GROUP_NAMES

#LSTM and GRU
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1, bidirectional=True, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0)
        self.fc = nn.Linear(hidden_dim * 2, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out) 

class GRUNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.dropout(out)
        return self.fc(out)


def get_trajectories(model_name, model, indices, pt_data=None, json_dataset=None):
    trajectories = []
    device = torch.device("cpu")
    model.eval()
    
    with torch.no_grad():
        for idx in indices:
            preds = []
            
            # Transformer 用.json
            if model_name == "Transformer":
                try:
                    sample = json_dataset[idx]
                    real_len = sample['length']

                    batch = collate_fn([sample])
                    batch_input = {
                        "groups": {k: v.to(device) for k, v in batch["groups"].items()},
                        "stage_ids": batch["stage_ids"].to(device),
                        "time_norm": batch["time_norm"].to(device),
                        "padding_mask": batch["padding_mask"].to(device),
                        "labels": batch["labels"].to(device)
                    }

                    probs, _, _, _ = model(batch_input)
                    preds = probs[0, :real_len].numpy()
                except Exception as e:
                    print(f"⚠️ Transformer ships {idx}: {e}")
                    continue
            
            #  LSTM & GRU 用 .pt
            else:
                X_all, lengths_all = pt_data
                sample_x = X_all[idx:idx+1]
                real_len = lengths_all[idx].item()
                
                if model_name == "LSTM":
                    logits = model(sample_x)
                    if logits.dim() == 3: logits = logits.squeeze(-1)
                    preds = torch.sigmoid(logits)[0, :real_len].numpy()
                    
                else: 
                    logits = model(sample_x)
                    if logits.dim() == 3: logits = logits.squeeze(-1)
                    preds = torch.sigmoid(logits)[0, :real_len].numpy()

            if len(preds) < 2: continue

            x_old = np.linspace(0, 1, num=len(preds))
            x_new = np.linspace(0, 1, num=100)
            f = interp.interp1d(x_old, preds, kind='linear')
            traj = f(x_new)
 
            traj = gaussian_filter1d(traj, sigma=2)
            trajectories.append(traj)
            
    return np.array(trajectories)


if __name__ == "__main__":
  
    # load .pt
    try:
        X_pt = torch.load("team_seq_features.pt").float()
        y_pt = torch.load("team_seq_labels.pt").float()
        l_pt = torch.load("team_seq_lengths.pt").long()
    except:
        print("❌ fail to find .pt files!")
        exit()

    # load .json
    try:
        tf_dataset = HoKDataset("hok_sample_100.json")
    except Exception as e:
        print(f"❌ fail to find .json files! {e}")
        exit()

    input_dim = X_pt.shape[2]
    device = torch.device("cpu")
    
    lstm = LSTMClassifier(input_dim, 64).to(device)
    gru = GRUNetwork(input_dim, 128, 1).to(device)

    sample = tf_dataset[0]
    group_dims = [sample["groups"][gi].shape[-1] for gi in range(len(GROUP_NAMES))]
    tf_model = DualTransformerWinPredictor(group_dims, len(tf_dataset.stage2idx)).to(device)
    

    models_to_load = [
        ("LSTM", lstm, "lstm_model.pth"), 
        ("GRU", gru, "gru_model_with_dropout.pt"), 
        ("Transformer", tf_model, "dual_transformer_hok.pth") 
    ]

    for name, model, path in models_to_load:
        load_path = path if os.path.exists(path) else "model/" + path
        if os.path.exists(load_path):
            try:
                model.load_state_dict(torch.load(load_path, map_location=device))
                print(f"🔹 {name} sucessfuly load the weight")
            except Exception as e:
                print(f"⚠️ {name} : {e}")
        else:
            print(f"⚠️ {name} fail to find the model file at{load_path}")#randomly initialized weights

   
    # 20% test set
    num_samples = min(len(y_pt), len(tf_dataset))
    indices = np.arange(num_samples)
    split = int(num_samples * 0.8)
    test_idx = indices[split:]
 
    win_idx = test_idx[y_pt[test_idx].numpy() == 1]
    lose_idx = test_idx[y_pt[test_idx].numpy() == 0]
    
    print(f"the size of test: {len(test_idx)} (Win: {len(win_idx)}, Lose: {len(lose_idx)})")

  
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=120)
    
    config = {
        "LSTM": (lstm, (X_pt, l_pt), "#3498db"),      
        "GRU": (gru, (X_pt, l_pt), "#e74c3c"),        
        "Transformer": (tf_model, tf_dataset, "#2ecc71") 
    }

    for i, (title, idxs) in enumerate([("Scenario A: True Result = WIN", win_idx), ("Scenario B: True Result = LOSE", lose_idx)]):
        ax = axes[i]
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5) 
        
        for name, (model, data, color) in config.items():
            if name == "Transformer":
                trajs = get_trajectories(name, model, idxs, json_dataset=data)
            else:
                trajs = get_trajectories(name, model, idxs, pt_data=data)
            
            if len(trajs) > 0:
                mean_traj = trajs.mean(axis=0)
                std_traj = trajs.std(axis=0)
                
                #  (Mean +/- Std)
                ax.plot(np.linspace(0, 100, 100), mean_traj, label=name, color=color, linewidth=3)
                ax.fill_between(np.linspace(0, 100, 100), 
                                mean_traj - std_traj*0.2, 
                                mean_traj + std_traj*0.2, 
                                color=color, alpha=0.15)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Game Progress (%)")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        if i == 0: 
            ax.set_ylabel("Predicted Win Probability")
            ax.legend(loc="upper left")

    plt.tight_layout()
    plt.savefig("evaluation_2.png")
    print("✅ saved as: evaluation_2.png")
    plt.show()