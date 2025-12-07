import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_curve, auc, confusion_matrix,classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random
# transformer can be imported directly
from dual_transformer import DualTransformerWinPredictor, HoKDataset, collate_fn, GROUP_NAMES

# LSTM 
class LSTMClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=1,
                 bidirectional=True, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.bidirectional = bidirectional
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), 1)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        if self.bidirectional:
            h_last = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            h_last = h_n[-1]
        h_last = self.dropout(h_last)
        logits = self.fc(h_last).squeeze(1)
        return logits

#GRU
class GRUNetworkWithDropout(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.2):
        super(GRUNetworkWithDropout, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True, bidirectional=False)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x, lengths=None):
        output, _ = self.gru(x)
        if lengths is not None:
            out = output[torch.arange(len(output)), lengths - 1]
        else:
            out = output[:, -1, :]           
        out = self.dropout(out)
        out = self.fc(out)
        return out

def get_metrics(y_true, y_scores):
    # y_scores 是概率值 (0~1)
    y_pred = (y_scores > 0.5).astype(int)
    acc = accuracy_score(y_true, y_pred)
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['Lose', 'Win'])
    return {"acc": acc, "auc": roc_auc, "cm": cm, "report": report, "fpr": fpr, "tpr": tpr}
    #return {"acc": acc, "auc": roc_auc, "fpr": fpr, "tpr": tpr, "cm": cm}

def evaluate_rnn_model(model, name, X_test, y_test, lengths_test=None):
    model.eval()
    with torch.no_grad():
        if name == "GRU":
            # GRU needs lengths
            logits = model(X_test, lengths_test)
        else:
            logits = model(X_test)      
        probs = torch.sigmoid(logits).squeeze().numpy()
        
    return get_metrics(y_test.numpy(), probs)

def evaluate_transformer_model(model, dataloader, device):
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) 
                         else {kk: vv.to(device) for kk, vv in v.items()})
                     for k, v in batch.items()}
            
            p, _, _, _ = model(batch) # p 已经是 sigmoid 后
            lengths = (~batch["padding_mask"]).sum(dim=1)
            
            for i in range(len(p)):
                length = lengths[i]
                last_prob = p[i, length-1].item()
                label = batch["labels"][i, length-1].item()
                all_probs.append(last_prob)
                all_labels.append(label)
                
    return get_metrics(np.array(all_labels), np.array(all_probs))

if __name__ == "__main__":
    device = torch.device("cpu")
    print("evaluating...")
    results = {} 
    # LSTM & GRU
    print("\n[1/2] evaluating LSTM and GRU...")
    try:
        X = torch.load("team_seq_features.pt").float()
        y = torch.load("team_seq_labels.pt").float()
        lengths = torch.load("team_seq_lengths.pt")
        
        # test datasets : 20%
        X_train, X_test, y_train, y_test, len_train, len_test = train_test_split(
            X, y, lengths, test_size=0.2, random_state=42, stratify=y
        )
        
        input_dim = X.shape[2]
        
        # --- LSTM ---
        lstm = LSTMClassifier(input_dim=input_dim, hidden_dim=64, num_layers=1)
        lstm.load_state_dict(torch.load("lstm_model.pth")) 
        results["LSTM"] = evaluate_rnn_model(lstm, "LSTM", X_test, y_test)
        print(f"   -> LSTM Accuracy: {results['LSTM']['acc']:.4f}")

        # --- GRU ---
        gru = GRUNetworkWithDropout(input_size=input_dim, hidden_size=128, output_size=1)
        gru.load_state_dict(torch.load("gru_model_with_dropout.pt")) 
        results["GRU"] = evaluate_rnn_model(gru, "GRU", X_test, y_test, len_test)
        print(f"   -> GRU  Accuracy: {results['GRU']['acc']:.4f}")

    except Exception as e:
        print(f"⚠️ LSTM/GRU 评估出错: {e}")

    #evalute transformer
    print("\n[2/2] evaluating Dual Transformer...")
    try:
        # 重新构建 Dataset
        dataset = HoKDataset("hok_sample_100.json")
        
        example = dataset[0]
        group_input_dims = [example["groups"][gi].shape[-1] for gi in range(len(GROUP_NAMES))]
        stage_vocab_size = len(dataset.stage2idx)
        
        tf_model = DualTransformerWinPredictor(group_input_dims, stage_vocab_size)
        
        try:
            tf_model.load_state_dict(torch.load("dual_transformer_hok.pth", map_location=device))
        except:
            tf_model.load_state_dict(torch.load("model/dual_transformer_hok.pth", map_location=device))
            
        indices = list(range(len(dataset)))
        random.seed(42)
        random.shuffle(indices)
        test_indices = indices[:int(0.2 * len(indices))] # 20%
 
        test_loader = DataLoader(Subset(dataset, test_indices), batch_size=8, collate_fn=collate_fn)
        results["Transformer"] = evaluate_transformer_model(tf_model, test_loader, device)
        print(f"   -> Transformer Accuracy: {results['Transformer']['acc']:.4f}")
        
    except Exception as e:
        print(f"⚠️ Transformer: {e}")

    # draw ROC curves
    plt.figure(figsize=(10, 8))
    colors = {'LSTM': 'blue', 'GRU': 'green', 'Transformer': 'red'}
    
    for name, res in results.items():
        plt.plot(res['fpr'], res['tpr'], 
                 label=f"{name} (AUC = {res['auc']:.3f})", 
                 color=colors.get(name, 'black'), linewidth=2)
        
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Model Comparison: ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig('evaluation_ROC.png')
    print("✅ ROC curves are saved as .png")
    plt.show()
    print("\n[3/3] Drawing Confusion Matrices...")
    
    num_models = len(results)
    
    if num_models > 0:
        fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 4))

        if num_models == 1:
            axes = [axes]
        if isinstance(axes, np.ndarray):
            axes_iter = axes.flat
        else:
            axes_iter = axes

        for ax, (name, metrics) in zip(axes_iter, results.items()):
            cm = metrics['cm'] 
            acc = metrics['acc']
 
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, annot_kws={"size": 14})
 
            ax.set_title(f"{name}\nAcc: {acc:.2%}", fontsize=14, fontweight='bold')
            ax.set_xlabel("Predicted Label", fontsize=12)

            is_first = (ax == (axes[0] if isinstance(axes, list) else axes.flat[0]))
            if is_first:
                ax.set_ylabel("True Label", fontsize=12)
            else:
                ax.set_ylabel("")

        plt.tight_layout()
        plt.savefig('evaluation_CM.png')
        print("✅ Confusion Matrices saved as.png")
        plt.show()
        
    else:
        print("!!No models evaluated!!!")

    