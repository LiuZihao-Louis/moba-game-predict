import json
import math
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset

# ----------------------
# Global config
# ----------------------

GROUP_NAMES = ["gold", "kill", "tower", "wild", "soldier", "heroes"]
PROF_LIST = ["tank", "warrior", "assassin", "mage", "marksman", "support"]
PROF_INDEX = {p: i for i, p in enumerate(PROF_LIST)}


# ----------------------
# Feature engineering
# ----------------------

def as_float(v) -> float:
    """Robust conversion to float."""
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return 0.0
    if isinstance(v, dict):
        if "state" in v:
            return 1.0 if v["state"] else 0.0
    return 0.0


def extract_features_side(frame_side: Dict, frame_opp: Dict) -> Tuple[np.ndarray, ...]:
    """
    从一帧中提取 6 个 feature group（视角：frame_side，相对方：frame_opp）。
    每个 group 返回一个 np.ndarray，最后组成长度为 6 的 tuple。
    """
    me = frame_side
    opp = frame_opp

    # 1) Gold 组
    gold_vec = np.array(
        [as_float(me.get("gold", 0)) - as_float(opp.get("gold", 0))],
        dtype=np.float32,
    )

    # 2) Kill 组
    kill_vec = np.array(
        [
            as_float(me.get("kill", 0)) - as_float(opp.get("kill", 0)),
            as_float(me.get("participation", 0))
            - as_float(opp.get("participation", 0)),
        ],
        dtype=np.float32,
    )

    # 3) Tower 组
    tower_vec = np.array(
        [
            as_float(me.get("tower_cnt", 0)) - as_float(opp.get("tower_cnt", 0)),
            as_float(me.get("middleTower_cnt", 0))
            - as_float(opp.get("middleTower_cnt", 0)),
            as_float(me.get("crystalTower_cnt", 0))
            - as_float(opp.get("crystalTower_cnt", 0)),
        ],
        dtype=np.float32,
    )

    # 4) Wild Resource 组
    wild_vec = np.array(
        [
            as_float(me.get("num_baron", 0)) - as_float(opp.get("num_baron", 0)),
            as_float(me.get("num_darkTyrant", 0))
            - as_float(opp.get("num_darkTyrant", 0)),
            as_float(me.get("num_dragon", 0))
            - as_float(opp.get("num_dragon", 0)),
            as_float(me.get("baron", 0)) - as_float(opp.get("baron", 0)),
            as_float(me.get("darkTyrant", 0))
            - as_float(opp.get("darkTyrant", 0)),
        ],
        dtype=np.float32,
    )

    # 5) Soldier 组（线权推进）
    me_soldier = me.get("soldier.distance", {})
    opp_soldier = opp.get("soldier.distance", {})
    soldier_vec = np.array(
        [
            as_float(me_soldier.get("down", 0))
            - as_float(opp_soldier.get("down", 0)),
            as_float(me_soldier.get("mid", 0))
            - as_float(opp_soldier.get("mid", 0)),
            as_float(me_soldier.get("up", 0)) - as_float(opp_soldier.get("up", 0)),
        ],
        dtype=np.float32,
    )

    # 6) Heroes 组（阵容 + AD/AP/Jungle）
    counts_me = np.zeros(len(PROF_LIST), dtype=np.float32)
    counts_opp = np.zeros(len(PROF_LIST), dtype=np.float32)

    for hero in me.get("hero", {}).values():
        for p in hero.get("profession", []):
            if p in PROF_INDEX:
                counts_me[PROF_INDEX[p]] += 1.0

    for hero in opp.get("hero", {}).values():
        for p in hero.get("profession", []):
            if p in PROF_INDEX:
                counts_opp[PROF_INDEX[p]] += 1.0

    prof_diff = counts_me - counts_opp

    ad_me = 1.0 if me.get("AD", {}).get("state", False) else 0.0
    ap_me = 1.0 if me.get("AP", {}).get("state", False) else 0.0
    jungle_me = 1.0 if me.get("Jungle", {}).get("state", False) else 0.0

    ad_opp = 1.0 if opp.get("AD", {}).get("state", False) else 0.0
    ap_opp = 1.0 if opp.get("AP", {}).get("state", False) else 0.0
    jungle_opp = 1.0 if opp.get("Jungle", {}).get("state", False) else 0.0

    comp_vec = np.array(
        [ad_me - ad_opp, ap_me - ap_opp, jungle_me - jungle_opp],
        dtype=np.float32,
    )

    heroes_vec = np.concatenate([prof_diff, comp_vec]).astype(np.float32)

    return (gold_vec, kill_vec, tower_vec, wild_vec, soldier_vec, heroes_vec)


# ----------------------
# Dataset & DataLoader
# ----------------------

class HoKDataset(Dataset):
    """将 hok_sample_100.json 转成时序样本。

    每一场对局生成两条样本：
    - 以 wincp 为视角：特征 = wincp - losecp，label=1
    - 以 losecp 为视角：特征 = losecp - wincp，label=0
    """

    def __init__(self, json_path: str, max_games: int = None):
        self.records: List[Tuple[str, Dict]] = []
        with open(json_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    game_id, j = line.split(None, 1)
                except ValueError:
                    # 跳过异常行
                    continue
                rec = json.loads(j)
                self.records.append((game_id, rec))
                if max_games is not None and len(self.records) >= max_games:
                    break

        # 构建阶段（sum）词表
        stage_set = set()
        for game_id, rec in self.records:
            for t, fr in rec.items():
                stage_set.add(fr["wincp"]["sum"])
        self.stage2idx = {s: i for i, s in enumerate(sorted(stage_set))}
        self.idx2stage = {i: s for s, i in self.stage2idx.items()}

        # 展开成样本列表
        self.samples: List[Dict] = []
        for game_id, rec in self.records:
            times = sorted(rec.keys(), key=lambda x: float(x))
            T = len(times)
            if T == 0:
                continue

            groups_win = [[] for _ in GROUP_NAMES]
            groups_lose = [[] for _ in GROUP_NAMES]
            stage_ids: List[int] = []
            time_vals: List[float] = []

            for ts in times:
                fr = rec[ts]
                w = fr["wincp"]
                l = fr["losecp"]

                feats_w = extract_features_side(w, l)
                feats_l = extract_features_side(l, w)

                for gi in range(len(GROUP_NAMES)):
                    groups_win[gi].append(feats_w[gi])
                    groups_lose[gi].append(feats_l[gi])

                stage_ids.append(self.stage2idx[w["sum"]])
                time_vals.append(float(ts))

            time_vals = np.array(time_vals, dtype=np.float32)
            if T > 1 and (time_vals[-1] - time_vals[0]) != 0:
                time_norm = (time_vals - time_vals[0]) / (
                    time_vals[-1] - time_vals[0]
                )
            else:
                time_norm = np.zeros_like(time_vals, dtype=np.float32)

            stage_ids = np.array(stage_ids, dtype=np.int64)
            groups_win = [np.stack(g, axis=0) for g in groups_win]
            groups_lose = [np.stack(g, axis=0) for g in groups_lose]

            labels_win = np.ones(T, dtype=np.float32)
            labels_lose = np.zeros(T, dtype=np.float32)

            # 视角：赢家
            self.samples.append(
                {
                    "groups": groups_win,
                    "stage_ids": stage_ids,
                    "time_norm": time_norm,
                    "labels": labels_win,
                    "length": T,
                    "game_id": game_id,
                    "view": "win",
                }
            )
            # 视角：输家
            self.samples.append(
                {
                    "groups": groups_lose,
                    "stage_ids": stage_ids,
                    "time_norm": time_norm,
                    "labels": labels_lose,
                    "length": T,
                    "game_id": game_id,
                    "view": "lose",
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        return self.samples[idx]


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """把不同长度的序列 pad 成 batch。"""
    batch_size = len(batch)
    max_len = max(item["length"] for item in batch)

    group_tensors: Dict[str, torch.Tensor] = {}
    for gi, gname in enumerate(GROUP_NAMES):
        dim = batch[0]["groups"][gi].shape[-1]
        group_tensors[gname] = torch.zeros(
            batch_size, max_len, dim, dtype=torch.float32
        )

    stage_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    time_norm = torch.zeros(batch_size, max_len, 1, dtype=torch.float32)
    labels = torch.zeros(batch_size, max_len, dtype=torch.float32)
    padding_mask = torch.ones(batch_size, max_len, dtype=torch.bool)  # True=pad

    for i, sample in enumerate(batch):
        L = sample["length"]
        padding_mask[i, : L] = False
        for gi, gname in enumerate(GROUP_NAMES):
            arr = sample["groups"][gi]
            group_tensors[gname][i, : L, :] = torch.from_numpy(arr)
        stage_ids[i, : L] = torch.from_numpy(sample["stage_ids"])
        time_norm[i, : L, 0] = torch.from_numpy(sample["time_norm"])
        labels[i, : L] = torch.from_numpy(sample["labels"])

    return {
        "groups": group_tensors,  # dict name -> [B,T,dim]
        "stage_ids": stage_ids,   # [B,T]
        "time_norm": time_norm,   # [B,T,1]
        "labels": labels,         # [B,T]
        "padding_mask": padding_mask,  # [B,T] bool
    }


# ----------------------
# Model: Dual Transformer
# ----------------------

class SpatialTransformer(nn.Module):
    """空间维度 Transformer：6 个 feature group 之间做注意力。"""

    def __init__(
        self,
        group_input_dims: List[int],
        d_model: int = 64,
        num_layers: int = 1,
        nhead: int = 4,
    ):
        super().__init__()
        self.num_groups = len(group_input_dims)
        self.d_model = d_model

        self.group_mlps = nn.ModuleList()
        for in_dim in group_input_dims:
            self.group_mlps.append(
                nn.Sequential(
                    nn.Linear(in_dim, d_model),
                    nn.ReLU(),
                    nn.Linear(d_model, d_model),
                    nn.ReLU(),
                )
            )

        # 区分不同 group 的 embedding，相当于 group-type embedding
        self.group_embed = nn.Embedding(self.num_groups, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出 win-score 的 head（所有 group 共享一套参数）
        self.score_head = nn.Linear(d_model, 1)

    def forward(self, group_feats_dict: Dict[str, torch.Tensor]):
        # group_feats_dict: name -> [B,T,dim]
        feats_list = [group_feats_dict[name] for name in GROUP_NAMES]
        B, T, _ = feats_list[0].shape

        # 按 frame，把 6 个 group 堆成 token
        tokens_list = []
        for gi, x in enumerate(feats_list):
            h = self.group_mlps[gi](x)  # [B,T,d_model]
            e = self.group_embed.weight[gi]  # [d_model]
            h = h + e.view(1, 1, -1)
            tokens_list.append(h)

        tokens = torch.stack(tokens_list, dim=2)  # [B,T,G,d_model]
        tokens = tokens.view(B * T, self.num_groups, self.d_model)  # [B*T,G,d_model]

        out = self.encoder(tokens)  # [B*T,G,d_model]
        out = out.view(B, T, self.num_groups, self.d_model)  # [B,T,G,d_model]

        # 每个 group 一个 win-score，范围 [-1,1]
        s = self.score_head(out).squeeze(-1)  # [B,T,G]
        s = torch.tanh(s)

        # frame 级别的表示，用来喂 Temporal Transformer
        frame_repr = out.mean(dim=2)  # [B,T,d_model]

        return s, frame_repr


class TemporalTransformer(nn.Module):
    """时间维度 Transformer：在时序上建模不同帧之间的依赖关系。"""

    def __init__(
        self,
        frame_dim: int,
        stage_vocab_size: int,
        d_model: int = 64,
        num_layers: int = 2,
        nhead: int = 4,
    ):
        super().__init__()
        self.stage_embed = nn.Embedding(stage_vocab_size, 4)

        # 输入 = frame_repr + time_norm(1) + stage_embed(4)
        self.input_proj = nn.Linear(frame_dim + 1 + 4, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出 group 权重的 head，输入是 [h_t, s(t)]
        self.weight_head = nn.Sequential(
            nn.Linear(d_model + len(GROUP_NAMES), d_model),
            nn.ReLU(),
            nn.Linear(d_model, len(GROUP_NAMES)),
        )

    def forward(
        self,
        frame_repr: torch.Tensor,  # [B,T,frame_dim]
        s: torch.Tensor,           # [B,T,G]
        stage_ids: torch.Tensor,   # [B,T]
        time_norm: torch.Tensor,   # [B,T,1]
        padding_mask: torch.Tensor = None,  # [B,T] bool, True=pad
    ) -> torch.Tensor:
        stage_emb = self.stage_embed(stage_ids)  # [B,T,4]
        x = torch.cat([frame_repr, time_norm, stage_emb], dim=-1)  # [B,T,frame_dim+5]
        h = self.input_proj(x)  # [B,T,d_model]

        h = self.encoder(h, src_key_padding_mask=padding_mask)  # [B,T,d_model]

        # 拼上当前帧的各 group win-score，再输出权重
        h_and_s = torch.cat([h, s], dim=-1)  # [B,T,d_model+G]
        logits_w = self.weight_head(h_and_s)  # [B,T,G]
        w = torch.softmax(logits_w, dim=-1)   # [B,T,G], 每一帧 6 个权重和为 1
        return w


class DualTransformerWinPredictor(nn.Module):
    """整体模型：Spatial Transformer + Temporal Transformer。"""

    def __init__(
        self,
        group_input_dims: List[int],
        stage_vocab_size: int,
        spatial_d_model: int = 64,
        temporal_d_model: int = 64,
    ):
        super().__init__()
        self.spatial = SpatialTransformer(
            group_input_dims, d_model=spatial_d_model
        )
        self.temporal = TemporalTransformer(
            frame_dim=spatial_d_model,
            stage_vocab_size=stage_vocab_size,
            d_model=temporal_d_model,
        )

    def forward(self, batch: Dict[str, torch.Tensor]):
        groups = batch["groups"]
        stage_ids = batch["stage_ids"]
        time_norm = batch["time_norm"]
        padding_mask = batch["padding_mask"]

        # 空间阶段：6 个 feature group 之间做注意力，得到 win-score 和 frame 表示
        s, frame_repr = self.spatial(groups)  # s: [B,T,G]; frame_repr: [B,T,D]

        # 时间阶段：在时序上做 Transformer，得到 group 权重 w_t
        w = self.temporal(
            frame_repr, s, stage_ids, time_norm, padding_mask
        )  # [B,T,G]

        # 贡献 & 最终 logit
        contrib = w * s  # [B,T,G]
        F = contrib.sum(dim=-1)  # [B,T]
        p = torch.sigmoid(F)     # [B,T] 胜率（以当前视角阵营获胜为正类）

        return p, s, w, contrib


# ----------------------
# 训练 & 评估
# ----------------------

def compute_loss(
    p: torch.Tensor,
    y: torch.Tensor,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    """只在非 pad 位置上计算 BCE。"""
    bce = nn.BCELoss(reduction="none")
    loss = bce(p, y)  # [B,T]
    loss = loss.masked_fill(padding_mask, 0.0)
    denom = (~padding_mask).sum().clamp(min=1).float()
    return loss.sum() / denom


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in dataloader:
        batch = {
            k: (
                v.to(device)
                if isinstance(v, torch.Tensor)
                else {kk: vv.to(device) for kk, vv in v.items()}
            )
            for k, v in batch.items()
        }

        p, s, w, contrib = model(batch)
        loss = compute_loss(p, batch["labels"], batch["padding_mask"])

        mask = ~batch["padding_mask"]
        total_loss += loss.item() * mask.sum().item()

        preds = (p >= 0.5).float()
        total_correct += ((preds == batch["labels"]) & mask).sum().item()
        total_count += mask.sum().item()

    if total_count == 0:
        return 0.0, 0.0

    avg_loss = total_loss / total_count
    acc = total_correct / total_count
    return avg_loss, acc


def train_model(
    json_path: str,
    num_epochs: int = 20,
    batch_size: int = 8,
    lr: float = 1e-3,
    max_games: int = None,
    seed: int = 42,
) -> DualTransformerWinPredictor:
    # 固定随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # 数据集
    dataset = HoKDataset(json_path, max_games=max_games)
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty, please check json_path.")

    example = dataset[0]
    group_input_dims = [
        example["groups"][gi].shape[-1] for gi in range(len(GROUP_NAMES))
    ]
    stage_vocab_size = len(dataset.stage2idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DualTransformerWinPredictor(
        group_input_dims, stage_vocab_size
    ).to(device)

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split = int(0.8 * len(indices))
    if split == 0 or split == len(indices):
        # 样本太少就不拆 train/val
        train_idx = indices
        val_idx = []
    else:
        train_idx, val_idx = indices[:split], indices[split:]

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    val_loader = (
        DataLoader(
            Subset(dataset, val_idx),
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
        if len(val_idx) > 0
        else None
    )

    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        model.train()
        for batch in train_loader:
            batch = {
                k: (
                    v.to(device)
                    if isinstance(v, torch.Tensor)
                    else {kk: vv.to(device) for kk, vv in v.items()}
                )
                for k, v in batch.items()
            }

            optimizer.zero_grad()
            p, s, w, contrib = model(batch)
            loss = compute_loss(p, batch["labels"], batch["padding_mask"])
            loss.backward()
            optimizer.step()

        if val_loader is not None:
            val_loss, val_acc = evaluate(model, val_loader, device)
            print(
                f"Epoch {epoch:02d} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
            )
        else:
            print(f"Epoch {epoch:02d} finished (no val set).")

    return model


if __name__ == "__main__":
    # 使用示例：把下面这个路径改成你的 hok_sample_100.json 路径
    json_path = "hok_sample_100.json"

    # 训练模型（max_games=None 会用完所有数据；先可以用 max_games=50 快速试跑）
    model = train_model(
        json_path=json_path,
        num_epochs=20,
        batch_size=8,
        lr=1e-3,
        max_games=None,
    )

    # 训练完之后你可以把 model 保存下来：
    torch.save(model.state_dict(), "model/dual_transformer_hok.pth")
    print("Model saved to dual_transformer_hok.pth")
