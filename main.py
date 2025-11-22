# -*- coding: utf-8 -*-
import json
import numpy as np
import torch
from typing import Dict, Any, List

# =======================
# 配置
# =======================
ALL_ROLES = ["mage", "warrior", "tank", "marksman", "assassin", "support"]
SUM_TAGS = ["early", "midTerm", "late", "midTerm.late", "late.midTerm"]


def safe_get(d: Dict[str, Any], key: str, default=None):
    """安全取值"""
    return d[key] if key in d else default


# =======================
# 单侧（losecp / wincp）特征
# =======================
def build_side_features(side: Dict[str, Any]) -> np.ndarray:
    feats: List[float] = []

    # 1. 地图资源
    feats += [
        float(safe_get(side, "num_baron", 0)),
        float(int(safe_get(side, "baron", "0"))),
        float(safe_get(side, "num_darkTyrant", 0)),
        float(int(safe_get(side, "darkTyrant", "0"))),
        float(safe_get(side, "num_dragon", 0)),
    ]

    # 2. 经济、击杀、塔
    feats += [
        float(safe_get(side, "gold", 0)),
        float(safe_get(side, "kill", 0)),
        float(safe_get(side, "crystalTower_cnt", 0)),
        float(safe_get(side, "middleTower_cnt", 0)),
        float(safe_get(side, "tower_cnt", 0)),
    ]

    # 3. Buff
    feats += [
        float(safe_get(side, "num_redbuf", 0)),
        float(safe_get(side, "num_bluebuf", 0)),
    ]

    # 4. 兵线距离 + mask
    sd = safe_get(side, "soldier.distance", {})
    for lane in ["down", "mid", "up"]:
        v = float(safe_get(sd, lane, -1))
        if v < 0:  # 缺失
            feats.append(0.0)
            feats.append(1.0)
        else:
            feats.append(v)
            feats.append(0.0)

    # 5. 参团率 participation
    p = float(safe_get(side, "participation", -1))
    if p < 0:
        feats.append(0.0)  # value
        feats.append(1.0)  # missing
    else:
        feats.append(p)
        feats.append(0.0)

    # 6. AD / AP / Jungle 标记
    for k in ["AD", "AP", "Jungle"]:
        info = safe_get(side, k, {"state": False, "id": -1})
        feats.append(1.0 if info.get("state") else 0.0)
        feats.append(float(info.get("id", -1)))

    # 7. 英雄数据
    heroes = safe_get(side, "hero", {})
    hero_list = list(heroes.values())

    if len(hero_list) == 0:
        feats += [0.0] * 10
    else:
        golds = [float(h.get("gold", 0)) for h in hero_list]
        kills = [float(h.get("killCnt", 0)) for h in hero_list]
        deaths = [float(h.get("deadCnt", 0)) for h in hero_list]
        assists = [float(h.get("assistCnt", 0)) for h in hero_list]

        feats += [
            sum(golds),
            np.mean(golds),
            max(golds),
            sum(kills),
            sum(deaths),
            sum(assists),
            np.mean(kills),
            np.mean(deaths),
            np.mean(assists),
            len(hero_list),
        ]

    # 8. 职业分布
    role_counts = {r: 0 for r in ALL_ROLES}
    for h in hero_list:
        for r in h.get("profession", []):
            if r in role_counts:
                role_counts[r] += 1
    feats += [float(role_counts[r]) for r in ALL_ROLES]

    # 9. sum 标签 one-hot
    sum_tag = safe_get(side, "sum", "")
    feats += [1.0 if sum_tag == tag else 0.0 for tag in SUM_TAGS]

    return np.array(feats, dtype=np.float32)


# =======================
# 从一局游戏 → (T, D) tensor
# game_data = {"0.0": {...}, "0.5": {...}, ...}
# =======================
def game_to_tensor(game_data: Dict[str, Any]) -> torch.Tensor:
    # 按时间排序
    time_keys = sorted(game_data.keys(), key=lambda x: float(x))

    all_feats = []
    for t in time_keys:
        frame = game_data[t]
        lose_feat = build_side_features(frame["losecp"])
        win_feat = build_side_features(frame["wincp"])

        # 拼接成一条特征（你可以改成差值版本）
        feat = np.concatenate([lose_feat, win_feat], axis=0)
        all_feats.append(feat)

    arr = np.stack(all_feats, axis=0)  # (T, D)
    return torch.from_numpy(arr)       # FloatTensor


# =======================
# 从 hok_sample_100.json 文件中读取所有游戏
# =======================
def load_all_games(path: str):
    game_ids = []
    tensors = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 文件中是：game_id<多个空格>{json}
            game_id, json_str = line.split(maxsplit=1)

            game_data = json.loads(json_str)
            tensor = game_to_tensor(game_data)

            game_ids.append(game_id)
            tensors.append(tensor)

    return game_ids, tensors


# =======================
# 测试用
# =======================
if __name__ == "__main__":
    game_ids, tensors = load_all_games("hok_sample_100.json")

    print("总局数：", len(game_ids))
    print("第一局 ID：", game_ids[0])
    print("第一局张量 shape：", tensors[0].shape)
