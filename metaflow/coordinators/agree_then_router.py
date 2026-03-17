from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class _DisagreementRouter(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


class AgreeThenRouterCoordinator:
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
        self.router = _DisagreementRouter(in_features=10)
        self.is_trained = False
        self.feature_mean = None
        self.feature_std = None
        self.stats = {
            "picked": [],
            "total": 0,
            "disagree": 0,
            "router_train_samples": 0,
            "router_train_pos_rate": None,
            "router_val_acc": None,
            "router_margin_baseline_acc": None,
            "router_enabled": False,
        }

    def _expert_features(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits / self.temperature, dim=-1)
        top2 = probs.topk(k=2, dim=-1).values
        confidence = top2[:, 0]
        margin = top2[:, 0] - top2[:, 1]
        entropy = -(probs * (probs.clamp_min(1e-12).log())).sum(dim=-1)
        logit_norm = logits.norm(p=2, dim=-1)
        top_logit = logits.max(dim=-1).values
        return torch.stack([top_logit, margin, entropy, logit_norm, confidence], dim=-1)

    def _pair_features(self, logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
        feat_a = self._expert_features(logits_a)
        feat_b = self._expert_features(logits_b)
        return torch.cat([feat_a, feat_b], dim=-1)

    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        if self.feature_mean is None or self.feature_std is None:
            return features
        return (features - self.feature_mean) / self.feature_std

    def fit_from_models(
        self,
        model_a,
        model_b,
        dataset,
        device: str,
        batch_size: int = 256,
        epochs: int = 25,
        lr: float = 1e-2,
        seed: int = 42,
        min_gain_over_margin: float = 0.002,
    ):
        model_a.eval()
        model_b.eval()

        g = torch.Generator()
        g.manual_seed(seed)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)

        all_features = []
        all_labels = []

        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)

                logits_a = model_a(x)
                logits_b = model_b(x)

                pred_a = logits_a.argmax(dim=-1)
                pred_b = logits_b.argmax(dim=-1)

                disagree = pred_a != pred_b
                if disagree.sum().item() == 0:
                    continue

                a_correct = pred_a == y
                b_correct = pred_b == y

                one_correct = disagree & (a_correct ^ b_correct)
                if one_correct.sum().item() == 0:
                    continue

                feat = self._pair_features(logits_a, logits_b)
                labels = b_correct[one_correct].float()

                all_features.append(feat[one_correct].cpu())
                all_labels.append(labels.cpu())

        if not all_features:
            self.is_trained = False
            self.stats["router_train_samples"] = 0
            self.stats["router_train_pos_rate"] = None
            self.stats["router_val_acc"] = None
            self.stats["router_margin_baseline_acc"] = None
            self.stats["router_enabled"] = False
            return

        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)

        self.stats["router_train_samples"] = int(labels.numel())
        self.stats["router_train_pos_rate"] = float(labels.mean().item())

        num_samples = labels.numel()
        if num_samples < 20:
            self.is_trained = False
            self.stats["router_val_acc"] = None
            self.stats["router_margin_baseline_acc"] = None
            self.stats["router_enabled"] = False
            return

        g_split = torch.Generator()
        g_split.manual_seed(seed)
        perm = torch.randperm(num_samples, generator=g_split)
        val_size = max(1, int(0.2 * num_samples))
        train_idx = perm[val_size:]
        val_idx = perm[:val_size]

        train_features = features[train_idx]
        train_labels = labels[train_idx]
        val_features = features[val_idx]
        val_labels = labels[val_idx]

        feature_mean = train_features.mean(dim=0, keepdim=True)
        feature_std = train_features.std(dim=0, keepdim=True).clamp_min(1e-6)

        train_features = (train_features - feature_mean) / feature_std
        val_features = (val_features - feature_mean) / feature_std

        margin_a = features[val_idx][:, 1]
        margin_b = features[val_idx][:, 6]
        margin_choose_b = margin_b > margin_a
        margin_acc = (margin_choose_b == (val_labels > 0.5)).float().mean().item()

        self.router = _DisagreementRouter(in_features=features.size(1)).to(device)
        optimizer = torch.optim.Adam(self.router.parameters(), lr=lr)

        train_features = train_features.to(device)
        train_labels = train_labels.to(device)
        val_features = val_features.to(device)
        val_labels = val_labels.to(device)

        pos = train_labels.sum().item()
        neg = train_labels.numel() - pos
        pos_weight_value = (neg / max(pos, 1.0))
        pos_weight = torch.tensor(pos_weight_value, device=device)

        self.router.train()
        best_state = None
        best_val_acc = -1.0
        for _ in range(epochs):
            logits = self.router(train_features)
            loss = F.binary_cross_entropy_with_logits(logits, train_labels, pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                self.router.eval()
                val_logits = self.router(val_features)
                val_pred = torch.sigmoid(val_logits) >= 0.5
                val_acc = (val_pred == (val_labels > 0.5)).float().mean().item()
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.detach().cpu().clone() for k, v in self.router.state_dict().items()}
                self.router.train()

        if best_state is not None:
            self.router.load_state_dict(best_state)

        self.router.eval()
        self.feature_mean = feature_mean.to(device)
        self.feature_std = feature_std.to(device)
        self.stats["router_val_acc"] = float(best_val_acc)
        self.stats["router_margin_baseline_acc"] = float(margin_acc)

        self.is_trained = best_val_acc >= (margin_acc + min_gain_over_margin)
        self.stats["router_enabled"] = bool(self.is_trained)

    def combine(self, logits_list: List[torch.Tensor]) -> torch.Tensor:
        if len(logits_list) == 0:
            raise ValueError("logits_list is empty")
        if len(logits_list) == 1:
            return logits_list[0]
        if len(logits_list) != 2:
            raise ValueError("AgreeThenRouterCoordinator currently supports exactly 2 experts.")

        stacked = torch.stack(logits_list, dim=0)  # [2, B, C]
        logits_a = stacked[0]
        logits_b = stacked[1]

        pred_a = logits_a.argmax(dim=-1)
        pred_b = logits_b.argmax(dim=-1)

        agree = pred_a == pred_b
        disagree = ~agree

        choose_b = torch.zeros_like(pred_a, dtype=torch.bool)

        if disagree.any():
            if self.is_trained:
                with torch.no_grad():
                    pair_feat = self._pair_features(logits_a[disagree], logits_b[disagree])
                    pair_feat = self._normalize_features(pair_feat)
                    router_logits = self.router(pair_feat)
                    choose_b_disagree = torch.sigmoid(router_logits) >= 0.5
                choose_b[disagree] = choose_b_disagree
            else:
                probs = torch.softmax(stacked / self.temperature, dim=-1)
                top2 = probs.topk(k=2, dim=-1).values
                margin = top2[..., 0] - top2[..., 1]  # [2, B]
                choose_b[disagree] = margin[1, disagree] > margin[0, disagree]

        best_idx = choose_b.long()

        if (not self.stats["picked"]) or (len(self.stats["picked"]) != 2):
            self.stats["picked"] = [0, 0]
        self.stats["picked"][0] += (best_idx == 0).sum().item()
        self.stats["picked"][1] += (best_idx == 1).sum().item()
        self.stats["total"] += pred_a.size(0)
        self.stats["disagree"] += disagree.sum().item()

        gather_idx = best_idx.view(1, -1, 1).expand(1, stacked.size(1), stacked.size(2))
        return torch.gather(stacked, dim=0, index=gather_idx).squeeze(0)
