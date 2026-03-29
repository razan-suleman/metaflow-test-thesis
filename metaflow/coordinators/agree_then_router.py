from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class _DisagreementRouter(nn.Module):
    """
    Simple binary classifier that decides whether to pick Model A or Model B.
    Uses a single linear layer to output a score (0 = pick A, 1 = pick B).
    """
    def __init__(self, in_features: int):
        super().__init__()
        # Single linear layer: input features -> binary decision
        self.linear = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply linear layer and remove last dimension
        # Input: [batch_size, 10] -> Output: [batch_size, 1] -> squeeze -> [batch_size]
        return self.linear(x).squeeze(-1)


class AgreeThenRouterCoordinator:
    """
    Two-stage coordinator: If models agree, use that prediction.
    If they disagree, use a trained neural router to pick the better model.
    """
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature  # Temperature for softmax scaling
        
        # Router: 10 features (5 from each model) -> binary decision
        self.router = _DisagreementRouter(in_features=10)
        self.is_trained = False  # Whether router beat baseline and is enabled
        
        # Feature normalization parameters (computed during training)
        self.feature_mean = None
        self.feature_std = None
        
        # Statistics tracking for analysis and debugging
        self.stats = {
            "picked": [],              # Count of how many times each model was selected
            "total": 0,                # Total predictions made
            "disagree": 0,             # How many times models disagreed
            "router_train_samples": 0, # Number of training samples collected
            "router_train_pos_rate": None,  # Class balance (% pick B)
            "router_val_acc": None,    # Router's validation accuracy
            "router_margin_baseline_acc": None,  # Baseline heuristic accuracy
            "router_enabled": False,   # Whether router is active
        }

    def _expert_features(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Extract 5 statistical features from a model's logits to measure confidence/uncertainty.
        These features help the router learn when to trust each model.
        """
        # Convert logits to probabilities with temperature scaling
        probs = torch.softmax(logits / self.temperature, dim=-1)
        
        # Get top 2 probabilities for each sample
        top2 = probs.topk(k=2, dim=-1).values
        
        # Feature 1: Confidence - highest probability (how sure is the top prediction?)
        confidence = top2[:, 0]
        
        # Feature 2: Margin - gap between top 2 predictions (how much better is #1 vs #2?)
        margin = top2[:, 0] - top2[:, 1]
        
        # Feature 3: Entropy - uncertainty measure (are probabilities spread out or focused?)
        entropy = -(probs * (probs.clamp_min(1e-12).log())).sum(dim=-1)
        
        # Feature 4: Logit norm - overall magnitude of the logit vector
        logit_norm = logits.norm(p=2, dim=-1)
        
        # Feature 5: Top logit - strongest signal before softmax
        top_logit = logits.max(dim=-1).values
        
        # Stack all 5 features: [batch_size, 5]
        return torch.stack([top_logit, margin, entropy, logit_norm, confidence], dim=-1)

    def _pair_features(self, logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
        """
        Combine features from both models into a single feature vector.
        Output: [toplogit_A, margin_A, entropy_A, norm_A, conf_A,
                 toplogit_B, margin_B, entropy_B, norm_B, conf_B]
        """
        feat_a = self._expert_features(logits_a)  # [batch_size, 5]
        feat_b = self._expert_features(logits_b)  # [batch_size, 5]
        return torch.cat([feat_a, feat_b], dim=-1)  # [batch_size, 10]

    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Normalize features using training set statistics (mean=0, std=1).
        Returns unnormalized features if router hasn't been trained yet.
        """
        if self.feature_mean is None or self.feature_std is None:
            return features
        # Standardization: z = (x - μ) / σ
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
        # Set models to evaluation mode (disable dropout, etc.)
        model_a.eval()
        model_b.eval()

        # Create deterministic random generator for reproducible data loading
        g = torch.Generator()
        g.manual_seed(seed)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)

        # Collect training data from disagreement cases
        all_features = []  # Will store feature vectors
        all_labels = []    # Will store binary labels (0=pick A, 1=pick B)

        # Collect training data without computing gradients
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)

                # Get predictions from both models
                logits_a = model_a(x)
                logits_b = model_b(x)

                pred_a = logits_a.argmax(dim=-1)  # Model A's predicted class
                pred_b = logits_b.argmax(dim=-1)  # Model B's predicted class

                # Filter 1: Only keep disagreement cases
                disagree = pred_a != pred_b  # Boolean: True where models disagree
                if disagree.sum().item() == 0:
                    continue  # No disagreements in this batch, skip

                # Check which model(s) are correct
                a_correct = pred_a == y  # Boolean: True where A is correct
                b_correct = pred_b == y  # Boolean: True where B is correct

                # Filter 2: XOR - only keep cases where EXACTLY one model is correct
                # This gives clean training data with definitive right answers
                one_correct = disagree & (a_correct ^ b_correct)
                if one_correct.sum().item() == 0:
                    continue  # No clean samples in this batch, skip

                # Extract features for all samples (before filtering)
                feat = self._pair_features(logits_a, logits_b)
                
                # Create labels: 1.0 if B is correct, 0.0 if A is correct
                # Only for the filtered samples where exactly one is correct
                labels = b_correct[one_correct].float()

                # Store filtered features and labels
                all_features.append(feat[one_correct].cpu())
                all_labels.append(labels.cpu())

        # Safety check: If no training data collected, disable router
        if not all_features:
            self.is_trained = False
            self.stats["router_train_samples"] = 0
            self.stats["router_train_pos_rate"] = None
            self.stats["router_val_acc"] = None
            self.stats["router_margin_baseline_acc"] = None
            self.stats["router_enabled"] = False
            return

        # Concatenate all batches into single tensors
        features = torch.cat(all_features, dim=0)  # [total_samples, 10]
        labels = torch.cat(all_labels, dim=0)      # [total_samples]

        # Record training dataset statistics
        self.stats["router_train_samples"] = int(labels.numel())  # Total samples
        self.stats["router_train_pos_rate"] = float(labels.mean().item())  # Class balance

        # Safety check: Need minimum 20 samples for meaningful train/val split
        num_samples = labels.numel()
        if num_samples < 20:
            # Not enough data to train reliably, disable router
            self.is_trained = False
            self.stats["router_val_acc"] = None
            self.stats["router_margin_baseline_acc"] = None
            self.stats["router_enabled"] = False
            return

        # Create reproducible train/validation split (80/20)
        g_split = torch.Generator()
        g_split.manual_seed(seed)
        perm = torch.randperm(num_samples, generator=g_split)  # Random shuffle of indices
        val_size = max(1, int(0.2 * num_samples))  # 20% validation, minimum 1 sample
        train_idx = perm[val_size:]  # First 80% for training
        val_idx = perm[:val_size]    # Last 20% for validation

        # Split data using shuffled indices
        train_features = features[train_idx]
        train_labels = labels[train_idx]
        val_features = features[val_idx]
        val_labels = labels[val_idx]

        # Feature normalization: compute statistics from TRAINING set only
        feature_mean = train_features.mean(dim=0, keepdim=True)  # Mean of each feature
        feature_std = train_features.std(dim=0, keepdim=True).clamp_min(1e-6)  # Std, avoid div by 0

        # Apply standardization (mean=0, std=1) to both sets using TRAIN statistics
        train_features = (train_features - feature_mean) / feature_std
        val_features = (val_features - feature_mean) / feature_std

        # Compute baseline accuracy: simple "pick the model with higher margin" heuristic
        margin_a = features[val_idx][:, 1]  # Model A's margin feature (column 1)
        margin_b = features[val_idx][:, 6]  # Model B's margin feature (column 6)
        margin_choose_b = margin_b > margin_a  # True if margin heuristic picks B
        margin_acc = (margin_choose_b == (val_labels > 0.5)).float().mean().item()

        # Initialize router and optimizer
        self.router = _DisagreementRouter(in_features=features.size(1)).to(device)
        optimizer = torch.optim.Adam(self.router.parameters(), lr=lr)

        # Move data to GPU/CPU
        train_features = train_features.to(device)
        train_labels = train_labels.to(device)
        val_features = val_features.to(device)
        val_labels = val_labels.to(device)

        # Handle class imbalance: weight positive class (pick B) inversely to its frequency
        pos = train_labels.sum().item()  # Number of "pick B" labels
        neg = train_labels.numel() - pos  # Number of "pick A" labels
        pos_weight_value = (neg / max(pos, 1.0))  # Higher weight if B is rare
        pos_weight = torch.tensor(pos_weight_value, device=device)

        # Training loop: train router and track best validation accuracy
        self.router.train()
        best_state = None
        best_val_acc = -1.0
        
        for _ in range(epochs):
            # Forward pass
            logits = self.router(train_features)
            
            # Compute loss with class balancing
            loss = F.binary_cross_entropy_with_logits(logits, train_labels, pos_weight=pos_weight)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Validation: check performance on held-out data
            with torch.no_grad():
                self.router.eval()
                val_logits = self.router(val_features)
                val_pred = torch.sigmoid(val_logits) >= 0.5  # Convert to binary predictions
                val_acc = (val_pred == (val_labels > 0.5)).float().mean().item()
                
                # Save best model (early stopping based on validation accuracy)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_state = {k: v.detach().cpu().clone() for k, v in self.router.state_dict().items()}
                self.router.train()

        # Restore best model from training
        if best_state is not None:
            self.router.load_state_dict(best_state)

        # Set to eval mode and save normalization parameters
        self.router.eval()
        self.feature_mean = feature_mean.to(device)
        self.feature_std = feature_std.to(device)
        
        # Record final statistics
        self.stats["router_val_acc"] = float(best_val_acc)
        self.stats["router_margin_baseline_acc"] = float(margin_acc)

        # Adaptive enabling: only use router if it beats baseline by threshold
        # This prevents using an overfitted router that doesn't genuinely help
        self.is_trained = best_val_acc >= (margin_acc + min_gain_over_margin)
        self.stats["router_enabled"] = bool(self.is_trained)

    def combine(self, logits_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Main prediction method: Combine predictions from two models.
        Strategy: If agree, use that. If disagree, use router (or margin fallback).
        """
        # Input validation
        if len(logits_list) == 0:
            raise ValueError("logits_list is empty")
        if len(logits_list) == 1:
            return logits_list[0]
        if len(logits_list) != 2:
            raise ValueError("AgreeThenRouterCoordinator currently supports exactly 2 experts.")

        # Stack logits: [2, batch_size, num_classes]
        stacked = torch.stack(logits_list, dim=0)
        logits_a = stacked[0]  # Model A's logits
        logits_b = stacked[1]  # Model B's logits

        # Get predicted classes
        pred_a = logits_a.argmax(dim=-1)
        pred_b = logits_b.argmax(dim=-1)

        # Check agreement
        agree = pred_a == pred_b
        disagree = ~agree

        # Initialize: assume we pick Model A (choose_b = False)
        choose_b = torch.zeros_like(pred_a, dtype=torch.bool)

        # Handle disagreements: use router or fallback to margin heuristic
        if disagree.any():
            if self.is_trained:
                # Router is enabled: use neural network to decide
                with torch.no_grad():
                    # Extract and normalize features for disagreement cases only
                    pair_feat = self._pair_features(logits_a[disagree], logits_b[disagree])
                    pair_feat = self._normalize_features(pair_feat)
                    
                    # Router prediction: sigmoid >= 0.5 means pick B
                    router_logits = self.router(pair_feat)
                    choose_b_disagree = torch.sigmoid(router_logits) >= 0.5
                
                # Update decisions for disagreement cases
                choose_b[disagree] = choose_b_disagree
            else:
                # Router disabled: fall back to margin heuristic (pick more confident model)
                probs = torch.softmax(stacked / self.temperature, dim=-1)
                top2 = probs.topk(k=2, dim=-1).values  # Top 2 probs for each model
                margin = top2[..., 0] - top2[..., 1]   # Confidence margin [2, batch_size]
                
                # Pick B if B's margin > A's margin
                choose_b[disagree] = margin[1, disagree] > margin[0, disagree]

        # Convert boolean to integer indices: 0=A, 1=B
        best_idx = choose_b.long()

        # Update statistics
        if (not self.stats["picked"]) or (len(self.stats["picked"]) != 2):
            self.stats["picked"] = [0, 0]
        self.stats["picked"][0] += (best_idx == 0).sum().item()  # Count A selections
        self.stats["picked"][1] += (best_idx == 1).sum().item()  # Count B selections
        self.stats["total"] += pred_a.size(0)
        self.stats["disagree"] += disagree.sum().item()

        # Use best_idx to select the appropriate logits from stacked tensor
        # Reshape indices to match stacked dimensions and gather along model dimension
        gather_idx = best_idx.view(1, -1, 1).expand(1, stacked.size(1), stacked.size(2))
        return torch.gather(stacked, dim=0, index=gather_idx).squeeze(0)
