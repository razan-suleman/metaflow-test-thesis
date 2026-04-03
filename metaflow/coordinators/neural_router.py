"""
Neural Router Coordinator - Learns optimal routing from expert logits.
Better than heuristics because it learns patterns from data.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class RouterNetwork(nn.Module):
    """Small MLP that learns which expert to trust based on logits."""
    
    def __init__(self, input_dim=20, hidden_dim=64):
        super().__init__()
        # Input: concatenated logits from both experts (2 * 10 = 20)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)  # Output: choose expert A or B
        )
    
    def forward(self, logits_a, logits_b):
        """
        Args:
            logits_a: (batch, 10) logits from expert A
            logits_b: (batch, 10) logits from expert B
        Returns:
            (batch, 2) scores for choosing A vs B
        """
        x = torch.cat([logits_a, logits_b], dim=-1)  # (batch, 20)
        return self.net(x)


class NeuralRouterCoordinator:
    """Coordinator that uses learned neural router."""
    
    def __init__(self, model_a, model_b, device="cpu"):
        self.model_a = model_a
        self.model_b = model_b
        self.device = device
        self.router = None
        
    def train_router(self, probe_loader, epochs=50, lr=0.001):
        """
        Train the router on probe dataset.
        
        Strategy: For each sample, determine which expert is actually correct,
        and train router to predict that choice.
        """
        print(f"\n{'='*60}")
        print("Training Neural Router")
        print(f"{'='*60}")
        
        # Collect logits and determine oracle choices
        all_logits_a = []
        all_logits_b = []
        all_oracle_choices = []  # 0 for A, 1 for B
        
        self.model_a.eval()
        self.model_b.eval()
        
        print("Collecting expert predictions...")
        with torch.no_grad():
            for x, y in probe_loader:
                x, y = x.to(self.device), y.to(self.device)
                
                logits_a = self.model_a(x)
                logits_b = self.model_b(x)
                
                pred_a = logits_a.argmax(dim=-1)
                pred_b = logits_b.argmax(dim=-1)
                
                correct_a = (pred_a == y).float()
                correct_b = (pred_b == y).float()
                
                # Oracle choice: pick the expert that's correct
                # If both correct or both wrong, use confidence as tiebreaker
                conf_a = torch.softmax(logits_a, dim=-1).max(dim=-1)[0]
                conf_b = torch.softmax(logits_b, dim=-1).max(dim=-1)[0]
                
                oracle_choice = torch.where(
                    correct_a > correct_b,
                    torch.zeros_like(correct_a, dtype=torch.long),  # Choose A
                    torch.where(
                        correct_b > correct_a,
                        torch.ones_like(correct_b, dtype=torch.long),   # Choose B
                        (conf_b > conf_a).long()  # Tiebreaker: higher confidence
                    )
                )
                
                all_logits_a.append(logits_a.cpu())
                all_logits_b.append(logits_b.cpu())
                all_oracle_choices.append(oracle_choice.cpu())
        
        # Create training dataset
        logits_a_train = torch.cat(all_logits_a, dim=0)
        logits_b_train = torch.cat(all_logits_b, dim=0)
        choices_train = torch.cat(all_oracle_choices, dim=0)
        
        train_dataset = TensorDataset(logits_a_train, logits_b_train, choices_train)
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Oracle choice distribution: A={100*(choices_train==0).float().mean():.1f}%, B={100*(choices_train==1).float().mean():.1f}%")
        
        # Initialize router
        self.router = RouterNetwork(input_dim=20, hidden_dim=64).to(self.device)
        optimizer = optim.Adam(self.router.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        self.router.train()
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            total = 0
            
            for logits_a, logits_b, choice in train_loader:
                logits_a = logits_a.to(self.device)
                logits_b = logits_b.to(self.device)
                choice = choice.to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass
                router_scores = self.router(logits_a, logits_b)
                loss = criterion(router_scores, choice)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                # Accuracy
                pred_choice = router_scores.argmax(dim=-1)
                correct += (pred_choice == choice).sum().item()
                total += choice.size(0)
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg_loss = total_loss / len(train_loader)
                acc = 100 * correct / total
                print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {acc:.2f}%")
        
        print(f"{'='*60}")
        print(f"✓ Router training complete! Final accuracy: {acc:.2f}%")
        print(f"{'='*60}\n")
        
        self.router.eval()
    
    def predict(self, test_loader, return_stats=False):
        """
        Make predictions using the trained neural router.
        
        Returns:
            predictions: (N,) tensor of predicted classes
            stats: dict with routing statistics (if return_stats=True)
        """
        if self.router is None:
            raise RuntimeError("Router not trained! Call train_router() first.")
        
        all_preds = []
        picks_a = 0
        picks_b = 0
        
        self.model_a.eval()
        self.model_b.eval()
        self.router.eval()
        
        with torch.no_grad():
            for x, _ in test_loader:
                x = x.to(self.device)
                
                # Get logits from both experts
                logits_a = self.model_a(x)
                logits_b = self.model_b(x)
                
                # Router decides which expert to use
                router_scores = self.router(logits_a, logits_b)
                choice = router_scores.argmax(dim=-1)  # 0=A, 1=B
                
                # Select predictions based on router choice
                pred_a = logits_a.argmax(dim=-1)
                pred_b = logits_b.argmax(dim=-1)
                
                preds = torch.where(choice == 0, pred_a, pred_b)
                all_preds.append(preds.cpu())
                
                picks_a += (choice == 0).sum().item()
                picks_b += (choice == 1).sum().item()
        
        predictions = torch.cat(all_preds, dim=0)
        
        if return_stats:
            total = picks_a + picks_b
            stats = {
                'picks_a': picks_a,
                'picks_b': picks_b,
                'pct_a': 100 * picks_a / total,
                'pct_b': 100 * picks_b / total
            }
            return predictions, stats
        
        return predictions


def create_neural_router_coordinator(model_a, model_b, probe_loader, device="cpu", epochs=50):
    """
    Helper function to create and train a neural router coordinator.
    
    Args:
        model_a: Expert A model
        model_b: Expert B model
        probe_loader: DataLoader for training the router
        device: Device to use
        epochs: Number of training epochs
    
    Returns:
        Trained NeuralRouterCoordinator
    """
    coordinator = NeuralRouterCoordinator(model_a, model_b, device)
    coordinator.train_router(probe_loader, epochs=epochs)
    return coordinator


class NeuralRouterWrapper:
    """
    Wrapper to make NeuralRouter compatible with MetaFlow coordinator interface.
    
    This adapter allows the router to work with MetaFlow's combine() protocol
    even though it needs access to the underlying models for routing decisions.
    """
    
    def __init__(self, neural_router_coordinator):
        self.neural_router = neural_router_coordinator
        self.stats = {"picked": [0, 0], "total": 0, "disagree": 0}
    
    def combine(self, logits_list):
        """
        Combine logits by routing to the appropriate expert.
        
        Args:
            logits_list: List of [logits_a, logits_b]
        
        Returns:
            Selected logits based on router decision
        """
        logits_a, logits_b = logits_list[0], logits_list[1]
        
        # Get router decision
        with torch.no_grad():
            if self.neural_router.router is None:
                # Fallback to confidence if router not trained
                conf_a = torch.softmax(logits_a, dim=-1).max(dim=-1)[0]
                conf_b = torch.softmax(logits_b, dim=-1).max(dim=-1)[0]
                choice = (conf_b > conf_a).long()
            else:
                router_scores = self.neural_router.router(logits_a, logits_b)
                choice = router_scores.argmax(dim=-1)  # 0=A, 1=B
        
        # Select logits based on choice
        batch_size = logits_a.size(0)
        selected_logits = torch.where(
            choice.unsqueeze(1).expand_as(logits_a) == 0,
            logits_a,
            logits_b
        )
        
        # Update stats
        self.stats["picked"][0] += (choice == 0).sum().item()
        self.stats["picked"][1] += (choice == 1).sum().item()
        self.stats["total"] += batch_size
        
        # Check disagreement
        pred_a = logits_a.argmax(dim=-1)
        pred_b = logits_b.argmax(dim=-1)
        self.stats["disagree"] += (pred_a != pred_b).sum().item()
        
        return selected_logits
