import torch
import torch.nn as nn

def compute_weights(class_weights):
    w_0 = 1.0 / class_weights[0]
    w_1 = 1.0 / class_weights[1]
    sum_w = w_0 + w_1
    return torch.tensor([w_0 / sum_w, w_1 / sum_w], dtype=torch.float32)

# Test BreastMNIST
w_breast = compute_weights([0.632, 0.368])
print("BreastMNIST weights [benign, malignant]:", w_breast.tolist())
assert round(w_breast[0].item(), 3) == 0.368
assert round(w_breast[1].item(), 3) == 0.632
criterion_breast = nn.CrossEntropyLoss(weight=w_breast)
logits = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
targets = torch.tensor([0, 1])
loss = criterion_breast(logits, targets)
print("BreastMNIST CrossEntropy loss:", loss.item())

# Test PneumoniaMNIST
w_pneu = compute_weights([0.35, 0.65])
print("PneumoniaMNIST weights [normal, pneumonia]:", w_pneu.tolist())
assert round(w_pneu[0].item(), 2) == 0.65
assert round(w_pneu[1].item(), 2) == 0.35

# Test FracAtlas
w_frac = compute_weights([0.79, 0.21])
print("FracAtlas weights [non-fractured, fractured]:", w_frac.tolist())
assert round(w_frac[0].item(), 2) == 0.21
assert round(w_frac[1].item(), 2) == 0.79

print(">>> ALL MATHEMATICAL & LOSS TESTS PASSED 100%!")
