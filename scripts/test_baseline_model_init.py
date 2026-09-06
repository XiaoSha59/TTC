import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from models.baseline import SupervisedImageClassifier

print(">>> Testing SupervisedImageClassifier initialization with class_weights=[0.632, 0.368]...")
model = SupervisedImageClassifier(
    num_classes=2,
    max_epochs=100,
    lr=1e-4,
    class_weights=[0.632, 0.368]
)

print("Criterion:", model.criterion)
print("Criterion weight:", model.criterion.weight)
expected_w0 = (1.0 / 0.632) / (1.0 / 0.632 + 1.0 / 0.368)
expected_w1 = (1.0 / 0.368) / (1.0 / 0.632 + 1.0 / 0.368)
print(f"Calculated weight: w0={model.criterion.weight[0]:.4f} (expected {expected_w0:.4f}), w1={model.criterion.weight[1]:.4f} (expected {expected_w1:.4f})")
assert torch.isclose(model.criterion.weight[0], torch.tensor(expected_w0, dtype=torch.float32))
assert torch.isclose(model.criterion.weight[1], torch.tensor(expected_w1, dtype=torch.float32))

# Test forward pass with dummy data
dummy_x = torch.randn(4, 3, 224, 224)
dummy_y = torch.tensor([0, 1, 0, 1])
out = model(dummy_x)
print("Output logits shape:", out.shape)
loss = model.criterion(out, dummy_y)
print("Loss computed successfully:", loss.item())
print("MODEL INITIALIZATION & LOSS PASS: 100% VERIFIED!")
