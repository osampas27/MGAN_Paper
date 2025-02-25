import torch
from models.mgan_model import MGAN
from data.preprocessing import DataPreprocessor

# Load Model
model = MGAN(input_dim=128, hidden_dim=256, output_dim=2, num_heads=8, num_layers=4)
model.load_state_dict(torch.load("results/mgan_model.pth"))
model.eval()

# Load Test Data
test_loader = DataPreprocessor.load_data('data/test.csv', batch_size=32)

# Evaluate
correct = 0
total = 0
for x, adj, y in test_loader:
    output = model(x, adj)
    predictions = torch.argmax(output, dim=1)
    correct += (predictions == y).sum().item()
    total += y.size(0)

print(f"Accuracy: {correct / total:.4f}")
