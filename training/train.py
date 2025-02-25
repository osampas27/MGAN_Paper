import torch
import torch.optim as optim
import torch.nn as nn
from models.mgan_model import MGAN
from models.loss_functions import CrossEntropyLossWithLabelSmoothing
from data.preprocessing import DataPreprocessor

# Hyperparameters
input_dim = 128
hidden_dim = 256
output_dim = 2  # Binary classification
num_heads = 8
num_layers = 4
dropout = 0.5
epochs = 50
batch_size = 32
learning_rate = 0.001

# Initialize Model
model = MGAN(input_dim, hidden_dim, output_dim, num_heads, num_layers, dropout)
criterion = CrossEntropyLossWithLabelSmoothing(smoothing=0.1)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Load Data (Example)
train_loader, val_loader = DataPreprocessor.load_data('data/train.csv', batch_size=batch_size)

# Training Loop
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for x, adj, y in train_loader:
        optimizer.zero_grad()
        output = model(x, adj)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")
