import torch
import torch.nn as nn
import torch.optim as optim
import random

# Define a simple neural network
class MyModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Hyperparameters
batch_size = 32
learning_rate = 3e-4
max_iters = 3000
eval_iters = 50

# Create a sample dataset and dataloaders (you will need to replace this with your own data)
input_size = 64
output_size = 10
dataset_size = 1000

# Create a random dataset
data = [(torch.randn(input_size), torch.randint(0, output_size, (1,))) for _ in range(dataset_size)]

# Split the dataset into training and validation sets
train_size = int(0.8 * len(data))
train_data, val_data = data[:train_size], data[train_size:]

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

# Initialize the model and optimizer
model = MyModel(input_size, 128, output_size)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(max_iters):
    model.train()
    total_loss = 0

    for batch_inputs, batch_labels in train_loader:
        optimizer.zero_grad()
        batch_outputs = model(batch_inputs)
        loss = nn.CrossEntropyLoss()(batch_outputs, batch_labels.squeeze())
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % eval_iters == 0:
        # Evaluate the model on the validation set
        model.eval()
        with torch.no_grad():
            val_loss = 0
            correct = 0
            total = 0

            for batch_inputs, batch_labels in val_loader:
                batch_outputs = model(batch_inputs)
                val_loss += nn.CrossEntropyLoss()(batch_outputs, batch_labels.squeeze()).item()
                _, predicted = batch_outputs.max(1)
                total += batch_labels.size(0)
                correct += predicted.eq(batch_labels.squeeze()).sum().item()

            accuracy = 100 * correct / total
            print(f"Epoch [{epoch + 1}/{max_iters}] Loss: {total_loss / len(train_loader)} Validation Accuracy: {accuracy}%")
