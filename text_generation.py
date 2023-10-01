import torch
import torch.nn as nn
import numpy as np

# Define a simple character-level RNN model
class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, h):
        out, h = self.rnn(x, h)
        out = self.fc(out)
        return out, h

# Hyperparameters
input_size = 100  # Example: Size of the character vocabulary
hidden_size = 128
output_size = 100  # Example: Size of the character vocabulary
sequence_length = 20
num_layers = 2
num_epochs = 100

# Create a toy dataset (sequence prediction task)
# ...

# Initialize the model and optimizer
model = CharRNN(input_size, hidden_size, output_size, num_layers)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training loop for text generation
# ...

# Generating text with the trained RNN
# ...
