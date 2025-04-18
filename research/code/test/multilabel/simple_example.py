import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# Sample dataset
class MultiLabelDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
labels = [[1, 0, 1], [0, 1, 0], [1, 1, 0]]

dataset = MultiLabelDataset(data, labels)
dataloader = DataLoader(dataset, batch_size=2)


# Model
class MultiLabelModel(nn.Module):
    def __init__(self, input_size, num_labels):
        super(MultiLabelModel, self).__init__()
        self.linear = nn.Linear(input_size, num_labels)

    def forward(self, x):
        return self.linear(x)


input_size = 3
num_labels = 3
model = MultiLabelModel(input_size, num_labels)

# Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
epochs = 10
for epoch in range(epochs):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")

# Prediction
with torch.no_grad():
    sample_input = torch.tensor([[2, 4, 6]], dtype=torch.float32)
    output = model(sample_input)
    probabilities = torch.sigmoid(output)
    predicted_labels = (probabilities > 0.5).int()
    print("Predicted probabilities:", probabilities)
    print("Predicted labels:", predicted_labels)
