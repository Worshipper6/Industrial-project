import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv

# Load the Cora dataset
dataset = Planetoid(root='data/Cora', name='Cora')
data = dataset[0]

# Define the SAGE model
class SAGEModel(torch.nn.Module):
    def __init__(self, in_features, hidden_units, num_classes):
        super(SAGEModel, self).__init__()
        self.conv1 = SAGEConv(in_features, hidden_units)
        self.conv2 = SAGEConv(hidden_units, num_classes)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create the model and move it to the device
model = SAGEModel(dataset.num_features, hidden_units=16, num_classes=dataset.num_classes).to(device)

# Create the data loader
loader = DataLoader(dataset, batch_size=128, shuffle=True)

# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0005)
criterion = torch.nn.NLLLoss()

# Training loop
def train():
    model.train()

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        for param in model.parameters():
            loss += 0.001 * torch.norm(param, p=2)  # L2 regularization
        loss.backward()
        optimizer.step()

# Evaluation loop
def test():
    model.eval()
    logits, accs = model(data.x.to(device), data.edge_index.to(device)), []

    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)

    return accs

# Train the model
for epoch in range(1, 201):
    train()
    if epoch % 10 == 0:
        train_acc, val_acc, test_acc = test()
        print(f'Epoch: {epoch:03d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')