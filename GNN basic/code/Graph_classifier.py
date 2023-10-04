import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv, GraphConv, global_mean_pool
from torch.nn import ReLU
import torch.nn.functional as F
from torch_geometric.loader import DataLoader

dataset = TUDataset(root='data/TUDataset', name='PROTEINS', transform=NormalizeFeatures())
dataset = dataset.shuffle()

train_dataset = dataset[:800]
test_dataset = dataset[800:]

train_loader = DataLoader(train_dataset, batch_size=800, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=313, shuffle=False)

class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        self.conv1 = GraphConv(dataset.num_features, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, dataset.num_classes)
    
    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = ReLU()(x)
        x = self.conv2(x, edge_index)
        x = ReLU()(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        return global_mean_pool(x, batch)
    
model = GCN(hidden_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()
    
def train():
    model.train()
    for data in train_loader:
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
    

def test():
    model.eval()
    for data in test_loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct = int((pred == data.y).sum())  # Check against ground-truth labels.
        acc = correct / len(pred)
        return acc
    


for epoch in range(1,101):
    train()
    print(test())