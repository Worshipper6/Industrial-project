import torch
from torch_geometric.datasets import Planetoid#下载数据集用的
from torch_geometric.transforms import NormalizeFeatures
from torch.nn import Linear, ReLU, Dropout, Module
from torch_geometric.nn import GCNConv

dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())#transform预处理

print()
print(f'Dataset: {dataset}:')
print('======================')
print(f'Number of graphs: {len(dataset)}')
print(f'Number of features: {dataset.num_features}')
print(f'Number of classes: {dataset.num_classes}')

# class MLP(Module):
#     def __init__(self):
#         super().__init__()
#         self.linear = torch.nn.Sequential(
#             Linear(dataset.num_features, 32),
#             ReLU(),
#             #Dropout(p=0.5),
#             #Linear(32,32),
#             #ReLU(),
#             Linear(32, dataset.num_classes)
#         )
    
#     def forward(self, x):
#         return self.linear(x)

class MLP(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        torch.manual_seed(12345)
        self.lin1 = Linear(dataset.num_features, hidden_channels)
        self.lin2 = Linear(hidden_channels, dataset.num_classes)

    def forward(self, x):
        x = self.lin1(x)
        x = x.relu()
        x = torch.nn.functional.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x
    
model_MLP = MLP(16)
criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model_MLP.parameters(), lr=0.01, weight_decay=5e-4)  # Define optimizer.

for i in range(1,201):
    model_MLP.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model_MLP(dataset.x)  # Perform a single forward pass.
    loss = criterion(out[dataset.train_mask], dataset.y[dataset.train_mask])  # Compute the loss solely based on the training nodes.
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    print(f'Epoch: {i}, Loss: {loss:.4f}')


out = model_MLP(dataset.x)
pred = out.argmax(dim=1)
test_correct = pred[dataset.test_mask] == dataset.y[dataset.test_mask]  # Check against ground-truth labels.
test_acc = int(test_correct.sum()) / int(dataset.test_mask.sum())  # Derive ratio of correct predictions.
print(test_acc)