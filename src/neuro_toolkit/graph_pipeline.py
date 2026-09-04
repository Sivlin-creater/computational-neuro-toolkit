import torch
import torch.nn as nn
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv


class SimpleGNN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super().__init__()
        self.conv1 = GCNConv(in_feats, hidden_feats)
        self.conv2 = GCNConv(hidden_feats, out_feats)

    def forward(self, x, edge_index):
        h = torch.relu(self.conv1(x, edge_index))
        out = self.conv2(h, edge_index)
        return out


def run_gnn_pipeline(epochs=50, data_dir="/tmp/Cora"):
    dataset = Planetoid(root=data_dir, name="Cora")
    data = dataset[0]

    model = SimpleGNN(dataset.num_features, 16, dataset.num_classes)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=-1)
        acc = (pred[data.test_mask] == data.y[data.test_mask]
               ).sum() / data.test_mask.sum()
    return acc.item()


if __name__ == "__main__":
    accuracy = run_gnn_pipeline()
    print(f"Project 2: Cora Test Accuracy = {accuracy * 100:.2f}%")
