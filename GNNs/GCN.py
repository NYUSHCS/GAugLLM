import torch
from torch_geometric.datasets import Planetoid, NELL
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, GATConv
import data_utils.logistic_regression_eval as eval
from data_utils.load import load_llm_feature_and_data
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 512)  # Change the hidden size to 64
        self.conv2 = GCNConv(512, num_classes)
        self.norm = torch.nn.BatchNorm1d(512)  # Adjust BatchNorm1d size to 64

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.norm(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.8, training=self.training)  # Set dropout rate to 0.8
        x = self.conv2(x, edge_index)
        return x



class GIN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GIN, self).__init__()
        nn1 = torch.nn.Sequential(torch.nn.Linear(num_node_features, 256), torch.nn.ReLU(), torch.nn.Linear(256, 256))
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(torch.nn.Linear(256, num_classes))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_node_features, 32, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(32 * 8, num_classes, heads=1, concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def split_data_k(y, k_shot=20, data_random_seed=0):
    np.random.seed(data_random_seed)
    num_classes = y.max() + 1
    all_indices = np.arange(len(y))

    train_indices = []

    for i in range(num_classes):
        class_indices = np.where(y == i)[0]
        if len(class_indices) < k_shot:
            raise ValueError(f"Not enough samples in class {i} for k-shot learning")
        class_train_indices = np.random.choice(class_indices, k_shot, replace=False)
        train_indices.extend(class_train_indices)

    all_indices = np.setdiff1d(all_indices, train_indices)

    val_indices = []

    for i in range(num_classes):
        class_indices = np.where(y == i)[0]
        class_indices = np.setdiff1d(class_indices, train_indices)  # remove already chosen train_indices
        class_val_indices = np.random.choice(class_indices, 30, replace=False)
        val_indices.extend(class_val_indices)

    val_indices = np.array(val_indices)
    all_indices = np.setdiff1d(all_indices, val_indices)

    # All remaining indices will be for testing
    test_indices = all_indices

    train_mask = np.isin(np.arange(len(y)), train_indices)
    val_mask = np.isin(np.arange(len(y)), val_indices)
    test_mask = np.isin(np.arange(len(y)), test_indices)

    return train_mask, val_mask, test_mask


def train(model, data, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    loss_function = torch.nn.CrossEntropyLoss().to(device)
    #train_mask,val_mask,test_mask = adjust_masks(data)
    train_mask, val_mask, test_mask = split_data_k(data.y, k_shot=20, data_random_seed=0)
    best_val_accuracy = 0.0
    model.train()
    for epoch in range(200):
        out = model(data)
        optimizer.zero_grad()
        loss = loss_function(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        # Evaluate on validation data
        val_accuracy = evaluate(model, data, val_mask)
        print('Epoch {:03d} loss {:.4f}, Val Accuracy: {:.4f}'.format(epoch, loss.item(), val_accuracy))

        #Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = model.state_dict()  # Save the model parameters

    #Load the best model for testing
    model.load_state_dict(best_model)
    return test_mask

def evaluate(model, data, val_mask):
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = int(pred[val_mask].eq(data.y[val_mask]).sum().item())
    acc = correct / int(val_mask.sum())
    #print('Val Accuracy: {:.4f}'.format(acc))
    return acc

def test(model, data, test_mask):
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = int(pred[test_mask].eq(data.y[test_mask]).sum().item())
    acc = correct / int(test_mask.sum())
    print('GCN Accuracy: {:.4f}'.format(acc))


def main(args):

    dataset = load_llm_feature_and_data(
        dataset_name=args.dataset,
        lm_model_name='microsoft/deberta-base',
        feature_type=args.feature_type,
        device='cpu',
        use_BoW=args.use_BoW, )

    num_node_features = dataset.x.shape[1]
    num_classes =dataset.y.max().item() + 1

    device = args.device

    if args.model_type == 'GCN':
        model = GCN(num_node_features, num_classes).to('cpu')
    elif args.model_type == 'GIN':
        model = GIN(num_node_features, num_classes).to('cpu')
    elif args.model_type == 'GAT':
        model = GAT(num_node_features, num_classes).to('cpu')

    test_mask = train(model, dataset, device)

    test(model, dataset, test_mask)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Graph Neural Network Training")
    parser.add_argument('--dataset', type=str, default = 'cora', help="Name of the dataset")
    parser.add_argument('--feature_type', type=str, default='ogb', help="Feature type for dataset")
    parser.add_argument('--use_BoW', type=bool, default=True,  help="Flag to use BoW")
    parser.add_argument('--device', type=str, default='cpu', help="Device to use for training")
    parser.add_argument('--model_type', type=str, default='GCN', help="Language model name")

    args = parser.parse_args()
    main(args)