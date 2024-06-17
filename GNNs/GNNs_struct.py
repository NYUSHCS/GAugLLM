from sklearn.model_selection import train_test_split
from torch_geometric.nn import GCNConv, GINConv, GATConv
import argparse
import numpy as np
import torch
import torch.nn.functional as F
import os
import sys
import json
from torch_geometric.utils.sparse import to_edge_index
import random
sys.path.append("..") # TODO merge TAPE into current repo
from data_utils.load import load_llm_feature_and_data
from data_utils.dataset import check_candidate_lists,modify_edge_index_one_time_ratio
'''
# os.chdir(os.getcwd()+'/GNNs')
class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, args.dim_hidden)
        self.conv2 = GCNConv(args.dim_hidden, num_classes)
        self.norm = torch.nn.BatchNorm1d(args.dim_hidden)

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv1(x, edge_index)
        x = self.norm(x)
        x = F.relu(x)
        x = F.dropout(x, p=args.dropout, training=self.training)

        x = self.conv2(x, edge_index)
        return x
'''

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, number_of_layers):
        super(GCN, self).__init__()

        self.number_of_layers = number_of_layers
        self.convs = torch.nn.ModuleList()
        
        # First layer
        self.convs.append(GCNConv(num_node_features, args.dim_hidden))
        
        # Intermediate layers
        for _ in range(number_of_layers - 2):
            self.convs.append(GCNConv(args.dim_hidden, args.dim_hidden))

        # Last layer
        self.convs.append(GCNConv(args.dim_hidden, num_classes))
        self.norm = torch.nn.BatchNorm1d(args.dim_hidden)

    def forward(self, data):
        x, edge_index = data.x.float(), data.edge_index

        for i, conv in enumerate(self.convs):
            x = F.dropout(x, p=args.dropout, training=self.training)
            x = conv(x, edge_index)
            if i < self.number_of_layers - 1:
                x = self.norm(x)
                x = F.relu(x)

        return x

class GIN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GIN, self).__init__()
        nn1 = torch.nn.Sequential(torch.nn.Linear(num_node_features, args.dim_hidden), torch.nn.ReLU(),
                                  torch.nn.Linear(args.dim_hidden, args.dim_hidden))
        self.conv1 = GINConv(nn1)
        self.conv2 = GINConv(torch.nn.Linear(args.dim_hidden, num_classes))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x

'''
class GAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GAT, self).__init__()
        self.conv1 = GATConv(num_node_features, 8, heads=8, dropout=args.dropout)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, num_classes, heads=1, concat=False, dropout=args.dropout)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

'''
class GAT(torch.nn.Module):
    def __init__(self, num_node_features, num_classes, number_of_layers):
        super(GAT, self).__init__()

        self.number_of_layers = number_of_layers
        heads_intermediate = 8  # Number of heads for intermediate layers
        dim_intermediate = 8    # Dimension for intermediate layers

        self.convs = torch.nn.ModuleList()

        # First layer
        self.convs.append(GATConv(num_node_features, dim_intermediate, heads=heads_intermediate, dropout=args.dropout))

        # Intermediate layers
        for _ in range(number_of_layers - 2):
            self.convs.append(GATConv(dim_intermediate * heads_intermediate, dim_intermediate, heads=heads_intermediate, dropout=args.dropout))

        # Last layer
        self.convs.append(GATConv(dim_intermediate * heads_intermediate, num_classes, heads=1, concat=False, dropout=args.dropout))

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < self.number_of_layers - 1:
                x = F.elu(x)
                x = F.dropout(x, training=self.training)
            else:
                x = F.log_softmax(x, dim=1)

        return x



def split_data_k(y,  data_random_seed, k_shot=20):
    if data_random_seed != 99999:
        #print("Random Seed")
        #print(data_random_seed)
        random.seed(data_random_seed)
        np.random.seed(data_random_seed)
        torch.manual_seed(data_random_seed)
        num_classes = y.max() + 1
        all_indices = np.arange(len(y))
        y = y.cpu()
        train_indices = []
    
        for i in range(num_classes):
            class_indices = np.where(y == i)[0]
            if len(class_indices) < k_shot:
                #print(f"Not enough samples in class {i} for {k_shot}-shot learning, using all as train")
                class_train_indices = np.random.choice(class_indices, len(class_indices), replace=False)
            else: 
                class_train_indices = np.random.choice(class_indices, k_shot, replace=False)
            train_indices.extend(class_train_indices)
    
        all_indices = np.setdiff1d(all_indices, train_indices)
    
        val_indices = []
    
        for i in range(num_classes):
            class_indices = np.where(y == i)[0]
            class_indices = np.setdiff1d(class_indices, train_indices)  # remove already chosen train_indices
            
            #! if val is not sufficient , use rest as val
            class_val_indices = np.random.choice(class_indices, len(class_indices) if len(class_indices)<30 else 30, replace=False)
            val_indices.extend(class_val_indices)
    
        val_indices = np.array(val_indices)
        all_indices = np.setdiff1d(all_indices, val_indices)
    
        # All remaining indices will be for testing
        test_indices = all_indices
    
        train_mask = np.isin(np.arange(len(y)), train_indices)
        val_mask = np.isin(np.arange(len(y)), val_indices)
        test_mask = np.isin(np.arange(len(y)), test_indices)
    else:
        with open('/LLM4GCL_update/data_utils/ogbn_arxiv_split_idx.json', 'r') as f:
            split_idx = json.load(f)

        # Extract indices from the JSON file
        train_idx = torch.tensor(split_idx['train'])
        val_idx = torch.tensor(split_idx['valid'])
        test_idx = torch.tensor(split_idx['test'])

        # Convert indices to boolean masks
        train_mask = torch.zeros(len(y), dtype=torch.bool)
        val_mask = torch.zeros(len(y), dtype=torch.bool)
        test_mask = torch.zeros(len(y), dtype=torch.bool)

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


def split_data_s(y, data_random_seed=0):
    # train：0.2   val：0.2   test：0.6
    rng = np.random.RandomState(data_random_seed)  # this will ensure the dataset will be split exactly the same
    indices = np.arange(len(y))
    train_indices, temp_indices, y_train, y_temp = train_test_split(indices, y, test_size=0.8, random_state=rng)
    val_indices, test_indices, y_val, y_test = train_test_split(temp_indices, y_temp, test_size=0.75, random_state=rng)
    # Create train_mask, val_mask, and test_mask
    train_mask = np.zeros(len(y), dtype=bool)
    val_mask = np.zeros(len(y), dtype=bool)
    test_mask = np.zeros(len(y), dtype=bool)
    train_mask[train_indices] = True
    val_mask[val_indices] = True
    test_mask[test_indices] = True
    return train_mask, val_mask, test_mask


def train(model, data, device, dataseed):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_function = torch.nn.CrossEntropyLoss().to(device)

    train_mask, val_mask, test_mask = split_data_k(data.y, k_shot=args.k_shot, data_random_seed=dataseed)
    best_val_accuracy = 0.0

    model.train()

    for epoch in range(args.epochs):
        out = model(data)
        optimizer.zero_grad()
        loss = loss_function(out[train_mask], data.y[train_mask])
        loss.backward()
        optimizer.step()

        # Evaluate on validation data
        val_accuracy = evaluate(model, data, val_mask)
        #print('Epoch {:03d} loss {:.4f}, Val Accuracy: {:.4f}'.format(epoch, loss.item(), val_accuracy))

        # Check for improvement
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_model = model.state_dict()  # Save the model parameters

    # Load the best model for testing
    model.load_state_dict(best_model)
    return test_mask


def evaluate(model, data, val_mask):
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = int(pred[val_mask].eq(data.y[val_mask]).sum().item())
    acc = correct / int(val_mask.sum())
    return acc


def test(model, data, test_mask):
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = int(pred[test_mask].eq(data.y[test_mask]).sum().item())
    acc = correct / int(test_mask.sum())
    #print('TEST Accuracy: {:.4f}'.format(acc))
    return acc


def main(args):
    dataset = load_llm_feature_and_data(
        dataset_name=args.dataset,
        lm_model_name='microsoft/deberta-base',
        feature_type='attention-1',
        #feature_type='GIA',
        #feature_type='BOW',
        device=args.device)

    #dataset.edge_index,_ = to_edge_index(dataset.edge_index)
    candidate_lists = check_candidate_lists(args.dataset)
    edge_index = dataset.edge_index
    edge_index_aug = modify_edge_index_one_time_ratio(args.dataset,edge_index,candidate_lists,ratio=args.edge_ratio)
    print(edge_index.shape)
    print(edge_index_aug.shape)
    dataset.edge_index = edge_index_aug.to(args.device)
    num_node_features = dataset.x.shape[1]
    num_classes = dataset.y.max().item() + 1
    number_of_layers = args.num_layers
    device = args.device

    accs = []
    dataseeds = [0,1,2,3,4]
    if args.dataset == 'ogbn-arxiv':
        dataseeds = [99999]
    for dataseed in dataseeds:
        if args.model_type == 'GCN':
            model = GCN(num_node_features, num_classes,number_of_layers).to(device)
        elif args.model_type == 'GIN':
            model = GIN(num_node_features, num_classes,number_of_layers).to(device)
        elif args.model_type == 'GAT':
            model = GAT(num_node_features, num_classes,number_of_layers).to(device)

        
        test_mask = train(model, dataset, device, dataseed)

        acc = test(model, dataset, test_mask)
        accs.append(acc)

    avg_accuracy = np.mean(accs)
    std_deviation = np.std(accs)
    print('Average accuracy: {:.4f}'.format(avg_accuracy))
    print('Standard deviation: {:.4f}'.format(std_deviation))
    return {'avg_accuracy': avg_accuracy, 'std_deviation': std_deviation}


def save_args_to_file(args, results, output_folder="configs"):
    filename = args.dataset + '_' + args.model_type + '_' + args.feature_type + '_' + str(args.k_shot) + 'shot.config'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_file = os.path.join(output_folder, filename)
    
    # 尝试读取现有的平均精度值
    existing_accuracy = None
    if os.path.exists(output_file):
        with open(output_file, 'r') as file:
            for line in file:
                if "Average accuracy:" in line:
                    existing_accuracy = float(line.split(":")[1].strip())
                    break

    # 如果文件不存在或新的平均精度更高，写入新的内容
    if existing_accuracy is None or results['avg_accuracy'] > existing_accuracy:
        with open(output_file, 'w') as file:
            for arg in vars(args):
                file.write(f"{arg}: {getattr(args, arg)}\n")
            # Add results to the file
            file.write("\nResults:\n")
            file.write(f"Average accuracy: {results['avg_accuracy']:.4f}\n")
            file.write(f"Standard deviation: {results['std_deviation']:.4f}\n")
    else:
        print("Existing record has higher or equal average accuracy. No update made.")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Graph Neural Network Training")
    parser.add_argument('--dataset', type=str, default='amazon-photo', help="Name of the dataset")
    parser.add_argument('--feature_type', type=str, default='ogb', help="Feature type for dataset")
    parser.add_argument('--device', type=int, default=0, help="Device to use for training")
    parser.add_argument('--model_type', type=str, default='GCN', help="Language model name")
    parser.add_argument('--num_layers', type=int, default=2, help="Number of layers in GCN")
    parser.add_argument('--dim_hidden', type=int, default=64, help="Hidden dimension in GCN")
    parser.add_argument('--dropout', type=float, default=0.8, help="Dropout probability")
    parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.005, help="Weight decay for Adam optimizer")
    parser.add_argument('--epochs', type=int, default=500, help="Number of epochs")
    parser.add_argument('--k_shot', type=int, default=50, help="Number of epochs")
    parser.add_argument('--edge_ratio', type=float, default=0.4, help="Number of epochs")
    args = parser.parse_args()
    results = main(args)
    save_args_to_file(args, results)
    
