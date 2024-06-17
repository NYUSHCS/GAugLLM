import argparse
import os
import torch

from torch.utils.data import DataLoader
import numpy as np
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.datasets import Amazon, Coauthor, Planetoid, Reddit
import time
from torch_geometric.data import Data
from model import LPDecoder_ogb as LPDecoder
from model import GCN_mgaev3 as GCN
from model import SAGE_mgaev2 as SAGE
from model import GIN_mgaev2 as GIN
from torch_geometric.utils import to_undirected, add_self_loops
from torch_sparse import SparseTensor
from sklearn.metrics import roc_auc_score
from utils import edgemask_um, edgemask_dm, do_edge_split_nc
from sklearn.model_selection import KFold
from sklearn import svm
from sklearn.metrics import f1_score
import os.path as osp
import concurrent.futures
from torch import Tensor
from typing import List, Tuple, Dict
import json
from torch_geometric.utils import degree
import sys
sys.path.append("..") # TODO merge TAPE into current repo
from BGRL.bgrl.transforms_new import get_graph_drop_transform_new,remove_del_candidate_from_edges
from data_utils.dataset import check_candidate_lists,modified_edge_index_tensor
from data_utils.load import load_llm_feature_and_data
from torch_geometric.utils.sparse import to_edge_index
import data_utils.logistic_regression_eval as eval
from data_utils.logistic_regression_eval import Multi_K_Shot_Evaluator  

def candidates_tensor(
    edges_to_add: Tensor = None,
    edges_to_del: Tensor = None,
    degrees_list: List[int] = [],
) -> torch.Tensor:
    
    add_p = int(1)
    del_p = int(1)
    if not isinstance(degrees_list, torch.Tensor):
        degrees_list = torch.tensor(degrees_list, dtype=torch.float32)

    number_of_adds = torch.clamp(torch.round(add_p * degrees_list).to(torch.int64), max=10)
    number_of_dels = torch.clamp(torch.round(del_p * degrees_list).to(torch.int64), max=10)

    edges_to_add_selected = add_and_sort_by_random_dimension(edges_to_add)
    edges_to_del_selected = add_and_sort_by_random_dimension(edges_to_del)

    edges_to_add_selected = apply_top_k_mask(edges_to_add, number_of_adds)
    edges_to_del_selected = apply_top_k_mask(edges_to_del, number_of_dels)

    add_edges_tensor = add_edges(edges_to_add_selected)
    del_edges_tensor = del_edges(edges_to_del_selected)
    
    return add_edges_tensor,del_edges_tensor



import torch
import json

def random_edge_mask(args, edge_index, device, num_nodes):
    num_edge = len(edge_index)
    index = np.arange(num_edge)
    np.random.shuffle(index)
    mask_num = int(num_edge * args.mask_ratio)
    pre_index = torch.from_numpy(index[0:-mask_num])
    mask_index = torch.from_numpy(index[-mask_num:])
    edge_index_train = edge_index[pre_index].t()
    edge_index_mask = edge_index[mask_index].to(device)

    edge_index_train, _ = add_self_loops(edge_index_train, num_nodes=num_nodes)
    adj = SparseTensor.from_edge_index(edge_index_train).t()
    return adj, edge_index_train, edge_index_mask

def add_and_sort_by_random_dimension(edges_tensor, max_len=10):
    #print(edges_tensor)
    num_nodes, num_features, max_len = edges_tensor.shape

    # 生成随机索引排列
    random_indices = torch.argsort(torch.rand(num_nodes, max_len), dim=1)

    # 为每个特征维度重复随机索引
    random_indices_expanded = random_indices.unsqueeze(1).expand(-1, num_features, -1)

    # 使用高级索引重排最后一个维度
    shuffled = edges_tensor.gather(2, random_indices_expanded)

    # 对每个节点，将 -1 值移动到末尾
    mask = shuffled != -1
    sorted_indices = mask.argsort(dim=2, descending=True)
    shuffled_sorted = shuffled.gather(2, sorted_indices)

    #print(shuffled_sorted)

    return shuffled_sorted

def apply_top_k_mask(tensor, number_of_mask):
    batch_size, _, max_len = tensor.shape

    # Convert number_of_mask to a tensor and clamp the values to max_len
    k_values = torch.tensor(number_of_mask, device=tensor.device).clamp(max=max_len)

    # Create range tensor to compare with k_values
    range_tensor = torch.arange(max_len, device=tensor.device).expand(batch_size, max_len)

    # Create a mask where elements are within the top k for each batch
    mask = range_tensor < k_values[:, None]

    # Create a tensor filled with -1
    masked_tensor = torch.full_like(tensor, -1)

    # Copy only top-k elements to the masked tensor
    # Ensure that the mask used for indexing is boolean and correctly shaped
    bool_mask = mask.unsqueeze(1).to(torch.bool)
    bool_mask = bool_mask.expand_as(tensor)
    masked_tensor[bool_mask] = tensor[bool_mask]

    return masked_tensor

def add_edges(edges_to_add):
    # 创建 mask 标记所有第二行为 1 的元素
    #print(edges_to_add)
    mask = edges_to_add[:, 1, :] == 1
    
    # 使用 mask 来选择有效的源节点和目标节点索引
    source_indices = torch.arange(edges_to_add.size(0)).unsqueeze(1).expand_as(mask)
    target_indices = edges_to_add[:, 0, :]

    # 应用 mask 并重塑结果
    source_indices = source_indices[mask].view(1, -1)
    target_indices = target_indices[mask].view(1, -1)

    # 将源节点和目标节点索引堆叠起来
    result_tensor = torch.cat([source_indices, target_indices], dim=0)
    #print(result_tensor)
    return result_tensor


def del_edges(edges_to_del):
    # 需要减去所有标记为0的元素 = 减去所有元素（在一开始） + 加上所有标记为1的元素
    # 创建 mask 标记所有第二行为 1 的元素
    #print(edges_to_del)
    mask = edges_to_del[:, 1, :] == 1
    # 使用 mask 来选择有效的源节点和目标节点索引
    source_indices = torch.arange(edges_to_del.size(0)).unsqueeze(1).expand_as(mask)
    target_indices = edges_to_del[:, 0, :]

    # 应用 mask 并重塑结果
    source_indices = source_indices[mask].view(1, -1)
    target_indices = target_indices[mask].view(1, -1)

    # 将源节点和目标节点索引堆叠起来
    result_tensor = torch.cat([source_indices, target_indices], dim=0)
    #print(result_tensor)
    return result_tensor

def update_edge_index_new(edge_index, add_edges_tensor, del_edges_tensor):
    device = edge_index.device

    # 先在 PyTorch 中连接 edge_index 和 add_edges_tensor
    updated_edge_index = torch.cat([edge_index, add_edges_tensor.to(device)], dim=1)
    updated_edge_index = torch.cat([edge_index, del_edges_tensor.to(device)], dim=1)

    return updated_edge_index


def edit_edges_batch(
    add_p: float = 0.5,
    del_p: float = 0.5,
    edges_to_add: Tensor = None,
    edges_to_del: Tensor = None,
    degrees_list: List[int] = [],
) -> torch.Tensor:


    if not isinstance(degrees_list, torch.Tensor):
        degrees_list = torch.tensor(degrees_list, dtype=torch.float32)

    number_of_adds = torch.clamp(torch.round(add_p * degrees_list).to(torch.int64), max=10)
    number_of_dels = torch.clamp(torch.round(del_p * degrees_list).to(torch.int64), max=10)

    edges_to_add_selected = add_and_sort_by_random_dimension(edges_to_add)
    edges_to_del_selected = add_and_sort_by_random_dimension(edges_to_del)

    edges_to_add_selected = apply_top_k_mask(edges_to_add, number_of_adds)
    edges_to_del_selected = apply_top_k_mask(edges_to_del, number_of_dels)

    add_edges_tensor = add_edges(edges_to_add_selected)
    del_edges_tensor = del_edges(edges_to_del_selected)

    return add_edges_tensor,del_edges_tensor

def train(model, predictor, data, edge_index, optimizer, edges_to_add, edges_to_del, add_edge_p, del_edge_p,degrees_list,add_candidate_all_set,del_candidate_all_set,args):
    model.train()
    predictor.train()

    total_loss = total_examples = 0
    add_edges_tensor,del_edges_tensor = edit_edges_batch(del_p = del_edge_p, add_p = add_edge_p, edges_to_add = edges_to_add, edges_to_del = edges_to_del, degrees_list = degrees_list)


    add_edges_set = set(tuple(edge) for edge in add_edges_tensor.t().tolist())
    del_edges_set = set(tuple(edge) for edge in del_edges_tensor.t().tolist())

    masked_add_edges_set = add_candidate_all_set - add_edges_set
    masked_del_edges_set = del_candidate_all_set - del_edges_set

    #pos_train_edge_set = masked_add_edges_set + masked_del_edges_set
    pos_train_edge_set = masked_add_edges_set.union(masked_del_edges_set)
    pos_train_edge = torch.tensor(list(pos_train_edge_set))


    edge_index = update_edge_index_new(edge_index.t(), add_edges_tensor,del_edges_tensor).t()
    edge_index, _ = add_self_loops(edge_index.t(), num_nodes=data.x.shape[0])
    adj = SparseTensor.from_edge_index(edge_index).t()

    #original code
    #adj, _, pos_train_edge = edgemask_dm(args.mask_ratio, edge_index, data.x.device, data.x.shape[0])
    adj = adj.to(data.x.device)


    for perm in DataLoader(range(pos_train_edge.size(0)), args.batch_size,
                           shuffle=True):
        optimizer.zero_grad()

        h = model(data.x, adj)

        edge = pos_train_edge[perm].t()

        #print("edge in DataLoader:",edge)
        #print("edge in DataLoader Shape:",edge.shape)

        pos_out = predictor(h, edge)
        pos_loss = -torch.log(pos_out + 1e-15).mean()

        # Just do some trivial random sampling.
        edge = torch.randint(0, data.x.shape[0], edge.size(), dtype=torch.long,
                             device=data.x.device)
        neg_out = predictor(h, edge)
        neg_loss = -torch.log(1 - neg_out + 1e-15).mean()

        loss = pos_loss + neg_loss
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(predictor.parameters(), 1.0)

        optimizer.step()

        num_examples = pos_out.size(0)
        total_loss += loss.item() * num_examples
        total_examples += num_examples

    return total_loss / total_examples


@torch.no_grad()
def test(model, predictor, data, pos_test_edge, neg_test_edge, batch_size):
    model.eval()
    predictor.eval()

    h = model(data.x, data.full_adj_t)

    pos_test_edge = pos_test_edge.to(data.x.device)
    neg_test_edge = neg_test_edge.to(data.x.device)

    pos_test_preds = []
    for perm in DataLoader(range(pos_test_edge.size(0)), batch_size):
        edge = pos_test_edge[perm].t()
        pos_test_preds += [predictor(h, edge).squeeze().cpu()]
    pos_test_pred = torch.cat(pos_test_preds, dim=0)

    neg_test_preds = []
    for perm in DataLoader(range(neg_test_edge.size(0)), batch_size):
        edge = neg_test_edge[perm].t()
        neg_test_preds += [predictor(h, edge).squeeze().cpu()]
    neg_test_pred = torch.cat(neg_test_preds, dim=0)

    test_pred = torch.cat([pos_test_pred, neg_test_pred], dim=0)
    test_true = torch.cat([torch.ones_like(pos_test_pred), torch.zeros_like(neg_test_pred)], dim=0)
    test_auc = roc_auc_score(test_true, test_pred)
    return test_auc


def extract_feature_list_layer2(feature_list):
    xx_list = []
    xx_list.append(feature_list[-1])
    tmp_feat = torch.cat(feature_list, dim=-1)
    xx_list.append(tmp_feat)
    return xx_list


def accuracy(preds, labels):
    correct = (preds == labels).astype(float)
    correct = correct.sum()
    return correct / len(labels)


def classify_fold(feature, labels, train_index, test_index):
    train_X, train_y = feature[train_index], labels[train_index]
    test_X, test_y = feature[test_index], labels[test_index]
    clf = svm.SVC(kernel='rbf', decision_function_shape='ovo')
    clf.fit(train_X, train_y)
    preds = clf.predict(test_X)

    micro = f1_score(test_y, preds, average='micro')
    macro = f1_score(test_y, preds, average='macro')
    acc = accuracy(preds, test_y)
    return micro, macro, acc

def test_classify(feature, labels, args):
    f1_mac = []
    f1_mic = []
    accs = []
    kf = KFold(n_splits=5, random_state=42, shuffle=True)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = []
        for train_index, test_index in kf.split(feature):
            future = executor.submit(classify_fold, feature, labels, train_index, test_index)
            futures.append(future)

        for future in concurrent.futures.as_completed(futures):
            micro, macro, acc = future.result()
            accs.append(acc)
            f1_mac.append(macro)
            f1_mic.append(micro)
        
    f1_mic = np.array(f1_mic)
    f1_mac = np.array(f1_mac)
    accs = np.array(accs)
    f1_mic = np.mean(f1_mic)
    f1_mac = np.mean(f1_mac)
    accs = np.mean(accs)
    print('Testing based on svm: ',
          'f1_micro=%.4f' % f1_mic,
          'f1_macro=%.4f' % f1_mac,
          'acc=%.4f' % accs)
    return f1_mic, f1_mac, accs


#add和del都得乘以node_degree,
hyperparams_space = {
    'lr': [5e-4],
    'add_edge_p': [0.2],
    'del_edge_p': [0.2],  
    'emb_epoch': [1]
}

results = {
    "best_acc_search": 0.0,
    "variance": 0.0,
    "best_hyperparams": {}
}


def main():

    #Grid Search
    global results
    var_with_best_acc = 0.0
    best_acc_search = 0.0
    best_hyperparams = {}

    parser = argparse.ArgumentParser(description='S2-GAE (GNN)')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--log_steps', type=int, default=1)
    parser.add_argument('--use_sage', type=str, default='GCN')
    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--use_valedges_as_input', type=bool, default=False)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--decode_layers', type=int, default=3)
    parser.add_argument('--hidden_channels', type=int, default=128)
    parser.add_argument('--decode_channels', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--data_seeds', type=list, default=[0,1,2,3,4])
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--mask_type', type=str, default='dm',
                        help='dm | um')  # whether to use mask features
    parser.add_argument('--patience', type=int, default=50,
                        help='Use attribute or not')
    parser.add_argument('--mask_ratio', type=float, default=0.8)
    parser.add_argument("--feature_type", type=str, required=True)
    parser.add_argument("--eval_multi_k", action="store_true")
    #parser.add_argument("--logdir", type=str, default='runs/')
    args = parser.parse_args()
    print(args)

    if args.eval_multi_k:
        multi_k_eval=Multi_K_Shot_Evaluator()

    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)

    path = osp.join('dataset/class')

    # edge_index = data.edge_index

    for lr in hyperparams_space['lr']:
        for add_edge_p in hyperparams_space['add_edge_p']:
            for del_edge_p in hyperparams_space['del_edge_p']:
                for emb_epoch in hyperparams_space['emb_epoch']:
                        hyperparams = {
                            'lr': lr,
                            'add_edge_p': add_edge_p,
                            'del_edge_p': del_edge_p,
                            'emb_epoch':emb_epoch,
                            }
                        args.lr = lr
                        add_edge_p = add_edge_p
                        del_edge_p = del_edge_p
                        emb_epoch = emb_epoch

                        if args.dataset in {'ogbn-arxiv', 'products', 'mag'}:
                            print('loading ogb dataset...')
                            data = load_llm_feature_and_data(
                                dataset_name = args.dataset, 
                                lm_model_name='microsoft/deberta-base',
                                feature_type=f'attention-{emb_epoch}',
                                device=args.device)

                        elif args.dataset in {'Cora', 'Citeseer', 'pubmed'} or 'amazon' in args.dataset:
                            data = load_llm_feature_and_data(
                                dataset_name = args.dataset.lower(), 
                                lm_model_name='microsoft/deberta-base',
                                feature_type=f'attention-{emb_epoch}',
                                device=args.device)

                        else:
                            raise ValueError(args.dataset)


                        degrees_list = degree(data.edge_index[0], num_nodes=data.num_nodes).tolist()
                        candidate_lists = check_candidate_lists(args.dataset)
                        edges_to_del,edges_to_add = modified_edge_index_tensor(args.dataset, candidate_lists)
                        add_candidate_all_tensor,del_candidate_all_tensor = candidates_tensor(edges_to_add,edges_to_del,degrees_list)

                        add_candidate_all_set = set(tuple(edge) for edge in add_candidate_all_tensor.t().tolist())
                        del_candidate_all_set = set(tuple(edge) for edge in del_candidate_all_tensor.t().tolist())

                        #首先基础的edge_index是删除了所有的add和del的candidates的纯净版 后面每个epoch中edge_index再加上选出来的边 没选上的作为mask掉的
                        updated_edge_index = remove_del_candidate_from_edges(data.edge_index,edges_to_del)
                        
                        data.edge_index = updated_edge_index


                        if isinstance(data.edge_index,SparseTensor):
                            data.edge_index,_ = to_edge_index(data.edge_index)
                        
                        if data.is_undirected():
                            edge_index = data.edge_index
                        else:
                            print('### Input graph {} is directed'.format(args.dataset))
                            edge_index = to_undirected(data.edge_index)
                        data.full_adj_t = SparseTensor.from_edge_index(edge_index).t()

                        edge_index, test_edge, test_edge_neg = do_edge_split_nc(edge_index, data.x.shape[0])
                        
                        labels = data.y.view(-1)

                        save_path_model = 'weight/s2gaesvm-' + args.use_sage +'_'+args.dataset + args.feature_type + '_{}_{}'.format(args.dataset, args.mask_type) + '_{}'.format(
                            args.num_layers) + '_hidd{}-{}-{}-{}'.format(args.hidden_channels, args.mask_ratio, args.decode_layers,
                                                                        args.decode_channels) + '_model.pth'
                        save_path_predictor = 'weight/s2gaesvm' + args.use_sage +'_'+ args.dataset + args.feature_type + '_{}_{}'.format(args.dataset,
                                                                                            args.mask_type) + '_{}'.format(
                            args.num_layers) + '_hidd{}-{}-{}-{}'.format(args.hidden_channels, args.mask_ratio, args.decode_layers,
                                                                        args.decode_channels) + '_pred.pth'

                        out2_dict = {0: 'last', 1: 'combine'}
                        result_dict = out2_dict
                        svm_result_final = np.zeros(shape=[args.runs, len(out2_dict)])
                        # Use training + validation edges for inference on test set.

                        data = data.to(device)

                        if args.use_sage == 'SAGE':
                            model = SAGE(data.num_features, args.hidden_channels,
                                        args.hidden_channels, args.num_layers,
                                        args.dropout).to(device)
                        elif args.use_sage == 'GIN':
                            model = GIN(data.num_features, args.hidden_channels,
                                        args.hidden_channels, args.num_layers,
                                        args.dropout).to(device)
                        else:
                            model = GCN(data.num_features, args.hidden_channels,
                                        args.hidden_channels, args.num_layers,
                                        args.dropout).to(device)

                        predictor = LPDecoder(args.hidden_channels, args.decode_channels, 1, args.num_layers,
                                                args.decode_layers, args.dropout).to(device)

                        print('Start training with mask ratio={} # optimization edges={} / {}'.format(args.mask_ratio,
                                                                                                int(args.mask_ratio *
                                                                                                    edge_index.shape[0]), edge_index.shape[0]))
                        final_acc_list = []
                        early_stp_acc_list=[]
                        for run in range(args.runs):
                            model.reset_parameters()
                            predictor.reset_parameters()
                            optimizer = torch.optim.Adam(
                                list(model.parameters()) + list(predictor.parameters()),
                                lr=args.lr)

                            best_valid = 0.0
                            best_epoch = 0
                            cnt_wait = 0
                            for epoch in range(1, 1 + args.epochs):
                                t1 = time.time()
                                loss = train(model, predictor, data, edge_index, optimizer,edges_to_add,edges_to_del,del_edge_p,add_edge_p,degrees_list,add_candidate_all_set,del_candidate_all_set,args)
                                t2 = time.time()
                                auc_test = test(model, predictor, data, test_edge, test_edge_neg,
                                            args.batch_size)

                                if auc_test > best_valid:
                                    best_valid = auc_test
                                    best_epoch = epoch
                                    torch.save(model.state_dict(), save_path_model)
                                    torch.save(predictor.state_dict(), save_path_predictor)
                                    cnt_wait = 0
                                else:
                                    cnt_wait += 1

                                print(f'Run: {run + 1:02d}, '
                                    f'Epoch: {epoch:02d}, '
                                    f'Best_epoch: {best_epoch:02d}, '
                                    f'Best_valid: {100 * best_valid:.2f}%, '
                                    f'Loss: {loss:.4f}, ')
                                print('***************')
                                if cnt_wait == 50:
                                    print('Early stop at {}'.format(epoch))
                                # break

                            print('##### Testing on {}/{}'.format(run, args.runs))

                            model.load_state_dict(torch.load(save_path_model))
                            predictor.load_state_dict(torch.load(save_path_predictor))
                            feature = model(data.x, data.full_adj_t)
                            feature = [feature_.detach() for feature_ in feature]

                            feature_list = extract_feature_list_layer2(feature)

                            
                            for i, feature_tmp in enumerate(feature_list):
                                
                                if args.eval_multi_k:
                                    multi_k_eval.multi_k_fit_logistic_regression_new(features=feature_tmp,labels=labels,dataset_name=args.dataset,data_random_seeds=args.data_seeds,device=device)
                                
                                                        
                        if args.eval_multi_k:
                            #multi_k_eval.save_csv_results(dataset_name=args.dataset,experience_name="S2GAE",feature_type=args.feature_type)
                            acc_1 = multi_k_eval.save_csv_results(dataset_name=args.dataset, experience_name="S2GAE",feature_type=args.feature_type + 'attention')
                            second_part_1 = acc_1.split('/')[1]
                            number_str_1 = second_part_1.split('±')[0]
                            acc = float(number_str_1)
                            var = float(second_part_1.split('±')[1])
                            if acc > best_acc_search:
                                best_acc_search = acc
                                var_with_best_acc = var
                                best_hyperparams = hyperparams

                                print("best_hyperparams updated!")
                                results["best_acc_search"] = best_acc_search
                                results["best_hyperparams"] = best_hyperparams
                                results["variance"] = var_with_best_acc
                                # 输出结果到文件
                                output_file_name = "S2GAE_" + args.dataset + "_new_attention_struct"+".txt"
                                with open(output_file_name, "w") as file:
                                    file.write("Best Accuracy: {}\n".format(results["best_acc_search"]))
                                    file.write("Variance with Best Acc: {}\n".format(results["variance"]))
                                    file.write("Best Hyperparameters:\n")
                                    for param, value in results["best_hyperparams"].items():
                                        file.write("{}: {}\n".format(param, value))

                                print(f"Results saved to {output_file_name}")



                        if osp.exists(save_path_model):
                            os.remove(save_path_model)
                            os.remove(save_path_predictor)
                            print('Successfully delete the saved models')

    
if __name__ == "__main__":
    main()
