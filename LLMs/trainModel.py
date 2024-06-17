from GiantTrainer import GiantTrainer
from GiantTrainer_new import GiantTrainer_new
import argparse
import torch
import pandas as pd
import os
import sys
from torch_geometric.utils.sparse import to_edge_index
from data_utils.dataset import check_candidate_lists,modify_edge_index_one_time
from data_utils.load import load_llm_feature_and_data
from utils import get_neighbors_list,reorder_text_by_neighbors
os.environ['CURL_CA_BUNDLE'] = ''

def insert_add_candidate_from_edges(edge_index, add_edges_tensor):
    device = edge_index.device

    add_set = set()
    for node_index, sub_tensor in enumerate(add_edges_tensor):
        for index in sub_tensor[0]:
            if index != -1:
                add_set.add((node_index, index.item()))

    original_edges = set()
    for col in range(edge_index.shape[1]):
        original_edges.add((edge_index[0, col].item(), edge_index[1, col].item()))
    
    updated_edge_set = original_edges | add_set  # 使用并集操作添加新边
    # 分离源节点和目标节点
    source_nodes = [pair[0] for pair in updated_edge_set]
    target_nodes = [pair[1] for pair in updated_edge_set]

    # 转换为张量
    updated_edge_index_tensor = torch.tensor([source_nodes, target_nodes]).to(device)

    return updated_edge_index_tensor


def remove_del_candidate_from_edges(edge_index, del_edges_tensor):
    device = edge_index.device

    del_set = set()
    for node_index, sub_tensor in enumerate(del_edges_tensor):
        for index in sub_tensor[0]:
            if index != -1:
                del_set.add((node_index, index.item()))

    original_edges = set()
    for col in range(edge_index.shape[1]):
        original_edges.add((edge_index[0, col].item(), edge_index[1, col].item()))
    
    updated_edge_set = original_edges - del_set
    # 分离源节点和目标节点
    source_nodes = [pair[0] for pair in updated_edge_set]
    target_nodes = [pair[1] for pair in updated_edge_set]

    # 转换为张量
    updated_edge_index_tensor = torch.tensor([source_nodes, target_nodes]).to(device)

    return updated_edge_index_tensor

def update_edge_index_new(edge_index, add_edges_tensor, del_edges_tensor):
    device = edge_index.device

    # 先在 PyTorch 中连接 edge_index 和 add_edges_tensor
    updated_edge_index = torch.cat([edge_index, add_edges_tensor.to(device)], dim=1)
    updated_edge_index = torch.cat([edge_index, del_edges_tensor.to(device)], dim=1)

    return updated_edge_index

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    torch.cuda.empty_cache()
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--LLM_type', type=str, default='cora')
    
    args = parser.parse_args()
    
    dataset_name = args.dataset
    sys.path.append("..")
    if dataset_name == 'pubmed':
        from pubmedTrainer import GiantTrainer_new
    elif dataset_name == 'amazon-photo':
        from photoTrainer import GiantTrainer_new
    elif dataset_name == 'amazon-history':
        from historyTrainer import GiantTrainer_new
    elif dataset_name == 'ogbn-arxiv':
        from arxivTrainer import GiantTrainer_new
    elif dataset_name == 'amazon-computers':
        from computersTrainer import GiantTrainer_new
    torch.cuda.empty_cache()
    dataset = load_llm_feature_and_data(dataset_name,feature_type="BOW" if 'amazon' in args.dataset else 'ogb' ,use_text=True,device=device,use_explanation=False,use_generate_text=False
                                        ,use_summery_text=False)

    #用新的augmented structure进行GIANT
    if dataset_name == 'ogbn-arxiv':
        dataset.edge_index,_ = to_edge_index(dataset.edge_index)
    
    candidate_lists = check_candidate_lists(dataset_name)
    dataset.edge_index = modify_edge_index_one_time(dataset_name,dataset.edge_index,candidate_lists)
    
    
    data = dataset
    text = data.text
    labels = data.y.tolist()

    seed = 0


    if args.LLM_type == 'GIA_NEW':
        trainer = GiantTrainer_new(data,text,dataset_name, device)
    elif args.LLM_type == 'GIA':
        trainer = GiantTrainer(data,text,dataset_name, device)

    trainer.train()
    emb = trainer.eval_and_save()
    print(emb.shape)


