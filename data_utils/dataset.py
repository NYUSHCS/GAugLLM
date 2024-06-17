import json
import dgl
import torch
from torch.utils.data import Dataset as TorchDataset
import os
import re
import random
import math
import ipdb
from torch_geometric.utils import degree
# convert PyG dataset to DGL dataset
class CustomDGLDataset(TorchDataset):
    def __init__(self, name, pyg_data):
        self.name = name
        self.pyg_data = pyg_data

    def __len__(self):
        return 1

    def __getitem__(self, idx):

        data = self.pyg_data
        g = dgl.DGLGraph()
        if self.name == 'ogbn-arxiv':
            edge_index = data.edge_index.to_torch_sparse_coo_tensor().coalesce().indices()
        else:
            edge_index = data.edge_index
        g.add_nodes(data.num_nodes)
        g.add_edges(edge_index[0], edge_index[1])

        if data.edge_attr is not None:
            g.edata['feat'] = torch.FloatTensor(data.edge_attr)
        if self.name == 'ogbn-arxiv':
            g = dgl.to_bidirected(g)
            print(
                f"Using GAT based methods,total edges before adding self-loop {g.number_of_edges()}")
            g = g.remove_self_loop().add_self_loop()
            print(f"Total edges after adding self-loop {g.number_of_edges()}")
        g.ndata['feat'] = torch.FloatTensor(data.x)
        g.ndata['label'] = torch.LongTensor(data.y).squeeze()
        
        g.ndata["train_mask"] = data.train_mask
        g.ndata["val_mask"] = data.val_mask
        g.ndata["test_mask"] = data.test_mask
        
        return g

    @property
    def train_mask(self):
        return self.pyg_data.train_mask

    @property
    def val_mask(self):
        return self.pyg_data.val_mask

    @property
    def test_mask(self):
        return self.pyg_data.test_mask


# Create torch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['node_id'] = idx
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return len(self.encodings["input_ids"])



def check_candidate_lists(dataset_name):

    candidate_lists = []

    generated_texts_dir = f"../prompts/{dataset_name}"

    all_files = [f for f in os.listdir(generated_texts_dir) if f.startswith("generated_") and f.endswith(".txt")]
    sorted_files = sorted(all_files, key=lambda x: int(x.split("generated_")[1].split(".txt")[0]))

    for filename in sorted_files:
        with open(os.path.join(generated_texts_dir, filename), 'r') as f:
            content = f.read().strip()
            numbers = re.findall(r"\d+", content)
            candidate_list = list(map(int, numbers))

            candidate_lists.append(candidate_list)

    return candidate_lists

def modified_edge_index(dataset_name,candidate_lists):
    with open(f"../prompts/{dataset_name}_new.json", 'r') as f:
        node_results = json.load(f)

    correct_predictions_for_neighbors = 0
    correct_predictions_for_non_neighbors = 0
    total_neighbors = 0
    total_non_neighbors = 0
    edges_to_add = {}
    edges_to_del = {}
    # 遍历 candidate_lists 和 candidate_neighbors_lists 来更新边
    #candidates
    for i, candidates in enumerate(candidate_lists):
        node_data = node_results[str(i)]
        neighbor_index_list = node_data['most_dissimilar_neighbors']
        non_neighbor_index_list = node_data['most_similar_non_neighbors']
        # 根据 neighbor_list 和 non_neighbor_list 的长度选择边
        if len(neighbor_index_list) >= 5:
            neighbor_candidates_list = candidates[:len(neighbor_index_list)]
            none_neighbor_candidates_list = candidates[len(neighbor_index_list):]
        else:
            neighbor_candidates_list = []
            none_neighbor_candidates_list = candidates



        total_neighbors += len(neighbor_candidates_list)
        total_non_neighbors += len(none_neighbor_candidates_list) 
        
        for j, is_connected in enumerate(neighbor_candidates_list):
            candidate = neighbor_index_list[j]
            if i not in edges_to_del:
                edges_to_del[i] = []
            if is_connected:
                edges_to_del[i].append((i, candidate, True))
                correct_predictions_for_neighbors += 1
            else:
                edges_to_del[i].append((i, candidate, False))

        for j, is_connected in enumerate(none_neighbor_candidates_list):
            candidate = non_neighbor_index_list[j]
            if i not in edges_to_add:
                edges_to_add[i] = []
            if is_connected:
                edges_to_add[i].append((i, candidate, True))
            else:
                edges_to_add[i].append((i, candidate, False))
                correct_predictions_for_non_neighbors += 1
   
    return edges_to_del, edges_to_add


import torch
import json

def modified_edge_index_tensor(dataset_name, candidate_lists):
    with open(f"../prompts/{dataset_name}_new.json", 'r') as f:
        node_results = json.load(f)

    num_nodes = len(node_results)
    max_neighbor_length = max(len(node_data['most_dissimilar_neighbors']) for node_data in node_results.values())
    max_non_neighbor_length = max(len(node_data['most_similar_non_neighbors']) for node_data in node_results.values())

    edges_to_del_tensor = torch.full((num_nodes, 2, max_neighbor_length), -1, dtype=torch.int32)
    edges_to_add_tensor = torch.full((num_nodes, 2, max_non_neighbor_length), -1, dtype=torch.int32)


    for i, candidates in enumerate(candidate_lists):
        node_data = node_results[str(i)]
        neighbor_index_list = node_data['most_dissimilar_neighbors']
        non_neighbor_index_list = node_data['most_similar_non_neighbors']
        
        # 根据 neighbor_list 和 non_neighbor_list 的长度选择边
        if len(neighbor_index_list) >= 5:
            for j, candidate in enumerate(neighbor_index_list):
                edges_to_del_tensor[i, 0, j] = candidate
                edges_to_del_tensor[i, 1, j] = int(j < len(candidates) and candidates[j])

            for j, candidate in enumerate(non_neighbor_index_list):
                idx = j + len(neighbor_index_list)
                edges_to_add_tensor[i, 0, j] = candidate
                edges_to_add_tensor[i, 1, j] = int(idx < len(candidates) and candidates[idx])
        else:
            for j, candidate in enumerate(neighbor_index_list):
                edges_to_del_tensor[i, 0, j] = -1
                edges_to_del_tensor[i, 1, j] = -1

            for j, candidate in enumerate(non_neighbor_index_list):
                idx = j + len(neighbor_index_list)
                edges_to_add_tensor[i, 0, j] = candidate
                edges_to_add_tensor[i, 1, j] = int(idx < len(candidates) and candidates[idx])


    return edges_to_del_tensor, edges_to_add_tensor


def modify_edge_index_one_time_ratio(dataset_name, edge_index, candidate_lists, ratio):
    # 将原始边转换为集合，便于添加和移除操作
    original_edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    with open(f"../prompts/{dataset_name}_new.json", 'r') as f:
        node_results = json.load(f)

    # 用于保存需要添加或删除的边
    edges_to_add = set()
    edges_to_del = set()

    print(ratio*10)
    # 遍历 candidate_lists 和 candidate_neighbors_lists 来更新边
    for i, candidates in enumerate(candidate_lists):
        node_data = node_results[str(i)]
        neighbor_index_list = node_data['most_dissimilar_neighbors']
        non_neighbor_index_list = node_data['most_similar_non_neighbors']
        
        if len(neighbor_index_list) >= 5:
            neighbor_candidates_list = candidates[:len(neighbor_index_list)]
            none_neighbor_candidates_list = candidates[len(neighbor_index_list)+1:]
        else:
            neighbor_candidates_list = []
            none_neighbor_candidates_list = candidates
        
        for j, is_connected in enumerate(neighbor_candidates_list):
            candidate = neighbor_index_list[j] 
            if not is_connected:
                edges_to_del.add((i, candidate))

        for j, is_connected in enumerate(none_neighbor_candidates_list):
            candidate = non_neighbor_index_list[j] 
            if is_connected:
                edges_to_add.add((i, candidate))

    # 按比例选择需要添加和删除的边
    edges_to_add = set(random.sample(edges_to_add, int(len(edges_to_add) * ratio)))
    edges_to_del = set(random.sample(edges_to_del, int(len(edges_to_del) * ratio)))

    # 添加和删除边
    for edge in edges_to_add:
        original_edges.add(edge)
    
    for edge in edges_to_del:
        original_edges.discard(edge)

    # 将更新后的边集合转换回tensor格式
    new_src, new_dest = zip(*original_edges)
    new_src = torch.tensor(new_src, dtype=torch.long)
    new_dest = torch.tensor(new_dest, dtype=torch.long)

    # 组合 new_src 和 new_dest 形成新的 edge_index
    new_edge_index = torch.stack([new_src, new_dest], dim=0)

    return new_edge_index


def modify_edge_index_one_time(dataset_name,edge_index, candidate_lists):
    # 将原始边转换为集合，便于添加和移除操作
    original_edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))

    with open(f"../prompts/{dataset_name}_new.json", 'r') as f:
        node_results = json.load(f)

    correct_predictions_for_neighbors = 0
    correct_predictions_for_non_neighbors = 0
    total_neighbors = 0
    total_non_neighbors = 0
    # 遍历 candidate_lists 和 candidate_neighbors_lists 来更新边
    #candidates
    for i, candidates in enumerate(candidate_lists):
        edges_to_modify = []
        
        node_data = node_results[str(i)]
        neighbor_index_list = node_data['most_dissimilar_neighbors']
        non_neighbor_index_list = node_data['most_similar_non_neighbors']
        
        if len(neighbor_index_list) >= 5:
            neighbor_candidates_list = candidates[:len(neighbor_index_list)]
            none_neighbor_candidates_list = candidates[len(neighbor_index_list)+1:]
        else:
            neighbor_candidates_list = []
            none_neighbor_candidates_list = candidates

        total_neighbors += len(neighbor_candidates_list)
        total_non_neighbors += len(non_neighbor_index_list) 
        
        for j, is_connected in enumerate(neighbor_candidates_list):
            candidate = neighbor_candidates_list[j] 
            if candidate > 1000000:
                print(i)
            if is_connected:
                edges_to_modify.append((i, candidate, True))
                correct_predictions_for_neighbors += 1
            else:
                edges_to_modify.append((i, candidate, False))
                
                
        for j, is_connected in enumerate(none_neighbor_candidates_list):
            candidate = none_neighbor_candidates_list[j] 
            if candidate > 1000000:
                print(i)
            if is_connected:
                edges_to_modify.append((i, candidate, True))
            else:
                edges_to_modify.append((i, candidate, False))
                correct_predictions_for_non_neighbors += 1



        # 根据edges_to_modify来更新original_edges
        for src, dest, add_edge in edges_to_modify:
            if add_edge:
                original_edges.add((src, dest))
            else:
                original_edges.discard((src, dest))

    # 将更新后的边集合转换回tensor格式
    new_src, new_dest = zip(*original_edges)
    new_src = torch.tensor(new_src, dtype=torch.long)
    new_dest = torch.tensor(new_dest, dtype=torch.long)

    # 组合 new_src 和 new_dest 形成新的 edge_index
    new_edge_index = torch.stack([new_src, new_dest], dim=0)

    return new_edge_index




import numpy as np
from scipy.sparse import coo_matrix

def modify_adj(adj, candidate_lists, candidate_neighbors_lists):
    # Convert adj to coo_matrix format
    adj_coo = adj.tocoo()

    # Extract current edges
    src, dest = adj_coo.row, adj_coo.col

    # Process candidate_lists and candidate_neighbors_lists to generate new edges
    new_src, new_dest = [], []
    for i, candidates in enumerate(candidate_lists):
        for j, is_connected in enumerate(candidates):
            if is_connected:
                neighbor = candidate_neighbors_lists[i][j]
                new_src.append(i)
                new_dest.append(neighbor)

    # Convert lists to arrays
    new_src = np.array(new_src, dtype=np.int64)
    new_dest = np.array(new_dest, dtype=np.int64)

    # Create new adjacency matrix
    data = np.ones(len(new_src))
    new_adj = coo_matrix((data, (new_src, new_dest)), shape=adj.shape)

    return new_adj
