from typing import List, Tuple, Dict
import copy
from torch import Tensor
import torch
from torch_geometric.utils.dropout import dropout_adj
from torch_geometric.transforms import Compose
from torch_geometric.utils import degree
import random
import cProfile
from collections import defaultdict
import time
#from memory_profiler import profile
import numpy as np
class DropFeatures:
    r"""Drops node features with probability p."""
    def __init__(self, p=None, precomputed_weights=True):
        assert 0. < p < 1., 'Dropout probability has to be between 0 and 1, but got %.2f' % p
        self.p = p

    def __call__(self, data):
        drop_mask = torch.empty((data.x.size(1),), dtype=torch.float32, device=data.x.device).uniform_(0, 1) < self.p
        #data.x = data.x.clone()  # 克隆 data.x
        data.x[:, drop_mask] = 0  # 修改克隆后的张量
        return data

    def __repr__(self):
        return '{}(p={})'.format(self.__class__.__name__, self.p)


def tensor_to_set(tensor):
    edge_set = set()
    num_sources, _, num_targets = tensor.shape

    for source_idx in range(num_sources):
        for target_idx in range(num_targets):
            target_node = tensor[source_idx, 0, target_idx].item()
            # 如果遇到 -1，则停止处理当前 source_node 的剩余 target_node
            if target_node == -1:
                break
            value = tensor[source_idx, 1, target_idx].item()
            # 如果值为 1，则添加边
            if value == 1:
                edge_set.add((source_idx, int(target_node)))
    return edge_set




def dict_to_tensor(edges_dict_add, edges_dict_del, max_len=10):
    all_keys = sorted(set(edges_dict_add.keys()) | set(edges_dict_del.keys()))
    num_keys = len(all_keys)
    key_to_index = {key: i for i, key in enumerate(all_keys)}

    edges_tensor_add = torch.zeros((num_keys, 2, max_len), dtype=torch.float32)
    edges_tensor_del = torch.zeros((num_keys, 2, max_len), dtype=torch.float32) - 1  # 初始化为 -1

    # 填充 edges_tensor_add
    for key, tuples in edges_dict_add.items():
        i = key_to_index[key]
        padded_tuples = tuples + [(key, -1, False)] * (max_len - len(tuples))
        indices, bools = zip(*[(t[1], float(t[2])) for t in padded_tuples])
        edges_tensor_add[i, 0, :] = torch.tensor(indices, dtype=torch.float32)
        edges_tensor_add[i, 1, :] = torch.tensor(bools, dtype=torch.float32)

    # 填充 edges_tensor_del
    for key, tuples in edges_dict_del.items():
        i = key_to_index[key]
        padded_tuples = tuples + [(key, -1, False)] * (max_len - len(tuples))
        indices, bools = zip(*[(t[1], float(t[2])) for t in padded_tuples])
        edges_tensor_del[i, 0, :] = torch.tensor(indices, dtype=torch.float32)
        edges_tensor_del[i, 1, :] = torch.tensor(bools, dtype=torch.float32)

    return edges_tensor_add, edges_tensor_del




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




def create_mask_and_apply(edges_to_add, number_of_adds):
    num_nodes, _, max_length = edges_to_add.shape
    mask = torch.zeros_like(edges_to_add, dtype=torch.bool)

    # 将number_of_adds转换为字典，键为索引数量，值为节点索引列表
    adds_dict = defaultdict(list)
    for idx, num in enumerate(number_of_adds):
        adds_dict[num].append(idx)

    # 对每个唯一的索引数量执行批处理操作
    for num_adds, indices in adds_dict.items():
        if num_adds > 0:
            # 直接选择每个节点的前num_adds个位置
            mask[indices, :, :num_adds] = 1

    # 应用掩码
    masked_tensor = edges_to_add * mask
    masked_tensor[masked_tensor == 0] = -1

    # 保持原始为-1的位置不变
    original_minus_one = edges_to_add == 0
    masked_tensor[original_minus_one] = 0

    return masked_tensor



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


def edit_edges_batch(
    edge_index: Tensor,
    add_p: float = 0.5,
    del_p: float = 0.5,
    edges_to_add: Tensor = None,
    edges_to_del: Tensor = None,
    degrees_list: List[int] = [],
) -> torch.Tensor:

    #start_time = time.time()

    if not isinstance(degrees_list, torch.Tensor):
        degrees_list = torch.tensor(degrees_list, dtype=torch.float32)

    number_of_adds = torch.clamp(torch.round(add_p * degrees_list).to(torch.int64), max=10)
    number_of_dels = torch.clamp(torch.round(del_p * degrees_list).to(torch.int64), max=10)

    #end_time = time.time()
    #print(f"Initialization Time: {end_time - start_time} seconds")

    #start_time = time.time()
    edges_to_add_selected = add_and_sort_by_random_dimension(edges_to_add)
    edges_to_del_selected = add_and_sort_by_random_dimension(edges_to_del)
    #end_time = time.time()
    #print(f"Add and Sort by Random Dimension Time: {end_time - start_time} seconds")

    #start_time = time.time()
    edges_to_add_selected = apply_top_k_mask(edges_to_add, number_of_adds)
    edges_to_del_selected = apply_top_k_mask(edges_to_del, number_of_dels)
    #end_time = time.time()
    #print(f"Create Mask and Apply Time: {end_time - start_time} seconds")

    #start_time = time.time()
    add_edges_tensor = add_edges(edges_to_add_selected)
    del_edges_tensor = del_edges(edges_to_del_selected)
    #end_time = time.time()
    #print(f"Add/Del Edges Time: {end_time - start_time} seconds")


    #start_time = time.time()
    new_edge_index = update_edge_index_new(edge_index, add_edges_tensor, del_edges_tensor)
    #end_time = time.time()
    #print(f"Update Edge Index Optimized Time: {end_time - start_time} seconds")

    return new_edge_index



def edit_edges(
    edge_index: Tensor,
    add_p: float = 0.5,
    del_p: float = 0.5,
    edges_to_add: Dict[int, List[Tuple[int, int, bool]]] = {},  # dict of lists of tuples (index, candidate_index, True/False)
    edges_to_del: Dict[int, List[Tuple[int, int, bool]]] = {},
    degrees_list: List[int] = [], 
    number_of_nodes: int = 0,
) -> Tuple[Tensor]:
    '''
    profiler = cProfile.Profile()
    profiler.enable()
    '''
    modified_edges = edge_index
    modified_edge_index = []

    print()
    # 将原始边集合转换为集合形式，用于快速查找
    original_edges = edge_index

    for index in range(number_of_nodes):
        # Randomly Choose Edges Based on Degree
        edges_to_add_for_this_node = edges_to_add.get(index, [])
        edges_to_del_for_this_node = edges_to_del.get(index, [])

        node_degree = degrees_list[index]
        number_of_adds = min(round(add_p * node_degree), len(edges_to_add_for_this_node))
        number_of_dels = min(round(del_p * node_degree), len(edges_to_del_for_this_node))

        #得看怎么优化sample
        edges_to_add_selected_for_this_node = random.sample(edges_to_add_for_this_node, number_of_adds)
        edges_to_del_selected_for_this_node = random.sample(edges_to_del_for_this_node, number_of_dels)
        modified_edge_index.extend(edges_to_add_selected_for_this_node)
        modified_edge_index.extend(edges_to_del_selected_for_this_node)
        '''
        if index == 5:
            print(modified_edge_index)
        '''
    modified_count = 0
    # 根据edges_to_modify来更新existing_edges
    for content in modified_edge_index:
        src = content[0]
        dest = content[1]
        state = content[2]
        edge = (src,dest)
        if state == True:
            if edge not in original_edges:  # 如果是新添加的边
                modified_edges.add(edge)
                modified_count += 1
        else:
            if edge in original_edges:  # 如果是被删除的边
                modified_edges.discard(edge)
                modified_count += 1

    # 将更新后的边集合转换回tensor格式
    new_src, new_dest = zip(*modified_edges)
    new_src = torch.tensor(new_src, dtype=torch.long)
    new_dest = torch.tensor(new_dest, dtype=torch.long)

    # 组合 new_src 和 new_dest 形成新的 edge_index
    new_edge_index = torch.stack([new_src, new_dest], dim=0)
    total_edges = len(original_edges)
    modified_percentage = (modified_count / total_edges) * 100 if total_edges > 0 else 0
    print("/n")
    print("********")
    print(modified_percentage)
    print("********")

    '''
    profiler.disable()
    profiler.print_stats(sort='cumtime')
    '''

    return new_edge_index




class EditEdges:
    r"""Edit edges with probability Ps."""
    def __init__(self, del_p, add_p, del_tensor, add_tensor, force_undirected=False):
        assert 0. <= del_p <= 1., 'Drop Probability has to be between 0 and 1, but got %.2f' % del_p
        assert 0. <= add_p <= 1., 'Add Probability has to be between 0 and 1, but got %.2f' % add_p
        self.del_p = del_p
        self.add_p = add_p
        self.add_tensor = add_tensor
        self.del_tensor = del_tensor
        self.force_undirected = force_undirected

    def __call__(self, data):
        edge_index = data.edge_index
        number_of_nodes = data.num_nodes
        degrees_list = degree(edge_index[0], num_nodes=number_of_nodes).tolist()
        #edge_index = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        #modified_edge_index = edit_edges(edge_index, del_p=self.del_p, add_p=self.add_p, edges_to_add = self.add_list, edges_to_del = self.del_list, degrees_list = degrees_list, number_of_nodes = number_of_nodes)

        #edge_index_without_del_candidate = remove_del_candidate_from_edges(edge_index, self.del_tensor)
        modified_edge_index = edit_edges_batch(edge_index, del_p=self.del_p, add_p=self.add_p, edges_to_add = self.add_tensor, edges_to_del = self.del_tensor, degrees_list = degrees_list)
        data.edge_index = modified_edge_index
        return data

    def __repr__(self):
        return '{}(p={}, force_undirected={})'.format(self.__class__.__name__, self.p, self.force_undirected)





def get_graph_drop_transform_new(del_edge_p, add_edge_p, drop_feat_p, del_tensor, add_tensor):
    transforms = list()

    # make copy of graph
    transforms.append(copy.deepcopy)

    # drop edges
    if del_edge_p > 0. or add_edge_p > 0.:
        transforms.append(EditEdges(del_edge_p,add_edge_p,del_tensor,add_tensor))

    # drop features
    if drop_feat_p > 0.:
        transforms.append(DropFeatures(drop_feat_p))
    return Compose(transforms)