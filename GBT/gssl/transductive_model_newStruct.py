from copy import deepcopy
from typing import Dict, Optional, Tuple, Union
from torch import Tensor
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import Data
from torch_geometric import nn as tgnn
from tqdm.auto import tqdm
from typing import List
from GBT.gssl.loss import get_loss
from torch_geometric.utils import degree

'''
apply GBT original loss

'''

class Model:
    def __init__(
        self,
        feature_dim: int,
        emb_dim: int,
        loss_name: str,
        p_x: float,
        p_e: float,
        lr_base: float,
        total_epochs: int,
        warmup_epochs: int,
        edges_to_add:Tensor,
        edges_to_del:Tensor,
        degrees_list:List,
        add_p: float,
        del_p:float,
    ):
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self._encoder = GCNEncoder(
            in_dim=feature_dim, out_dim=emb_dim
        ).to(self._device)

        self._loss_fn = get_loss(loss_name=loss_name)

        self._optimizer = torch.optim.AdamW(
            params=self._encoder.parameters(),
            lr=lr_base,
            weight_decay=1e-5,
        )
        self._scheduler = LinearWarmupCosineAnnealingLR(
            optimizer=self._optimizer,
            warmup_epochs=warmup_epochs,
            max_epochs=total_epochs,
        )

        self._p_x = p_x
        self._p_e = p_e
        self.add_p = add_p
        self.del_p = del_p
        self._total_epochs = total_epochs

        self._use_pytorch_eval_model = False

        self.edges_to_add = edges_to_add
        self.edges_to_del = edges_to_del


    def fit(
        self,
        data: Data,
        logger: Optional[SummaryWriter] = None,
        log_interval: Optional[int] = None,
        trainer_LM =None,
        co_train = False,
        masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> dict:
        self._encoder.train()
        logs = {
            "log_epoch": [],
            "train_accuracies": [],
            "val_accuracies": [],
            "test_accuracies": [],
            "z": [],
        }

        data = data.to(self._device)
        z = []
        degrees_list = degree(data.edge_index[0], num_nodes=data.num_nodes).tolist()

        for epoch in tqdm(iterable=range(self._total_epochs)):
            self._optimizer.zero_grad()

            (x_a, ei_a), (x_b, ei_b) = augment(
                data=data, p_x=self._p_x, p_e=self._p_e, add_p=self.add_p, del_p=self.del_p, edges_to_add = self.edges_to_add, edges_to_del = self.edges_to_del, degrees_list= degrees_list
            )

            z_a = self._encoder(x=x_a, edge_index=ei_a)
            z_b = self._encoder(x=x_b, edge_index=ei_b)

            loss = self._loss_fn(z_a=z_a, z_b=z_b)

            loss.backward()


            self._optimizer.step()
            self._scheduler.step()
            
            if (epoch+1) % (self._total_epochs //10) ==0:
                z1.append(deepcopy(self.predict(data=data)))
                z2.append(deepcopy(self.predict(data=data2)))

        data = data.to("cpu")

        return logs,z

    def predict(self, data: Data) -> torch.Tensor:
        self._encoder.eval()

        with torch.no_grad():
            z = self._encoder(
                x=data.x.to(self._device),
                edge_index=data.edge_index.to(self._device),
            )

            return z.cpu()


class GCNEncoder(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self._conv1 = tgnn.GCNConv(in_dim, 2 * out_dim)
        self._conv2 = tgnn.GCNConv(2 * out_dim, out_dim)

        self._bn1 = nn.BatchNorm1d(2 * out_dim, momentum=0.01)  # same as `weight_decay = 0.99`

        self._act1 = nn.PReLU()

    def forward(self, x, edge_index):
        x = self._conv1(x, edge_index)
        x = self._bn1(x)
        x = self._act1(x)

        x = self._conv2(x, edge_index)

        return x


def augment(data: Data, p_x: float, p_e: float, add_p:float,del_p:float,edges_to_add: Tensor, edges_to_del:Tensor, degrees_list:List[int]):
    device = data.x.device

    x = data.x
    num_fts = x.size(-1)

    ei = data.edge_index
    num_edges = ei.size(-1)

    x_a = bernoulli_mask(size=(1, num_fts), prob=p_x).to(device) * x
    x_b = bernoulli_mask(size=(1, num_fts), prob=p_x).to(device) * x

    ei_a = ei[:, bernoulli_mask(size=num_edges, prob=p_e).to(device) == 1.]
    #ei_b = ei[:, bernoulli_mask(size=num_edges, prob=p_e).to(device) == 1.]

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

    ei_b = update_edge_index_new(ei, add_edges_tensor, del_edges_tensor)



    return (x_a, ei_a), (x_b, ei_b)


def bernoulli_mask(size: Union[int, Tuple[int, ...]], prob: float):
    return torch.bernoulli((1 - prob) * torch.ones(size))


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