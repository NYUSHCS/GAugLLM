from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import torch
from torch import nn, Tensor
from torch_geometric import nn as tgnn
from typing import List
from GBT.gssl.loss import get_loss
from GBT.gssl.transductive_model_GBT import Model


class ArxivModel(Model):

    def __init__(
        self,
        feature_dim: int,
        emb_dim: int,
        loss_name: str,
        p_x_1: float,
        p_e_1: float,
        p_x_2: float,
        p_e_2: float,
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

        self._encoder = ThreeLayerGCNEncoder(
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

        self._p_x_1 = p_x_1
        self._p_e_1 = p_e_1
        self._p_x_2 = p_x_2
        self._p_e_2 = p_e_2
        self.edges_to_add = edges_to_add
        self.edges_to_del = edges_to_del 
        self.degrees_list = degrees_list
        self.add_p = add_p
        self.del_p = del_p

        self._total_epochs = total_epochs



class ThreeLayerGCNEncoder(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self._conv1 = tgnn.GCNConv(in_dim, out_dim)
        self._conv2 = tgnn.GCNConv(out_dim, out_dim)
        self._conv3 = tgnn.GCNConv(out_dim, out_dim)

        self._bn1 = nn.BatchNorm1d(out_dim, momentum=0.01)
        self._bn2 = nn.BatchNorm1d(out_dim, momentum=0.01)

        self._act1 = nn.PReLU()
        self._act2 = nn.PReLU()

    def forward(self, x, edge_index):
        x = self._conv1(x, edge_index)
        x = self._bn1(x)
        x = self._act1(x)

        x = self._conv2(x, edge_index)
        x = self._bn2(x)
        x = self._act2(x)

        x = self._conv3(x, edge_index)

        return x
