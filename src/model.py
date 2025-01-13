import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F

from data_set import DataSet
from utils import BPRLoss, EmbLoss

from torch_geometric.typing import Adj, OptTensor

from torch import Tensor
from torch.nn import Parameter
from torch_scatter import scatter_add
from torch_geometric.nn.inits import zeros
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils.num_nodes import maybe_num_nodes


class GCNConv(MessagePassing):
    """
    This class is the graph convolutional operator
    
    .. math::
    \mathbf{E}^{k,l+1} = \mathbf{D}_{k}^{-1/2} \mathbf{\hat{A}}_k
    \mathbf{D}_{k}^{-1/2} \mathbf{E}^{k,l},
    where :math:`\mathbf{A}_{k}` denotes the adjacency matrix and
    :math:`\hat{D}_{k}^{-1/2}` is diagonal degree matrix and

    The new adjacency matrix contains user-item interactions of current and previous behaviors to represent edge weights
    :math:`\mathbf{\hat{A}}_{k} = \mathbf{\hat{A}}_1` if k=1,
    :math:`\mathbf{\hat{A}}_{k} = \mathbf{A}_{k} + \mathbf{A}_{k-1}` otherwise.

    Args:
        in_channels (int): Size of each input sample, or :obj:`-1` to derive
            the size from the first input(s) to the forward method.
        out_channels (int): Size of each output sample.
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    def __init__(self, in_channels: int, out_channels: int, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(GCNConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.bias = Parameter(torch.Tensor(out_channels))
        zeros(self.bias)

    def forward(self, x: Tensor, edge_index: Adj, prior_index: Adj = None) -> Tensor:
        """_summary_

        Args:
            x (Tensor):  Input features
            edge_index (Adj): Graph edge indices
            prior_index (Adj, optional): Prior graph edge indices (optional). Defaults to None.

        Returns:
            Tensor: output tensor after applying GCNConv
        """
        num_nodes = maybe_num_nodes(edge_index, x.size(self.node_dim))

        edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        
        if prior_index is not None:
            edge_index = torch.cat((edge_index, prior_index), dim=1)
            edge_weight = torch.cat([edge_weight, torch.full((prior_index.size(1), ), 1, device=edge_index.device)])

        row, col = edge_index[0], edge_index[1]
        deg = scatter_add(edge_weight, col, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
        edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)
        out += self.bias

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

class GraphEncoder(nn.Module):
    """
    Given data batch, this class builds GCN layers and forwards the layer 
    """
    def __init__(self, layers, hidden_dim, dropout):
        super(GraphEncoder, self).__init__()
        self.gnn_layers = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim) for i in range(layers)])
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, edge_index, prior_index):
        """Performs a forward pass of the GraphEncoder.

        Args:
            x (Tensor): Input features
            edge_index (Adj): Graph edge indices.
            prior_index (Adj): Prior graph edge indices.

        Returns:
            Tensor: the output tensor after applying GCN layers.
        """
        for i in range(len(self.gnn_layers)):
            x = self.gnn_layers[i](x=x, edge_index=edge_index, prior_index=prior_index)
            #x = self.dropout(x)
        return x


class MBA(nn.Module):
    """
    This class represents implementation of our MBA model.
    It consists of cascading GCN blocks and attention networks.
    """
    def __init__(self, args, dataset: DataSet):
        super(MBA, self).__init__()

        self.device = args.device
        self.layers = args.layers
        self.node_dropout = args.node_dropout
        self.n_users = dataset.user_count
        self.n_items = dataset.item_count
        self.edge_index = dataset.edge_index
        self.behaviors = args.behaviors
        self.n_behaviors = len(self.behaviors)
        self.embedding_size = args.embedding_size
        self.user_embedding = nn.Embedding(self.n_users + 1, self.embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)
        self.user_transformation = nn.ModuleList([nn.Linear(self.embedding_size, self.embedding_size) for _ in range(self.n_behaviors-1)])
        self.item_transformation = nn.ModuleList([nn.Linear(self.embedding_size, self.embedding_size) for _ in range(self.n_behaviors-1)])

        self.Graph_encoder = nn.ModuleDict({
            behavior: GraphEncoder(self.layers[index], self.embedding_size, self.node_dropout) for index, behavior in enumerate(self.behaviors)
        })
        self.additive_attn = AdditiveAttention(query_dim=self.embedding_size, embed_dim=self.embedding_size)

        self.reg_weight = args.reg_weight
        self.bpr_loss = BPRLoss()
        self.emb_loss = EmbLoss()

        self.model_path = args.model_path
        self.check_point = args.check_point
        self.if_load_model = args.if_load_model

        self.storage_all_embeddings = None

        self.apply(self._init_weights)

        self._load_model()

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            nn.init.xavier_uniform_(module.weight.data)


    def _load_model(self):
        """Loads the pre-trained model if available."""
        if self.if_load_model:
            parameters = torch.load(os.path.join(self.model_path, self.check_point))
            self.load_state_dict(parameters, strict=False)

    def gcn_propagate(self):
        """
        Performs GCN propagation.
        Returns the final graph embedding.
        """
        all_embeddings = {}
        total_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        prior_index = None
        for i, behavior in enumerate(self.behaviors):
            indices = self.edge_index[behavior].to(self.device)
            
            total_embeddings = self.Graph_encoder[behavior](total_embeddings, indices, prior_index)
            all_embeddings[behavior] = total_embeddings
            
            if i<self.n_behaviors-1:
                user_embedding, item_embedding = torch.split(all_embeddings[behavior], [self.n_users + 1, self.n_items + 1])
                total_embeddings = torch.cat([self.user_transformation[i](user_embedding), self.item_transformation[i](item_embedding)], dim=0)
                prior_index = indices

        all_embeddings = torch.stack([all_embeddings[behavior] for behavior in self.behaviors[:self.n_behaviors]], dim=1)
        user_all_embedding, item_all_embedding = torch.split(all_embeddings, [self.n_users + 1, self.n_items + 1])
        user_all_embedding = self.additive_attn(embeddings=user_all_embedding)
        item_all_embedding = torch.sum(item_all_embedding, dim=1)
        final_embedding = torch.cat((user_all_embedding, item_all_embedding),dim=0)
        return final_embedding

    def forward(self, batch_data):
        """Performs a forward pass of the MBA model.

        Args:
            batch_data: Batch data containing user and item interactions.

        Returns:
            the total loss
        """
        self.storage_all_embeddings = None

        final_embedding = self.gcn_propagate()
        data = batch_data[:, -1]
        users = data[:, 0].long()
        items = data[:, 1:].long()
        user_all_embedding, item_all_embedding = torch.split(final_embedding, [self.n_users + 1, self.n_items + 1])

        user_feature = user_all_embedding[users.view(-1, 1)].expand(-1, items.shape[1], -1)
        item_feature = item_all_embedding[items]
        
        scores = torch.sum(user_feature * item_feature, dim=2)
        bpr_loss = self.bpr_loss(scores[:, 0], scores[:, 1])
        transformation_loss = sum(self.emb_loss(layer.weight.data) for layer in self.user_transformation) + sum(self.emb_loss(layer.weight.data) for layer in self.item_transformation)
        total_loss = bpr_loss + self.reg_weight * (self.emb_loss(self.user_embedding.weight, self.item_embedding.weight) + transformation_loss)

        return total_loss
    
    def full_predict(self, users):
        """Predicts scores for user-item interactions.

        Args:
            users (Tensor): User indices.

        Returns:
            prediction scores for user-item interactions.
        """
        if self.storage_all_embeddings is None:
            self.storage_all_embeddings = self.gcn_propagate()

        user_embedding, item_embedding = torch.split(self.storage_all_embeddings, [self.n_users + 1, self.n_items + 1])
        user_emb = user_embedding[users.long()]
        scores = torch.matmul(user_emb, item_embedding.transpose(0, 1))
        return scores
    
class AdditiveAttention(nn.Module):
    """
    This class represents user with the importance of behaviors by using attention network
    """
    def __init__(self, query_dim: int, embed_dim: int):
        """Initialization
        
        Args:
            query_dim:  the dimension of the additive attention query vectors.
            embed_dim: the dimension of the ``embeddings``.
        """
        super().__init__()
        self.projection = nn.Linear(in_features=embed_dim, out_features=query_dim)
        self.query_vector = nn.Parameter(nn.init.xavier_uniform_(torch.empty(query_dim, 1),
                                                                 gain=nn.init.calculate_gain('tanh')).squeeze())

    def forward(self, embeddings: Tensor):
        """Computes attention-weighted sequence representations

        Args:
            embeddings (Tensor): Input embeddings.

        Returns:
            attention-weighted representations.
        """
        attn_weight = torch.matmul(torch.tanh(self.projection(embeddings)), self.query_vector)
        # attn_weight.masked_fill_(~mask, 1e-30)
        attn_weight = F.softmax(attn_weight, dim=1)
        seq_repr = torch.bmm(attn_weight.unsqueeze(dim=1), embeddings).squeeze(dim=1)
        return seq_repr
