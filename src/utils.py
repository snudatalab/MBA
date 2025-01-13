import torch
import torch.nn as nn
import numpy as np


class BPRLoss(nn.Module):
    """
    This class defines the BPR loss, which is used as a ranking-based loss for recommendation systems. 
    It takes positive and negative scores and computes the BPR loss, which is then returned as a scalar tensor.
    """
    def __init__(self):
        super(BPRLoss, self).__init__()
        self.gamma = 1e-10

    def forward(self, p_score, n_score):
        loss = -torch.log(self.gamma + torch.sigmoid(p_score - n_score))
        loss = loss.mean()

        return loss


class EmbLoss(nn.Module):
    """
    This class defines an embedding loss, which is used for regularization on embeddings in recommendation models. 
    It takes a variable number of embeddings and computes the regularization loss, which is then returned as a scalar tensor.
    """

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings):
        emb_loss = torch.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            emb_loss += torch.norm(embedding, p=self.norm)
        emb_loss /= embeddings[-1].shape[0]
        return emb_loss

def hit_(pos_index, pos_len):
    """
    Hit_ (also known as hit ratio at :math:`N`) is a way of calculating how many 'hits' you have
    in an n-sized list of ranked items.

    .. math::
        \mathrm {HR@K} =\frac{Number \space of \space Hits @K}{|GT|}

    :math:`HR` is the number of users with a positive sample in the recommendation list.
    :math:`GT` is the total number of samples in the test set.

    """    
    result = np.cumsum(pos_index, axis=1)
    return (result > 0).astype(int)

def ndcg_(pos_index, pos_len):
    """
    NDCG_ (also known as normalized discounted cumulative gain) is a measure of ranking quality.
    Through normalizing the score, users and their recommendation list results in the whole test set can be evaluated.

    .. math::
        \begin{gather}
            \mathrm {DCG@K}=\sum_{i=1}^{K} \frac{2^{rel_i}-1}{\log_{2}{(i+1)}}\\
            \mathrm {IDCG@K}=\sum_{i=1}^{K}\frac{1}{\log_{2}{(i+1)}}\\
            \mathrm {NDCG_u@K}=\frac{DCG_u@K}{IDCG_u@K}\\
            \mathrm {NDCG@K}=\frac{\sum \nolimits_{u \in U^{te}NDCG_u@K}}{|U^{te}|}
        \end{gather}

    :math:`K` stands for recommending :math:`K` items.
    And the :math:`rel_i` is the relevance of the item in position :math:`i` in the recommendation list.
    :math:`{rel_i}` equals to 1 if the item is ground truth otherwise 0.
    :math:`U^{te}` stands for all users in the test set.

    """
    len_rank = np.full_like(pos_len, pos_index.shape[1])
    idcg_len = np.where(pos_len > len_rank, len_rank, pos_len)

    iranks = np.zeros_like(pos_index, dtype=np.float)
    iranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    idcg = np.cumsum(1.0 / np.log2(iranks + 1), axis=1)
    for row, idx in enumerate(idcg_len):
        idcg[row, idx:] = idcg[row, idx - 1]

    ranks = np.zeros_like(pos_index, dtype=np.float)
    ranks[:, :] = np.arange(1, pos_index.shape[1] + 1)
    dcg = 1.0 / np.log2(ranks + 1)
    dcg = np.cumsum(np.where(pos_index, dcg, 0), axis=1)

    result = dcg / idcg
    return result

def metrics():
    """
    This function returns a dictionary of metric functions, 
    including 'ndcg' and 'hit', 
    which is used for evaluating a model.
    """
    metrics_dict = {
        'ndcg': ndcg_,
        'hit': hit_
    }
    return metrics_dict
