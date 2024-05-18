import torch
import torch.nn as nn
import math


class TransE(nn.Module):
    def __init__(self, num_ent, num_rel, emb_dim, device):
        super(TransE, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.emb_dim = emb_dim
        self.device = device

        self.ent_embs = nn.Embedding(num_ent, emb_dim).to(device)
        self.rel_embs = nn.Embedding(num_rel, emb_dim).to(device)

        sqrt_size = 6.0 / math.sqrt(emb_dim)
        nn.init.uniform_(self.ent_embs.weight.data, -sqrt_size, sqrt_size)
        nn.init.uniform_(self.rel_embs.weight.data, -sqrt_size, sqrt_size)

    def forward(self, heads, rels, tails):
        h_embs = self.ent_embs(heads)
        r_embs = self.rel_embs(rels)
        t_embs = self.ent_embs(tails)

        scores = torch.norm(h_embs + r_embs - t_embs, p=1, dim=1)
        return scores

    def l2_loss(self):
        return (torch.norm(self.ent_embs.weight, p=2) ** 2 + torch.norm(self.rel_embs.weight, p=2) ** 2) / 2
