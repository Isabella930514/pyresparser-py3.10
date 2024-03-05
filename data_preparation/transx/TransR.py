import torch
import torch.nn as nn


class TransR(nn.Module):
    def __init__(self, num_ent, num_rel, ent_dim, rel_dim, device):
        super(TransR, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.ent_dim = ent_dim
        self.rel_dim = rel_dim
        self.device = device

        self.ent_embs = nn.Embedding(num_ent, ent_dim).to(device)
        self.rel_embs = nn.Embedding(num_rel, rel_dim).to(device)
        self.transfer_matrix = nn.Embedding(num_rel, ent_dim * rel_dim).to(device)

        nn.init.xavier_uniform_(self.ent_embs.weight.data)
        nn.init.xavier_uniform_(self.rel_embs.weight.data)
        nn.init.xavier_uniform_(self.transfer_matrix.weight.data)

    def forward(self, heads, rels, tails):
        h_embs = self.ent_embs(heads)
        r_embs = self.rel_embs(rels)
        t_embs = self.ent_embs(tails)
        M_r = self.transfer_matrix(rels)

        M_r = M_r.view(-1, self.rel_dim, self.ent_dim)
        h_embs = torch.matmul(h_embs.unsqueeze(1), M_r).squeeze(1)
        t_embs = torch.matmul(t_embs.unsqueeze(1), M_r).squeeze(1)

        scores = torch.norm(h_embs + r_embs - t_embs, p=1, dim=1)
        return scores

    def l2_loss(self):
        return (torch.norm(self.ent_embs.weight, p=2) ** 2 +
                torch.norm(self.rel_embs.weight, p=2) ** 2 +
                torch.norm(self.transfer_matrix.weight, p=2) ** 2) / 2
