import torch
import torch.nn as nn


class TransH(nn.Module):
    def __init__(self, num_ent, num_rel, emb_dim, device):
        super(TransH, self).__init__()
        self.num_ent = num_ent
        self.num_rel = num_rel
        self.emb_dim = emb_dim
        self.device = device

        self.ent_embs = nn.Embedding(num_ent, emb_dim).to(device)
        self.rel_embs = nn.Embedding(num_rel, emb_dim).to(device)
        self.norm_vector = nn.Embedding(num_rel, emb_dim).to(device)

        # Initialize embeddings
        nn.init.xavier_uniform_(self.ent_embs.weight.data)
        nn.init.xavier_uniform_(self.rel_embs.weight.data)
        nn.init.xavier_uniform_(self.norm_vector.weight.data)

    def forward(self, heads, rels, tails):
        # Get embeddings for heads, relations, and tails
        h_embs = self.ent_embs(heads)
        r_embs = self.rel_embs(rels)
        t_embs = self.ent_embs(tails)
        n_vector = self.norm_vector(rels)

        # Project entities onto the relation-specific hyperplane
        def project(ent_emb, norm_vect):
            return ent_emb - torch.sum(ent_emb * norm_vect, dim=1, keepdim=True) * norm_vect

        h_proj = project(h_embs, n_vector)
        t_proj = project(t_embs, n_vector)

        # Calculate scores based on the TransH scoring function
        scores = torch.norm(h_proj + r_embs - t_proj, p=1, dim=1)
        return scores

    def l2_loss(self):
        return (torch.norm(self.ent_embs.weight, p=2) ** 2 +
                torch.norm(self.rel_embs.weight, p=2) ** 2 +
                torch.norm(self.norm_vector.weight, p=2) ** 2) / 2
