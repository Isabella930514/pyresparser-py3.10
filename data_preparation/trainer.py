import os
import torch
import torch.nn.functional as F
from data_preparation.SimplE import SimplE
from data_preparation.transx.TransE import TransE
from data_preparation.transx.TransH import TransH
from data_preparation.transx.TransR import TransR


class Trainer:
    def __init__(self, dataset, args, kge_model):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if kge_model == "TransE":
            self.model = TransE(dataset.num_ent(), dataset.num_rel(), args.emb_dim, self.device)
        elif kge_model == "SimplE":
            self.model = SimplE(dataset.num_ent(), dataset.num_rel(), args.emb_dim, self.device)
        elif kge_model == "TransH":
            self.model = TransH(dataset.num_ent(), dataset.num_rel(), args.emb_dim, self.device)
        elif kge_model == "TransR":
            self.model = TransR(dataset.num_ent(), dataset.num_rel(), args.emb_dim, self.device)
        else:
            raise ValueError("Unsupported KGE model type: {}".format(kge_model))
        self.dataset = dataset
        self.args = args

    def train(self, kge_model):
        self.model.train()

        optimizer = torch.optim.Adagrad(
            self.model.parameters(),
            lr=self.args.lr,
            weight_decay=0,
            initial_accumulator_value=0.1  # this is added because of the consistency to the original tensorflow code
        )

        for epoch in range(1, self.args.ne + 1):
            last_batch = False
            total_loss = 0.0

            while not last_batch:
                h, r, t, l = self.dataset.next_batch(self.args.batch_size, neg_ratio=self.args.neg_ratio,
                                                     device=self.device)
                last_batch = self.dataset.was_last_batch()
                optimizer.zero_grad()
                scores = self.model(h, r, t)
                loss = torch.sum(F.softplus(-l * scores)) + (
                        self.args.reg_lambda * self.model.l2_loss() / self.dataset.num_batch(self.args.batch_size))
                loss.backward()
                optimizer.step()
                total_loss += loss.cpu().item()

            print("Loss in iteration " + str(epoch) + ": " + str(
                total_loss) + "(" + self.dataset.name + " and " + kge_model + ")")

            if epoch % self.args.save_each == 0:
                self.save_model(epoch)

    def save_model(self, chkpnt):
        print("Saving the model")
        directory = "models/" + self.dataset.name + "/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.model, directory + str(chkpnt) + ".chkpnt")
