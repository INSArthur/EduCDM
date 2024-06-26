# coding: utf-8
# 2021/3/23 @ tongshiwei

import logging
import numpy as np
import torch
from torcheval.metrics import BinaryAUROC, BinaryPrecision, BinaryRecall, BinaryF1Score
from tqdm import tqdm
from torch import nn
from EduCDM import CDM
from sklearn.metrics import roc_auc_score, accuracy_score

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

class MFNet(nn.Module):
    """Matrix Factorization Network"""

    def __init__(self, user_num, item_num, latent_dim):
        super(MFNet, self).__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.latent_dim = latent_dim
        self.user_embedding = nn.Embedding(self.user_num, self.latent_dim)
        self.item_embedding = nn.Embedding(self.item_num, self.latent_dim)
        self.response = nn.Linear(2 * self.latent_dim, 1)

    def forward(self, user_id, item_id):
        user = self.user_embedding(user_id)
        item = self.item_embedding(item_id)
        return torch.squeeze(torch.sigmoid(self.response(torch.cat([user, item], dim=-1))), dim=-1)


class MCD(CDM):
    """Matrix factorization based Cognitive Diagnosis Model"""

    def __init__(self, user_num, item_num, latent_dim, common=None):
        super(MCD, self).__init__()
        self.mf_net = MFNet(user_num, item_num, latent_dim)

    def train(self, train_data, test_data=None, *, epoch: int, device="cpu", lr=0.001,eval_freq=5,quit_delta=30) -> ...:
        self.mf_net = self.mf_net.to(device)
        loss_function = nn.BCELoss()

        trainer = torch.optim.Adam(self.mf_net.parameters(), lr)

        best_acc = 0
        best_ite=0
        best_metrics = []

        for e in range(epoch):
            losses = []
            for batch_data in tqdm(train_data, "Epoch %s" % e):
                user_id, item_id, response,_,_ = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                predicted_response: torch.Tensor = self.mf_net(user_id, item_id)
                response: torch.Tensor = response.to(device)
                loss = loss_function(predicted_response, response)

                # back propagation
                trainer.zero_grad()
                loss.backward()
                trainer.step()

                losses.append(loss.mean().item())
            print("[Epoch %d] LogisticLoss: %.6f" % (e, float(np.mean(losses))))

            if test_data is not None and e %eval_freq == 0:
                correctness,users,auc,rmse = self.eval(test_data, device=device)
                acc = self.evaluate_overall_acc(correctness)
                #print("[Epoch %d] auc: %.6f, accuracy: %.6f" % (e, auc, accuracy))

                if acc> best_acc :
                    best_acc = acc
                    best_ite = e
                    best_metrics = [correctness, users, auc,rmse]

                if e-best_ite >= quit_delta :
                    break

        if test_data is not None :
            best_metrics.append(best_ite)
            return best_metrics
        else :
            return self.mf_net.user_embedding.weight.data.numpy(), self.mf_net.item_embedding.weight.data.numpy()

    def eval(self, test_data, device="cpu") -> tuple:
        metric = BinaryAUROC()
        precision = BinaryPrecision()
        recall = BinaryRecall()
        f1 = BinaryF1Score()
        self.mf_net = self.mf_net.to(device)
        self.mf_net.eval()
        y_pred = []
        y_true = []
        users = []
        for batch_data in tqdm(test_data, "evaluating"):
            user_id, item_id, response,_,_ = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            pred: torch.Tensor = self.mf_net(user_id, item_id)
            y_pred.extend(pred.tolist())
            y_true.extend(response.tolist())
            users.extend(user_id.tolist())

        self.mf_net.train()
        metric.update(torch.tensor(y_pred), torch.tensor(y_true))
        y_pred_rounded = torch.round(torch.tensor(y_pred)).int()
        y_true_rounded = torch.round(torch.tensor(y_true)).int()
        precision.update(y_pred_rounded, y_true_rounded)
        recall.update(y_pred_rounded, y_true_rounded)
        f1.update(y_pred_rounded,y_true_rounded)
        correctness = (np.array(y_true) == (np.array(y_pred) >= 0.5))
        rmse = np.sqrt(np.mean(np.power(np.array(y_true) - np.array(y_pred), 2)))
        return correctness, np.array((users)), metric.compute().item(), rmse, precision.compute().item(), recall.compute().item(), f1.compute().item()

    def save(self, filepath):
        torch.save(self.mf_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.mf_net.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)
