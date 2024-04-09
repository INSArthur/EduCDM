# coding: utf-8
# 2021/6/21 @ tongshiwei

import logging
import numpy as np
import torch
from torcheval.metrics import BinaryAUROC, BinaryPrecision, BinaryRecall, BinaryF1Score

from EduCDM import CDM
from torch import nn
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
import torch.autograd as autograd
import torch.nn.functional as F

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

class DINANet(nn.Module):
    def __init__(self, user_num, item_num, hidden_dim, max_slip=0.4, max_guess=0.4, *args, **kwargs):
        super(DINANet, self).__init__()
        self._user_num = user_num
        self._item_num = item_num
        self.step = 0
        self.max_step = 1000
        self.max_slip = max_slip
        self.max_guess = max_guess

        self.guess = nn.Embedding(self._item_num, 1)
        self.slip = nn.Embedding(self._item_num, 1)
        self.theta = nn.Embedding(self._user_num, hidden_dim)

    def forward(self, user, item, knowledge, *args):
        theta = self.theta(user)
        slip = torch.squeeze(torch.sigmoid(self.slip(item)) * self.max_slip)
        guess = torch.squeeze(torch.sigmoid(self.guess(item)) * self.max_guess)
        if self.training:
            n = torch.sum(knowledge * (torch.sigmoid(theta) - 0.5), dim=1)
            t, self.step = max((np.sin(2 * np.pi * self.step / self.max_step) + 1) / 2 * 100,
                               1e-6), self.step + 1 if self.step < self.max_step else 0
            return torch.sum(
                torch.stack([1 - slip, guess]).T * torch.softmax(torch.stack([n, torch.zeros_like(n)]).T / t, dim=-1),
                dim=1
            )
        else:
            n = torch.prod(knowledge * (theta >= 0) + (1 - knowledge), dim=1)
            return (1 - slip) ** n * guess ** (1 - n)


class STEFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        return F.hardtanh(grad_output)


class StraightThroughEstimator(nn.Module):
    def __init__(self):
        super(StraightThroughEstimator, self).__init__()

    def forward(self, x):
        x = STEFunction.apply(x)
        return x


class STEDINANet(DINANet):
    def __init__(self, user_num, item_num, hidden_dim, max_slip=0.4, max_guess=0.4, *args, **kwargs):
        super(STEDINANet, self).__init__(user_num, item_num, hidden_dim, max_slip, max_guess, *args, **kwargs)
        self.sign = StraightThroughEstimator()

    def forward(self, user, item, knowledge, *args):
        theta = self.sign(self.theta(user))
        slip = torch.squeeze(torch.sigmoid(self.slip(item)) * self.max_slip)
        guess = torch.squeeze(torch.sigmoid(self.guess(item)) * self.max_guess)
        mask_theta = (knowledge == 0) + (knowledge == 1) * theta
        n = torch.prod((mask_theta + 1) / 2, dim=-1)
        return torch.pow(1 - slip, n) * torch.pow(guess, 1 - n)


class DINA(CDM):
    def __init__(self, user_num, item_num, hidden_dim, ste=False, common=None):
        super(DINA, self).__init__()
        if ste:
            self.dina_net = STEDINANet(user_num, item_num, hidden_dim)
        else:
            self.dina_net = DINANet(user_num, item_num, hidden_dim)

        self.common = common

    def train(self, train_data, test_data=None, *, epoch: int, device="cpu", lr=0.001,eval_freq=5,quit_delta=30) -> ...:
        self.dina_net = self.dina_net.to(device)
        loss_function = nn.BCELoss()

        trainer = torch.optim.Adam(self.dina_net.parameters(), lr)

        best_acc = 0
        best_ite=0
        best_metrics = []

        for e in range(epoch):
            losses = []
            for batch_data in tqdm(train_data, "Epoch %s" % e):
                user_id, item_id, response, knowledge, _ = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                knowledge: torch.Tensor = knowledge.to(device)
                predicted_response: torch.Tensor = self.dina_net(user_id, item_id, knowledge)
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
                acc = self.common.evaluate_overall_acc(correctness)
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
            return self.dina_net.theta.weight.data.numpy(), self.dina_net.guess.weight.data.numpy()

    def eval(self, test_data, device="cpu") -> tuple:
        metric = BinaryAUROC()
        precision = BinaryPrecision()
        recall = BinaryRecall()
        f1 = BinaryF1Score()
        self.dina_net = self.dina_net.to(device)
        self.dina_net.eval()
        y_pred = []
        y_true = []
        users = []
        for batch_data in tqdm(test_data, "evaluating"):
            user_id, item_id, response, knowledge, _ = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            knowledge: torch.Tensor = knowledge.to(device)
            pred: torch.Tensor = self.dina_net(user_id, item_id, knowledge)
            y_pred.extend(pred.tolist())
            y_true.extend(response.tolist())
            users.extend(user_id.tolist())


        self.dina_net.train()
        metric.update(torch.tensor(y_pred), torch.tensor(y_true))
        y_pred_rounded = torch.round(torch.tensor(y_pred)).int()
        y_true_rounded = torch.round(torch.tensor(y_true)).int()
        precision.update(y_pred_rounded, y_true_rounded)
        recall.update(y_pred_rounded, y_true_rounded)
        f1.update(y_pred_rounded,y_true_rounded)
        correctness = (np.array(y_true) == (np.array(y_pred) >= 0.5))
        rmse = np.sqrt(np.mean(np.power(np.array(y_true) - np.array(y_pred), 2)))
        return correctness, np.array((users)),metric.compute().item(),rmse, precision.compute().item(), recall.compute().item(), f1.compute().item()

    def save(self, filepath):
        torch.save(self.dina_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.dina_net.load_state_dict(torch.load(filepath))
        logging.info("load parameters from %s" % filepath)
