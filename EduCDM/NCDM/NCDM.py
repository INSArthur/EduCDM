# coding: utf-8
# 2021/4/1 @ WangFei

import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from torcheval.metrics import metric, BinaryAUROC, BinaryPrecision, BinaryRecall
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score
from EduCDM import CDM

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

class PosLinear(nn.Linear):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight = 2 * F.relu(1 * torch.neg(self.weight)) + self.weight
        return F.linear(input, weight, self.bias)


class Net(nn.Module):

    def __init__(self, knowledge_n, exer_n, student_n):
        self.knowledge_dim = knowledge_n
        self.exer_n = exer_n
        self.emb_num = student_n
        self.stu_dim = self.knowledge_dim
        self.prednet_input_len = self.knowledge_dim
        self.prednet_len1, self.prednet_len2 = 512, 256  # changeable

        super(Net, self).__init__()

        # prediction sub-net
        self.student_emb = nn.Embedding(self.emb_num, self.stu_dim).to(device)
        self.k_difficulty = nn.Embedding(self.exer_n, self.knowledge_dim).to(device)
        self.e_difficulty = nn.Embedding(self.exer_n, 1).to(device)
        self.prednet_full1 = PosLinear(self.prednet_input_len, self.prednet_len1)
        self.drop_1 = nn.Dropout(p=0.5)
        self.prednet_full2 = PosLinear(self.prednet_len1, self.prednet_len2)
        self.drop_2 = nn.Dropout(p=0.5)
        self.prednet_full3 = PosLinear(self.prednet_len2, 1)

        # initialize
        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(self, stu_id, input_exercise, input_knowledge_point):
        # before prednet
        stu_emb = self.student_emb(stu_id)
        stat_emb = torch.sigmoid(stu_emb)
        k_difficulty = torch.sigmoid(self.k_difficulty(input_exercise))
        e_difficulty = torch.sigmoid(self.e_difficulty(input_exercise))  # * 10
        # prednet
        input_x = e_difficulty * (stat_emb - k_difficulty) * input_knowledge_point
        input_x = self.drop_1(torch.sigmoid(self.prednet_full1(input_x)))
        input_x = self.drop_2(torch.sigmoid(self.prednet_full2(input_x)))
        output_1 = torch.sigmoid(self.prednet_full3(input_x))

        return output_1.view(-1)


class NCDM(CDM):
    '''Neural Cognitive Diagnosis Model'''

    def __init__(self, knowledge_n, exer_n, student_n, common=None):
        super(NCDM, self).__init__()
        self.ncdm_net = Net(knowledge_n, exer_n, student_n)
        self.common = common

    def train(self, train_data, test_data=None, epoch=10, device="cpu", lr=0.002, silence=False,eval_freq=5,quit_delta=30):
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.train()

        loss_function = nn.BCELoss()
        optimizer = optim.Adam(self.ncdm_net.parameters(), lr=lr)
        
        best_acc = 0
        best_ite=0
        best_metrics = []
        
        for e in range(epoch):
            epoch_losses = []
            batch_count = 0
            for batch_data in tqdm(train_data, "Epoch %s" % e):
                batch_count += 1
                user_id, item_id, y, knowledge_emb, _ = batch_data
                user_id: torch.Tensor = user_id.to(device)
                item_id: torch.Tensor = item_id.to(device)
                knowledge_emb: torch.Tensor = knowledge_emb.to(device)
                y: torch.Tensor = y.to(device)
                pred: torch.Tensor = self.ncdm_net(user_id, item_id, knowledge_emb)
                loss = loss_function(pred, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_losses.append(loss.mean().item())

            if test_data is not None and e % eval_freq == 0:
                correctness, users, auc, rmse = self.eval(test_data, device=device)
                acc = self.common.evaluate_overall_acc(correctness)


                if acc > best_acc:
                    best_acc = acc
                    best_ite = e
                    best_metrics = [correctness, users, auc, rmse]

                print("[Epoch %d] auc: %.6f, accuracy: %.6f, best_ite: %.6f" % (e, auc, acc,best_ite))

                if e - best_ite >= quit_delta:
                    break

        if test_data is not None :
            best_metrics.append(best_ite)
            return best_metrics
        else :
            return self.ncdm_net.student_emb.weight.data.numpy(),self.ncdm_net.e_difficulty.weight.data.numpy()

    def eval(self, test_data, device="cpu"):
        metric = BinaryAUROC()
        precision = BinaryPrecision()
        recall = BinaryRecall()
        self.ncdm_net = self.ncdm_net.to(device)
        self.ncdm_net.eval()
        y_true, y_pred, users = [], [], []
        for batch_data in tqdm(test_data, "Evaluating"):
            user_id, item_id, y, knowledge_emb, _ = batch_data
            user_id: torch.Tensor = user_id.to(device)
            item_id: torch.Tensor = item_id.to(device)
            knowledge_emb: torch.Tensor = knowledge_emb.to(device)
            pred: torch.Tensor = self.ncdm_net(user_id, item_id, knowledge_emb)
            y_pred.extend(pred.detach().cpu().tolist())
            y_true.extend(y.tolist())
            users.extend(user_id.tolist())

        metric.update(torch.tensor(y_pred), torch.tensor(y_true))
        precision.update(torch.tensor(y_pred), torch.tensor(y_true))
        recall.update(torch.tensor(y_pred), torch.tensor(y_true))
        self.ncdm_net.train()
        correctness = (np.array(y_true) == (np.array(y_pred) >= 0.5))
        rmse = np.sqrt(np.mean(np.power(np.array(y_true) - np.array(y_pred), 2)))
        return correctness, np.array((users)), metric.compute().item(), rmse, precision.compute().item(), recall.compute().item()

    def save(self, filepath):
        torch.save(self.ncdm_net.state_dict(), filepath)
        logging.info("save parameters to %s" % filepath)

    def load(self, filepath):
        self.ncdm_net.load_state_dict(torch.load(filepath))  # , map_location=lambda s, loc: s
        logging.info("load parameters from %s" % filepath)
