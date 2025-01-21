from scipy.optimize import brent
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score
from torch.autograd import grad
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model, ensemble, kernel_approximation, svm
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import xgboost as xgb
from fairlearn.reductions import DemographicParity, EqualizedOdds, \
    ErrorRateParity
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, DemographicParity
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.utils.data import random_split
from .robust_loss import RobustLoss, group_dro_criterion, cvar_doro_criterion, chi_square_doro_criterion
from .marginal_dro_criterion import LipLoss, opt_model, marginal_dro_criterion
from .model_util import *


TRAIN_FRAC = 0.9

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes, num_units=16, nonlin=nn.ReLU(), dropout_ratio=0.1):
        super().__init__()

        self.dense0 = nn.Linear(input_dim, num_units)
        self.nonlin = nonlin
        self.dropout = nn.Dropout(dropout_ratio)
        self.dense1 = nn.Linear(num_units, num_units)
        self.output = nn.Linear(num_units, num_classes)

    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X.float()))
        X = self.dropout(X)
        X = self.nonlin(self.dense1(X))
        X = self.output(X)
        return X


class MLP_simple(nn.Module):
    def __init__(self, input_dim, num_classes, num_units=16, nonlin=nn.ReLU()):
        super(MLP_simple, self).__init__()

        self.dense0 = nn.Linear(input_dim, num_units)
        self.nonlin = nonlin
        self.output = nn.Linear(num_units, num_classes)
    def forward(self, X, **kwargs):
        X = self.nonlin(self.dense0(X.float()))
        X = self.output(X)
        return X


class Linear(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super().__init__()
        self.dense0 = nn.Linear(input_dim, num_classes)
        
    def forward(self, X, **kwargs):
        return self.dense0(X.float())

# class LogisticRegression():
#     def __init__(self, input_dim=9, num_classes=2):
#         self.model = Linear(input_dim, num_classes)
#         self.criterion = nn.CrossEntropyLoss()
#
#     def update(self, config):
#         self.lr = config["lr"]
#         self.batch_size = config["batch_size"]
#         self.train_epochs = config["train_epochs"]
#
#     def predict(self, X):
#         if not isinstance(X, torch.Tensor):
#             X = torch.tensor(X)
#         inputs = X.to(self.device)
#         self.model = self.model.to(self.device)
#         outputs = self.model(inputs)
#         _, predicted = torch.max(outputs.data, 1)
#         return predicted.detach().cpu().numpy()
#
#     def score(self, X, y):
#         if not isinstance(X, torch.Tensor):
#             X = torch.tensor(X)
#             y = torch.tensor(y)
#         inputs, labels = X.to(self.device), y.to(self.device)
#         self.model = self.model.to(self.device)
#         outputs = self.model(inputs)
#         _, predicted = torch.max(outputs.data, 1)
#         correct = (predicted == labels).sum().item()
#         total = y.shape[0]
#         return correct / total
#
#     def f1score(self, X, y):
#         if not isinstance(X, torch.Tensor):
#             X = torch.tensor(X)
#             y = torch.tensor(y)
#         inputs, labels = X.to(self.device), y.to(self.device)
#         self.model = self.model.to(self.device)
#         outputs = self.model(inputs)
#         _, predicted = torch.max(outputs.data, 1)
#         return f1_score(y.cpu().numpy(), predicted.cpu().numpy(), average='micro')
#
#
#     def fit(self, X, y, train_ratio=TRAIN_FRAC, device='cpu'):
#         self.device = device
#
#         if not isinstance(X, torch.Tensor):
#             X = torch.tensor(X)
#             y = torch.tensor(y)
#
#         self.model = self.model.to(self.device)
#         optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
#         # criterion = nn.CrossEntropyLoss()
#
#         trainset = TensorDataset(X, y)
#         test_abs = int(len(trainset) * train_ratio)
#         train_subset, val_subset = random_split(trainset, [test_abs, len(trainset) - test_abs])
#         trainloader = torch.utils.data.DataLoader(train_subset,batch_size=self.batch_size, shuffle=True, num_workers=8)
#         valloader = torch.utils.data.DataLoader(val_subset, batch_size=self.batch_size, shuffle=True, num_workers=8)
#         for epoch in range(0,self.train_epochs+1):
#             running_loss = 0.0
#             epoch_steps = 0
#             for i, data in enumerate(trainloader, 0):
#                 inputs, labels = data
#                 inputs, labels = inputs.to(self.device), labels.to(self.device)
#
#                 optimizer.zero_grad()
#                 outputs = self.model(inputs)
#                 loss = self.criterion(outputs, labels)
#                 loss.backward()
#                 optimizer.step()
#
#                 running_loss += loss.item()
#                 epoch_steps += 1

class RF():
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)

    def fit_weight(self, X_train, y_train, weight):
        self.model.fit(X_train, y_train, weight)

    def score(self, X_test, y_test):
        score = self.model.score(X_test, y_test)
        return score

    def predict_proba(self, X_test):
        probas = self.model.predict_proba(X_test)
        return probas

    def predict(self, X_test):
        result = self.model.predict(X_test)
        return result


class Logistic():
    def __init__(self):
        self.model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=100)

    def fit_weight(self, X_train, y_train, weight):
        self.model.fit(X_train, y_train, weight)

    def score(self, X_test, y_test):
        score = self.model.score(X_test, y_test)
        return score

    def predict_proba(self, X_test):
        probas = self.model.predict_proba(X_test)
        return probas

    def predict(self, X_test):
        result = self.model.predict(X_test)
        return result




class MLPClassifier():
    def __init__(self, input_dim=9, num_classes=2, lr=0.0002, batch_size=128, train_epochs=30, hidden_size=16, dropout_ratio=0.1,
                 device='cuda', decay=1e-3):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()
        self.lr = lr
        self.batch_size = batch_size
        self.train_epochs = train_epochs
        self.model = MLP(self.input_dim, self.num_classes, num_units=hidden_size,
                         dropout_ratio=dropout_ratio)
        # self.model = MLP_simple(self.input_dim, self.num_classes, num_units=hidden_size)
        self.device = device
        self.decay = decay

    def update(self, config):
        self.lr = config["lr"]
        self.batch_size = config["batch_size"]
    
    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        inputs = X.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.detach().cpu().numpy()

    def predict_logits(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        inputs = X.to(self.device)
        outputs = self.model(inputs)
        # print(outputs.shape)
        # softmax = nn.Softmax(dim=1)
        # outputs = softmax(outputs)
        return outputs.detach().cpu().numpy()

# MLP的输出仅仅为logits，并非概率，因此在该函数中加入softmax
    def predict_proba(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        inputs = X.to(self.device)
        outputs = self.model(inputs)
        # print(outputs.shape)
        softmax = nn.Softmax(dim=1)
        outputs = softmax(outputs)
        return outputs.detach().cpu().numpy()


    def score(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)
        inputs, labels = X.to(self.device), y.to(self.device)
        self.model = self.model.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == labels).sum().item()
        total = y.shape[0]
        return correct / total

    def f1score(self, X, y):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
            y = torch.tensor(y)
        inputs, labels = X.to(self.device), y.to(self.device)
        self.model = self.model.to(self.device)
        outputs = self.model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        return f1_score(y.cpu().numpy(), predicted.cpu().numpy(), average='macro')

    def fit(self, X, y, train_ratio=TRAIN_FRAC, device='cpu'):
        self.device = device 

        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
            y = torch.tensor(y)

        self.model = self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # criterion = nn.CrossEntropyLoss()
        
        trainset = TensorDataset(X, y)
        test_abs = int(len(trainset) * train_ratio)
        train_subset, val_subset = random_split(trainset, [test_abs, len(trainset) - test_abs])
        trainloader = torch.utils.data.DataLoader(train_subset,batch_size=self.batch_size, shuffle=True, num_workers=1)
        valloader = torch.utils.data.DataLoader(val_subset, batch_size=self.batch_size, shuffle=True, num_workers=1)
        for epoch in tqdm(range(0,self.train_epochs+1)):
            running_loss = 0.0
            epoch_steps = 0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epoch_steps += 1
                
            # print("[%d] loss: %.3f" % (epoch + 1,running_loss))
            
            
        total = 0
        correct = 0
        for i, data in enumerate(valloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        # print("accuracy", correct / total)

        return correct / total

    def fit_weight(self, X, y, weights, train_ratio=TRAIN_FRAC, device='cpu'):
        self.device = device 
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights).float()

        self.model = self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.decay)
        # criterion = nn.CrossEntropyLoss()
        
        # print(X.shape, y.shape, weights.shape)
        trainset = TensorDataset(X, y, weights)
        test_abs = int(len(trainset) * train_ratio)
        train_subset, val_subset = random_split(trainset, [test_abs, len(trainset) - test_abs])
        trainloader = torch.utils.data.DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=1)
        valloader = torch.utils.data.DataLoader(val_subset, batch_size=self.batch_size, shuffle=True, num_workers=1)
        self.train_loader = trainloader
        self.valid_loader = valloader
        train_errors = np.zeros(self.train_epochs + 1)
        for epoch in range(0, self.train_epochs+1):
            running_loss = 0.0
            epoch_steps = 0
            self.model.train()
            for i, data in enumerate(trainloader, 0):
                inputs, labels, weights_batch = data
                inputs, labels, weights_batch = inputs.to(self.device), labels.to(self.device), weights_batch.to(self.device)
                labels = labels.long()

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss = torch.dot(loss.reshape(-1), weights_batch.reshape(-1))/loss.shape[0]
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epoch_steps += 1
            print("epochs{}:{}".format(epoch, running_loss))
            train_errors[epoch] = running_loss
            # print("[%d] loss: %.3f" % (epoch + 1,running_loss))
            
            self.model.eval()
            total = 0
            correct = 0
            loss_eval = 0
            for i, data in enumerate(valloader, 0):
                with torch.no_grad():
                    inputs, labels, weights_batch = data
                    inputs, labels, weights_batch = inputs.to(self.device), labels.to(self.device), weights_batch.to(self.device)
                    labels = labels.long()

                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss = torch.dot(loss.reshape(-1), weights_batch.reshape(-1)) / loss.shape[0]
                    loss_eval += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

        return train_errors

class MLPRegressor():
    def __init__(self, input_dim=9, output_dim=1, lr=0.0002, batch_size=128, train_epochs=30, hidden_size=16,
                 dropout_ratio=0.1, device='cuda', decay=1e-3):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.criterion = nn.MSELoss()
        self.lr = lr
        self.batch_size = batch_size
        self.train_epochs = train_epochs
        self.model = MLP(self.input_dim, self.output_dim, num_units=hidden_size,
                         dropout_ratio=dropout_ratio)
        self.device = device
        self.decay = decay

    def fit_weight(self, X, y, weights, train_ratio=TRAIN_FRAC, device='cpu'):
        self.device = device
        self.criterion = nn.MSELoss(reduction='none')
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y)
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights).float()

        self.model = self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.decay)


        trainset = TensorDataset(X, y, weights)
        test_abs = int(len(trainset) * train_ratio)
        train_subset, val_subset = random_split(trainset, [test_abs, len(trainset) - test_abs])
        trainloader = torch.utils.data.DataLoader(train_subset, batch_size=self.batch_size, shuffle=True, num_workers=1)
        valloader = torch.utils.data.DataLoader(val_subset, batch_size=self.batch_size, shuffle=True, num_workers=1)
        self.train_loader = trainloader
        self.valid_loader = valloader
        train_errors = np.zeros(self.train_epochs + 1)
        for epoch in tqdm(range(0, self.train_epochs + 1)):
            running_loss = 0.0
            epoch_steps = 0
            self.model.train()
            for i, data in enumerate(trainloader, 0):
                inputs, labels, weights_batch = data
                inputs, labels, weights_batch = inputs.to(self.device), labels.to(self.device), weights_batch.to(
                    self.device)

                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs.squeeze(), labels)
                # print(loss.shape, inputs.shape, outputs.shape, weights_batch.shape, labels.shape)
                loss = torch.dot(loss.reshape(-1), weights_batch.reshape(-1)) / loss.shape[0]
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                epoch_steps += 1
            print("epochs{}:{}".format(epoch, running_loss))
            train_errors[epoch] = running_loss

        return train_errors

    def predict(self, X):
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X)
        inputs = X.to(self.device)
        outputs = self.model(inputs)
        outputs = outputs.reshape(-1)
        return outputs.detach().cpu().numpy()