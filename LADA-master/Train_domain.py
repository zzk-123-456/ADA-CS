import torch
import torch.nn as nn
from torch.utils.data import Dataset, TensorDataset
from torch.utils.data import random_split


class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim=100):
        super(Discriminator, self).__init__()
        self.dense0 = nn.Linear(input_dim, hidden_dim)
        self.nonlinear = nn.ReLU()
        self.dense1 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, 2)

    def forward(self, X):
        X = self.nonlinear(self.dense0(X.float()))
        X = self.nonlinear(self.dense1(X))
        X = self.output(X)
        return X


class Train_discriminator():
    def __init__(self, input_dim, batch_size, epochs, lr, device, hidden_dim):
        self.input_dim = input_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = device
        self.hidden_dim = hidden_dim
        self.model = Discriminator(self.input_dim, self.hidden_dim)

    def fit(self, embeddings, labels, train_ratio=0.9):
        if not isinstance(embeddings, torch.Tensor):
            X = torch.tensor(embeddings)
        else:
            X = embeddings
        if not isinstance(labels, torch.Tensor):
            y = torch.tensor(labels)
        else:
            y = labels
        self.model = self.model.to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        trainset = TensorDataset(X, y)
        test_abs = int(len(trainset) * train_ratio)
        train_subset, val_subset = random_split(trainset, [test_abs, len(trainset) - test_abs])
        trainloader = torch.utils.data.DataLoader(train_subset, batch_size=self.batch_size, shuffle=False)
        valloader = torch.utils.data.DataLoader(val_subset, batch_size=self.batch_size, shuffle=False)
        self.train_loader = trainloader
        self.valid_loader = valloader
        self.criterion = nn.CrossEntropyLoss()
        # self.model.train()
        for epoch in range(self.epochs):
            running_loss = 0.0
            for batch_inputs, batch_labels in trainloader:
                batch_inputs = batch_inputs.to(self.device)
                batch_labels = batch_labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.model(batch_inputs)
                loss = self.criterion(outputs, batch_labels)
                running_loss += loss.item()
                loss.backward()
                optimizer.step()
            print(f'Epoch [{epoch + 1}/{self.epochs}], Loss: {running_loss:.4f}')

