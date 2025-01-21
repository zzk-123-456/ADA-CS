import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import os
# from model import get_model
import torch.optim as optim
# from solver import get_solver
import logging
import torch.nn.functional as F

from scipy.special import kl_div

def resetRNGseed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(model, device, train_loader, optimizer, epoch):
    """
    Test model on provided data for single epoch
    """
    model.train()
    total_loss, correct = 0.0, 0
    for batch_idx, (data, target, _) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = nn.CrossEntropyLoss()(output, target)
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        corr = pred.eq(target.view_as(pred)).sum().item()
        correct += corr
        loss.backward()
        optimizer.step()

    train_acc = 100. * correct / len(train_loader.sampler)
    avg_loss = total_loss / len(train_loader.sampler)
    logging.info('Train Epoch: {} | Avg. Loss: {:.3f} | Train Acc: {:.3f}'.format(epoch, avg_loss, train_acc))
    return avg_loss

def test(model, device, test_loader, split="target test"):
    """
    Test model on provided data
    """
    # logging.info('Evaluating model on {}...'.format(split))
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target, _ in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = nn.CrossEntropyLoss()(output, target)
            test_loss += loss.item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            corr = pred.eq(target.view_as(pred)).sum().item()
            correct += corr
            del loss, output

    test_loss /= len(test_loader.sampler)
    test_acc = 100. * correct / len(test_loader.sampler)

    return test_acc, test_loss


def get_optim(name, *args, **kwargs):
    if name == 'Adadelta':
        return optim.Adadelta(*args, **kwargs)
    elif name == 'Adam':
        return optim.Adam(*args, **kwargs)
    elif name == 'SGD':
        return optim.SGD(*args, **kwargs, momentum=0.9, nesterov=True)


def shared_reweight(source_X, other_X, disc, device):
    source_X = torch.tensor(source_X).to(device)
    other_X = torch.tensor(other_X).to(device)
    piA = F.softmax(disc(source_X), dim=1)[:, 0]
    piB = F.softmax(disc(other_X), dim=1)[:, 0]

    print(piA)
    print(piB)

    # print(piA)
    alpha = (other_X.shape[0]) / (source_X.shape[0] + other_X.shape[0])
    wA = piA / ((1 - alpha) * piA + alpha * (1 - piA))
    wB = (1 - piB) / ((1 - alpha) * piB + alpha * (1 - piB))
    wA = wA / wA.sum()
    wB = wB / wB.sum()

    source_X = source_X.cpu().numpy()
    other_X = other_X.cpu().numpy()
    wA = wA.cpu().detach().numpy()
    wB = wB.cpu().detach().numpy()
    new_X = np.concatenate([source_X, other_X], axis=0)
    new_weights = np.concatenate([wA, wB])
    new_weights /= np.sum(new_weights)
    return wA, wB, new_X, new_weights

# 要求输入格式为两个等形状的向量，维度为num_samples*num_class
def JS_div(P, Q):
    emb1 = P.shape[1]
    emb2 = Q.shape[1]
    max_emb = max(emb1, emb2)
    if emb1 < max_emb:
        P = np.pad(P, ((0, 0), (0, max_emb - emb1)), 'constant', constant_values=0)
        P /= P.sum(axis=1, keepdims=True)
    if emb2 < max_emb:
        Q = np.pad(Q, ((0, 0), (0, max_emb - emb2)), 'constant', constant_values=0)
        Q /= Q.sum(axis=1, keepdims=True)
    M = 0.5 * (P + Q)
    div1 = np.sum(kl_div(P, M), axis=1)
    div2 = np.sum(kl_div(Q, M), axis=1)
    return 0.5*(div1+div2)


def KL_div(P, Q):
    return np.sum(kl_div(P, Q), axis=1)



def get_embedding(model, loader, device, num_classes, batch_size, with_emb=False, emb_dim=512):
    model.eval()
    embedding = torch.zeros([len(loader.sampler), num_classes])
    embedding_pen = torch.zeros([len(loader.sampler), emb_dim])
    labels = torch.zeros(len(loader.sampler))
    preds = torch.zeros(len(loader.sampler))
    batch_sz = batch_size
    with torch.no_grad():
        for batch_idx, (data, target, _) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            if with_emb:
                # e1的维度为num_cls, e2的维度为emb_dim
                e1, e2 = model(data, with_emb=True)
                embedding_pen[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, e2.shape[0]), :] = e2.cpu()
            else:
                e1 = model(data, with_emb=False)

            embedding[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, e1.shape[0]), :] = e1.cpu()
            labels[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, e1.shape[0])] = target
            preds[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, e1.shape[0])] = e1.argmax(dim=1, keepdim=True).squeeze()

    return embedding, labels, preds, embedding_pen