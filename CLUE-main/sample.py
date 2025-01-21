# -*- coding: utf-8 -*-
"""
Implements active learning sampling strategies
Adapted from https://github.com/ej0cl6/deep-active-learning
"""

import os
import copy
import random
import numpy as np

import scipy
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances
from sklearn.tree import DecisionTreeRegressor
from scipy.special import kl_div

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
from torch.utils.data.sampler import Sampler, SubsetRandomSampler

import utils
from utils import ActualSequentialSampler, JS_div
from adapt.solvers.solver import get_solver
from methods.model_family import *
from Tempreture_Scaling import ModelWithTemperature
from data import DomainDataset
from Train_domain import Discriminator, Train_discriminator

torch.manual_seed(1234)
torch.cuda.manual_seed(1234)
random.seed(1234)
np.random.seed(1234)

al_dict = {}
def register_strategy(name):
    def decorator(cls):
        al_dict[name] = cls
        return cls
    return decorator

def get_strategy(sample, *args):
    if sample not in al_dict: raise NotImplementedError
    return al_dict[sample](*args)

class SamplingStrategy:
    """
    Sampling Strategy wrapper class
    此处的train_idx指的是target数据的index，用于后续训练
    """
    def __init__(self, dset, train_idx, model, discriminator, device, args, balanced=False):
        self.dset = dset
        if dset.name == 'DomainNet':
            self.num_classes = self.dset.get_num_classes()
        else:
            self.num_classes = len(set(dset.targets.numpy()))
        self.train_idx = np.array(train_idx)
        self.model = model
        self.discriminator = discriminator
        self.device = device
        self.args = args
        self.idxs_lb = np.zeros(len(self.train_idx), dtype=bool)
        self.decay = args.weight_decay

    def query(self, n):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = np.copy(idxs_lb)

    def pred(self, idxs=None, with_emb=False):
        if idxs is None:
            idxs = np.arange(len(self.train_idx))[~self.idxs_lb]

        train_sampler = ActualSequentialSampler(self.train_idx[idxs])
        data_loader = torch.utils.data.DataLoader(self.dset, sampler=train_sampler,
                                                  batch_size=self.args.batch_size, drop_last=False)
        self.model.eval()
        all_log_probs = []
        all_scores = []
        all_embs = []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                if with_emb:
                    scores, embs = self.model(data, with_emb=True)
                    all_embs.append(embs.cpu())
                else:
                    scores = self.model(data, with_emb=False)
                log_probs = nn.LogSoftmax(dim=1)(scores)
                all_log_probs.append(log_probs)
                all_scores.append(scores)

        all_log_probs = torch.cat(all_log_probs)
        all_probs = torch.exp(all_log_probs)
        all_scores = torch.cat(all_scores)
        if with_emb:
            all_embs = torch.cat(all_embs)
            return idxs, all_probs, all_log_probs, all_scores, all_embs
        else:
            return idxs, all_probs, all_log_probs, all_scores


    def train(self, target_train_dset, da_round=1, src_loader=None, src_model=None):
        """
        Driver train method
        """
        best_val_acc, best_model = 0.0, None

        train_sampler = SubsetRandomSampler(self.train_idx[self.idxs_lb])
        tgt_sup_loader = torch.utils.data.DataLoader(target_train_dset, sampler=train_sampler, num_workers=4,
                                                     batch_size=self.args.batch_size, drop_last=False)
        tgt_unsup_loader = torch.utils.data.DataLoader(target_train_dset, shuffle=True, num_workers=4,
                                                       batch_size=self.args.batch_size, drop_last=False)
        opt_net_tgt = optim.Adam(self.model.parameters(), lr=self.args.adapt_lr, weight_decay=self.args.wd)

        # Update discriminator adversarially with classifier
        lr_scheduler = optim.lr_scheduler.StepLR(opt_net_tgt, 20, 0.5)
        solver = get_solver(self.args.da_strat, self.model, src_loader, tgt_sup_loader, tgt_unsup_loader,
                            self.train_idx, opt_net_tgt, da_round, self.device, self.args)

        for epoch in range(self.args.adapt_num_epochs):
            if self.args.da_strat == 'dann':
                opt_dis_adapt = optim.Adam(self.discriminator.parameters(), lr=self.args.adapt_lr,
                                           betas=(0.9, 0.999), weight_decay=0)
                solver.solve(epoch, self.discriminator, opt_dis_adapt)
            elif self.args.da_strat in ['ft', 'mme']:
                solver.solve(epoch)
            else:
                raise NotImplementedError

            lr_scheduler.step()

        return self.model

    # get weights for source and target samples
    def get_weight(self, src_emb, tgt_emb):
        if self.args.domain_class == "discriminator_default":
            wA, wB, new_X, new_weights = utils.shared_reweight(src_emb, tgt_emb, self.discriminator, self.device)
        if self.args.domain_class == "discriminator":
            target_unsup_score = torch.tensor(tgt_emb).to(self.device)
            source_train_score = torch.tensor(src_emb).to(self.device)
            target_dom_s = torch.ones(len(source_train_score)).long().to(self.device)
            target_dom_t = torch.zeros(len(target_unsup_score)).long().to(self.device)
            score_concat = torch.cat((source_train_score, target_unsup_score), 0)
            label_concat = torch.cat((target_dom_s, target_dom_t), 0)
            train_discriminator = Train_discriminator(input_dim=self.num_classes, batch_size=self.args.batch_size,
                                                    epochs=self.args.domain_epochs, lr=2e-5,
                                                      device=self.device, hidden_dim=self.args.domain_hidden)
            train_discriminator.fit(score_concat, label_concat)
            wA, wB, new_X, new_weights = utils.shared_reweight(src_emb, tgt_emb, train_discriminator.model, self.device)
        if self.args.domain_class == "xgb":
            wA, wB, new_X, new_weights = utils.shared_reweight(src_emb, tgt_emb, None, self.device)
        return wA, wB, new_X, new_weights

    # annotate samples using ADA-CS
    def reweight(self, source_model, target_model, new_X, new_weights, target_dset, target_weight, n2, method="mlp"):
        # # idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]
        # if self.args.cnn == 'LeNet':
        # 	emb_dim = 500
        # elif self.args.cnn == 'ResNet34':
        # 	emb_dim = 512
        # tgt_emb, tgt_lab, tgt_preds, tgt_pen_emb = utils.get_embedding(self.model, target_train_loader, self.device,
        # 															   self.num_classes,
        # 															   self.args.batch_size, with_emb=True, emb_dim=emb_dim)
        # src_emb, src_lab, src_preds, src_pen_emb = utils.get_embedding(self.model, src_train_loader, self.device,
        # 															   self.num_classes,
        # 															   self.args.batch_size, with_emb=True, emb_dim=emb_dim)
        # target_sup_dset = tgt_pen_emb[self.train_idx[self.idxs_lb_one]]
        # target_unsup_dset = tgt_pen_emb[self.train_idx[~self.idxs_lb_one]]
        # source_train_dset = src_pen_emb
        # wA, wB, new_X, new_weights = self.get_weight(src_emb, tgt_emb)
        # # if self.args.domain_class == "discriminator_default":
        # # 	wA, wB, new_X, new_weights = utils.shared_reweight(src_emb, tgt_emb, self.discriminator, self.device)
        # # if self.args.domain_class == "discriminator":
        # # 	target_unsup_score = tgt_emb[self.train_idx[~self.idxs_lb]].to(self.device)
        # # 	source_train_score = src_emb.to(self.device)
        # # 	target_dom_s = torch.ones(len(source_train_score)).long().to(self.device)
        # # 	target_dom_t = torch.zeros(len(target_unsup_score)).long().to(self.device)
        # # 	score_concat = torch.cat((source_train_score, target_unsup_score), 0)
        # # 	label_concat = torch.cat((target_dom_s, target_dom_t), 0)
        # # 	train_discriminator = Train_discriminator(input_dim=10, batch_size=self.args.batch_size,
        # # 											epochs=self.args.domain_epochs, lr=2e-5,
        # # 											  device=self.device, hidden_dim=self.args.domain_hidden)
        # # 	train_discriminator.fit(score_concat, label_concat)
        # # 	wA, wB, new_X, new_weights = utils.shared_reweight(src_emb, tgt_emb, train_discriminator.model, self.device)
        # # if self.args.domain_class == "xgb":
        # # 	wA, wB, new_X, new_weights = utils.shared_reweight(src_emb, tgt_emb, None, self.device)
        # source_train_index = src_train_loader.sampler.indices
        # # get labeled and unlabeled target data
        # wB_sup = wB[self.train_idx[self.idxs_lb_one]]
        # wB_unsup = wB[self.train_idx[~self.idxs_lb_one]]
        # # get labeled source data and label
        # source_train_label = src_train_loader.dataset.labels[source_train_index]
        # # get labeled target label
        # target_sup_label = target_train_loader.dataset.targets[self.train_idx[self.idxs_lb_one]]
        # # print(wA.shape, wB_sup.shape)
        # source_model = MLPClassifier(input_dim=emb_dim, num_classes=self.num_classes, lr=self.args.mlp_lr1
        # 							 , batch_size=self.args.batch_size, train_epochs= self.args.training_epochs1
        # 							 ,hidden_size=self.args.hidden_size1, device=self.device, decay=self.decay)
        # target_model = MLPClassifier(input_dim=emb_dim, num_classes=self.num_classes, lr=self.args.mlp_lr2
        # 							 , batch_size=self.args.batch_size, train_epochs= self.args.training_epochs2
        # 							 ,hidden_size=self.args.hidden_size2, device=self.device, decay=self.decay)
        #
        # source_model.fit_weight(source_train_dset, source_train_label, wA.reshape(-1))
        # target_model.fit_weight(target_sup_dset, target_sup_label, wB_sup.reshape(-1))
        # if self.args.temperature:
        # 	source_model = ModelWithTemperature(source_model.model, source_model.device).set_temperature(source_model.valid_loader)
        # 	target_model = ModelWithTemperature(target_model.model, target_model.device).set_temperature(target_model.valid_loader)
        #
        # source_acc = source_model.score(source_train_dset, source_train_label)
        # target_acc = source_model.score(target_sup_dset, target_sup_label)
        #
        # print("source model source acc is %.4f" % source_acc)
        # print("source model target acc is %.4f" % target_acc)
        #
        # source_acc2 = target_model.score(source_train_dset, source_train_label)
        # target_acc2 = target_model.score(target_sup_dset, target_sup_label)
        # print("tgt model source acc is %.4f" % source_acc2)
        # print("tgt model target acc is %.4f" % target_acc2)
        #
        # source_pred = (source_model.predict(source_train_dset) == source_train_label)
        # target_pred = (source_model.predict(target_sup_dset) == target_sup_label)
        # sx_source = np.dot(wA, source_pred)
        # sx_target = np.dot(wB_sup, target_pred)
        # proportion = (sx_source-sx_target)/(source_pred-target_pred)
        # print("proportion of Y|X shift is {0}".format(proportion))
        # new_X = np.concatenate([source_train_dset, target_sup_dset], axis=0)
        # new_weights = np.concatenate([wA, wB_sup], axis=0)
        # new_X = target_unsup_dset
        # new_weights = wB_unsup
        # predict_proba中包含softmax层
        idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]
        proba_P2Q = source_model.predict_proba(new_X)
        proba_Q2P = target_model.predict_proba(new_X)
        new_Y = JS_div(proba_P2Q, proba_Q2P)
        sorts = np.argsort(new_Y)
        for i in range(1,4):
            index = sorts[-1*i]
            print(proba_P2Q[index])
            print(proba_Q2P[index])
        region_model = DecisionTreeRegressor(max_depth=20, min_samples_leaf=100, min_samples_split=200,
                                         min_weight_fraction_leaf=0.05, ccp_alpha=0.0001).fit(new_X, new_Y,
                                                                                              sample_weight=new_weights)
        risk_region = region_model.predict(target_dset)
        if self.args.weight:
            risk_region = risk_region*target_weight
        index_maximum = np.argsort(risk_region)[-n2:]
        index_query_second = idxs_unlabeled[index_maximum]
        return index_query_second

    def cal_proportion(self, idx, idxs_lb, src_train_loader, target_train_loader, n2):
        idx_lb_now = np.zeros(len(self.train_idx), dtype=bool)
        idx_lb_now[idx] = True
        labeled_idx_before = np.arange(len(self.train_idx))[self.idxs_lb]
        labeled_idx_now = np.arange(len(self.train_idx))[idx_lb_now]
        unlabeled_idx_before = np.arange(len(self.train_idx))[~self.idxs_lb]
        wB_idx_now = idx_lb_now[~self.idxs_lb]
        self.idxs_lb = np.copy(idxs_lb)
        unlabeled_idx_now = np.arange(len(self.train_idx))[~self.idxs_lb]
        print("length of labeled idx now{0}".format(len(labeled_idx_now)))
        print("length of labeled idx before{0}".format(len(labeled_idx_before)))
        print("length of unlabeled idx now{0}".format(len(unlabeled_idx_now)))
        print("length of unlabeled idx before{0}".format(len(unlabeled_idx_before)))

        if self.args.cnn == 'LeNet':
            emb_dim = 500
        elif self.args.cnn == 'ResNet34':
            emb_dim = 512


        sup_sampler_before = ActualSequentialSampler(self.train_idx[labeled_idx_before])
        sup_loader_before = torch.utils.data.DataLoader(self.dset, sampler=sup_sampler_before, batch_size=self.args.batch_size,
                                                  drop_last=False)
        sup_emb_before, sup_lab_before, _, sup_pen_emb_before = utils.get_embedding(self.model, sup_loader_before, self.device,
                                                                       self.num_classes,
                                                                       self.args.batch_size, with_emb=True, emb_dim=emb_dim)


        sup_sampler_now = ActualSequentialSampler(self.train_idx[labeled_idx_now])
        sup_loader_now = torch.utils.data.DataLoader(self.dset, sampler=sup_sampler_now, num_workers=4,
                                                     batch_size=self.args.batch_size, drop_last=False)
        sup_emb_now, sup_lab_now, _, sup_pen_emb_now = utils.get_embedding(self.model, sup_loader_now, self.device,
                                                                       self.num_classes,
                                                                       self.args.batch_size, with_emb=True, emb_dim=emb_dim)


        unsup_sampler_before = ActualSequentialSampler(self.train_idx[unlabeled_idx_before])
        unsup_loader_before = torch.utils.data.DataLoader(self.dset, sampler=unsup_sampler_before, num_workers=4,
                                                          batch_size=self.args.batch_size, drop_last=False)
        unsup_emb_before, unsup_lab_before, _, unsup_pen_emb_before = utils.get_embedding(self.model, unsup_loader_before, self.device,
                                                                       self.num_classes,
                                                                       self.args.batch_size, with_emb=True, emb_dim=emb_dim)


        unsup_sampler_now = ActualSequentialSampler(self.train_idx[unlabeled_idx_now])
        unsup_loader_now = torch.utils.data.DataLoader(self.dset, sampler=unsup_sampler_now, num_workers=4,
                                                       batch_size=self.args.batch_size, drop_last=False)
        unsup_emb_now, unsup_lab_now, _, unsup_pen_emb_now = utils.get_embedding(self.model, unsup_loader_now, self.device,
                                                                       self.num_classes,
                                                                       self.args.batch_size, with_emb=True, emb_dim=emb_dim)


        src_emb, src_lab, _, src_pen_emb = utils.get_embedding(self.model, src_train_loader, self.device,
                                                                       self.num_classes,
                                                                       self.args.batch_size, with_emb=True, emb_dim=emb_dim)

        # Treat the source identically to the samples selected from the previous iteration
        source_emb = np.concatenate([src_emb, sup_emb_before], axis=0)
        wA, wB, new_X, new_weights = self.get_weight(source_emb, unsup_emb_before)

        source_train_dset = src_pen_emb
        source_train_dset = np.concatenate([source_train_dset, sup_pen_emb_before], axis=0)
        source_train_label = np.concatenate([src_lab, sup_lab_before], axis=0)

        # The selected samples in this round and the total number of unselected samples after these samples are selected are recorded
        wB_sup_now = wB[wB_idx_now]
        wB_unsup_now = wB[~wB_idx_now]

        source_model = MLPClassifier(input_dim=emb_dim, num_classes=self.num_classes, lr=self.args.mlp_lr1
                                     , batch_size=self.args.batch_size, train_epochs= self.args.training_epochs1
                                     ,hidden_size=self.args.hidden_size1, device=self.device, decay=self.decay)
        target_model = MLPClassifier(input_dim=emb_dim, num_classes=self.num_classes, lr=self.args.mlp_lr2
                                     , batch_size=self.args.batch_size, train_epochs= self.args.training_epochs2
                                     ,hidden_size=self.args.hidden_size2, device=self.device, decay=self.decay)

        source_model.fit_weight(source_train_dset, source_train_label, wA.reshape(-1))
        target_model.fit_weight(sup_pen_emb_now, sup_lab_now, wB_sup_now.reshape(-1))
        # The model was calibrated using Temperature Scaling
        if self.args.temperature:
            source_model = ModelWithTemperature(source_model.model, source_model.device).set_temperature(source_model.valid_loader)
            target_model = ModelWithTemperature(target_model.model, target_model.device).set_temperature(target_model.valid_loader)
        source_acc = source_model.score(source_train_dset, source_train_label)
        target_acc = source_model.score(sup_pen_emb_now, sup_lab_now)

        print("source model source acc is %.4f" % source_acc)
        print("source model target acc is %.4f" % target_acc)

        source_acc2 = target_model.score(source_train_dset, source_train_label)
        target_acc2 = target_model.score(sup_pen_emb_now, sup_lab_now)

        print("tgt model source acc is %.4f" % source_acc2)
        print("tgt model target acc is %.4f" % target_acc2)

        new_X = np.concatenate([source_train_dset, sup_pen_emb_now], axis=0)
        new_weights = np.concatenate([wA, wB_sup_now], axis=0)
        index_query_second = self.reweight(source_model, target_model, new_X, new_weights, unsup_pen_emb_now, wB_unsup_now, n2)
        return index_query_second




@register_strategy('uniform')
class RandomSampling(SamplingStrategy):
    """
    Uniform sampling
    """
    def __init__(self, dset, train_idx, model, discriminator, device, args, balanced=False):
        super(RandomSampling, self).__init__(dset, train_idx, model, discriminator, device, args)
        self.labels = dset.labels if dset.name == 'DomainNet' else dset.targets
        self.classes = np.unique(self.labels)
        self.dset = dset
        self.balanced = balanced

    def query(self, n):
        return np.random.choice(np.where(self.idxs_lb==0)[0], n, replace=False)

@register_strategy('AADA')
class AADASampling(SamplingStrategy):
    """
    Implements Active Adversarial Domain Adaptation (https://arxiv.org/abs/1904.07848)
    """
    def __init__(self, dset, train_idx, model, discriminator, device, args, balanced=False):
        super(AADASampling, self).__init__(dset, train_idx, model, discriminator, device, args)
        self.D = None
        self.E = None

    def query(self, n):
        """
        s(x) = frac{1-G*_d}{G_f(x))}{G*_d(G_f(x))} [Diversity] * H(G_y(G_f(x))) [Uncertainty]
        """
        self.model.eval()
        idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]
        train_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])
        data_loader = torch.utils.data.DataLoader(self.dset, sampler=train_sampler, num_workers=4, batch_size=64, drop_last=False)

        # Get diversity and entropy
        all_log_probs, all_scores = [], []
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                scores = self.model(data)
                log_probs = nn.LogSoftmax(dim=1)(scores)
                all_scores.append(scores)
                all_log_probs.append(log_probs)

        all_scores = torch.cat(all_scores)
        all_log_probs = torch.cat(all_log_probs)

        all_probs = torch.exp(all_log_probs)
        disc_scores = nn.Softmax(dim=1)(self.discriminator(all_scores))
        # Compute diversity
        self.D = torch.div(disc_scores[:, 0], disc_scores[:, 1])
        # Compute entropy
        self.E = -(all_probs*all_log_probs).sum(1)
        scores = (self.D*self.E).sort(descending=True)[1]
        # Sample from top-2 % instances, as recommended by authors
        top_N = int(len(scores) * 0.05)
        q_idxs = np.random.choice(scores[:top_N].cpu().numpy(), n, replace=False)

        return idxs_unlabeled[q_idxs]

@register_strategy('BADGE')
class BADGESampling(SamplingStrategy):
    """
    Implements BADGE: Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds (https://arxiv.org/abs/1906.03671)
    """
    def __init__(self, dset, train_idx, model, discriminator, device, args, balanced=False):
        super(BADGESampling, self).__init__(dset, train_idx, model, discriminator, device, args)

    def query(self, n):
        idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]
        train_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])
        data_loader = torch.utils.data.DataLoader(self.dset, sampler=train_sampler, num_workers=4, batch_size=self.args.batch_size, drop_last=False)
        self.model.eval()

        if self.args.cnn == 'LeNet':
            emb_dim = 500
        elif self.args.cnn == 'ResNet34':
            emb_dim = 512

        tgt_emb = torch.zeros([len(data_loader.sampler), self.num_classes])
        tgt_pen_emb = torch.zeros([len(data_loader.sampler), emb_dim])
        tgt_lab = torch.zeros(len(data_loader.sampler))
        tgt_preds = torch.zeros(len(data_loader.sampler))
        batch_sz = self.args.batch_size

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                e1, e2 = self.model(data, with_emb=True)
                tgt_pen_emb[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, e2.shape[0]), :] = e2.cpu()
                tgt_emb[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, e1.shape[0]), :] = e1.cpu()
                tgt_lab[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, e1.shape[0])] = target
                tgt_preds[batch_idx*batch_sz:batch_idx*batch_sz + min(batch_sz, e1.shape[0])] = e1.argmax(dim=1, keepdim=True).squeeze()

        # Compute uncertainty gradient
        tgt_scores = nn.Softmax(dim=1)(tgt_emb)
        tgt_scores_delta = torch.zeros_like(tgt_scores)
        tgt_scores_delta[torch.arange(len(tgt_scores_delta)), tgt_preds.long()] = 1

        # Uncertainty embedding
        badge_uncertainty = (tgt_scores-tgt_scores_delta)

        # Seed with maximum uncertainty example
        max_norm = utils.row_norms(badge_uncertainty.cpu().numpy()).argmax()

        _, q_idxs = utils.kmeans_plus_plus_opt(badge_uncertainty.cpu().numpy(), tgt_pen_emb.cpu().numpy(), n, init=[max_norm])

        return idxs_unlabeled[q_idxs]

@register_strategy('CLUE')
class CLUESampling(SamplingStrategy):
    """
    Implements CLUE: CLustering via Uncertainty-weighted Embeddings
    """
    def __init__(self, dset, train_idx, model, discriminator, device, args, balanced=False):
        super(CLUESampling, self).__init__(dset, train_idx, model, discriminator, device, args)
        self.random_state = np.random.RandomState(1234)
        self.T = self.args.clue_softmax_t
        self.decay = args.weight_decay

    def query(self, n):
        idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]
        train_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])
        data_loader = torch.utils.data.DataLoader(self.dset, sampler=train_sampler, num_workers=4, \
                                                  batch_size=self.args.batch_size, drop_last=False)
        self.model.eval()

        if self.args.cnn == 'LeNet':
            emb_dim = 500
        elif self.args.cnn == 'ResNet34':
            emb_dim = 512

        # Get embedding of target instances
        tgt_emb, tgt_lab, tgt_preds, tgt_pen_emb = utils.get_embedding(self.model, data_loader, self.device, self.num_classes, \
                                                                       self.args.batch_size, with_emb=True, emb_dim=emb_dim)
        tgt_pen_emb = tgt_pen_emb.cpu().numpy()
        tgt_scores = nn.Softmax(dim=1)(tgt_emb / self.T)
        tgt_scores += 1e-8
        sample_weights = -(tgt_scores*torch.log(tgt_scores)).sum(1).cpu().numpy()

        # Run weighted K-means over embeddings
        km = KMeans(n)
        km.fit(tgt_pen_emb, sample_weight=sample_weights)

        # Find nearest neighbors to inferred centroids
        dists = euclidean_distances(km.cluster_centers_, tgt_pen_emb)
        sort_idxs = dists.argsort(axis=1)
        q_idxs = []
        ax, rem = 0, n
        while rem > 0:
            q_idxs.extend(list(sort_idxs[:, ax][:rem]))
            q_idxs = list(set(q_idxs))
            rem = n-len(q_idxs)
            ax += 1
        return idxs_unlabeled[q_idxs]






@register_strategy('CLUE_shift')
class CLUESampling_Shift(SamplingStrategy):
    """
    Implements CLUE: CLustering via Uncertainty-weighted Embeddings
    """

    def __init__(self, dset, train_idx, model, discriminator, device, args, balanced=False):
        super(CLUESampling_Shift, self).__init__(dset, train_idx, model, discriminator, device, args)
        self.random_state = np.random.RandomState(1234)
        self.T = self.args.clue_softmax_t

    def query(self, n):
        idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]
        train_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])
        data_loader = torch.utils.data.DataLoader(self.dset, sampler=train_sampler, num_workers=4, \
                                                  batch_size=self.args.batch_size, drop_last=False)
        self.model.eval()

        if self.args.cnn == 'LeNet':
            emb_dim = 500
        elif self.args.cnn == 'ResNet34':
            emb_dim = 512

        # Get embedding of target instances
        tgt_emb, tgt_lab, tgt_preds, tgt_pen_emb = utils.get_embedding(self.model, data_loader, self.device,
                                                                       self.num_classes,
                                                                       self.args, with_emb=True, emb_dim=emb_dim)
        print(tgt_emb)
        print(tgt_pen_emb)
        tgt_pen_emb = tgt_pen_emb.cpu().numpy()
        tgt_scores = nn.Softmax(dim=1)(tgt_emb / self.T)
        tgt_scores += 1e-8
        sample_weights = -(tgt_scores * torch.log(tgt_scores)).sum(1).cpu().numpy()

        # Run weighted K-means over embeddings
        km = KMeans(n)
        km.fit(tgt_pen_emb, sample_weight=sample_weights)

        # Find nearest neighbors to inferred centroids
        dists = euclidean_distances(km.cluster_centers_, tgt_pen_emb)
        sort_idxs = dists.argsort(axis=1)
        q_idxs = []
        ax, rem = 0, n
        while rem > 0:
            q_idxs.extend(list(sort_idxs[:, ax][:rem]))
            q_idxs = list(set(q_idxs))
            rem = n - len(q_idxs)
            ax += 1
        print(idxs_unlabeled[q_idxs])
        return idxs_unlabeled[q_idxs]


@register_strategy('kmeans')
class KmeansSampling(SamplingStrategy):
    """
    Implements CLUE: CLustering via Uncertainty-weighted Embeddings
    """

    def __init__(self, dset, train_idx, model, discriminator, device, args, balanced=False):
        super(KmeansSampling, self).__init__(dset, train_idx, model, discriminator, device, args)

    def query(self, n):
        idxs_unlabeled, _, _, _, all_embs = self.pred(with_emb=True)
        all_embs = all_embs.numpy()

        # Run weighted K-means over embeddings
        km = KMeans(n_clusters=n)
        km.fit(all_embs)

        # use below code to match CLUE implementation
        # Find nearest neighbors to inferred centroids
        dists = euclidean_distances(km.cluster_centers_, all_embs)
        sort_idxs = dists.argsort(axis=1)
        q_idxs = []
        ax, rem = 0, n
        while rem > 0:
            q_idxs.extend(list(sort_idxs[:, ax][:rem]))
            q_idxs = list(set(q_idxs))
            rem = n - len(q_idxs)
            ax += 1

        return idxs_unlabeled[q_idxs]


@register_strategy('entropy')
class EntropySampling(SamplingStrategy):
    """
    Implements entropy based sampling
    """

    def __init__(self, dset, train_idx, model, discriminator, device, args, balanced=False):
        super(EntropySampling, self).__init__(dset, train_idx, model, discriminator, device, args)

    def query(self, n):
        idxs_unlabeled, all_probs, all_log_probs, _ = self.pred()
        # Compute entropy
        entropy = -(all_probs * all_log_probs).sum(1)
        q_idxs = (entropy).sort(descending=True)[1][:n]
        q_idxs = q_idxs.cpu().numpy()
        return idxs_unlabeled[q_idxs]

@register_strategy('margin')
class MarginSampling(SamplingStrategy):
    """
    Implements margin based sampling
    """

    def __init__(self, dset, train_idx, model, discriminator, device, args, balanced=False):
        super(MarginSampling, self).__init__(dset, train_idx, model, discriminator, device, args)

    def query(self, n):
        idxs_unlabeled, all_probs, _, _ = self.pred()
        # Compute BvSB margin
        top2 = torch.topk(all_probs, 2).values
        BvSB_scores = 1-(top2[:, 0] - top2[:, 1])  # use minus for descending sorting
        q_idxs = (BvSB_scores).sort(descending=True)[1][:n]
        q_idxs = q_idxs.cpu().numpy()
        return idxs_unlabeled[q_idxs]


@register_strategy('leastConfidence')
class LeastConfidenceSampling(SamplingStrategy):
    def __init__(self, dset, train_idx, model, discriminator, device, args, balanced=False):
        super(LeastConfidenceSampling, self).__init__(dset, train_idx, model, discriminator, device, args)

    def query(self, n):
        idxs_unlabeled, all_probs, _, _ = self.pred()
        confidences = -all_probs.max(1)[0]  # use minus for descending sorting
        q_idxs = (confidences).sort(descending=True)[1][:n]
        q_idxs = q_idxs.cpu().numpy()
        return idxs_unlabeled[q_idxs]


@register_strategy('coreset')
class CoreSetSampling(SamplingStrategy):
    def __init__(self, dset, train_idx, model, discriminator, device, args, balanced=False):
        super(CoreSetSampling, self).__init__(dset, train_idx, model, discriminator, device, args)

    def furthest_first(self, X, X_lb, n):
        m = np.shape(X)[0]
        if np.shape(X_lb)[0] == 0:
            min_dist = np.tile(float("inf"), m)
        else:
            dist_ctr = pairwise_distances(X, X_lb)
            min_dist = np.amin(dist_ctr, axis=1)

        idxs = []

        for i in range(n):
            idx = min_dist.argmax()
            idxs.append(idx)
            dist_new_ctr = pairwise_distances(X, X[[idx], :])
            for j in range(m):
                min_dist[j] = min(min_dist[j], dist_new_ctr[j, 0])

        return idxs

    def query(self, n):
        idxs = np.arange(len(self.train_idx))
        idxs_unlabeled, _, _, _, all_embs = self.pred(idxs=idxs, with_emb=True)
        all_embs = all_embs.numpy()
        q_idxs = self.furthest_first(all_embs[~self.idxs_lb, :], all_embs[self.idxs_lb, :], n)
        return idxs_unlabeled[~self.idxs_lb][q_idxs]

@register_strategy('LAS')
class LASSampling(SamplingStrategy):
    '''
    Implement Local context-aware sampling (LAS)
    '''

    def __init__(self, dset, train_idx, model, discriminator, device, args, balanced=False):
        super(LASSampling, self).__init__(dset, train_idx, model, discriminator, device, args)

    def query(self, n):
        idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]
        train_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])
        data_loader = torch.utils.data.DataLoader(self.dset, sampler=train_sampler, num_workers=4, \
                                                  batch_size=self.args.batch_size, drop_last=False)
        # build nearest neighbors
        self.model.eval()
        all_probs = []
        all_embs = []
        with torch.no_grad():
            for batch_idx, (data, target, *_) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                scores, embs = self.model(data, with_emb=True)
                all_embs.append(embs.cpu())
                probs = F.softmax(scores, dim=-1)
                all_probs.append(probs.cpu())

        all_probs = torch.cat(all_probs)
        all_embs = F.normalize(torch.cat(all_embs), dim=-1)

        # get Q_score
        sim = all_embs.cpu().mm(all_embs.transpose(1, 0))
        K = self.args.s_k
        sim_topk, topk = torch.topk(sim, k=K + 1, dim=1)
        sim_topk, topk = sim_topk[:, 1:], topk[:, 1:]
        wgt_topk = (sim_topk / sim_topk.sum(dim=1, keepdim=True))

        Q_score = -((all_probs[topk] * all_probs.unsqueeze(1)).sum(-1) * wgt_topk).sum(-1)

        # propagate Q_score
        for i in range(self.args.s_prop_iter):
            Q_score += (wgt_topk * Q_score[topk]).sum(-1) * self.args.s_prop_coef

        m_idxs = Q_score.sort(descending=True)[1]

        # oversample and find centroids
        M = self.args.s_m
        m_topk = m_idxs[:n * (1 + M)]
        km = KMeans(n_clusters=n)
        km.fit(all_embs[m_topk])
        dists = euclidean_distances(km.cluster_centers_, all_embs[m_topk])
        sort_idxs = dists.argsort(axis=1)
        q_idxs = []
        ax, rem = 0, n
        while rem > 0:
            q_idxs.extend(list(sort_idxs[:, ax][:rem]))
            q_idxs = list(set(q_idxs))
            rem = n - len(q_idxs)
            ax += 1

        q_idxs = m_idxs[q_idxs].cpu().numpy()
        self.dset.rand_transform = None

        return idxs_unlabeled[q_idxs]


class MHPSampling(SamplingStrategy):
    '''
    Implements MHPL: Minimum Happy Points Learning for Active Source Free Domain Adaptation (CVPR'23)
    '''

    def __init__(self, dset, train_idx, model, discriminator, device, args):
        super(MHPSampling, self).__init__(dset, train_idx, model, discriminator, device, args)

    def query(self, n):
        idxs_unlabeled = np.arange(len(self.train_idx))[~self.idxs_lb]
        train_sampler = ActualSequentialSampler(self.train_idx[idxs_unlabeled])
        data_loader = torch.utils.data.DataLoader(self.dset, sampler=train_sampler, num_workers=4,
                                                  batch_size=self.args.batch_size, drop_last=False)
        self.model.eval()
        all_probs = []
        all_embs = []
        with torch.no_grad():
            for batch_idx, (data, target, *_) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                scores, embs = self.model(data, with_emb=True)
                all_embs.append(embs.cpu())
                probs = F.softmax(scores, dim=-1)
                all_probs.append(probs.cpu())

        all_probs = torch.cat(all_probs)
        all_embs = F.normalize(torch.cat(all_embs), dim=-1)

        # find KNN
        sim = all_embs.cpu().mm(all_embs.transpose(1, 0))
        K = self.args.s_k
        sim_topk, topk = torch.topk(sim, k=K + 1, dim=1)
        sim_topk, topk = sim_topk[:, 1:], topk[:, 1:]
        sim_topk = sim_topk.to(self.device)
        topk = topk.to(self.device)

        all_probs = all_probs.to(self.device)
        all_embs = all_embs.to(self.device)

        # get NP scores
        all_preds = all_probs.argmax(-1)
        Sp = (torch.eye(self.num_classes).to(self.device)[all_preds[topk]]).sum(1)
        Sp = Sp / Sp.sum(-1, keepdim=True)
        NP = -(torch.log(Sp+1e-9)*Sp).sum(-1)

        # get NA scores
        NA = sim_topk.sum(-1) / K
        NAU = NP*NA
        sort_idxs = NAU.argsort(descending=True)

        q_idxs = []
        ax, rem = 0, n
        while rem > 0:
            if topk[sort_idxs[ax]][0] not in q_idxs:
                q_idxs.append(sort_idxs[ax].cpu())
            rem = n - len(q_idxs)
            ax += 1
        q_idxs = np.array(q_idxs)

        return idxs_unlabeled[q_idxs]


al_dict['MHP'] = MHPSampling