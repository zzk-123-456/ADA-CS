# -*- coding: utf-8 -*-
"""
Implements active learning sampling strategies
Adapted from https://github.com/ej0cl6/deep-active-learning
"""

import os
import copy
import random
import numpy as np

from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from torch.utils.data import DataLoader
import logging

import utils.utils as utils
from solver import get_solver
from model import get_model
from dataset.image_list import ImageList
from .utils import row_norms, kmeans_plus_plus_opt, get_embedding, ActualSequentialSampler
from utils.utils import JS_div, KL_div

from Tempreture_Scaling import ModelWithTemperature
from Train_domain import Train_discriminator
from methods.model_family import *
from calculate_shift import get_weights
import warnings
warnings.filterwarnings("ignore")

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
    # print(al_dict)
    if sample not in al_dict: raise NotImplementedError
    return al_dict[sample](*args)


class SamplingStrategy:
    """
    Sampling Strategy wrapper class
    """

    def __init__(self, src_dset, tgt_dset, source_model, device, num_classes, cfg):
        self.src_dset = src_dset
        self.tgt_dset = tgt_dset
        self.num_classes = num_classes
        self.model = copy.deepcopy(source_model) # initialized with source model
        self.device = device
        self.cfg = cfg
        self.discrim = nn.Sequential(
                    nn.Linear(self.cfg.DATASET.NUM_CLASS, 500),
                    nn.ReLU(),
                    nn.Linear(500, 500),
                    nn.ReLU(),
                    nn.Linear(500, 2)).to(self.device) # should be initialized by running train_uda
        self.idxs_lb = np.zeros(len(self.tgt_dset.train_idx), dtype=bool)
        self.solver = None
        self.lr_scheduler = None
        self.opt_discrim = None
        self.opt_net_tgt = None
        self.query_dset = tgt_dset.get_dsets()[1] # change to query dataset

    def query(self, n, epoch):
        pass

    def update(self, idxs_lb):
        self.idxs_lb = np.copy(idxs_lb)

    def pred(self, idxs=None, with_emb=False):
        if idxs is None:
            idxs = np.arange(len(self.tgt_dset.train_idx))[~self.idxs_lb]

        train_sampler = ActualSequentialSampler(self.tgt_dset.train_idx[idxs])
        data_loader = torch.utils.data.DataLoader(self.query_dset, sampler=train_sampler, num_workers=self.cfg.DATALOADER.NUM_WORKERS,
                                                  batch_size=self.cfg.DATALOADER.BATCH_SIZE, drop_last=False)
        self.model.eval()
        all_log_probs = []
        all_scores = []
        all_embs = []
        with torch.no_grad():
            for batch_idx, (data, target, _) in enumerate(data_loader):
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

    def train_uda(self, epochs=1):
        """
            Unsupervised adaptation of source model to target at round 0
            Returns:
                Model post adaptation
        """
        source = self.cfg.DATASET.SOURCE_DOMAIN
        target = self.cfg.DATASET.TARGET_DOMAIN
        uda_strat = self.cfg.ADA.UDA

        adapt_dir = os.path.join('checkpoints', 'adapt')
        adapt_net_file = os.path.join(adapt_dir, '{}_{}_{}_{}_{}.pth'.format(uda_strat, source, target,
                                                        self.cfg.MODEL.BACKBONE.NAME, self.cfg.TRAINER.MAX_UDA_EPOCHS))

        if not os.path.exists(adapt_dir):
            os.makedirs(adapt_dir)

        if self.cfg.TRAINER.LOAD_FROM_CHECKPOINT and os.path.exists(adapt_net_file):
            logging.info('Found pretrained uda checkpoint, loading...')
            adapt_model = get_model('AdaptNet', num_cls=self.num_classes, weights_init=adapt_net_file,
                                    model=self.cfg.MODEL.BACKBONE.NAME)
        else:
            src_loader, tgt_loader, tgt_sup_loader, tgt_unsup_loader, joint_sup_loader = self.build_loaders()

            src_train_loader = self.src_dset.get_loaders()[0]
            target_train_dset = self.tgt_dset.get_dsets()[0]
            train_sampler = SubsetRandomSampler(self.tgt_dset.train_idx[self.idxs_lb])
            tgt_sup_loader = torch.utils.data.DataLoader(target_train_dset, sampler=train_sampler,
                                                         num_workers=self.cfg.DATALOADER.NUM_WORKERS, \
                                                         batch_size=self.cfg.DATALOADER.BATCH_SIZE, drop_last=False)
            tgt_unsup_loader = torch.utils.data.DataLoader(target_train_dset, shuffle=True,
                                                           num_workers=self.cfg.DATALOADER.NUM_WORKERS, \
                                                           batch_size=self.cfg.DATALOADER.BATCH_SIZE, drop_last=False)

            logging.info('No pretrained checkpoint found, training...')
            adapt_model = get_model('AdaptNet', num_cls=self.num_classes, src_weights_init=self.model,
                                    model=self.cfg.MODEL.BACKBONE.NAME, normalize=self.cfg.MODEL.NORMALIZE, temp=self.cfg.MODEL.TEMP)
            opt_net_tgt = utils.get_optim(self.cfg.OPTIM.UDA_NAME, adapt_model.tgt_net.parameters(self.cfg.OPTIM.UDA_LR, self.cfg.OPTIM.BASE_LR_MULT),
                                         lr=self.cfg.OPTIM.UDA_LR)
            uda_solver = get_solver(uda_strat, adapt_model.tgt_net, src_loader, tgt_loader, tgt_sup_loader, tgt_unsup_loader,
                       joint_sup_loader, opt_net_tgt, False, self.device, self.cfg)

            for epoch in range(epochs):
                print("Running uda epoch {}/{}".format(epoch, epochs))
                if uda_strat in ['dann']:
                    opt_discrim = optim.Adadelta(adapt_model.discrim.parameters(), lr=self.cfg.OPTIM.UDA_LR)
                    uda_solver.solve(epoch, adapt_model.discrim, opt_discrim)
                elif uda_strat in ['mme']:
                    uda_solver.solve(epoch)
                else:
                    logging.info('Warning: no uda training with {}, skipped'.format(uda_strat))
                    return self.model

            adapt_model.save(adapt_net_file)

        self.model = adapt_model.tgt_net
        return self.model

    def build_loaders(self):
        src_loader = self.src_dset.get_loaders()[0]
        tgt_loader = self.tgt_dset.get_loaders()[0]

        target_train_dset = self.tgt_dset.get_dsets()[0]
        train_sampler = SubsetRandomSampler(self.tgt_dset.train_idx[self.idxs_lb])
        tgt_sup_loader = DataLoader(target_train_dset, sampler=train_sampler,
                                                     num_workers=self.cfg.DATALOADER.NUM_WORKERS, \
                                                     batch_size=self.cfg.DATALOADER.BATCH_SIZE, drop_last=False)
        train_sampler = SubsetRandomSampler(self.tgt_dset.train_idx[~self.idxs_lb])
        tgt_unsup_loader = DataLoader(target_train_dset, sampler=train_sampler,
                                                     num_workers=self.cfg.DATALOADER.NUM_WORKERS, \
                                                     batch_size=self.cfg.DATALOADER.BATCH_SIZE*self.cfg.DATALOADER.TGT_UNSUP_BS_MUL,
                                                     drop_last=False)

        # create joint src_tgt_sup loader as commonly used
        joint_list = [self.src_dset.train_dataset.samples[_] for _ in self.src_dset.train_idx] + \
                        [self.tgt_dset.train_dataset.samples[_] for _ in self.tgt_dset.train_idx[self.idxs_lb]]

        # use source train transform
        join_transform = self.src_dset.get_dsets()[0].transform
        joint_train_ds = ImageList(joint_list, root=self.cfg.DATASET.ROOT, transform=join_transform)
        joint_sup_loader = DataLoader(joint_train_ds, batch_size=self.cfg.DATALOADER.BATCH_SIZE, shuffle=True,
                                          drop_last=False, num_workers=self.cfg.DATALOADER.NUM_WORKERS)

        return src_loader, tgt_loader, tgt_sup_loader, tgt_unsup_loader, joint_sup_loader

    def train(self, epoch):
        src_loader, tgt_loader, tgt_sup_loader, tgt_unsup_loader, joint_sup_loader = self.build_loaders()

        if self.opt_net_tgt is None:
            self.opt_net_tgt = utils.get_optim(self.cfg.OPTIM.NAME, self.model.parameters(self.cfg.OPTIM.ADAPT_LR,
                                   self.cfg.OPTIM.BASE_LR_MULT), lr=self.cfg.OPTIM.ADAPT_LR, weight_decay=0.00001)

        if self.opt_discrim is None:
            self.opt_discrim = utils.get_optim(self.cfg.OPTIM.NAME, self.discrim.parameters(), lr=self.cfg.OPTIM.ADAPT_LR, weight_decay=0)

        solver = get_solver(self.cfg.ADA.DA, self.model, src_loader, tgt_loader, tgt_sup_loader, tgt_unsup_loader,
                                     joint_sup_loader, self.opt_net_tgt, True, self.device, self.cfg)

        if self.cfg.ADA.DA in ['dann']:
            solver.solve(epoch, self.discrim, self.opt_discrim)
        elif self.cfg.ADA.DA in ['LAA', 'RAA']:
            train_sampler = ActualSequentialSampler(self.tgt_dset.train_idx)
            seq_query_loader = torch.utils.data.DataLoader(self.query_dset, sampler=train_sampler,
                                                           num_workers=self.cfg.DATALOADER.NUM_WORKERS,
                                                           batch_size=self.cfg.DATALOADER.BATCH_SIZE, drop_last=False)
            solver.solve(epoch, seq_query_loader)
        else:
            solver.solve(epoch)


        return self.model

    # get weights for source and target samples
    def get_weight(self, src_emb, tgt_emb):
        target_unsup_score = torch.tensor(tgt_emb).to(self.device)
        source_train_score = torch.tensor(src_emb).to(self.device)
        target_dom_s = torch.ones(len(source_train_score)).long().to(self.device)
        target_dom_t = torch.zeros(len(target_unsup_score)).long().to(self.device)
        score_concat = torch.cat((source_train_score, target_unsup_score), 0)
        label_concat = torch.cat((target_dom_s, target_dom_t), 0)
        train_discriminator = Train_discriminator(input_dim=self.num_classes, batch_size=self.cfg.DATALOADER.BATCH_SIZE,
                                                  epochs=self.cfg.SECOND.DIS_EPOCHS, lr=self.cfg.SECOND.DIS_LR,
                                                  device=self.device, hidden_dim=self.cfg.SECOND.DIS_HIDDENS)
        train_discriminator.fit(score_concat, label_concat)
        wA, wB, new_X, new_weights = utils.shared_reweight(src_emb, tgt_emb, train_discriminator.model, self.device)
        return wA, wB, new_X, new_weights

    # annotate samples using ADA-CS
    def reweight(self, source_model, target_model, new_X, new_weights, target_dset, target_weight, n2):
        idxs_unlabeled = np.arange(len(self.tgt_dset.train_idx))[~self.idxs_lb]
        proba_P2Q = source_model.predict_proba(new_X)
        proba_Q2P = target_model.predict_proba(new_X)
        alpha = self.cfg.SECOND.ALPHA

        if self.cfg.SECOND.DIV == 'js':
            new_Y = JS_div(proba_P2Q, proba_Q2P)
        if self.cfg.SECOND.DIV == 'kl':
            new_Y = KL_div(proba_P2Q, proba_Q2P)

        if self.cfg.SECOND.REGION == 'lasso':
            lasso_model = Lasso(alpha=alpha)
            lasso_model.fit(new_X, new_Y, new_weights)
            risk_region = lasso_model.predict(target_dset)
        if self.cfg.SECOND.REGION == 'ridge':
            ridge_model = Ridge(alpha=alpha)
            ridge_model.fit(new_X, new_Y, new_weights)
            risk_region = ridge_model.predict(target_dset)
        if self.cfg.SECOND.REGION == 'direct':
            proba_P2Q_ = source_model.predict_proba(target_dset)
            proba_Q2P_ = target_model.predict_proba(target_dset)
            risk_region = JS_div(proba_P2Q_, proba_Q2P_)

        if self.cfg.SECOND.REGION == 'tree':
            region_model = RandomForestRegressor(n_estimators=100, random_state=42)
            region_model.fit(new_X, new_Y, new_weights)
            risk_region = region_model.predict(target_dset)

        if self.cfg.SECOND.WEIGHT:
            risk_region = risk_region*target_weight
        index_maximum = np.argsort(risk_region)[-n2:]
        index_query_second = idxs_unlabeled[index_maximum]
        return index_query_second


    def cal_proportion(self, idx, idxs_lb, src_train_loader, target_train_loader, n2):
        idx_lb_now = np.zeros(len(self.tgt_dset.train_idx), dtype=bool)
        idx_lb_now[idx] = True
        labeled_idx_before = np.arange(len(self.tgt_dset.train_idx))[self.idxs_lb]
        labeled_idx_now = np.arange(len(self.tgt_dset.train_idx))[idx_lb_now]
        unlabeled_idx_before = np.arange(len(self.tgt_dset.train_idx))[~self.idxs_lb]
        wB_idx_now = idx_lb_now[~self.idxs_lb]
        self.idxs_lb = np.copy(idxs_lb)
        unlabeled_idx_now = np.arange(len(self.tgt_dset.train_idx))[~self.idxs_lb]
        print("length of labeled idx now{0}".format(len(labeled_idx_now)))
        print("length of labeled idx before{0}".format(len(labeled_idx_before)))
        print("length of unlabeled idx now{0}".format(len(unlabeled_idx_now)))
        print("length of unlabeled idx before{0}".format(len(unlabeled_idx_before)))

        if self.cfg.MODEL.BACKBONE.NAME == "ResNet34Fc":
            emb_dim = 512
        if self.cfg.MODEL.BACKBONE.NAME == "ResNet50Fc":
            emb_dim = 256



        sup_sampler_before = ActualSequentialSampler(self.tgt_dset.train_idx[labeled_idx_before])
        sup_loader_before = torch.utils.data.DataLoader(self.query_dset, sampler=sup_sampler_before,
                                                        batch_size=self.cfg.DATALOADER.BATCH_SIZE,
                                                        drop_last=False)
        sup_emb_before, sup_lab_before, _, sup_pen_emb_before = utils.get_embedding(self.model, sup_loader_before,
                                                                                    self.device,
                                                                                    self.num_classes,
                                                                                    self.cfg.DATALOADER.BATCH_SIZE, with_emb=True,
                                                                                    emb_dim=emb_dim)

        sup_sampler_now = ActualSequentialSampler(self.tgt_dset.train_idx[labeled_idx_now])
        sup_loader_now = torch.utils.data.DataLoader(self.query_dset, sampler=sup_sampler_now, num_workers=4,
                                                     batch_size=self.cfg.DATALOADER.BATCH_SIZE, drop_last=False)
        sup_emb_now, sup_lab_now, _, sup_pen_emb_now = utils.get_embedding(self.model, sup_loader_now, self.device,
                                                                           self.num_classes,
                                                                           self.cfg.DATALOADER.BATCH_SIZE, with_emb=True,
                                                                           emb_dim=emb_dim)

        unsup_sampler_before = ActualSequentialSampler(self.tgt_dset.train_idx[unlabeled_idx_before])
        unsup_loader_before = torch.utils.data.DataLoader(self.query_dset, sampler=unsup_sampler_before, num_workers=4,
                                                          batch_size=self.cfg.DATALOADER.BATCH_SIZE, drop_last=False)
        unsup_emb_before, unsup_lab_before, _, unsup_pen_emb_before = utils.get_embedding(self.model, unsup_loader_before,
                                                                                          self.device,
                                                                                          self.num_classes,
                                                                                          self.cfg.DATALOADER.BATCH_SIZE,
                                                                                          with_emb=True, emb_dim=emb_dim)

        unsup_sampler_now = ActualSequentialSampler(self.tgt_dset.train_idx[unlabeled_idx_now])
        unsup_loader_now = torch.utils.data.DataLoader(self.query_dset, sampler=unsup_sampler_now, num_workers=4,
                                                       batch_size=self.cfg.DATALOADER.BATCH_SIZE, drop_last=False)
        unsup_emb_now, unsup_lab_now, _, unsup_pen_emb_now = utils.get_embedding(self.model, unsup_loader_now, self.device,
                                                                                 self.num_classes,
                                                                                 self.cfg.DATALOADER.BATCH_SIZE, with_emb=True,
                                                                                 emb_dim=emb_dim)

        src_emb, src_lab, _, src_pen_emb = utils.get_embedding(self.model, src_train_loader, self.device,
                                                               self.num_classes,
                                                               self.cfg.DATALOADER.BATCH_SIZE, with_emb=True, emb_dim=emb_dim)
        # Treat the source identically to the samples selected from the previous iteration
        source_emb = np.concatenate([src_emb, sup_emb_before], axis=0)
        wA, wB, new_X, new_weights = self.get_weight(source_emb, unsup_emb_before)
        print(np.sort(wA))
        print(np.sort(wB))

        source_train_dset = src_pen_emb
        source_train_dset = np.concatenate([source_train_dset, sup_pen_emb_before], axis=0)
        source_train_label = np.concatenate([src_lab, sup_lab_before], axis=0)


        # target_train_dset = sup_emb_now
        target_train_dset = sup_pen_emb_now

        # The selected samples in this round and the total number of unselected samples after these samples are selected are recorded
        wB_sup_now = wB[wB_idx_now]
        wB_unsup_now = wB[~wB_idx_now]
        wB_sup_now /= np.sum(wB_sup_now)
        wB_unsup_now /= np.sum(wB_unsup_now)

        source_model = MLPClassifier(input_dim=emb_dim, num_classes=self.num_classes, lr=self.cfg.SECOND.MLP_LR1
                                     , batch_size=self.cfg.DATALOADER.BATCH_SIZE, train_epochs=self.cfg.SECOND.EPOCHS_1
                                     , hidden_size=self.cfg.SECOND.HIDDENS_1, device=self.device, decay=self.cfg.SECOND.DECAY)
        # target_model = MLPClassifier(input_dim=emb_dim, num_classes=self.num_classes, lr=self.cfg.SECOND.MLP_LR2
        #                              , batch_size=self.cfg.DATALOADER.BATCH_SIZE, train_epochs=self.cfg.SECOND.EPOCHS_2
        #                              , hidden_size=self.cfg.SECOND.HIDDENS_2, device=self.device, decay=self.cfg.SECOND.DECAY)

        source_model.fit_weight(source_train_dset, source_train_label, wA.reshape(-1))
        # target_model.fit_weight(sup_pen_emb_now, sup_lab_now, wB_sup_now.reshape(-1))
        target_model = copy.deepcopy(source_model)
        target_model.lr = self.cfg.SECOND.MLP_LR2
        target_model.train_epochs = self.cfg.SECOND.EPOCHS_2
        target_model.fit_weight(target_train_dset, sup_lab_now, wB_sup_now.reshape(-1))
        if self.cfg.SECOND.TEM:
            source_model = ModelWithTemperature(source_model.model, source_model.device).set_temperature(
                source_model.valid_loader)
            target_model = ModelWithTemperature(target_model.model, target_model.device).set_temperature(
                target_model.valid_loader)

        source_acc = source_model.score(source_train_dset, source_train_label)
        target_acc = source_model.score(target_train_dset, sup_lab_now)

        print("source model source acc is %.4f" % source_acc)
        print("source model target acc is %.4f" % target_acc)

        source_acc2 = target_model.score(source_train_dset, source_train_label)
        target_acc2 = target_model.score(target_train_dset, sup_lab_now)
        print("tgt model source acc is %.4f" % source_acc2)
        print("tgt model target acc is %.4f" % target_acc2)

        new_X = np.concatenate([source_train_dset, target_train_dset, unsup_pen_emb_now], axis=0)
        # new_weights = np.concatenate([wA, wB], axis=0)
        # new_weights /= np.sum(new_weights)

        target_query_dset = unsup_pen_emb_now
        # target_query_dset = unsup_emb_now

        index_query_second = self.reweight(source_model, target_model, new_X, new_weights, target_query_dset, wB_unsup_now, n2)
        return index_query_second

    # calculate CSS
    def calcu_shift(self, src_train_loader):
        if self.cfg.MODEL.BACKBONE.NAME == "ResNet34Fc":
            emb_dim = 512
        if self.cfg.MODEL.BACKBONE.NAME == "ResNet50Fc":
            emb_dim = 256
        source_model2 = get_model(self.cfg.MODEL.BACKBONE.NAME, num_cls=self.cfg.DATASET.NUM_CLASS, normalize=self.cfg.MODEL.NORMALIZE,
                                 temp=self.cfg.MODEL.TEMP).to(self.device)
        source_file = '{}_{}_source_{}.pth'.format(self.cfg.DATASET.SOURCE_DOMAIN, self.cfg.MODEL.BACKBONE.NAME,
                                                   self.cfg.TRAINER.MAX_SOURCE_EPOCHS)
        source_dir = os.path.join('checkpoints', 'source')
        source_path = os.path.join(source_dir, source_file)
        source_model2.load_state_dict(torch.load(source_path, map_location=self.device), strict=False)
        src_loader2, tgt_loader2, tgt_sup_loader2, tgt_unsup_loader2, joint_sup_loader2 = self.build_loaders()
        for epochs in range(20):
            source_model2.train()
            tgt_sup_iter2 = iter(tgt_sup_loader2)
            opt_net_tgt2 = utils.get_optim(self.cfg.OPTIM.NAME, source_model2.parameters(self.cfg.OPTIM.ADAPT_LR,
                                   self.cfg.OPTIM.BASE_LR_MULT), lr=self.cfg.OPTIM.ADAPT_LR, weight_decay=0.00001)
            while True:
                try:
                    data_t, target_t, _ = next(tgt_sup_iter2)
                    data_t, target_t = data_t.to(self.device), target_t.to(self.device)
                except:
                    break

                opt_net_tgt2.zero_grad()
                output = source_model2(data_t)
                loss = nn.CrossEntropyLoss()(output, target_t)
                loss.backward()
                opt_net_tgt2.step()
        src_train_loader2, src_valid_loader2, src_test_loader2 = src_train_loader.get_loaders(valid_type='val', valid_ratio=1.0)
        tgt_train_loader2, tgt_valid_loader2, tgt_test_loader2 = self.tgt_dset.get_loaders(valid_type='val', valid_ratio=1.0)
        src_emb2, src_lab2, src_pred2, src_pen_emb2 = utils.get_embedding(source_model2, src_test_loader2, self.device, self.num_classes,
                                                               self.cfg.DATALOADER.BATCH_SIZE,
                                                               with_emb=True,
                                                               emb_dim=emb_dim)
        tgt_emb2, tgt_lab2, tgt_pred2, tgt_pen_emb2 = utils.get_embedding(source_model2, tgt_test_loader2, self.device, self.num_classes,
                                                                self.cfg.DATALOADER.BATCH_SIZE,
                                                                with_emb=True,
                                                                emb_dim=emb_dim)
        wA, wB, new_X, new_weights = get_weights(src_emb2, tgt_emb2)
        wA = wA / np.sum(wA)
        wB = wB / np.sum(wB)

        src_src_acc, src_src_loss = utils.test(source_model2, self.device, src_test_loader2)
        src_tgt_acc, src_tgt_loss = utils.test(source_model2, self.device, tgt_test_loader2)
        src_src_pred = (src_pred2 == src_lab2)
        src_tgt_pred = (tgt_pred2 == tgt_lab2)
        src_sx_acc = np.dot(src_src_pred, wA)
        sx_tgt_acc = np.dot(src_tgt_pred, wB)
        print(src_src_acc, src_sx_acc, sx_tgt_acc)
        # conditional_shift = (src_sx_acc - sx_tgt_acc) / (src_src_acc / 100 - src_tgt_acc / 100)
        conditional_shift = src_sx_acc - sx_tgt_acc
        print(f'conditional shift is {conditional_shift}')


@register_strategy('random')
class RandomSampling(SamplingStrategy):
    """
    Uniform sampling
    """

    def __init__(self, src_dset, tgt_dset, model, device, num_classes, cfg):
        super(RandomSampling, self).__init__(src_dset, tgt_dset, model, device, num_classes, cfg)

    def query(self, n, epoch):
        return np.random.choice(np.where(self.idxs_lb == 0)[0], n, replace=False)



@register_strategy('entropy')
class EntropySampling(SamplingStrategy):
    """
    Implements entropy based sampling
    """

    def __init__(self, src_dset, tgt_dset, model, device, num_classes, cfg):
        super(EntropySampling, self).__init__(src_dset, tgt_dset, model, device, num_classes, cfg)

    def query(self, n, epoch):
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

    def __init__(self, src_dset, tgt_dset, model, device, num_classes, cfg):
        super(MarginSampling, self).__init__(src_dset, tgt_dset, model, device, num_classes, cfg)

    def query(self, n, epoch):
        idxs_unlabeled, all_probs, _, _ = self.pred()
        # Compute BvSB margin
        top2 = torch.topk(all_probs, 2).values
        BvSB_scores = 1-(top2[:,0] - top2[:,1]) # use minus for descending sorting
        q_idxs = (BvSB_scores).sort(descending=True)[1][:n]
        q_idxs = q_idxs.cpu().numpy()
        return idxs_unlabeled[q_idxs]


@register_strategy('leastConfidence')
class LeastConfidenceSampling(SamplingStrategy):
    def __init__(self, src_dset, tgt_dset, model, device, num_classes, cfg):
        super(LeastConfidenceSampling, self).__init__(src_dset, tgt_dset, model, device, num_classes, cfg)

    def query(self, n, epoch):
        idxs_unlabeled, all_probs, _, _ = self.pred()
        confidences = -all_probs.max(1)[0] # use minus for descending sorting
        q_idxs = (confidences).sort(descending=True)[1][:n]
        q_idxs = q_idxs.cpu().numpy()
        return idxs_unlabeled[q_idxs]


@register_strategy('coreset')
class CoreSetSampling(SamplingStrategy):
    def __init__(self, src_dset, tgt_dset, model, device, num_classes, cfg):
        super(CoreSetSampling, self).__init__(src_dset, tgt_dset, model, device, num_classes, cfg)

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

    def query(self, n, epoch):
        idxs = np.arange(len(self.tgt_dset.train_idx))
        idxs_unlabeled, _, _, _, all_embs = self.pred(idxs=idxs, with_emb=True)
        all_embs = all_embs.numpy()
        q_idxs = self.furthest_first(all_embs[~self.idxs_lb, :], all_embs[self.idxs_lb, :], n)
        return idxs_unlabeled[q_idxs]


@register_strategy('AADA')
class AADASampling(SamplingStrategy):
    """
    Implements Active Adversarial Domain Adaptation (https://arxiv.org/abs/1904.07848)
    """

    def __init__(self, src_dset, tgt_dset, model, device, num_classes, cfg):
        super(AADASampling, self).__init__(src_dset, tgt_dset, model, device, num_classes, cfg)

    def query(self, n, epoch):
        """
        s(x) = frac{1-G*_d}{G_f(x))}{G*_d(G_f(x))} [Diversity] * H(G_y(G_f(x))) [Uncertainty]
        """
        self.model.eval()
        idxs_unlabeled = np.arange(len(self.tgt_dset.train_idx))[~self.idxs_lb]
        train_sampler = ActualSequentialSampler(self.tgt_dset.train_idx[idxs_unlabeled])
        data_loader = torch.utils.data.DataLoader(self.query_dset, sampler=train_sampler, num_workers=4, batch_size=64,
                                                  drop_last=False)

        # Get diversity and entropy
        all_log_probs, all_scores = [], []
        with torch.no_grad():
            for batch_idx, (data, target, _) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                scores = self.model(data)
                log_probs = nn.LogSoftmax(dim=1)(scores)
                all_scores.append(scores)
                all_log_probs.append(log_probs)

        all_scores = torch.cat(all_scores)
        all_log_probs = torch.cat(all_log_probs)

        all_probs = torch.exp(all_log_probs)
        disc_scores = nn.Softmax(dim=1)(self.discrim(all_scores))
        # Compute diversity
        self.D = torch.div(disc_scores[:, 0], disc_scores[:, 1])
        # Compute entropy
        self.E = -(all_probs * all_log_probs).sum(1)
        scores = (self.D * self.E).sort(descending=True)[1]
        # Sample from top-2 % instances, as recommended by authors
        top_N = max(int(len(scores) * 0.02), n)
        q_idxs = np.random.choice(scores[:top_N].cpu().numpy(), n, replace=False)

        return idxs_unlabeled[q_idxs]


@register_strategy('BADGE')
class BADGESampling(SamplingStrategy):
    """
    Implements BADGE: Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds (https://arxiv.org/abs/1906.03671)
    """

    def __init__(self, src_dset, tgt_dset, model, device, num_classes, cfg):
        super(BADGESampling, self).__init__(src_dset, tgt_dset, model, device, num_classes, cfg)

    def query(self, n, epoch):
        idxs_unlabeled = np.arange(len(self.tgt_dset.train_idx))[~self.idxs_lb]
        train_sampler = ActualSequentialSampler(self.tgt_dset.train_idx[idxs_unlabeled])
        data_loader = torch.utils.data.DataLoader(self.query_dset, sampler=train_sampler, num_workers=self.cfg.DATALOADER.NUM_WORKERS,
                                                  batch_size=self.cfg.DATALOADER.BATCH_SIZE, drop_last=False)
        self.model.eval()

        if 'LeNet' in self.cfg.MODEL.BACKBONE.NAME:
            emb_dim = 500
        elif 'ResNet34' in self.cfg.MODEL.BACKBONE.NAME:
            emb_dim = 512
        elif 'ResNet50' in self.cfg.MODEL.BACKBONE.NAME:
            emb_dim = 256

        tgt_emb = torch.zeros([len(data_loader.sampler), self.num_classes])
        tgt_pen_emb = torch.zeros([len(data_loader.sampler), emb_dim])
        tgt_lab = torch.zeros(len(data_loader.sampler))
        tgt_preds = torch.zeros(len(data_loader.sampler))
        batch_sz = self.cfg.DATALOADER.BATCH_SIZE

        with torch.no_grad():
            for batch_idx, (data, target, _) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                e1, e2 = self.model(data, with_emb=True)
                tgt_pen_emb[batch_idx * batch_sz:batch_idx * batch_sz + min(batch_sz, e2.shape[0]), :] = e2.cpu()
                tgt_emb[batch_idx * batch_sz:batch_idx * batch_sz + min(batch_sz, e1.shape[0]), :] = e1.cpu()
                tgt_lab[batch_idx * batch_sz:batch_idx * batch_sz + min(batch_sz, e1.shape[0])] = target
                tgt_preds[batch_idx * batch_sz:batch_idx * batch_sz + min(batch_sz, e1.shape[0])] = e1.argmax(dim=1,
                                                         keepdim=True).squeeze()

        # Compute uncertainty gradient
        tgt_scores = nn.Softmax(dim=1)(tgt_emb)
        tgt_scores_delta = torch.zeros_like(tgt_scores)
        tgt_scores_delta[torch.arange(len(tgt_scores_delta)), tgt_preds.long()] = 1

        # Uncertainty embedding
        badge_uncertainty = (tgt_scores - tgt_scores_delta)

        # Seed with maximum uncertainty example
        max_norm = row_norms(badge_uncertainty.cpu().numpy()).argmax()

        _, q_idxs = kmeans_plus_plus_opt(badge_uncertainty.cpu().numpy(), tgt_pen_emb.cpu().numpy(), n,
                                               init=[max_norm])

        return idxs_unlabeled[q_idxs]


@register_strategy('kmeans')
class KmeansSampling(SamplingStrategy):
    """
    Implements CLUE: CLustering via Uncertainty-weighted Embeddings
    """

    def __init__(self, src_dset, tgt_dset, model, device, num_classes, cfg):
        super(KmeansSampling, self).__init__(src_dset, tgt_dset, model, device, num_classes, cfg)

    def query(self, n, epoch):
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


@register_strategy('CLUE')
class CLUESampling(SamplingStrategy):
    """
    Implements CLUE: CLustering via Uncertainty-weighted Embeddings
    """

    def __init__(self, src_dset, tgt_dset, model, device, num_classes, cfg):
        super(CLUESampling, self).__init__(src_dset, tgt_dset, model, device, num_classes, cfg)
        self.random_state = np.random.RandomState(1234)
        self.T = 0.1

    def query(self, n, epoch):
        idxs_unlabeled = np.arange(len(self.tgt_dset.train_idx))[~self.idxs_lb]
        train_sampler = ActualSequentialSampler(self.tgt_dset.train_idx[idxs_unlabeled])
        data_loader = torch.utils.data.DataLoader(self.query_dset, sampler=train_sampler, num_workers=self.cfg.DATALOADER.NUM_WORKERS, \
                                                  batch_size=self.cfg.DATALOADER.BATCH_SIZE, drop_last=False)
        self.model.eval()

        if 'LeNet' in self.cfg.MODEL.BACKBONE.NAME:
            emb_dim = 500
        elif 'ResNet34' in self.cfg.MODEL.BACKBONE.NAME:
            emb_dim = 512
        elif 'ResNet50' in self.cfg.MODEL.BACKBONE.NAME:
            emb_dim = 256

        # Get embedding of target instances
        tgt_emb, tgt_lab, tgt_preds, tgt_pen_emb = get_embedding(self.model, data_loader, self.device,
                                                                       self.num_classes, \
                                                                       self.cfg, with_emb=True, emb_dim=emb_dim)
        tgt_pen_emb = tgt_pen_emb.cpu().numpy()
        tgt_scores = torch.softmax(tgt_emb / self.T, dim=-1)
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

        return idxs_unlabeled[q_idxs]

class MHPSampling(SamplingStrategy):
    '''
    Implements MHPL: Minimum Happy Points Learning for Active Source Free Domain Adaptation (CVPR'23)
    '''

    def __init__(self, src_dset, tgt_dset, model, device, num_classes, cfg):
        super(MHPSampling, self).__init__(src_dset, tgt_dset, model, device, num_classes, cfg)

    def query(self, n, epoch):
        idxs_unlabeled = np.arange(len(self.tgt_dset.train_idx))[~self.idxs_lb]
        train_sampler = ActualSequentialSampler(self.tgt_dset.train_idx[idxs_unlabeled])
        data_loader = torch.utils.data.DataLoader(self.query_dset, sampler=train_sampler,
                                                  num_workers=self.cfg.DATALOADER.NUM_WORKERS,
                                                  batch_size=self.cfg.DATALOADER.BATCH_SIZE, drop_last=False)
        self.model.eval()
        all_probs = []
        all_embs = []
        with torch.no_grad():
            for batch_idx, (data, target, _) in enumerate(data_loader):
                data, target = data.to(self.device), target.to(self.device)
                scores, embs = self.model(data, with_emb=True)
                all_embs.append(embs.cpu())
                probs = F.softmax(scores, dim=-1)
                all_probs.append(probs)

        all_probs = torch.cat(all_probs)
        all_embs = F.normalize(torch.cat(all_embs), dim=-1)

        # find KNN
        sim = all_embs.cpu().mm(all_embs.transpose(1, 0))
        K = self.cfg.LADA.S_K
        sim_topk, topk = torch.topk(sim, k=K + 1, dim=1)
        sim_topk, topk = sim_topk[:, 1:], topk[:, 1:]
        sim_topk = sim_topk.to(self.device)
        topk = topk.to(self.device)

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