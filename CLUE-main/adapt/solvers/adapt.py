# -*- coding: utf-8 -*-
import sys
import random
import numpy as np
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .solver import register_solver
sys.path.append('../../')
import utils
# from utils import Entropy, BCE_softlabels, get_losses_unlabeled, inv_lr_scheduler, sigmoid_rampup, sigmoid_rampup

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)
torch.cuda.manual_seed(1234)

class BaseSolver:
	"""
	Base DA solver class
	"""
	def __init__(self, net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, da_round, device, args):
		self.net = net
		self.src_loader = src_loader
		self.tgt_sup_loader = tgt_sup_loader
		self.tgt_unsup_loader = tgt_unsup_loader
		self.train_idx = np.array(train_idx)
		self.tgt_opt = tgt_opt
		self.da_round = da_round
		self.device = device
		self.args = args

	def solve(self, epoch):
		pass

@register_solver('ft')
class TargetFTSolver(BaseSolver):
	"""
	Finetune on target labels
	"""
	def __init__(self, net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, da_round, device, args):
		super(TargetFTSolver, self).__init__(net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, da_round, device, args)
	
	def solve(self, epoch):
		"""
		Finetune on target labels
		"""		
		self.net.train()		
		if (self.da_round > 0): tgt_sup_iter = iter(self.tgt_sup_loader)
		info_str = '[Train target finetuning] Epoch: {}'.format(epoch)
		while True:
			try:
				data_t, target_t = next(tgt_sup_iter)
				data_t, target_t = data_t.to(self.device), target_t.to(self.device)
			except: break
			
			self.tgt_opt.zero_grad()
			output = self.net(data_t)
			loss = nn.CrossEntropyLoss()(output, target_t)
			info_str = '[Train target finetuning] Epoch: {}'.format(epoch)
			info_str += ' Target Sup. Loss: {:.3f}'.format(loss.item())
			
			loss.backward()
			self.tgt_opt.step()
		
		if epoch % 10 == 0: print(info_str)

@register_solver('dann')
class DANNSolver(BaseSolver):
	"""
	Implements DANN from Unsupervised Domain Adaptation by Backpropagation: https://arxiv.org/abs/1409.7495
	"""
	def __init__(self, net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, da_round, device, args):
		super(DANNSolver, self).__init__(net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, da_round, device, args)
	
	def solve(self, epoch, disc, disc_opt):
		"""
		Semisupervised adaptation via DANN: XE on labeled source + XE on labeled target + \
									ent. minimization on target + DANN on source<->target
		"""
		gan_criterion = nn.CrossEntropyLoss()
		cent = utils.ConditionalEntropyLoss().to(self.device)

		self.net.train()
		disc.train()
		
		if self.da_round == 0:
			src_sup_wt, lambda_unsup, lambda_cent = 1.0, 0.1, 0.01 # Hardcoded for unsupervised DA
		else:
			src_sup_wt, lambda_unsup, lambda_cent = self.args.src_sup_wt, self.args.unsup_wt, self.args.cent_wt
			tgt_sup_iter = iter(self.tgt_sup_loader)

		joint_loader = zip(self.src_loader, self.tgt_unsup_loader)		
		for batch_idx, ((data_s, label_s), (data_tu, label_tu)) in enumerate(joint_loader):
			data_s, label_s = data_s.to(self.device), label_s.to(self.device)
			data_tu = data_tu.to(self.device)

			if self.da_round > 0:
				try:
					data_ts, label_ts = next(tgt_sup_iter)
					data_ts, label_ts = data_ts.to(self.device), label_ts.to(self.device)
				except: break

			# zero gradients for optimizers
			self.tgt_opt.zero_grad()
			disc_opt.zero_grad()

			# Train with target labels
			score_s = self.net(data_s)
			xeloss_src = src_sup_wt*nn.CrossEntropyLoss()(score_s, label_s)

			info_str = "[Train DANN] Epoch: {}".format(epoch)
			info_str += " Src Sup loss: {:.3f}".format(xeloss_src.item())                    

			xeloss_tgt = 0
			if self.da_round > 0:
				score_ts = self.net(data_ts)
				xeloss_tgt = nn.CrossEntropyLoss()(score_ts, label_ts)
				info_str += " Tgt Sup loss: {:.3f}".format(xeloss_tgt.item())

			# extract and concat features
			score_tu = self.net(data_tu)
			f = torch.cat((score_s, score_tu), 0)

			# predict with discriminator
			f_rev = utils.ReverseLayerF.apply(f)
			pred_concat = disc(f_rev)

			target_dom_s = torch.ones(len(data_s)).long().to(self.device)
			target_dom_t = torch.zeros(len(data_tu)).long().to(self.device)
			label_concat = torch.cat((target_dom_s, target_dom_t), 0)

			# compute loss for disciminator
			loss_domain = gan_criterion(pred_concat, label_concat)
			loss_cent = cent(score_tu)

			loss_final = (xeloss_src + xeloss_tgt) + (lambda_unsup * loss_domain) + (lambda_cent * loss_cent)

			loss_final.backward()

			self.tgt_opt.step()
			disc_opt.step()
		
			# log net update info
			info_str += " DANN loss: {:.3f}".format(lambda_unsup * loss_domain.item())		
			info_str += " Ent Loss: {:.3f}".format(lambda_cent * loss_cent.item())		
		
		if epoch%10 == 0: print(info_str)

@register_solver('mme')
class MMESolver(BaseSolver):
	"""
	Implements MME from Semi-supervised Domain Adaptation via Minimax Entropy: https://arxiv.org/abs/1904.06487
	"""
	def __init__(self, net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, da_round, device, args):
		super(MMESolver, self).__init__(net, src_loader, tgt_sup_loader, tgt_unsup_loader, train_idx, tgt_opt, da_round, device, args)
	
	def solve(self, epoch):
		"""
		Semisupervised adaptation via MME: XE on labeled source + XE on labeled target + \
										adversarial ent. minimization on unlabeled target
		"""
		self.net.train()		
		src_sup_wt, lambda_adent = self.args.src_sup_wt, self.args.unsup_wt

		if self.da_round == 0:
			src_sup_wt, lambda_unsup = 1.0, 0.1
		else:
			src_sup_wt, lambda_unsup = self.args.src_sup_wt, self.args.unsup_wt
			tgt_sup_iter = iter(self.tgt_sup_loader)


		joint_loader = zip(self.src_loader, self.tgt_unsup_loader)
		for batch_idx, ((data_s, label_s), (data_tu, label_tu)) in enumerate(joint_loader):			
			data_s, label_s = data_s.to(self.device), label_s.to(self.device)
			data_tu = data_tu.to(self.device)
			
			if self.da_round > 0:
				try:
					data_ts, label_ts = next(tgt_sup_iter)
					data_ts, label_ts = data_ts.to(self.device), label_ts.to(self.device)
				except: break

			# zero gradients for optimizer
			self.tgt_opt.zero_grad()
					
			# log basic adapt train info
			info_str = "[Train Minimax Entropy] Epoch: {}".format(epoch)

			# extract features
			score_s = self.net(data_s)
			xeloss_src = src_sup_wt * nn.CrossEntropyLoss()(score_s, label_s)
			
			# log discriminator update info
			info_str += " Src Sup loss: {:.3f}".format(xeloss_src.item())
			
			xeloss_tgt = 0
			if self.da_round > 0:
				score_ts = self.net(data_ts)
				xeloss_tgt = nn.CrossEntropyLoss()(score_ts, label_ts)
				info_str += " Tgt Sup loss: {:.3f}".format(xeloss_tgt.item())

			xeloss = xeloss_src + xeloss_tgt
			xeloss.backward()
			self.tgt_opt.step()

			# Add adversarial entropy
			self.tgt_opt.zero_grad()

			score_tu = self.net(data_tu, reverse_grad=True)
			probs_tu = F.softmax(score_tu, dim=1)
			loss_adent = lambda_adent * torch.mean(torch.sum(probs_tu * (torch.log(probs_tu + 1e-5)), 1))
			loss_adent.backward()
			
			self.tgt_opt.step()
			
			# Log net update info
			info_str += " MME loss: {:.3f}".format(loss_adent.item())		
		
		if epoch%10 == 0: print(info_str)


# @register_solver('RAA')
# class RAASolver(BaseSolver):
#     """
#       Implement Random Anchor set Augmentation (RAA)
#     """
#
#     def __init__(self, net, src_loader, tgt_loader, tgt_sup_loader, tgt_unsup_loader, joint_sup_loader, tgt_opt,
#                  ada_stage, device, cfg, **kwargs):
#         super(RAASolver, self).__init__(net, src_loader, tgt_loader, tgt_sup_loader, tgt_unsup_loader,
#                                          joint_sup_loader, tgt_opt, ada_stage, device, cfg, **kwargs)
#
#     def solve(self, epoch, seq_query_loader):
#         K = self.cfg.LADA.A_K
#         th = self.cfg.LADA.A_TH
#
#         # create an anchor set
#         if len(self.tgt_sup_loader) > 0:
#             tgt_sup_dataset = self.tgt_sup_loader.dataset
#             tgt_sup_samples = [tgt_sup_dataset.samples[i] for i in self.tgt_sup_loader.sampler.indices]
#             seed_dataset = ImageList(tgt_sup_samples, root=tgt_sup_dataset.root, transform=tgt_sup_dataset.transform)
#             seed_dataset.rand_transform = rand_transform
#             seed_dataset.rand_num = self.cfg.LADA.A_RAND_NUM
#             seed_loader = torch.utils.data.DataLoader(seed_dataset, shuffle=True,
#                                           batch_size=self.tgt_sup_loader.batch_size, num_workers=self.tgt_sup_loader.num_workers)
#             seed_idxs = self.tgt_sup_loader.sampler.indices.tolist()
#             seed_iter = iter(seed_loader)
#             seed_labels = [seed_dataset.samples[i][1] for i in range(len(seed_dataset))]
#
#             if K > 0:
#                 # build nearest neighbors
#                 self.net.eval()
#                 tgt_idxs = []
#                 tgt_embs = []
#                 tgt_labels = []
#                 tgt_data = []
#                 seq_query_loader = copy.deepcopy(seq_query_loader)
#                 seq_query_loader.dataset.transform = copy.deepcopy(self.tgt_loader.dataset.transform)
#                 with torch.no_grad():
#                     for sample_ in seq_query_loader:
#                         sample = copy.deepcopy(sample_)
#                         del sample_
#                         data, label, idx = sample[0], sample[1], sample[2]
#                         data, label = data.to(self.device), label.to(self.device)
#                         score, emb = self.net(data, with_emb=True)
#                         tgt_embs.append(F.normalize(emb).detach().clone().cpu())
#                         tgt_labels.append(label.cpu())
#                         tgt_idxs.append(idx.cpu())
#                         tgt_data.append(data.cpu())
#
#                 tgt_embs = torch.cat(tgt_embs)
#                 tgt_data = torch.cat(tgt_data)
#                 tgt_idxs = torch.cat(tgt_idxs)
#
#         self.net.train()
#
#         src_iter = iter(self.src_loader)
#         iter_per_epoch = len(self.src_loader)
#
#         for batch_idx in range(iter_per_epoch):
#             if batch_idx % len(self.src_loader) == 0:
#                 src_iter = iter(self.src_loader)
#
#             data_s, label_s, _ = next(src_iter)
#             data_s, label_s = data_s.to(self.device), label_s.to(self.device)
#
#             self.tgt_opt.zero_grad()
#             output_s = self.net(data_s)
#             loss = nn.CrossEntropyLoss()(output_s, label_s)
#
#             if len(self.tgt_sup_loader) > 0:
#                 try:
#                     data_ts, label_ts, idx_ts, *data_rand_ts = next(seed_iter)
#                 except:
#                     seed_iter = iter(seed_loader)
#                     data_ts, label_ts, idx_ts, *data_rand_ts = next(seed_iter)
#
#
#                 if len(data_rand_ts)>0:
#                     for i, r_data in enumerate(data_rand_ts):
#                         alpha = 0.2
#                         mask = torch.FloatTensor(np.random.beta(alpha, alpha, size=(data_ts.shape[0], 1, 1, 1)))
#                         data_ts = (data_ts * mask) + (r_data * (1 - mask))
#                         data_ts, label_ts = data_ts.to(self.device), label_ts.to(self.device)
#                         output_ts, emb_ts = self.net(data_ts, with_emb=True)
#                         loss += nn.CrossEntropyLoss()(output_ts, label_ts)
#                 else:
#                     data_ts, label_ts = data_ts.to(self.device), label_ts.to(self.device)
#                     output_ts, emb_ts = self.net(data_ts, with_emb=True)
#                     loss += nn.CrossEntropyLoss()(output_ts, label_ts)
#
#             loss.backward()
#             self.tgt_opt.step()
#
#             if len(self.tgt_sup_loader) > 0 and K > 0 and len(seed_idxs) < tgt_embs.shape[0]:
#                 nn_idxs = torch.randint(0, tgt_data.shape[0], (data_ts.shape[0],)).to(self.device)
#
#                 data_nn = tgt_data[nn_idxs].to(self.device)
#
#                 with torch.no_grad():
#                     output_nn, emb_nn = self.net(data_nn, with_emb=True)
#                     prob_nn = torch.softmax(output_nn, dim=-1)
#                     tgt_embs[nn_idxs] = F.normalize(emb_nn).detach().clone().cpu()
#
#                 conf_samples = []
#                 conf_idx = []
#                 conf_pl = []
#                 dist = np.eye(prob_nn.shape[-1])[np.array(seed_labels)].sum(0) + 1
#                 sp = 1 - dist / dist.max() + dist.min() / dist.max()
#
#                 for i in range(prob_nn.shape[0]):
#                     idx = tgt_idxs[nn_idxs[i]].item()
#                     pl_i = prob_nn[i].argmax(-1).item()
#                     if np.random.random() <= sp[pl_i] and prob_nn[i].max(-1)[0] >= th and idx not in seed_idxs:
#                         conf_samples.append((self.tgt_loader.dataset.samples[idx][0], pl_i))
#                         conf_idx.append(idx)
#                         conf_pl.append(pl_i)
#
#                 seed_dataset.add_item(conf_samples)
#                 seed_idxs.extend(conf_idx)
#                 seed_labels.extend(conf_pl)


# @register_solver('LAA')
# class LAASolver(BaseSolver):
#     """
#       Local context-aware Anchor set Augmentation (LAA)
#     """
#
#     def __init__(self, net, src_loader, tgt_loader, tgt_sup_loader, tgt_unsup_loader, joint_sup_loader, tgt_opt,
#                  ada_stage, device, cfg, **kwargs):
#         super(LAASolver, self).__init__(net, src_loader, tgt_loader, tgt_sup_loader, tgt_unsup_loader,
#                                          joint_sup_loader, tgt_opt, ada_stage, device, cfg, **kwargs)
#
#     def solve(self, epoch, seq_query_loader):
#         K = self.cfg.LADA.A_K
#         th = self.cfg.LADA.A_TH
#
#         # create an anchor set
#         if len(self.tgt_sup_loader) > 0:
#             tgt_sup_dataset = self.tgt_sup_loader.dataset
#             tgt_sup_samples = [tgt_sup_dataset.samples[i] for i in self.tgt_sup_loader.sampler.indices]
#             seed_dataset = ImageList(tgt_sup_samples, root=tgt_sup_dataset.root, transform=tgt_sup_dataset.transform)
#             seed_dataset.rand_transform = rand_transform
#             seed_dataset.rand_num = self.cfg.LADA.A_RAND_NUM
#             seed_loader = torch.utils.data.DataLoader(seed_dataset, shuffle=True,
#                                           batch_size=self.tgt_sup_loader.batch_size, num_workers=self.tgt_sup_loader.num_workers)
#             seed_idxs = self.tgt_sup_loader.sampler.indices.tolist()
#             seed_iter = iter(seed_loader)
#             seed_labels = [seed_dataset.samples[i][1] for i in range(len(seed_dataset))]
#
#             if K > 0:
#                 # build nearest neighbors
#                 self.net.eval()
#                 tgt_idxs = []
#                 tgt_embs = []
#                 tgt_labels = []
#                 tgt_data = []
#                 seq_query_loader = copy.deepcopy(seq_query_loader)
#                 seq_query_loader.dataset.transform = copy.deepcopy(self.tgt_loader.dataset.transform)
#                 with torch.no_grad():
#                     for sample_ in seq_query_loader:
#                         sample = copy.deepcopy(sample_)
#                         del sample_
#                         data, label, idx = sample[0], sample[1], sample[2]
#                         data, label = data.to(self.device), label.to(self.device)
#                         score, emb = self.net(data, with_emb=True)
#                         tgt_embs.append(F.normalize(emb).detach().clone().cpu())
#                         tgt_labels.append(label.cpu())
#                         tgt_idxs.append(idx.cpu())
#                         tgt_data.append(data.cpu())
#
#                 tgt_embs = torch.cat(tgt_embs)
#                 tgt_data = torch.cat(tgt_data)
#                 tgt_idxs = torch.cat(tgt_idxs)
#
#         self.net.train()
#
#         src_iter = iter(self.src_loader)
#         iter_per_epoch = len(self.src_loader)
#
#         for batch_idx in range(iter_per_epoch):
#             if batch_idx % len(self.src_loader) == 0:
#                 src_iter = iter(self.src_loader)
#
#             data_s, label_s, _ = next(src_iter)
#             data_s, label_s = data_s.to(self.device), label_s.to(self.device)
#
#             self.tgt_opt.zero_grad()
#             output_s = self.net(data_s)
#             loss = nn.CrossEntropyLoss()(output_s, label_s)
#
#             if len(self.tgt_sup_loader) > 0:
#                 try:
#                     data_ts, label_ts, idx_ts, *data_rand_ts = next(seed_iter)
#                 except:
#                     seed_iter = iter(seed_loader)
#                     data_ts, label_ts, idx_ts, *data_rand_ts = next(seed_iter)
#
#                 if len(data_rand_ts) > 0:
#                     for i, r_data in enumerate(data_rand_ts):
#                         alpha = 0.2
#                         mask = torch.FloatTensor(np.random.beta(alpha, alpha, size=(data_ts.shape[0], 1, 1, 1)))
#                         data_ts = (data_ts * mask) + (r_data * (1 - mask))
#                         data_ts, label_ts = data_ts.to(self.device), label_ts.to(self.device)
#                         output_ts, emb_ts = self.net(data_ts, with_emb=True)
#                         loss += nn.CrossEntropyLoss()(output_ts, label_ts)
#                 else:
#                     data_ts, label_ts = data_ts.to(self.device), label_ts.to(self.device)
#                     output_ts, emb_ts = self.net(data_ts, with_emb=True)
#                     loss += nn.CrossEntropyLoss()(output_ts, label_ts)
#
#             loss.backward()
#             self.tgt_opt.step()
#
#             if len(self.tgt_sup_loader) > 0 and K > 0 and len(seed_idxs) < tgt_embs.shape[0]:
#                 mask = torch.ones(tgt_embs.shape[0])
#                 re_idxs = tgt_idxs[mask == 1]
#
#                 sim = F.normalize(emb_ts.cpu()).mm(tgt_embs[re_idxs].transpose(1, 0))
#                 sim_topk, topk = torch.topk(sim, k=K, dim=1)
#
#                 rand_nn = torch.randint(0, topk.shape[1], (topk.shape[0], 1))
#                 nn_idxs = torch.gather(topk, dim=-1, index=rand_nn).squeeze(1)
#                 nn_idxs = re_idxs[nn_idxs]
#
#                 data_nn = tgt_data[nn_idxs].to(self.device)
#
#                 with torch.no_grad():
#                     output_nn, emb_nn = self.net(data_nn, with_emb=True)
#                     prob_nn = torch.softmax(output_nn, dim=-1)
#                     tgt_embs[nn_idxs] = F.normalize(emb_nn).detach().clone().cpu()
#
#                 conf_samples = []
#                 conf_idx = []
#                 conf_pl = []
#                 dist = np.eye(prob_nn.shape[-1])[np.array(seed_labels)].sum(0) + 1
#                 dist = dist / dist.max()
#                 sp = 1 - dist / dist.max() + dist.min() / dist.max()
#
#                 for i in range(prob_nn.shape[0]):
#                     idx = tgt_idxs[nn_idxs[i]].item()
#                     pl_i = prob_nn[i].argmax(-1).item()
#                     if np.random.random() <= sp[pl_i] and prob_nn[i].max(-1)[0] >= th and idx not in seed_idxs:
#                         conf_samples.append((self.tgt_loader.dataset.samples[idx][0], pl_i))
#                         conf_idx.append(idx)
#                         conf_pl.append(pl_i)
#
#                 seed_dataset.add_item(conf_samples)
#                 seed_idxs.extend(conf_idx)
#                 seed_labels.extend(conf_pl)


# @register_solver('MCC')
# class MCCSolver(BaseSolver):
#     """
#     Implements MCC from Minimum Class Confusion for Versatile Domain Adaptation: https://arxiv.org/abs/1912.03699
#     https://github.com/thuml/Versatile-Domain-Adaptation
#     """
#
#     def __init__(self, net, src_loader, tgt_loader, tgt_sup_loader, tgt_unsup_loader, joint_sup_loader, tgt_opt,
#                  ada_stage, device, cfg, **kwargs):
#         super(MCCSolver, self).__init__(net, src_loader, tgt_loader, tgt_sup_loader, tgt_unsup_loader,
#                                          joint_sup_loader, tgt_opt, ada_stage, device, cfg, **kwargs)
#
#     def solve(self, epoch):
#         src_iter = iter(self.src_loader)
#         tgt_un_iter = iter(self.tgt_unsup_loader)
#         tgt_s_iter = iter(self.tgt_sup_loader)
#         iter_per_epoch = len(self.src_loader)
#
#         self.net.train()
#
#         for batch_idx in range(iter_per_epoch):
#             if batch_idx % len(self.src_loader) == 0:
#                 src_iter = iter(self.src_loader)
#
#             if batch_idx % len(self.tgt_unsup_loader) == 0:
#                 tgt_un_iter = iter(self.tgt_unsup_loader)
#
#             data_s, label_s, _ = next(src_iter)
#             data_s, label_s = data_s.to(self.device), label_s.to(self.device)
#
#             self.tgt_opt.zero_grad()
#             output_s = self.net(data_s)
#             loss = nn.CrossEntropyLoss()(output_s, label_s) * self.cfg.ADA.SRC_SUP_WT
#
#             if len(self.tgt_sup_loader) > 0:
#                 try:
#                     data_ts, label_ts, idx_ts = next(tgt_s_iter)
#                 except:
#                     tgt_s_iter = iter(self.tgt_sup_loader)
#                     data_ts, label_ts, idx_ts = next(tgt_s_iter)
#
#                 data_ts, label_ts = data_ts.to(self.device), label_ts.to(self.device)
#                 output_ts = self.net(data_ts)
#
#                 loss += nn.CrossEntropyLoss()(output_ts, label_ts)
#
#             data_tu, label_tu, _ = next(tgt_un_iter)
#             data_tu, label_tu = data_tu.to(self.device), label_tu.to(self.device)
#             output_tu = self.net(data_tu)
#
#             outputs_target_temp = output_tu / self.cfg.MODEL.TEMP
#             target_softmax_out_temp = nn.Softmax(dim=1)(outputs_target_temp)
#             target_entropy_weight = Entropy(target_softmax_out_temp).detach()
#             target_entropy_weight = 1 + torch.exp(-target_entropy_weight)
#             target_entropy_weight = self.cfg.DATALOADER.BATCH_SIZE * target_entropy_weight / torch.sum(target_entropy_weight)
#             cov_matrix_t = target_softmax_out_temp.mul(target_entropy_weight.view(-1, 1)).transpose(1, 0).mm(
#                 target_softmax_out_temp)
#             cov_matrix_t = cov_matrix_t / (torch.sum(cov_matrix_t, dim=1)+1e-12)
#             mcc_loss = (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / self.cfg.DATASET.NUM_CLASS
#
#             loss += mcc_loss
#
#             loss.backward()
#             self.tgt_opt.step()


# @register_solver('CDAC')
# class CDACSolver(BaseSolver):
#     """
#     Implements Cross-Domain Adaptive Clustering for Semi-Supervised Domain Adaptation: https://arxiv.org/abs/2104.09415
#     https://github.com/lijichang/CVPR2021-SSDA
#     """
#
#     def __init__(self, net, src_loader, tgt_loader, tgt_sup_loader, tgt_unsup_loader, joint_sup_loader, tgt_opt,
#                  ada_stage, device, cfg, **kwargs):
#         super(CDACSolver, self).__init__(net, src_loader, tgt_loader, tgt_sup_loader, tgt_unsup_loader,
#                                             joint_sup_loader, tgt_opt, ada_stage, device, cfg, **kwargs)
#
#     def solve(self, epoch):
#         self.net.train()
#
#         self.tgt_unsup_loader.dataset.rand_transform = rand_transform2
#         self.tgt_unsup_loader.dataset.rand_num = 2
#
#         data_iter_s = iter(self.src_loader)
#         data_iter_t = iter(self.tgt_sup_loader)
#         data_iter_t_unl = iter(self.tgt_unsup_loader)
#
#         len_train_source = len(self.src_loader)
#         len_train_target = len(self.tgt_sup_loader)
#         len_train_target_semi = len(self.tgt_unsup_loader)
#
#         BCE = BCE_softlabels().to(self.device)
#         criterion = nn.CrossEntropyLoss().to(self.device)
#
#         iter_per_epoch = len(self.src_loader)
#         for batch_idx in range(iter_per_epoch):
#             rampup = sigmoid_rampup(batch_idx+epoch*iter_per_epoch, 20000)
#             w_cons = 30.0 * rampup
#
#             self.tgt_opt = inv_lr_scheduler([0.1, 1.0, 1.0], self.tgt_opt, batch_idx+epoch*iter_per_epoch,
#                                             init_lr=0.01)
#
#             if len(self.tgt_sup_loader) > 0:
#                 if batch_idx % len_train_target == 0:
#                     data_iter_t = iter(self.tgt_sup_loader)
#                 if batch_idx % len_train_target_semi == 0:
#                     data_iter_t_unl = iter(self.tgt_unsup_loader)
#                 if batch_idx % len_train_source == 0:
#                     data_iter_s = iter(self.src_loader)
#                 data_t = next(data_iter_t)
#                 data_t_unl = next(data_iter_t_unl)
#                 data_s = next(data_iter_s)
#
#                 # load labeled source data
#                 x_s, target_s = data_s[0], data_s[1]
#                 im_data_s = x_s.to(self.device)
#                 gt_labels_s = target_s.to(self.device)
#
#                 # load labeled target data
#                 x_t, target_t = data_t[0], data_t[1]
#                 im_data_t = x_t.to(self.device)
#                 gt_labels_t = target_t.to(self.device)
#
#                 # load unlabeled target data
#                 x_tu, x_bar_tu, x_bar2_tu = data_t_unl[0], data_t_unl[3], data_t_unl[4]
#                 im_data_tu = x_tu.to(self.device)
#                 im_data_bar_tu = x_bar_tu.to(self.device)
#                 im_data_bar2_tu = x_bar2_tu.to(self.device)
#
#                 self.tgt_opt.zero_grad()
#                 # construct losses for overall labeled data
#                 data = torch.cat((im_data_s, im_data_t), 0)
#                 target = torch.cat((gt_labels_s, gt_labels_t), 0)
#                 out1 = self.net(data)
#                 ce_loss = criterion(out1, target)
#
#                 ce_loss.backward(retain_graph=True)
#                 self.tgt_opt.step()
#                 self.tgt_opt.zero_grad()
#
#                 # construct losses for unlabeled target data
#                 aac_loss, pl_loss, con_loss = get_losses_unlabeled(self.net, im_data=im_data_tu, im_data_bar=im_data_bar_tu,
#                                                                    im_data_bar2=im_data_bar2_tu, target=None, BCE=BCE,
#                                                                    w_cons=w_cons, device=self.device)
#                 loss = (aac_loss + pl_loss + con_loss) * self.cfg.ADA.UNSUP_WT * 10
#             else:
#                 if batch_idx % len_train_source == 0:
#                     data_iter_s = iter(self.src_loader)
#                 data_s, label_s, _ = next(data_iter_s)
#                 data_s, label_s = data_s.to(self.device), label_s.to(self.device)
#
#                 self.tgt_opt.zero_grad()
#                 output_s = self.net(data_s)
#                 loss = nn.CrossEntropyLoss()(output_s, label_s) * self.cfg.ADA.SRC_SUP_WT
#
#             loss.backward()
#             self.tgt_opt.step()