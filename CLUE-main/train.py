# -*- coding: utf-8 -*-
import os
import sys
import random
import distutils
from distutils import util
import argparse
from omegaconf import OmegaConf
import copy
import pprint
from collections import defaultdict
from tqdm import tqdm, trange
import time

import numpy as np
import torch

random.seed(1234)
torch.manual_seed(1234)
np.random.seed(1234)
torch.cuda.manual_seed(1234)

from adapt.models.models import get_model
import utils
from data import ASDADataset
from sample import *

def run_active_adaptation(args, source_model, src_dset, num_classes, device):
	"""
	Runs active domain adaptation experiments
	"""

	if_second = False
	if args.total_budget_second != 0:
		if_second = True
	# Load source data
	src_train_dset, _, _ = src_dset.get_dsets(apply_transforms=False)
	src_train_loader, _, src_test_loader, _ = src_dset.get_loaders()

	# Load target data
	# train数据集与原始数据集相比只是多加了一步transform
	# train_idx指的是trainloader中属于训练集的数据部分，虽然叫做trainloader，但其数据集中同时包含了train和valid两部分
	target_dset = ASDADataset(args.target, valid_ratio=0)
	target_train_dset, _, _ = target_dset.get_dsets(apply_transforms=False)
	target_train_loader, _, target_test_loader, train_idx = target_dset.get_loaders()

	# print("shape of data:")
	# print(src_train_loader.dataset.data.shape)
	# print(target_train_loader.dataset.data.shape)
	# Bookkeeping
	target_accs = defaultdict(list)
	ada_strat = '{}_{}'.format(args.model_init, args.al_strat)
	exp_name = '{}_{}_{}_{}_{}runs_{}rounds_{}budget'.format(args.id, args.model_init, args.al_strat, args.da_strat, \
															args.runs, args.num_rounds, args.total_budget)

	# Sample varying % of target data
	sampling_ratio = [(args.total_budget/args.num_rounds) for n in range(args.num_rounds+1)]
	sampling_ratio[0] = 0
	if if_second:
		sampling_ratio_all = [((args.total_budget_second+args.total_budget)/args.num_rounds) for n in range(args.num_rounds+1)]
		sampling_ratio_second = [(args.total_budget_second/args.num_rounds_2) for n in range(args.num_rounds_2+1)]
		sampling_ratio_second[0] = 0
		sampling_ratio_all[0] = 0
		while len(sampling_ratio_second) < len(sampling_ratio):
			sampling_ratio_second.append(0)
			# sampling_ratio_second.insert(0, 0)
		for i in range(len(sampling_ratio)):
			sampling_ratio[i] = sampling_ratio_all[i] - sampling_ratio_second[i]
		print(sampling_ratio_all, sampling_ratio, sampling_ratio_second)


	# Evaluate source model on target test
	transfer_perf, _ = utils.test(source_model, device, target_test_loader)
	out_str = '{}->{} performance (Before {}): Task={:.2f}'.format(args.source, args.target, args.da_strat, transfer_perf)
	print(out_str)

	print('------------------------------------------------------\n')
	print('Running strategy: Init={} AL={} Train={}'.format(args.model_init, args.al_strat, args.da_strat))
	print('\n------------------------------------------------------')	

	# Choose appropriate model initialization
	if args.model_init == 'scratch':
		model, src_model = get_model(args.cnn, num_cls=num_classes).to(device), source_model
	elif args.model_init == 'source':
		model, src_model = source_model, source_model

	# Run unsupervised DA at round 0, where applicable
	discriminator = None
	if args.da_strat != 'ft':
		print('Round 0: Unsupervised DA to target via {}'.format(args.da_strat))
		model, src_model, discriminator = utils.run_unsupervised_da(model, src_train_loader, None, target_train_loader, \
																	train_idx, num_classes, device, args)
	
		# Evaluate adapted source model on target test
		start_perf, _ = utils.test(model, device, target_test_loader)
		out_str = '{}->{} performance (After {}): {:.2f}'.format(args.source, args.target, args.da_strat, start_perf)
		print(out_str)
		print('\n------------------------------------------------------\n')
	else:
		start_perf, _ = utils.test(model, device, target_test_loader)
		out_str = '{}->{} performance (After {}): {:.2f}'.format(args.source, args.target, args.da_strat, start_perf)

	#################################################################
	# Main Active DA loop
	#################################################################


	tqdm_run = trange(args.runs)
	for run in tqdm_run: # Run over multiple experimental runs
		tqdm_run.set_description('Run {}'.format(str(run)))
		tqdm_run.refresh()
		tqdm_rat = trange(len(sampling_ratio[1:]))
		target_accs[0.0].append(start_perf)
		
		# Making a copy for current run
		curr_model = copy.deepcopy(model)
		curr_source_model = curr_model

		# Keep track of labeled vs unlabeled data
		idxs_lb = np.zeros(len(train_idx), dtype=bool)

		# Instantiate active sampling strategy
		sampling_strategy = get_strategy(args.al_strat, target_train_dset, train_idx, \
										 curr_model, discriminator, device, args)

		for ix in tqdm_rat: # Iterate over Active DA rounds
			ratio = sampling_ratio[ix+1]
			tqdm_rat.set_description('# Target labels={:d}'.format(int(ratio)))
			if if_second:
				ratio_second = sampling_ratio_second[ix+1]
				if ratio_second > 0:
					tqdm_rat.set_description('# \nTarget labels for second step={:d}'.format(int(ratio_second)))
			tqdm_rat.refresh()
			# Select instances via AL strategy
			print('\nSelecting instances...')
			idxs = sampling_strategy.query(int(ratio))
			idxs_lb[idxs] = True
			print("\ninstances for the first step {0}".format(int(ratio)))
			# Whether ADA-CS is used or not
			if if_second:
				if ratio_second > 0:
					print("\ninstances for the second step {0}".format(int(ratio)))
					idxs_2 = sampling_strategy.cal_proportion(idxs, idxs_lb,
															  src_train_loader, target_train_loader,
															  n2=int(ratio_second))
					print(int(ratio_second))
					idxs_lb[idxs_2] = True
					sampling_strategy.update(idxs_lb)
				else:
					sampling_strategy.update(idxs_lb)
			else:
				sampling_strategy.update(idxs_lb)

			# Update model with new data via DA strategy
			# 每轮会将新的样本加入labeled target集合，之后用这个新的集合训练最新的模型
			best_model = sampling_strategy.train(target_train_dset, da_round=(ix+1), \
												 src_loader=src_train_loader, \
												 src_model=curr_source_model)
			# Evaluate on target test and train splits
			test_perf, _ = utils.test(best_model, device, target_test_loader)
			train_perf, _ = utils.test(best_model, device, target_train_loader, split='train')

			out_str = '{}->{} Test performance (Round {}, # Target labels={:d}): {:.2f}'.format(args.source, args.target, ix, int(ratio), test_perf)
			out_str += '\n\tTrain performance (Round {}, # Target labels={:d}): {:.2f}'.format(ix, int(ratio), train_perf)
			print('\n------------------------------------------------------\n')
			print(out_str)

			target_accs[ratio].append(test_perf)

		# Log at the end of every run
		wargs = vars(args) if isinstance(args, argparse.Namespace) else dict(args)
		target_accs['args'] = wargs
		utils.log(target_accs, exp_name)

	return target_accs

def main():
	parser = argparse.ArgumentParser()

	# Experiment identifiers
	# parser.add_argument('--id', type=str, default='debug', help="Experiment identifier")
	parser.add_argument('--al_strat', type=str, default='uniform', help="Active learning strategy")
	parser.add_argument('--da_strat', type=str, default='mme', help="DA strat. Currently supports: {ft, DANN, MME}")
	# parser.add_argument('--model_init', type=str, default='source', help="Active DA model initialization")

	# Load existing configuration?
	parser.add_argument('--load_from_cfg', type=lambda x: bool(distutils.util.strtobool(x)), default=True, help="Load from config?")
	parser.add_argument('--cfg_file', type=str, help="Experiment configuration file", default="config/digits/clue_mme.yml")

	# Experimental details
	# parser.add_argument('--runs', type=int, default=1, help="Number of experimental runs")
	parser.add_argument('--source', default="svhn", help="Source dataset")
	parser.add_argument('--target', default="mnist", help="Target dataset")
	parser.add_argument('--total_budget', type=int, default=30, help="Total target budget")
	parser.add_argument('--total_budget_second', type=int, default=20, help="Total target budget for the second step")
	parser.add_argument('--num_rounds', type=int, default=5, help="Target dataset number of splits")
	parser.add_argument('--num_rounds_2', type=int, default=2, help="Target dataset number of splits in second step")

	parser.add_argument('--timestamp', default=time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime()),
						type=str, help='timestamp')
	
	# Load arguments from command line or via config file
	args_cmd = parser.parse_args()

	if args_cmd.load_from_cfg:
		args_cfg = dict(OmegaConf.load(args_cmd.cfg_file))
		args_cmd = vars(args_cmd)
		for k in args_cfg.keys():
			if args_cfg[k] is not None:
				if k not in args_cmd.keys():
					args_cmd[k] = args_cfg[k]
		args = OmegaConf.create(args_cmd)
	else: 
		args = args_cmd

	pp = pprint.PrettyPrinter()
	pp.pprint(args)
	logger = "{}_{}_{}_{}_{}_{}_{}_{}.txt".format(args.source, args.target, args.al_strat, args.da_strat
									  ,args.total_budget_second, args.total_budget, args.num_rounds_2, args.timestamp)
	log_path = './log/domainnet'
	logger_path = os.path.join(log_path, logger)
	# device = torch.device("cuda") if args.use_cuda else torch.device("cpu")
	device = torch.device("cuda")

	# Load source data
	src_dset = ASDADataset(args.source, batch_size=args.batch_size)
	src_train_loader, src_val_loader, src_test_loader, _ = src_dset.get_loaders()
	num_classes = src_dset.get_num_classes()
	print('Number of classes: {}'.format(num_classes))

	# Train / load a source model
	source_model = get_model(args.cnn, num_cls=num_classes).to(device)	
	source_file = '{}_{}_source.pth'.format(args.source, args.cnn)
	source_path = os.path.join('checkpoints', 'source', source_file)	

	if os.path.exists(source_path): # Load existing source model
		print('Loading source checkpoint: {}'.format(source_path))
		source_model.load_state_dict(torch.load(source_path, map_location=device), strict=False)
		best_source_model = source_model
	else:							# Train source model
		print('Training {} model...'.format(args.source))
		best_val_acc, best_source_model = 0.0, None
		source_optimizer = optim.Adam(source_model.parameters(), lr=args.lr, weight_decay=args.wd)

		for epoch in range(args.num_epochs):
			utils.train(source_model, device, src_train_loader, source_optimizer, epoch)
			val_acc, _ = utils.test(source_model, device, src_val_loader, split="val")
			out_str = '[Epoch: {}] Val Accuracy: {:.3f} '.format(epoch, val_acc)
			print(out_str)

			if (val_acc > best_val_acc):
				best_val_acc = val_acc
				best_source_model = copy.deepcopy(source_model)
				torch.save(best_source_model.state_dict(), os.path.join('checkpoints', 'source', source_file))

	# Evaluate on source test set
	test_acc, _ = utils.test(best_source_model, device, src_test_loader, split="test")
	out_str = '{} Test Accuracy: {:.3f} '.format(args.source, test_acc)
	print(out_str)

	# Run active adaptation experiments
	target_accs = run_active_adaptation(args, best_source_model, src_dset, num_classes, device)
	pp.pprint(target_accs)
	with open(logger_path, "w") as file:
		utils.write_recursive(target_accs, file)

if __name__ == '__main__':
	main()
