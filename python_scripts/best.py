'''
Importing modules
'''
import os
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import pickle
import datetime
import random

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
import numpy as np

from tqdm import tqdm
from skimage.transform import resize
from torch.utils.data import DataLoader, TensorDataset
from torch import Tensor
from torch.autograd import Variable

'''
Loading the input and output data
'''
root = os.getcwd()
ipath = os.path.join(root, 'projects/def-kjerbi/sirish01/Data/Processed/TrainData/')
opath = os.path.join(root, 'projects/def-kjerbi/sirish01/Data/Processed/TrainTruth/')

ifiles = sorted(os.listdir(ipath))
ofiles = sorted(os.listdir(opath))
input_data = []
output_data = []

for x in ifiles:
	input_data.append(np.load(ipath + x))
for x in ofiles:
	output_data.append(np.load(opath + x))

input_data = np.array(input_data)
output_data = np.array(output_data)

'''
LFADS module
'''
# print("Load Success")

#-------------------------
# COST FUNCTION COMPONENTS
#-------------------------
def KLCostGaussian(post_mu, post_lv, prior_mu, prior_lv):
	'''
	KLCostGaussian(post_mu, post_lv, prior_mu, prior_lv)

	KL-Divergence between a prior and posterior diagonal Gaussian distribution.

	Arguments:
		- post_mu (torch.Tensor): mean for the posterior
		- post_lv (torch.Tensor): logvariance for the posterior
		- prior_mu (torch.Tensor): mean for the prior
		- prior_lv (torch.Tensor): logvariance for the prior
	'''
	klc = 0.5 * (prior_lv - post_lv + torch.exp(post_lv - prior_lv) \
		 + ((post_mu - prior_mu)/torch.exp(0.5 * prior_lv)).pow(2) - 1.0).sum()
	return klc		
#--------
# NETWORK
#--------
class LFADS(nn.Module):
	
	#------------------------------------------------------------------------------
	#------------------------------------------------------------------------------
	def __init__(self, count, inputs_dim, output_dim, T, value1, value2, model_hyperparams = None,
				 device = 'cpu', save_variables = False,
				 seed = None):
		
		# -----------------------
		# BASIC INIT STUFF
		# -----------------------
		
		# call the nn.Modules constructor
		super(LFADS, self).__init__()
		
		# Default hyperparameters
		default_hyperparams  = {### DATA PARAMETERS ###
								'dataset_name'             : 'best',
								'run_name'                 : 'CV',
								
								### MODEL PARAMETERS ###
								'g_dim'                    : value1,
								'u_dim'                    : 2, 
								'factors_dim'              : value2,
								'g0_encoder_dim'           : value1,
								'c_encoder_dim'            : value1,
								'controller_dim'           : value1,
								'g0_prior_kappa'           : 0.1,
								'u_prior_kappa'            : 0.1,
								'keep_prob'                : 0.97,
								'clip_val'                 : 5.0,
								'max_norm'                 : 200,
			
								### OPTIMIZER PARAMETERS 
								'learning_rate'            : 0.001,
								'learning_rate_min'        : 1e-5,
								'learning_rate_decay'      : 0.95,
								'scheduler_on'             : True,
								'scheduler_patience'       : 4,
								'scheduler_cooldown'       : 4,
								'epsilon'                  : 0.1,
								'betas'                    : (0.9, 0.99),
								'l2_gen_scale'             : 0,
								'l2_con_scale'             : 0,
								'kl_weight_schedule_start' : 0,
								'kl_weight_schedule_dur'   : 2000,
								'l2_weight_schedule_start' : 0,
								'l2_weight_schedule_dur'   : 2000,
								'ew_weight_schedule_start' : 0,
								'ew_weight_schedule_dur'   : 2000}
		
		# Store the hyperparameters        
		self._update_params(count,default_hyperparams, model_hyperparams)
		
		self.inputs_dim                = inputs_dim
		self.output_dim                = output_dim
		self.T                         = T

		self.device                    = device
		self.save_variables            = save_variables
		self.seed                      = seed
		
		if self.seed is None:
			self.seed = random.randint(1, 10000)
			print('Random seed: {}'.format(self.seed))
		else:
			print('Preset seed: {}'.format(self.seed))
   
		random.seed(self.seed)
		np.random.seed(self.seed)
		torch.manual_seed(self.seed)
		if self.device == 'cuda':
			torch.cuda.manual_seed_all(self.seed)
		
		# Store loss
		self.full_loss_store = {'train_loss' : {}, 'train_recon_loss' : {}, 'train_kl_loss' : {},
								'valid_loss' : {}, 'valid_recon_loss' : {}, 'valid_kl_loss' : {},
								'l2_loss' : {}}

		self.train_loss_store = []
		self.valid_loss_store = []
		self.best = np.inf


		# Training variable
		self.epochs = 0
		self.current_step = 0
		self.last_decay_epoch = 0

		self.cost_weights = {'kl' : {'weight': 0, 'schedule_start': self.kl_weight_schedule_start,
									 'schedule_dur': self.kl_weight_schedule_dur},
							 'l2' : {'weight': 0, 'schedule_start': self.l2_weight_schedule_start,
									 'schedule_dur': self.l2_weight_schedule_dur}}
		
		# -----------------------
		# NETWORK LAYERS INIT
		# 
		# Notation:
		#
		#   layertype_outputvariable(_direction)
		#
		#   Examples: fc_factors = "fully connected layer, variable = factors"
		#             gru_Egen_forward = "gated recurrent unit layer, encoder for generator, forward direction"
		# -----------------------
		
		# ----
		# RNN layers
		# ----

		# Generator Forward Encoder
		self.gru_Egen_forward  = nn.GRUCell(input_size= self.inputs_dim, hidden_size= self.g0_encoder_dim)
		
		# Generator Backward Encoder
		self.gru_Egen_backward = nn.GRUCell(input_size= self.inputs_dim, hidden_size= self.g0_encoder_dim)
		
		# Controller Forward Encoder
		self.gru_Econ_forward  = nn.GRUCell(input_size= self.inputs_dim, hidden_size= self.c_encoder_dim)
		
		# Controller Backward Encoder
		self.gru_Econ_backward = nn.GRUCell(input_size= self.inputs_dim, hidden_size= self.c_encoder_dim)
		
		# Controller
		self.gru_controller    = nn.GRUCell(input_size= self.c_encoder_dim * 2 + self.factors_dim, hidden_size= self.controller_dim)
		
		# Generator
		self.gru_generator     = nn.GRUCell(input_size= self.u_dim, hidden_size= self.g_dim)
		# -----------
		# Fully connected layers
		# -----------
		
		# mean and logvar of the posterior distribution for the generator initial conditions (g0 from E_gen)
		# takes as inputs:
		#  - the forward encoder for g0 at time T (g0_enc_f_T)
		#  - the backward encoder for g0 at time 1 (g0_enc_b_0]

		self.fc_g0mean   = nn.Linear(in_features= 2 * self.g0_encoder_dim, out_features= self.g_dim)
		self.fc_g0logvar = nn.Linear(in_features= 2 * self.g0_encoder_dim, out_features= self.g_dim)
		
		# mean and logvar of the posterior distribution for the inferred inputs (u provided to g)
		# takes as inputs:
		#  - the controller at time t (c_t)

		self.fc_umean   = nn.Linear(in_features= self.controller_dim, out_features= self.u_dim)
		self.fc_ulogvar = nn.Linear(in_features= self.controller_dim, out_features= self.u_dim)
		
		# factors from generator output
		self.fc_factors = nn.Linear(in_features= self.g_dim, out_features= self.factors_dim)

		self.f1 = nn.Linear(in_features = self.factors_dim, out_features = 80)
		self.fc_clean   = nn.Linear(in_features= 80, out_features = self.output_dim)

		# -----------
		# Dropout layer
		# -----------
		self.dropout = nn.Dropout(1.0 - self.keep_prob)
		
		# -----------------------
		# WEIGHT INIT
		# 
		# The weight initialization is modified from the standard PyTorch, which is uniform. Instead,
		# the weights are drawn from a normal distribution with mean 0 and std = 1/sqrt(K) where K
		# is the size of the input dimension. This helps prevent vanishing/exploding gradients by
		# keeping the eigenvalues of the Jacobian close to 1.
		# -----------------------
		
		# Step through all layers and adjust the weight initiazition method accordingly
		for m in self.modules():
			
			# GRU layer, update using input weight and recurrent weight dimensionality
			if isinstance(m, nn.GRUCell):
				k_ih = m.weight_ih.shape[1] # dimensionality of the inputs to the GRU
				k_hh = m.weight_hh.shape[1] # dimensionality of the GRU outputs
				m.weight_ih.data.normal_(std = k_ih ** -0.5) # inplace resetting of W ~ N(0,1/sqrt(N))
				m.weight_hh.data.normal_(std = k_hh ** -0.5) # inplace resetting of W ~ N(0,1/sqrt(N))
			
			# FC layer, update using input dimensionality
			elif isinstance(m, nn.Linear):
				k = m.in_features # dimensionality of the inputs
				m.weight.data.normal_(std = k ** -0.5) # inplace resetting of W ~ N(0,1/sqrt(N))

		torch.nn.init.normal_(self.fc_factors.weight, std = 1.0)		
		torch.nn.init.normal_(self.f1.weight, std = 1.0)
		torch.nn.init.normal_(self.fc_clean.weight, std = 1.0)

		# Row-normalise fc_factors (See bullet-point 11 of section 1.9 of online methods)
		self.fc_factors.weight.data = F.normalize(self.fc_factors.weight.data, dim = 1)
		
		# --------------------------
		# LEARNABLE PRIOR PARAMETERS INIT
		# --------------------------
		
		self.g0_prior_mu = nn.parameter.Parameter(torch.tensor(0.0))
		self.u_prior_mu  = nn.parameter.Parameter(torch.tensor(0.0))
		
		from math import log
		self.g0_prior_logkappa = nn.parameter.Parameter(torch.tensor(log(self.g0_prior_kappa)))
		self.u_prior_logkappa  = nn.parameter.Parameter(torch.tensor(log(self.u_prior_kappa)))
		# --------------------------
		# OPTIMIZER INIT
		# --------------------------
		self.optimizer = opt.Adam(self.parameters(), lr=self.learning_rate, eps=self.epsilon, betas=self.betas)
		# --------------------------
		# LOSS FUNCTION
		# --------------------------
		self.loss = nn.MSELoss()
		
	#------------------------------------------------------------------------------
	#------------------------------------------------------------------------------
	def initialize(self, batch_size=None):
		'''
		initialize()
		
		Initialize dynamic model variables. These need to be reinitialized with each forward pass to
		ensure we don't need to retain graph between each .backward() call. 
		
		See https://discuss.pytorch.org/t/what-exactly-does-retain-variables-true-in-loss-backward-do/3508/2
		for discussion and explanation
		
		Note: The T + 1 terms  accommodate learnable biases for all variables, except for the generator,
		which is provided with a g0 estimate from the network
		
		optional arguments:
		  batch_size (int) : batch dimension. If None, use self.batch_size.
		
		'''
		
		batch_size = batch_size if batch_size is not None else self.batch_size
		
		self.g0_prior_mean = torch.ones(batch_size, self.g_dim).to(self.device)*self.g0_prior_mu            # g0 prior mean
		self.u_prior_mean  = torch.ones(batch_size, self.u_dim).to(self.device)*self.u_prior_mu             # u prior mean
		
		self.g0_prior_logvar = torch.ones(batch_size, self.g_dim).to(self.device)*self.g0_prior_logkappa    # g0 prior logvar
		self.u_prior_logvar  = torch.ones(batch_size, self.u_dim).to(self.device)*self.u_prior_logkappa     # u prior logvar
			
		self.c = Variable(torch.zeros((batch_size, self.controller_dim)).to(self.device))  # Controller hidden state

		self.efgen = Variable(torch.zeros((batch_size, self.g0_encoder_dim)).to(self.device))  # Forward generator encoder
		self.ebgen = Variable(torch.zeros((batch_size, self.g0_encoder_dim)).to(self.device))  # Backward generator encoder
		
		self.efcon = torch.zeros((batch_size, self.T+1, self.c_encoder_dim)).to(self.device)   # Forward controller encoder
		self.ebcon = torch.zeros((batch_size, self.T+1, self.c_encoder_dim)).to(self.device)   # Backward controller encoder

		self.raw_data		= torch.zeros(batch_size, self.T, self.inputs_dim)
		self.clean_data 	= torch.zeros(batch_size, self.T, self.output_dim)
		self.output        = torch.zeros(batch_size, self.T, self.output_dim)  
		self.factors       = torch.zeros(batch_size, self.T, self.factors_dim)
		self.inferred_inputs        = torch.zeros(batch_size, self.T, self.u_dim)
		self.inferred_inputs_mean   = torch.zeros(batch_size, self.T, self.u_dim)
		self.inferred_inputs_logvar = torch.zeros(batch_size, self.T, self.u_dim)
		
	def encode(self, x):
		'''
		encode(x)
		
		Function to encode the data with the forward and backward encoders.
		
		Arguments:
		  - x (torch.Tensor): Variable tensor of size batch size x time-steps x input dimension
		'''
		
		# Dropout some data
		if self.keep_prob < 1.0:
			x = self.dropout(x)
		
		# Encode data into forward and backward generator encoders to produce E_gen
		# for generator initial conditions.
		for t in range(1, self.T+1):
			
			# generator encoders
			self.efgen = torch.clamp(self.gru_Egen_forward(x[:, t-1], self.efgen),max=self.clip_val)
			self.ebgen = torch.clamp(self.gru_Egen_backward(x[:, -t], self.ebgen),max=self.clip_val)
			
			# controller encoders
			self.efcon[:, t]      = torch.clamp(self.gru_Econ_forward(x[:, t-1], self.efcon[:, t-1].clone()),max=self.clip_val)
			self.ebcon[:, -(t+1)] = torch.clamp(self.gru_Econ_backward(x[:, -t], self.ebcon[:, -t].clone()),max=self.clip_val)
		
		# Concatenate efgen_T and ebgen_1 for generator initial condition sampling
		egen = torch.cat((self.efgen, self.ebgen), dim=1)
		# Dropout the generator encoder output
		if self.keep_prob < 1.0:
			egen = self.dropout(egen)
			
		# Sample initial conditions for generator from g0 posterior distribution
		self.g0_mean   = self.fc_g0mean(egen)
		self.g0_logvar = torch.clamp(self.fc_g0logvar(egen), min=np.log(0.0001))
		self.g         = Variable(torch.randn(self.batch_size, self.g_dim).to(self.device))*torch.exp(0.5*self.g0_logvar)\
						 + self.g0_mean
		
		# KL cost for g(0)
#         pdb.set_trace()
		self.kl_loss   = KLCostGaussian(self.g0_mean, self.g0_logvar,
										self.g0_prior_mean, self.g0_prior_logvar)/x.shape[0]
		# print(x.shape[0])
		# Initialise factors
		self.f         = self.fc_factors(self.g)
	
	#------------------------------------------------------------------------------
	#------------------------------------------------------------------------------
	def generate(self, x, y):
		'''
		generate()
		
		Generates the rates using the controller encoder outputs and the sampled initial conditions for
		generator.
		'''
		
		self.recon_loss = 0
		
		for t in range(self.T):
			
			# Concatenate ebcon and efcon outputs at time t with factors at time t+1 as input to controller
			# Note: we take efcon at t+1, because the learnable biases are at first index for efcon
			econ_and_fac = torch.cat((self.efcon[:, t+1].clone(), self.ebcon[:,t].clone(), self.f), dim = 1)

			# Dropout the controller encoder outputs and factors
			if self.keep_prob < 1.0:
				econ_and_fac = self.dropout(econ_and_fac)
			
			# Update controller with controller encoder outputs
			self.c = torch.clamp(self.gru_controller(econ_and_fac, self.c),max=self.clip_val)

			# Calculate posterior distribution parameters for inferred inputs from controller state
			self.u_mean   = self.fc_umean(self.c)
			self.u_logvar = self.fc_ulogvar(self.c)
			# Sample inputs for generator from u(t) posterior distribution
			self.u = Variable(torch.randn(self.batch_size, self.u_dim).to(self.device))*torch.exp(0.5*self.u_logvar) \
						+ self.u_mean

			# KL cost for u(t)
			self.kl_loss = self.kl_loss + KLCostGaussian(self.u_mean, self.u_logvar,
										self.u_prior_mean, self.u_prior_logvar)/x.shape[0]

			# Update generator
			self.g = torch.clamp(self.gru_generator(self.u,self.g), min=0.0, max=self.clip_val)

			# Dropout on generator output
			if self.keep_prob < 1.0:
				self.g = self.dropout(self.g)

			# Generate factors from generator state
			self.f = self.fc_factors(self.g)
			# Generate rates from factor state
			out1 = F.relu(self.f1(self.f))
			self.res = self.fc_clean(out1)
			# if t and t % 10 == 0:
			# 	print("Epoch number is: %s"%(self.epochs))
			# 	print("Target Data")
			# 	print(y[:, t])
			# 	print(y[:, t].min(), y[:, t].max())
			# 	print("LFADS output")
			# 	print(self.res)
			# 	print(self.res.min(), self.res.max())
			
			L = self.loss(y[:, t], self.res)
			self.recon_loss = self.recon_loss + L
   
			self.raw_data[:,t] = x[:, t].detach().cpu()
			self.clean_data[:, t] = y[:, t].detach().cpu()
			self.output[:, t]   = self.res.detach().cpu()
			self.factors[:, t] = self.f.detach().cpu()
			self.inferred_inputs[:, t] = self.u.detach().cpu()
			self.inferred_inputs_mean[:, t] = self.u_mean.detach().cpu()
			self.inferred_inputs_logvar[:, t] = self.u_logvar.detach().cpu()
		self.recon_loss = self.recon_loss / self.T

						
	#------------------------------------------------------------------------------
	#------------------------------------------------------------------------------
	def forward(self, x, y):
		'''
		forward(x)
		
		Runs a forward pass through the network.
		
		Arguments:
		  - x (torch.Tensor): Single-trial spike data. Tensor of size batch size x time-steps x input dimension
		'''
		batch_size, steps_dim, inputs_dim = x.shape
		
		assert steps_dim  == self.T
		assert inputs_dim == self.inputs_dim
		
		self.batch_size = x.shape[0]
		self.initialize(batch_size=x.shape[0])
		self.encode(x)
		self.generate(x, y)

	def weight_schedule_fn(self, step):
		'''
		weight_schedule_fn(step)
		
		Calculate the KL and L2 regularization weights from the current training step number. Imposes
		linearly increasing schedule on regularization weights to prevent early pathological minimization
		of KL divergence and L2 norm before sufficient data reconstruction improvement. See bullet-point
		4 of section 1.9 in online methods
		
		required arguments:
		step (int) : training step number
		'''
		for cost_key in self.cost_weights.keys():
			# Get step number of scheduler
			weight_step = max(step - self.cost_weights[cost_key]['schedule_start'], 0)
			
			# Calculate schedule weight
			self.cost_weights[cost_key]['weight'] = min(weight_step/ self.cost_weights[cost_key]['schedule_dur'], 1.0)
	
	#------------------------------------------------------------------------------
	#------------------------------------------------------------------------------
	
	def apply_decay(self, current_loss):
		'''
		apply_decay(current_loss)
		
		Decrease the learning rate by a defined factor (self.learning_rate_decay) if loss is greater
		than the loss in the last six training steps and if the loss has not decreased in the last
		six training steps. See bullet point 8 of section 1.9 in online methods
		'''
		if len(self.train_loss_store) >= self.scheduler_patience:
			if all((current_loss > past_loss for past_loss in self.train_loss_store[-self.scheduler_patience:])):
				if self.epochs >= self.last_decay_epoch + self.scheduler_cooldown:
					self.learning_rate  = self.learning_rate * self.learning_rate_decay
					self.last_decay_epoch = self.epochs
					for g in self.optimizer.param_groups:
						g['lr'] = self.learning_rate
					print('Learning rate decreased to %.8f'%self.learning_rate)
	
	def test(self, valid_dl, batch_size):
		self.eval()
		self.batch_size = 64
		test_loss = 0
		test_recon_loss = 0
		test_kl_loss = 0
		for i, (xtest,ytest) in enumerate(valid_dl):
			with torch.no_grad():
				xtest = Variable(xtest)
				ytest = Variable(ytest)
				# print("Testing")
				self(xtest, ytest)
				loss = self.recon_loss
				# print("KL Loss per batch in test set: %.4f"%(self.kl_loss.data)) 
				test_loss += loss.data
				test_recon_loss += self.recon_loss.data
				test_kl_loss += self.kl_loss.data
		
		test_loss /= (i+1)
		test_recon_loss /= (i+1)
		test_kl_loss /= (i+1)
		return test_loss, test_recon_loss, test_kl_loss
			
	def randomloss(self, train_dl, test_dl, cv):
		self.batch_size = 64
		with torch.no_grad():
			train_loss = 0
			self.train()
			for i, (xtrain, ytrain) in enumerate(train_dl):
				xtrain = Variable(xtrain)
				ytrain = Variable(ytrain)
				self(xtrain, ytrain)
				loss = self.recon_loss
				train_loss += loss.data
			train_loss /= (i + 1)
			test_loss = 0
			self.eval()
			for i, (xtest, ytest) in enumerate(test_dl):
				xtest = Variable(xtest)
				ytest = Variable(ytest)
				self(xtest,ytest)
				loss = self.recon_loss 
				test_loss += loss.data
			test_loss /= (i + 1)
			print("Random Train Reconstruction Loss: %.4f, Random Test Reconstruction Loss: %.4f"%(train_loss, test_loss))
			random = {}
			random['Train'] = train_loss
			random['Test'] = test_loss
			save_loc = '%s/models/%s/%s/'%('./projects/def-kjerbi/sirish01', self.dataset_name, self.run_name)
			if not os.path.isdir(save_loc):
				os.makedirs(save_loc)
			with open(save_loc + 'randomloss_' + str(cv) + '.p','wb') as f:
					pickle.dump(random, f, protocol=pickle.HIGHEST_PROTOCOL)

			
	#------------------------------------------------------------------------------
	#------------------------------------------------------------------------------
	
	def fit(self, input_data, clean_data, train, test, cv, batch_size=64, max_epochs = 100, use_tensorboard=False, health_check=False, output='.'):
		
		self.batch_size = 64
		# create the dataloader
		
		# Initialize directory to save checkpoints
		save_loc = '%s/models/%s/%s/checkpoints/'%(output, self.dataset_name, self.run_name)
		# Create model_checkpoint directory if it doesn't exist
		if not os.path.isdir(save_loc):
			os.makedirs(save_loc)
		# print a message
		# for each epoch...   

		train_data = torch.tensor(np.concatenate(input_data[train])).type('torch.FloatTensor').to(device)
		train_truth = torch.tensor(np.concatenate(clean_data[train])).type('torch.FloatTensor').to(device)
		train_dataset = torch.utils.data.TensorDataset(train_data,train_truth)
		train_dl = torch.utils.data.DataLoader(train_dataset,batch_size = 64,shuffle = True)
  
		valid_data = torch.tensor(np.concatenate(input_data[test])).type('torch.FloatTensor').to(device)
		valid_truth = torch.tensor(np.concatenate(clean_data[test])).type('torch.FloatTensor').to(device)
		valid_dataset = torch.utils.data.TensorDataset(valid_data,valid_truth)
		valid_dl = torch.utils.data.DataLoader(valid_dataset,batch_size = 64,shuffle = True)
		self.randomloss(train_dl, valid_dl, cv)

		print(train_data.size(),train_truth.size())
		print(valid_data.size(),valid_truth.size())
		print(len(train_dataset),len(valid_dataset))
		for epoch in range(max_epochs):
			self.train()
			# If minimum learning rate reached, break training loop
			if self.learning_rate <= self.learning_rate_min:
				break
			# cumulative training loss for this epoch
			train_loss = 0
			train_recon_loss = 0
			train_kl_loss = 0

			for i, (xtrain,ytrain) in enumerate(train_dl, 0):
				self.current_step += 1							
				# apply Variable wrapper to batch
				xtrain = Variable(xtrain)
				ytrain = Variable(ytrain)

				# zero the parameter gradients
				self.optimizer.zero_grad()
				
				# Calculate regularizer weights
				self.weight_schedule_fn(self.current_step)

				# Forward
				# print("Training")
				self(xtrain, ytrain)
				# Calculate l2 regularisation penalty
				l2_loss = self.l2_gen_scale * self.gru_generator.weight_hh.norm(2)/self.gru_generator.weight_hh.numel() + \
						self.l2_con_scale * self.gru_controller.weight_hh.norm(2)/self.gru_controller.weight_hh.numel()
				
				# Collect separate weighted losses
				kl_weight = self.cost_weights['kl']['weight']
				l2_weight = self.cost_weights['l2']['weight']
				loss = self.recon_loss + kl_weight * self.kl_loss + l2_weight * l2_loss
				#print("KL Loss per batch in training set: %.4f"%(self.kl_loss.data))
				# return
				# Backward
				loss.backward()
				
				# clip gradient norm
				torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.max_norm)
				
				# update the weights
				self.optimizer.step()

				# Row-normalise fc_factors (See bullet-point 11 of section 1.9 of online methods)
				self.fc_factors.weight.data = F.normalize(self.fc_factors.weight.data, dim=1)
				
				if use_tensorboard:
					self.health_check(writer)
	 
				train_loss += loss.data
				train_recon_loss += self.recon_loss.data
				train_kl_loss += self.kl_loss.data
					
			train_loss /= (i+1)
			train_recon_loss /= (i+1)
			train_kl_loss /= (i+1)
   
			valid_loss, valid_recon_loss, valid_kl_loss = self.test(valid_dl,batch_size = 64)
			
			print('Epoch: %4d, tloss: %.3f, tr loss: %.3f, tkl loss: %.3f, vr loss: %.3f, v_loss: %.3f, vkl loss: %.3f' %(self.epochs+1, train_loss, train_recon_loss, train_kl_loss, valid_recon_loss,valid_loss, valid_kl_loss))
			# Apply learning rate decay function
			if self.scheduler_on:
				self.apply_decay(train_loss)
				
			# Store loss
			self.train_loss_store.append(float(train_loss))
			self.valid_loss_store.append(float(valid_loss))
			
			self.full_loss_store['train_loss'][self.epochs]       = float(train_loss)
			self.full_loss_store['train_recon_loss'][self.epochs] = float(train_recon_loss)
			self.full_loss_store['train_kl_loss'][self.epochs]    = float(train_kl_loss)
			self.full_loss_store['valid_loss'][self.epochs]       = float(valid_loss)
			self.full_loss_store['valid_recon_loss'][self.epochs] = float(valid_recon_loss)
			self.full_loss_store['valid_kl_loss'][self.epochs]    = float(valid_kl_loss)
			self.full_loss_store['l2_loss'][self.epochs]          = float(l2_loss.data)
			self.epochs += 1
			if self.current_step >= max(self.cost_weights['kl']['schedule_start'] + self.cost_weights['kl']['schedule_dur'],
										self.cost_weights['l2']['schedule_start'] + self.cost_weights['l2']['schedule_dur']):
				if self.valid_loss_store[-1] < self.best:
					self.last_saved = epoch
					self.best = self.valid_loss_store[-1]
					self.save_checkpoint(cv,output=output)

		import pandas as pd
		df = pd.DataFrame(self.full_loss_store)
		df.to_csv('%s/models/%s/%s/Total_%s.csv'%(output, self.dataset_name, self.run_name,cv), index_label='epoch')
		# Save a final checkpoint
		self.save_checkpoint(cv,force=True, output=output)
		
		print('...training complete.')
  
	def save_checkpoint(self,cv,force=False, purge_limit=50, output='.'):
		'''
		Save checkpoint of network parameters and optimizer state
		
		Arguments:
			force (bool) : force checkpoint to be saved (default = False)
			purge_limit (int) : delete previous checkpoint if there have been fewer
								epochs than this limit before saving again
		'''
		
		# output_filename of format [timestamp]_epoch_[epoch]_loss_[training].pth:
		#  - timestamp   (YYMMDDhhmm)
		#  - epoch       (int)
		#  - loss        (float with decimal point replaced by -)
		
		save_loc = '%s/models/%s/%s/checkpoints/'%(output, self.dataset_name, self.run_name)

		if force:
			pass
		else:
			if purge_limit:
				# Get checkpoint filenames
				try:
					_,_,filenames = list(os.walk(save_loc))[0]
					split_filenames = [os.path.splitext(fn)[0].split('_') for fn in filenames]
					avail_files = []
					for fn in split_filenames:
						if int(fn[6]) == cv:
							avail_files.append(fn)
					split_filenames = avail_files
					print(avail_files)
					epochs = [att[2] for att in split_filenames]
					epochs.sort()
					last_saved_epoch = epochs[-1]
					if self.epochs - 50 <= int(last_saved_epoch):
						rm_filename = [filename for filename in filenames if last_saved_epoch in filename][0]
						os.remove(save_loc+rm_filename)
					
				except IndexError:
					pass

		# Get current time in YYMMDDhhmm format
		timestamp = datetime.datetime.now().strftime('%y%m%d%H%M')
		
		# Get epoch_num as string
		epoch = str('%i'%self.epochs)
		
		# Get training_error as string
		loss = str(self.valid_loss_store[-1]).replace('.','-')
		
		output_filename = '%s_epoch_%s_loss_%s_CV_%s.pth'%(timestamp, epoch, loss, cv)
		assert os.path.splitext(output_filename)[1] == '.pth', 'Output filename must have .pth extension'
				
		# Create dictionary of training variables
		train_dict = {'best' : self.best, 'train_loss_store': self.train_loss_store,
					  'valid_loss_store' : self.valid_loss_store,
					  'full_loss_store' : self.full_loss_store,
					  'epochs' : self.epochs, 'current_step' : self.current_step,
					  'last_decay_epoch' : self.last_decay_epoch,
					  'learning_rate' : self.learning_rate,
					  'cost_weights' : self.cost_weights,
					  'dataset_name' : self.dataset_name}
		
		# Save network parameters, optimizer state, and training variables
		# torch.save({'net' : self.state_dict(), 'opt' : self.optimizer.state_dict(), 'train' : train_dict, 'time_series' : variables},
		# 		   save_loc+output_filename)

		torch.save({'net' : self.state_dict(), 'opt' : self.optimizer.state_dict(), 'train' : train_dict},
		   save_loc+output_filename)
	
	#------------------------------------------------------------------------------
	#------------------------------------------------------------------------------
	
	def _set_params(self, params):
		for k in params.keys():
			self.__setattr__(k, params[k])

	#------------------------------------------------------------------------------
	#------------------------------------------------------------------------------
	
	def _update_params(self, count,prev_params, new_params):
		if new_params:
			params = update_param_dict(prev_params, new_params)
		else:
			params = prev_params
		self._set_params(params)
		self.print_params(count,params)
  
	def print_params(self,cv,params):
		save_loc = '%s/models/%s/%s/'%('./projects/def-kjerbi/sirish01', self.dataset_name, self.run_name)
		if not os.path.isdir(save_loc):
			os.makedirs(save_loc)
		with open(save_loc + 'model' + str(cv) + '.txt','w') as f:
			for k in params.keys():
				f.write(str(k) + ':' + str(params[k]) +'\n')

'''
model run
'''
device = ("cuda" if torch.cuda.is_available() else "cpu")
outputpath = './projects/def-kjerbi/sirish01'
from sklearn.model_selection import KFold
from numpy.random import permutation
indices = np.arange(20)
print(indices)
kf = KFold(n_splits=3)
for cv, (train_index, test_index) in enumerate(kf.split(indices)):
	print(train_index,test_index)
	network = LFADS(count = cv,inputs_dim = 128,output_dim = 105,T = 50,value1 = 125,value2 = 10,model_hyperparams= None,device = device,save_variables = True).to(device)
	network.fit(input_data,output_data,train_index,test_index,cv,batch_size = 64,max_epochs=30,use_tensorboard = False,output = outputpath)
