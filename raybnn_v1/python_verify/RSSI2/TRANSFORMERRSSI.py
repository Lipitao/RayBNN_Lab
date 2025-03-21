

import numpy as np


import math
import time
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger



def main():
	print("Transformer RSSI")
	print(torch.cuda.get_device_name(0))

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



	input_size = int(sys.argv[1])
	
	fold = int(sys.argv[2])

	nhead = 4
	if input_size < 50:
		nhead = 2
	
	neuron_size = int( ((2**(input_size - 1).bit_length())/2) /nhead)*nhead
	output_size = 2

	if input_size < 40:
		neuron_size = 40


	train_X = np.loadtxt("Alcala_TRAINX.dat", delimiter=',').astype(np.float32)
	train_Y = np.loadtxt("Alcala_TRAINY.dat", delimiter=',').astype(np.float32)


	test_X = np.loadtxt("Alcala_TESTX.dat", delimiter=',').astype(np.float32)
	test_Y = np.loadtxt("Alcala_TESTY.dat", delimiter=',').astype(np.float32)



	train_data = np.concatenate((train_Y, train_X), axis=1)
	test_data = np.concatenate((test_Y, test_X), axis=1)




	print(train_data.shape)
	print(test_data.shape)


	traj_size = 20
	batch_size = 100

	sample_size = traj_size*batch_size

	trainloader = DataLoader(train_data, batch_size=sample_size, shuffle=False)
	testloader = DataLoader(test_data, batch_size=sample_size, shuffle=False)

	


	class Learner(pl.LightningModule):
		def __init__(self):
			super().__init__()
			self.lr = 1e-3
			self.iters = 0.


			self.act = nn.LeakyReLU(0.001)

			self.input_dense = nn.Linear(input_size, neuron_size)

			self.transformer_model = nn.Transformer(
				d_model=neuron_size, 
				num_encoder_layers=3,
				num_decoder_layers=3,
				dim_feedforward=neuron_size)

			self.output_dense = nn.Linear(neuron_size, output_size)


			self.predlist = np.zeros((2,output_size))
			self.actlist = np.zeros((2,output_size))

		def getTestData(self):
			return   self.predlist, self.actlist 

		def forward(self, x, y, y_mask):

			out0 = self.act(self.input_dense(x)) 

			outx = torch.zeros((traj_size+1, batch_size, neuron_size)).to(device)
			outx[:-1,:,:] = torch.reshape(out0, (traj_size, batch_size, neuron_size))
			outy = torch.zeros((traj_size+1, batch_size, neuron_size)).to(device)
			outy[1:,:,(neuron_size-1-output_size):(neuron_size-1)] = torch.reshape(y, (traj_size, batch_size, output_size))

			out1 = self.transformer_model(src=outx, tgt=outy, tgt_mask=y_mask)[:-1,:,:]

			out2 = torch.reshape(out1, (traj_size*batch_size, neuron_size))

			return self.output_dense(out2)

		def training_step(self, batch, batch_idx):
			self.iters += 1.

			x = batch[:,2:(2+input_size)]
			y = batch[:,:2]


			x, y = x.to(device), y.to(device)
			y_mask = (self.transformer_model.generate_square_subsequent_mask(traj_size+1)  ).to(device)

			y_hat = self.forward(x, y, y_mask)

			loss = 50.0*nn.MSELoss()(y_hat, y)
			epoch_progress = self.iters / self.loader_len

			tqdm_dict = {'train_MSE': loss}
			logs = {'train_MSE': loss, 'epoch': epoch_progress}
			return {'loss': loss, 'progress_bar': tqdm_dict, 'log': logs}

		def test_step(self, batch, batch_nb):
			x = batch[:,2:(2+input_size)]
			y = batch[:,:2]


			x, y = x.to(device), y.to(device)
			y_mask = (self.transformer_model.generate_square_subsequent_mask(traj_size+1)  ).to(device)

			y_hat = self.forward(x, y, y_mask)
			loss = nn.MSELoss()(y_hat, y)


			self.predlist = np.concatenate((self.predlist,y_hat.cpu().detach().numpy()),axis=0)

			self.actlist = np.concatenate((self.actlist,y.cpu().detach().numpy()),axis=0)
			return {'test_MSE': loss}


		def test_epoch_end(self, outputs):
			#print(outputs)
			loss_mean = torch.stack([x['test_MSE'] for x in outputs]).mean().cpu()

			logs = {'MSE_mean': loss_mean}
			return {'MSE_mean': loss_mean,
					'log': logs, 'progress_bar': logs}

		def configure_optimizers(self):
			opt = torch.optim.Adam(self.parameters(), lr=self.lr)
			return opt

		def train_dataloader(self):
			self.loader_len = len(trainloader)
			return trainloader

		def test_dataloader(self):
			self.test_loader_len = len(trainloader)
			return testloader







	learn = Learner()
	trainer = pl.Trainer(max_epochs=200,
				 gpus=1,
				 )


	pytorch_total_params = sum(p.numel() for p in learn.parameters() if p.requires_grad)


	st = time.process_time()
	trainer.fit(learn)
	elapsed_time = time.process_time() - st

	ret = trainer.test()


	predlist, actlist  =  learn.getTestData()


	act_name = "test_act_" + str(fold) + "_" + str(input_size) + ".dat"
	actlist = actlist[2:,:]
	print(actlist.shape)
	np.savetxt(act_name,actlist, delimiter=",", fmt='%.10f')

	pred_name = "test_pred_" + str(fold) + "_" + str(input_size) + ".dat"
	predlist = predlist[2:,:]
	print(predlist.shape)
	np.savetxt(pred_name,predlist, delimiter=",", fmt='%.10f')


	info_name = "info_" + str(fold) + "_" + str(input_size) + ".dat"
	np.savetxt(info_name, np.array([elapsed_time, pytorch_total_params]) , delimiter=",", fmt='%.10f')



if __name__ == '__main__':
	main()