

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
	print("CNN RSSI")
	print(torch.cuda.get_device_name(0))

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



	input_size = int(sys.argv[1])
	
	fold = int(sys.argv[2])

	CNN_channels = 4
	kernel_size = 5
	CNNsize = int( input_size - (kernel_size-1) )

	neuron_size = 2*input_size
	output_size = 2


	train_X = np.loadtxt("Alcala_TRAINX.dat", delimiter=',').astype(np.float32)
	train_Y = np.loadtxt("Alcala_TRAINY.dat", delimiter=',').astype(np.float32)


	test_X = np.loadtxt("Alcala_TESTX.dat", delimiter=',').astype(np.float32)
	test_Y = np.loadtxt("Alcala_TESTY.dat", delimiter=',').astype(np.float32)



	train_data = np.concatenate((train_Y, train_X), axis=1)
	test_data = np.concatenate((test_Y, test_X), axis=1)




	print(train_data.shape)
	print(test_data.shape)

	sample_size = 2000

	trainloader = DataLoader(train_data, batch_size=sample_size, shuffle=False)
	testloader = DataLoader(test_data, batch_size=sample_size, shuffle=False)




	class Learner(pl.LightningModule):
		def __init__(self, model:nn.Module):
			super().__init__()
			self.lr = 1e-3
			self.model = model
			self.iters = 0.

			self.predlist = np.zeros((2,output_size))
			self.actlist = np.zeros((2,output_size))

		def getTestData(self):
			return   self.predlist, self.actlist 

		def forward(self, x):
			return self.model(x)

		def training_step(self, batch, batch_idx):
			self.iters += 1.

			x = batch[:,2:(2+input_size)]
			x = x.reshape((x.shape[0], 1, x.shape[1]))
			y = batch[:,:2]


			x, y = x.to(device), y.to(device)
			y_hat = self.model(x)

			loss = 50.0*nn.MSELoss()(y_hat, y)
			epoch_progress = self.iters / self.loader_len

			tqdm_dict = {'train_MSE': loss}
			logs = {'train_MSE': loss, 'epoch': epoch_progress}
			return {'loss': loss, 'progress_bar': tqdm_dict, 'log': logs}

		def test_step(self, batch, batch_nb):
			x = batch[:,2:(2+input_size)]
			x = x.reshape((x.shape[0], 1, x.shape[1]))
			y = batch[:,:2]


			x, y = x.to(device), y.to(device)
			y_hat = self.model(x)
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



	

	model = nn.Sequential(
        nn.Conv1d(in_channels=1, out_channels=CNN_channels, kernel_size=kernel_size),
		nn.LeakyReLU(0.001),
		nn.Flatten(),

        nn.Linear(CNN_channels*CNNsize, neuron_size),
		nn.LeakyReLU(0.001),
        nn.Linear(neuron_size, neuron_size),
		nn.LeakyReLU(0.001),
        nn.Linear(neuron_size, output_size)
	).to(device)

	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)


	learn = Learner(model)
	trainer = pl.Trainer(max_epochs=200,
				 gpus=1,
				 )

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