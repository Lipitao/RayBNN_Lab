

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
	print("MLP EEG Optimize")
	print(torch.cuda.get_device_name(0))


	neuron_size_list = [8,16,32,64,128,256,512,1024,2048,4096]


	for neuron_size in neuron_size_list:
		device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


		input_dir = sys.argv[1]

		#neuron_size = int(sys.argv[2])

		fold = int(sys.argv[2])




		output_size = 1


		train_X = np.loadtxt(input_dir+"/X_train_"+str(fold)+".txt", delimiter=',').astype(np.float32)
		train_Y = np.loadtxt(input_dir+"/Y_train_"+str(fold)+".txt", delimiter=',').astype(np.float32)
		train_Y = train_Y[:, np.newaxis]
		print(train_X.shape)
		print(train_Y.shape)

		test_X = np.loadtxt(input_dir+"/X_test_"+str(fold)+".txt", delimiter=',').astype(np.float32)
		test_Y = np.loadtxt(input_dir+"/Y_test_"+str(fold)+".txt", delimiter=',').astype(np.float32)
		test_Y = test_Y[:, np.newaxis]

		input_size = train_X.shape[1]

		train_data = np.concatenate((train_Y, train_X), axis=1)
		test_data = np.concatenate((test_Y, test_X), axis=1)




		sample_size = 16

		trainloader = DataLoader(train_data, batch_size=sample_size, shuffle=False)
		testloader = DataLoader(test_data, batch_size=sample_size, shuffle=False)




		class Learner(pl.LightningModule):
			def __init__(self, model:nn.Module):
				super().__init__()
				self.lr = 0.0005
				self.lr = 0.001
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

				x = batch[:,1:]
				y = batch[:,0][:, np.newaxis]



				x, y = x.to(device), y.to(device)
				y_hat = self.model(x)

				loss = nn.BCELoss()(y_hat, y)
				epoch_progress = self.iters / self.loader_len

				tqdm_dict = {'train_MSE': loss}
				logs = {'train_MSE': loss, 'epoch': epoch_progress}
				return {'loss': loss, 'progress_bar': tqdm_dict, 'log': logs}

			def test_step(self, batch, batch_nb):
				x = batch[:,1:]
				y = batch[:,0][:, np.newaxis]


				x, y = x.to(device), y.to(device)
				y_hat = self.model(x)
				loss = nn.BCELoss()(y_hat, y)


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


		dropout = 0.5


		model = nn.Sequential(
			nn.Linear(input_size, neuron_size),
			nn.Dropout(p=dropout),
			nn.LeakyReLU(0.001),
			nn.Linear(neuron_size, neuron_size),
			nn.Dropout(p=dropout),
			nn.LeakyReLU(0.001),
			nn.Linear(neuron_size, output_size),
			nn.Sigmoid()
		).to(device)

		pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)


		learn = Learner(model)
		trainer = pl.Trainer(max_epochs=10,
					gpus=1,
					)

		st = time.process_time()
		trainer.fit(learn)
		elapsed_time = time.process_time() - st

		ret = trainer.test()


		predlist, actlist  =  learn.getTestData()


		act_name = "optim_act_" + str(neuron_size) + ".csv"
		actlist = actlist[2:,:]
		print(actlist.shape)
		np.savetxt(act_name,actlist, delimiter=",", fmt='%.10f')

		pred_name = "optim_pred_" + str(neuron_size) + ".csv"
		predlist = predlist[2:,:]
		print(predlist.shape)
		np.savetxt(pred_name,predlist, delimiter=",", fmt='%.10f')


		info_name = "optim_info_" + str(neuron_size) + ".csv"
		np.savetxt(info_name, np.array([elapsed_time, dropout*pytorch_total_params, neuron_size+neuron_size]) , delimiter=",", fmt='%.10f')



if __name__ == '__main__':
	main()
