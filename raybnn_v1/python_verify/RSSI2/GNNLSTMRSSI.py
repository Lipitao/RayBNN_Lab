

import numpy as np


import math
import time
import sys





import torch
import torch.nn.functional as F
import torch_geometric
from torch.nn import ModuleList, Embedding
from torch.nn import Sequential, ReLU, Linear
from torch_geometric.utils import degree
from torch_geometric.data import DataLoader
from torch_geometric.nn import PNAConv, BatchNorm, global_add_pool


from torch.nn import Linear
from torch_geometric.nn import GCN2Conv
from torch_geometric.nn.conv.gcn_conv import gcn_norm






from torch.utils.data import Dataset


import copy


import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger




class RSSIDataset(Dataset):

    def __init__(self, input_data):
        self.data = input_data

        self.size = len(self.data)

        

    def __len__(self):
        return self.size


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = int(idx)

        return self.data[idx]



def generate_graph(valX, valY, traj_size, batch_size):


	param_size = valX.shape[1]

	edge_index = np.zeros((2,  traj_size-1  ),dtype=np.int64)
	for j in range(traj_size - 1):
		edge_index[0,j] = j
		edge_index[1,j] = j + 1



	edge_index2 = np.zeros((2,  traj_size-1  ),dtype=np.int64)
	for j in range(traj_size - 1):
		edge_index2[0,j] = j + 1
		edge_index2[1,j] = j

	edge_index = np.concatenate((edge_index,edge_index2),axis=1)


	edge_attr = np.ones((edge_index.shape[1] )).astype(np.float32)


	graph_list = []







	totalnum = valX.shape[0]

	sample_size = traj_size*batch_size

	count = 0

	for j in range(int(totalnum/sample_size)):

		tempX = valX[count:(count+sample_size),:]
		tempY = valY[count:(count+sample_size),:]

		count = count + sample_size


		param_size = valX.shape[1]

		X = np.reshape(tempX, (traj_size, batch_size, param_size)).astype(np.float32)

		param_size = valY.shape[1]

		Y = np.reshape(tempY, (traj_size, batch_size, param_size)).astype(np.float32)

		
		for i in range(batch_size):
			graph = torch_geometric.data.Data(
									x=torch.tensor(X[:,i,:]),
									edge_index=torch.tensor(copy.deepcopy(edge_index)),
									edge_attr=torch.tensor(copy.deepcopy(edge_attr)),
									y=torch.tensor(Y[:,i,:]))

			graph_list.append(graph)




	return graph_list









def main():
	print("PNA RSSI")
	print(torch.cuda.get_device_name(0))

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



	input_size = int(sys.argv[1])
	
	fold = int(sys.argv[2])



	neuron_size = 2*input_size
	output_size = 2


	train_X = np.loadtxt("Alcala_TRAINX.dat", delimiter=',').astype(np.float32)
	train_Y = np.loadtxt("Alcala_TRAINY.dat", delimiter=',').astype(np.float32)


	test_X = np.loadtxt("Alcala_TESTX.dat", delimiter=',').astype(np.float32)
	test_Y = np.loadtxt("Alcala_TESTY.dat", delimiter=',').astype(np.float32)






	traj_size = 20
	batch_size = 100

	sample_size = traj_size*batch_size



	train_graph = RSSIDataset(generate_graph(train_X, train_Y, traj_size, batch_size))
	test_graph = RSSIDataset(generate_graph(test_X, test_Y, traj_size, batch_size))





	train_loader = DataLoader(train_graph, batch_size=batch_size, shuffle=False)
	test_loader = DataLoader(test_graph, batch_size=1, shuffle=False)



	class Net(torch.nn.Module):
		def __init__(self):
			super(Net, self).__init__()


			self.act = torch.nn.LeakyReLU(0.001)

			self.lins = torch.nn.ModuleList()
			self.lins.append(Linear(input_size, neuron_size))
			self.lins.append(Linear(neuron_size, output_size))

			self.LSTM0 = torch.nn.LSTM(neuron_size, neuron_size, 1)
			self.LSTM1 = torch.nn.LSTM(neuron_size, neuron_size, 1)

			self.convs = torch.nn.ModuleList()
			for layer in range(4):
				self.convs.append(
				GCN2Conv(neuron_size, layer + 1,
				shared_weights=True, normalize=False))
	
		def forward(self, graph):
			x = x_0 = self.act(self.lins[0](graph.x[:,:input_size]))

			h = x_0.unsqueeze(0)
			c = x_0.unsqueeze(0)

			for conv in self.convs:
				x = self.act(conv(x, x_0, graph.edge_index,  graph.edge_attr ))
				x, (h, c) = self.LSTM0(x.unsqueeze(0), (h, c) )
				x, (h, c) = self.LSTM1(x, (h, c) )
				x = x.squeeze(0)

			return self.lins[1](x)




	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = Net().to(device)
	optimizer = torch.optim.Adam([
		dict(params=model.convs.parameters()),
		dict(params=model.LSTM0.parameters()),
		dict(params=model.LSTM1.parameters()),
		dict(params=model.lins.parameters())
	], lr=0.001)



	def train(epoch):
		model.train()
		total_loss = 0.0
		for data in train_loader:
			data = data.to(device)
			optimizer.zero_grad()
			out = model(data)
			loss = 50.0*(out - data.y).abs().mean()
			loss.backward()
			total_loss += loss.item() 
			optimizer.step()
		return total_loss
		


	@torch.no_grad()
	def test(loader):

		predlist = np.zeros((2,output_size))
		actlist = np.zeros((2,output_size))

		model.eval()

		for data in loader:
			data = data.to(device)
			out = model(data)

			predlist = np.concatenate((predlist,out.cpu().detach().numpy()),axis=0)

			actlist = np.concatenate((actlist,data.y.cpu().detach().numpy()),axis=0)
		return  predlist, actlist





	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)


	st = time.process_time()

	for epoch in range(400):
		loss = train(epoch)
		print(loss)

	elapsed_time = time.process_time() - st




	predlist, actlist  = test(test_loader)


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