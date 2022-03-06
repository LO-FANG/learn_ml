#导入所需要的包
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

#定义真实的w和b
	true_w = ([2, -3.4])
	true_b = 4.2
	features, lables = d2l.synthetic_data(true_w, true_b, 1000)

#读取数据集
	def load_array(data_arrays, batch_size, is_train=True):
		#变量前加*号表示，该函数接收参数根据具体传参来定
		dataset = data.TensorDdataset(*data_arrays)
		return data.DataLoader(dataset, batch_size, shuffle=is_train)

	batch_size = 10
	data_iter = load_array((features, lables), batch_size)

	next(iter(data_iter))

#定义模型

	#nn是神经网络的缩写
	form torch import nn

	#Sequential是一个有序的Linear集合
	#net是Sequential类的一个实例，接受一个参数Linear，
	#Linear接受一个二维（features）的数据，输出一个一维标量(预测y)
	net = nn.Sequential(nn.Linear(2, 1))

	#初始化模型参数
	net[0].weight.data.normal_(0, 0.01)
	net[0].bias.data.fill_(0)

#定义损失函数 MSE
	loss = nn.MSELoss()

#定义优化算法 梯度下降
	trainer = torch.optim.SGD(net.parameters(), lr=0.03)

#训练
	num_epochs = 3
	for epoch in range(num_epochs):
		for X, y in data_iter:
			l = lss(net(X), y)
			trainer.zero_grad()
			#计算梯度
			l.backward()
			#模型更新
			trainer.step()
		#输入样本，每次迭代查看偏差
		l = loss(net(features), lables)
		print(f'epoch{epoch + 1}, loss{1:f}')
