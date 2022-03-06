#导入所需要的包

%matplotlib inline
import random
import torch

#定义一个生成随机数据集的函数
def synthetic_data(w, b, num_examples):

    #normal函数 返回一个张量，包含从给定参数means(均值),std(标准差)的离散正态分布中抽取随机数     
        #方差一般是用来度量随机变量和其数学期望（即均值）之间的偏离程度--离散程度。
        #统计中的方差（样本方差）是各个数据分别与其平均数之差的平方的和的平均数。
        #方差是衡量源数据和期望值（可近似看作平均值）相差的度量值。
        #标准差是方差的开方，描述的是样本集合的各个样本点到均值的距离之平均；同时，也可以反映一个数据集的离散程度
    X = torch.normal(0, 1, (num_examples, len(w))) #len(w) = 2
    y = torch.matmul(X, w) + b

    #给样本增加噪音
    y += torch.normal(0, 0.01, y.shape)

    #返回features和lables，reshape(-1, 1)将y转换成行向量
    return X, y.reshape(-1, 1)

#定义真实的权重w和偏差b
true_w = torch.tensor([2, -3.4])
true_b = 4.2

#获得features和lables    注意，[features中的每一行都包含一个二维数据样本， labels中的每一行都包含一维标签值（一个标量）]。
features, lables = synthetic_data(true_w, true_b, 1000)


#定义函数读取数据集
def data_iter(batch_size, features, lables):

    #获得样本的个数
    num_examples = len(features)

    #生成一个list，里面存放样本的index
    indices = list(range(num_examples))

    #将样本list的index打乱，达到随机读取的效果
    random.shuffle(indices)

    #从 0开始 ~ 到num_examples结束，每次跳batch_size个大小
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])

        #yield每次返回结果后，继续执行，直到for循环结束
        yield features[batch_indices], lables[batch_indices]

#定义batch_size
batch_size = 10

#调用data_iter函数，获取X,y 每次随即返回一个样本参与计算
for X, y in data_iter(batch_size, features, lables):
    print(X, '\n', y)
    break


#初始化参数模型
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

#定义linear-regression模型
def linreg(X, w, b):
    return torch.matmul(X, w) + b

#定义loss function 
# y_hat是预测值 y是真实值
def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


#定义优化算法梯度下降
# params是一个参数的list; lr是学习率
def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()


#训练

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

#轮询数据
for epoch in range(num_epochs):

    #取出一个子集
    for X, y in data_iter(batch_size, features, lables):

        #得到X，y的损失
        l = loss(net(X, w, b), y)

        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，并以此计算关于[w,b]的梯度
        l.sum().backward()

        #使用参数的梯度更新参数
        sgd([w,b], lr, batch_size)

    #评估进度    
    with torch.no_grad():
        train_l = loss(net(features, w, b), lables)
        print(f'epoch{epoch + 1}, loss{float(train_l.mean()):f}')


print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')