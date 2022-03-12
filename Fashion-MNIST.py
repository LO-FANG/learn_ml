# 老三样 导包
%matplotlib inline
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

d2l.use_svg_display()

# 通过ToTensor实例将图像数据从PIL类型变换成32位浮点数格式
# 并除以255是的所有像素的数值均在0到1之间
trans = transforms.ToTensor()
minst_train = torchvision.datasets.FashionMNIST(
	root="../data", train=True, transform=trans, download=True)
minst_test = torchvision.datasets.FashionMNIST(
	root="../data", train=False, transform=trans, download=True)

# 打印一下训练集和测试集的长度
len(minst_train), len(minst_test)

# 训练集第一张图片的形状
minst_train[0][0].shape

# 以下函数用于在数字标签索引及其文本名称之间进行转换
def get_fashion_mnist_lables(lables):
	text_lables = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
					'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
	return [text_lables[int(i)] for i in lables]

# 我们现在可以创建一个函数来可视化这些样本。
def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):  #@save
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

 # 几个样本的图像及其相应的标签
 X, y = next(iter(data.DataLoader(mnist_train, batch_size=18)))
 show_images(X.rreshape(18, 28, 28), 2, 9, title=get_fashion_mnist_lables(y))

 # 通过内置的数据迭代器随机读取小批量数据集
batch_size = 256

def get_dataLoader_worker():
	"""使用4个进程来读取数据"""
	return 4;

train_iter = data.DataLoader(mnist_train, batch_size, shuffle=True,
							 num_workers=get_dataLoader_worker())

# 看一下数据读取的时间
timer = d2l.Timer()
for X, y in train_iter:
	continue
f'{timer.stop():.2f} sec'