# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 22:40:39 2021

@author: MaYiming
"""
#导入库
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_loader import mydata, myTestdata
import matplotlib.pyplot as plt
import random

#设置所有需要用到的参数
params = {}
#训练集和测试机所需图像和标签的路径
params['train_image_path'] = "..\\minist\\train-images\\"
params['train_label_path'] = "..\\minist\\train-labels.txt"
params['test_image_path'] = "..\\minist\\t10k-images\\"
params['test_label_path'] = "..\\minist\\t10k-labels.txt"
#全连接输入隐藏输出参数设置
params['INPUT_SIZE'] = 28*28
params['HIDDEN_SIZE'] = 32*32
params['OUTPUT_SIZE'] = 10
#批量、学习率、训练迭代次数参数的设置
params['BATCH_SIZE'] = 100
params['LEARNING_RATE'] = 0.001
params['EPOCH'] = 10
#GPU是否可用
params['DEVICE'] = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#print(params['DEVICE'])
#查看train_data中的元素
#train_data = mydata(params)
#plt.imshow(train_data[1][0][0])
#导入训练集和测试集，并对集合进行批量、洗牌操作
train_loader = DataLoader(mydata(params),
                          shuffle = True,
                          batch_size = params['BATCH_SIZE'])
test_loader = DataLoader(myTestdata(params),
                          shuffle = False,
                          batch_size = 1)
#查看train_loader中的元素
#data = iter(train_loader)
#a=(inputs,labels)=next(data)
#print(a)

#model的搭建
class FCNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(FCNet, self).__init__()                          #继承
        self.layer1 = nn.Linear(input_size, hidden_size)       #输入层到隐藏层
        self.relu = nn.ReLU()                                  #激活函数
        self.dropout = nn.Dropout(0.5)                         #随机失活
        self.layer2 = nn.Linear(hidden_size, output_size)      #隐藏层到输出层
    
    def forward(self, x):
        #让x通过所有计算得到结果
        tem = self.layer1(x)
        tem = self.relu(tem)
        tem = self.dropout(tem)
        end = self.layer2(tem)
        return end

#实例化模型
model = FCNet(params['INPUT_SIZE'],params['HIDDEN_SIZE'],params['OUTPUT_SIZE']).to(params['DEVICE'])
#损失函数选择
criterion = nn.CrossEntropyLoss()
#优化器选择
optimizer = torch.optim.Adam(model.parameters(), lr=params['LEARNING_RATE'])
#查看参数
#print(model.parameters)

#用于记录损失和优化效果，图像输出使用
total_step = len(train_loader)
losses = []
losses_t =[]
accuracies = []
#开始训练
for epoch in range(params["EPOCH"]):
    #训练模式（开启dropout）
    model = model.train()
    los = 0
    for i,(sample, label) in enumerate(train_loader):
        #-1代表模糊处理，其实是批量大小params['BATCH_SIZE']，但是这样写不费脑子
        #to(device)表示在哪个上边训练、CPU或者GPU
        sample = sample.reshape(-1,params['INPUT_SIZE']).to(params['DEVICE'])
        label = label.to(params['DEVICE'])
        #计算差值
        output = model(sample)
        loss = criterion(output,label)
        los += loss.item()
        #梯度清零
        optimizer.zero_grad()
        #反向传播
        loss.backward()
        #优化
        optimizer.step()
        #输出训练过程
        if (i+1)%10 == 0:
            losses.append(los)
            los = 0
                #测试过程不需要梯度的计算
            with torch.no_grad():
                #测试模式（没有dropout）
                model = model.eval()
                los_t = 0
                for sample,label in test_loader:
                    rand = random.randint(1,1000)
                    if rand == 10:
                        sample = sample.reshape(-1,params['INPUT_SIZE']).to(params['DEVICE'])
                        label = label.to(params['DEVICE'])
                        output = model(sample)
                        loss_t = criterion(output,label)
                        los_t += loss_t.item()
                losses_t.append(los_t)
            model = model.train()
        if (i+1)%100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch+1, params['EPOCH'], i+1, total_step, loss.item()))
    #测试过程不需要梯度的计算
    with torch.no_grad():
        correct = 0
        total = 0
        #测试模式（没有dropout）
        model = model.eval()
        for sample,label in test_loader:
            sample = sample.reshape(-1,params['INPUT_SIZE']).to(params['DEVICE'])
            label = label.to(params['DEVICE'])
            output = model(sample)
            #通过max函数得到预测输出，返回data和下标，但是我们只想要下标
            #其中1表示对行求最大
            _,predict = torch.max(output.data,1)
            total +=1
            #预测正确就加一
            correct += (predict.item() == label.item())
        #正确率
        accuracy = correct/total
        accuracies.append(accuracy)
        print('Accuracy of the network on the 10000 test images: {} %'.format(100 * accuracy))

#打印每一百轮的loss
length = len(losses)
lis_x = list(range(1,length+1))
plt.figure()
plt.title('loss')

plt.plot(lis_x,losses_t, color = 'red')
plt.plot(lis_x,losses,color = 'blue')

#打印精确度
plt.figure()
plt.title('accuracy')
plt.plot(accuracies)
#输出
plt.show()
# 保存模型
torch.save(model.state_dict(), 'model.pth')