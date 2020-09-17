import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.optim as optim

kwargs = {}

train_data = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1397),(0.3081))])
    ),batch_size=64, shuffle=True, **kwargs)

test_data = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False,
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1397),(0.3081))])
    ),batch_size=64, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5)
        self.conv2 = nn.Conv2d(10,20, kernel_size=5)
        self.drop = nn.Dropout2d()
        self.fcon1 = nn.Linear(320, 60)
        self.fcon2 = nn.Linear(60,10)
    
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = self.drop(x)

        #print(x.size())
        #x = x.view(-1,320)
        x = torch.flatten(x, start_dim=1)
        x = self.fcon1(x)
        x = F.relu(x)

        x = self.fcon2(x)
        x = F.log_softmax(x, dim=1)
        return x

model = Net()
#model = torch.load('net1.pt')
#model = model()

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.8)

def train(epoch):
    model.train()
    for batch_id, (data, target) in enumerate(train_data):
        data = Variable(data)
        target = Variable(target)
        optimizer.zero_grad()
        out = model(data)
        criterion = F.nll_loss
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        print("epoch: " + str(epoch) + " loss: " + str(loss.item()))

for epoch in range(1, 5):
    train(epoch)

torch.save(Net, 'net1.pt')
