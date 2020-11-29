import os
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
from torchvision.models.resnet import Bottleneck, BasicBlock, ResNet
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub, fuse_modules

class QuantizableBasicBlock(BasicBlock):
    def __init__(self, *args, **kwargs):
        super(QuantizableBasicBlock, self).__init__(*args, **kwargs)
        self.add_relu = torch.nn.quantized.FloatFunctional()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.add_relu.add_relu(out, identity)

        return out

    def fuse_model(self):
        torch.quantization.fuse_modules(self, [['conv1', 'bn1', 'relu'],
                                               ['conv2', 'bn2']], inplace=True)
        if self.downsample:
            torch.quantization.fuse_modules(self.downsample, ['0', '1'], inplace=True)


class QuantizableBottleneck(Bottleneck):
    def __init__(self, *args, **kwargs):
        super(QuantizableBottleneck, self).__init__(*args, **kwargs)
        self.skip_add_relu = nn.quantized.FloatFunctional()
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.skip_add_relu.add_relu(out, identity)

        return out

    def fuse_model(self):
        fuse_modules(self, [['conv1', 'bn1', 'relu1'],
                            ['conv2', 'bn2', 'relu2'],
                            ['conv3', 'bn3']], inplace=True)
        if self.downsample:
            torch.quantization.fuse_modules(self.downsample, ['0', '1'], inplace=True)


class QuantizableResNet(ResNet):

    def __init__(self, *args, **kwargs):
        super(QuantizableResNet, self).__init__(*args, **kwargs)
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = super(QuantizableResNet,self).forward(x)
        x = self.dequant(x)
        return x

    def fuse_model(self):
        r"""Fuse conv/bn/relu modules in resnet models
        Fuse conv+bn+relu/ Conv+relu/conv+Bn modules to prepare for quantization.
        Model is modified in place.  Note that this operation does not change numerics
        and the model after modification is in floating point
        """

        fuse_modules(self, ['conv1', 'bn1', 'relu'], inplace=True)
        for m in self.modules():
            if type(m) == QuantizableBottleneck or type(m) == QuantizableBasicBlock:
                m.fuse_model()

def testing(model, testloader, writer=None, epoch=0):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    n_batches = 0
    test_loss = 0
    accuracy = 0

    for i, data in enumerate(testloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        n_batches += 1
        inputs, labels = data[0].to(device), data[1].to(device)

        # forward + backward + optimize
        outputs = model(inputs)
        test_loss += criterion(outputs, labels).item()
        accuracy += (F.softmax(outputs, dim=1).argmax(dim=1) == labels).float().mean()

    accuracy /= n_batches
    test_loss /= n_batches
    print('Test Loss', test_loss, epoch)
    print('Test Accuracy', accuracy.item(), epoch)
    if writer != None:
        writer.add_scalar('Test Loss', test_loss, epoch)
        writer.add_scalar('Test Accuracy', accuracy, epoch)
    

def training(model, optimizer, trainloader, writer, epoch):
    training_loss = 0.0
    accuracy = 0.0
    model.train()
    n_batches = 0
    criterion = nn.CrossEntropyLoss()
    for i, data in enumerate(trainloader, 0):
        n_batches += 1
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        training_loss += loss.item()
        loss.backward()
        optimizer.step()
        accuracy += (F.softmax(outputs, dim=1).argmax(dim=1) == labels).float().mean()

    training_loss = training_loss / n_batches
    accuracy = accuracy / n_batches
    print('Training Loss', training_loss, epoch)
    print('Training Accuracy', accuracy.item(), epoch)
    writer.add_scalar('Training Loss', training_loss, epoch)
    writer.add_scalar('Training Accuracy', accuracy, epoch)


if __name__ == '__main__':

    print(ResNet)

    models = ['resnet50']
    # run_name = f"{models[args.model]}_{args.name}_{args.batch_size}_{args.learning_rate}_{args.epochs}"
    run_name = 'quant_resnet50_CIFAR100_SGD_100_0.01_100'

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    batch_size = 100#args.batch_size
    trainset = torchvision.datasets.CIFAR100(root='./cifar100', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(root='./cifar100', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    float_model = QuantizableResNet(block=QuantizableBottleneck, layers=[3, 4, 6, 3]).to(device)
    # optimizer = optim.Adam(float_model.parameters(), lr=0.01)
    optimizer = optim.SGD(float_model.parameters(), lr=0.01, momentum=0.9)
    writer = SummaryWriter('quantized_resnet50/'+run_name+'/log')
    for epoch in range(100):  # loop over the dataset multiple times
        print(f'Epoch:{epoch + 1}')
        training(float_model,optimizer, trainloader, writer, epoch)
        testing(float_model, testloader, writer, epoch)
        writer.flush()
        if (epoch + 1) % 100 == 0:
            torch.save(float_model.state_dict(), f'./saved_models/{run_name}_{epoch+1}.pt')
    
writer.close()
print("Finished")

