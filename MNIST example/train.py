import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, (3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = F.relu
        self.flatten1 = torch.flatten
        # conv2 = nn.Conv2d(3, 3, (3, 3), padding=1)
        self.fc1 = nn.Linear(25088, 128)
        self.relu2 = F.relu
        self.fc2 = nn.Linear(128, 10)
        self.log_softmax = F.log_softmax
    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn1(x)
        print("Torch Conv\n", x)
        x = self.relu1(x)
        # print("After convolution:", x.size())
        # x = self.conv2(x)
        # x = F.relu(x)
        # x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        x = self.flatten1(x, 1)
        # x = x.unsqueeze(1)
        # print("After Flatten", x.size())
        x = self.fc1(x)
        print("After FC1", x.size())
        x = self.relu2(x)
        # print("Torch FC1:\n", x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        print("Torch FC2", x)
        output = self.log_softmax(x, dim=1)
        return output

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    kwargs = {'batch_size': 32}
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ]
    )
    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)
    model = Model().to(device)
    # model = Model()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.1)


    for epoch in range(10):
        train(model, device, test_loader, optimizer, epoch)
        test(model, device, test_loader)
    
    # x = torch.randn(10, 1, 28, 28, requires_grad=True)
    torch.save(model.state_dict(), "mnist_cnn_cpu.pt")
    # model.load_state_dict(torch.load('./mnist_cnn.pt'))
    
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        print(param_tensor, "\t", torch.max(model.state_dict()[param_tensor]), "\t", torch.min(model.state_dict()[param_tensor]))


    
if __name__ == '__main__':
    main()