import os
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from args import parser
import numpy as np
import torch
from torchvision.models.resnet import Bottleneck, BasicBlock, ResNet, model_urls
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url
from torch.quantization import QuantStub, DeQuantStub, fuse_modules
from torch._jit_internal import Optional

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


class QuantizableBottleneck(Bottleneck):
    def __init__(self, *args, **kwargs):
        # super(QuantizableBottleneck, self).__init__(*args, **kwargs)
        super().__init__(*args, **kwargs)
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
        # print("IN forward Pass")
        x = self.quant(x)
        # print("1", x.size())
        # Ensure scriptability
        # super(QuantizableResNet,self).forward(x)
        # is not scriptable
        # x = self._forward_impl(x)
        x = super(QuantizableResNet,self).forward(x)
        # print("2", x.shape)
        # x = self.forward(x)
        x = self.dequant(x)
        # print("3", x.shape)
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

def quantize_model(model, backend):
    _dummy_input_data = torch.rand(1, 3, 299, 299)
    if backend not in torch.backends.quantized.supported_engines:
        raise RuntimeError("Quantized backend not supported ")
    torch.backends.quantized.engine = backend
    model.eval()
    # Make sure that weight qconfig matches that of the serialized models
    if backend == 'fbgemm':
        model.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.default_observer,
            weight=torch.quantization.default_per_channel_weight_observer)
    elif backend == 'qnnpack':
        model.qconfig = torch.quantization.QConfig(
            activation=torch.quantization.default_observer,
            weight=torch.quantization.default_weight_observer)

    model.fuse_model()
    torch.quantization.prepare(model, inplace=True)
    model(_dummy_input_data)
    torch.quantization.convert(model, inplace=True)
    return model

def benchmark(model, testloader):

    model.eval()
    criterion = nn.CrossEntropyLoss()
    n_batches = 0
    test_loss = 0
    accuracy = 0
    device = 'cpu'

    for i, data in enumerate(testloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        n_batches += 1
        inputs, labels = data[0].to(device), data[1].to(device)
        # if quant == True:
        #     inputs = torch.quantize_per_tensor(inputs, scale=1e-3, zero_point=128, dtype=torch.quint8)
            # labels = torch.quantize_per_tensor(labels, scale=1e-3, zero_point=128, dtype=torch.quint8)

        outputs = model(inputs)
        test_loss += criterion(outputs, labels).item()
        accuracy += (F.softmax(outputs, dim=1).argmax(dim=1) == labels).float().mean()

    accuracy /= n_batches
    test_loss /= n_batches
    print('Test Loss', test_loss)
    print('Test Accuracy', accuracy.item())


def print_size_of_model(model, label=""):
    torch.save(model.state_dict(), "temp.p")
    size=os.path.getsize("temp.p")
    # print("model: ",label,' \t','Size(MB):{0:.2f}'.format(size/1e6))
    os.remove('temp.p')
    return size


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)


model_path = './saved_models/quant_resnet50_CIFAR10_ADAM_100_0.01_100_100.pt'


float_model = QuantizableResNet(block=QuantizableBottleneck, layers=[3, 4, 6, 3])

float_model.load_state_dict(torch.load(model_path))
print("Float Model(MB):{0:.2f}".format(print_size_of_model(float_model) / 1e6))

print("Test Results for Float Model:")
benchmark(float_model, testloader)

int_model = quantize_model(float_model, 'fbgemm')
print("Quantized Model(MB):{0:.2f}".format(print_size_of_model(int_model) / 1e6))
print("Test Results for Quantized Model:")
benchmark(int_model, testloader)
