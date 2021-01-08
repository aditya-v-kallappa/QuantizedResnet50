import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
import time
import copy
import time
from pytorch_mnist import Model

class Conv2d():
    def __init__(self, in_channels=1, out_channels=1, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), dilation=(1, 1), bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = tuple(kernel_size) if type(kernel_size) != int else (kernel_size, kernel_size)
        self.padding = tuple(padding) if type(padding) != int else (padding, padding)
        self.stride = stride if type(stride) != int else (stride, stride)
        self.dilation = dilation if type(dilation) != int else (dilation, dilation)

        self.weight = np.random.randint(low=-128, high=127, size=(out_channels, in_channels, kernel_size[0], kernel_size[1])) 
        
        if bias == True:
            # self.bias = np.random.randint(low=-128, high=127, size=out_channels)
            self.bias = np.zeros(shape=out_channels, dtype=int)
        elif bias == False or bias is None:
            self.bias = np.zeros(shape=out_channels, dtype=int)

    def __call__(self, input):
        N, C_in, H_in, W_in = input.shape
    
        C_out = self.out_channels
        H_out = (H_in + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1)) // self.stride[0]
        W_out = (W_in + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1)) // self.stride[1]

        output_shape = (N, C_out, H_out, W_out)
        # output = np.zeros(shape=output_shape, dtype=np.int16)  
        output = np.zeros(shape=output_shape)
        if self.padding[0] !=0 and self.padding[1] != 0:
            padded_input = self.pad_input(input)
        else:
            padded_input = input
        for n in range(N):
            for c in range(C_out):
                for i in range(H_out):
                    for j in range(W_out):
                        output[n, c, i, j] = \
                            np.sum(self.weight[c] * padded_input[n, :, i*self.stride[0]:i*self.stride[0]+self.kernel_size[0], j*self.stride[1]:j*self.stride[1]+self.kernel_size[1]]) + self.bias[c]
                    

        return output


    def pad_input(self, input):
        
        N, C, H, W = input.shape
        padded_input = np.zeros(shape=(N, C, H+2*self.padding[0], W+2*self.padding[1]), dtype=input.dtype)
        
        for n in range(N):
            for c in range(C):
                padded_input[n, c, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]] = input[n, c]

        return padded_input


class MaxPool2d():
    def __init__(self, kernel_size=(3, 3), padding=0, stride=0, dilation=(1, 1), ceil_mode=False):
        self.kernel_size = tuple(kernel_size) if type(kernel_size) != int else (kernel_size, kernel_size)
        self.dilation = dilation if type(dilation) != int else (dilation, dilation)
        self.ceil_mode = ceil_mode
        self.padding = padding if type(padding) != int else (padding, padding)
        if stride == 0:
            self.stride = kernel_size
        else:
            self.stride = stride if type(stride) != int else (stride, stride)

    def __call__(self, input):
        N, C_in, H_in, W_in = input.shape
    
        C_out = C_in
        if self.ceil_mode:
            H_out = np.ceil((H_in + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0])) / self.stride[0]) + 1
            W_out = np.ceil((W_in + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1])) / self.stride[1]) + 1
        else:
            H_out = np.floor((H_in + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0])) / self.stride[0]) + 1
            W_out = np.floor((W_in + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1])) / self.stride[1]) + 1
        
        
        # H_out = (H_in + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1) - 1) // self.stride[0] - 1
        # W_out = (W_in + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1) - 1) // self.stride[1] - 1
    
        H_out = int(H_out)
        W_out = int(W_out)
        # print(type(H_out))
        # print(type(W_out))

        output_shape = (N, C_out, H_out, W_out)
        print(output_shape)
        output = np.zeros(shape=output_shape)#, dtype=int)  
        if self.padding[0] != 0:
            padded_input = self.pad_input(input, self.padding)
        else:
            padded_input = input
        
        for n in range(N):
            for c in range(C_out):
                for h in range(H_out):
                    for w in range(W_out): 
                        # if h == 0 and w == 5:
                        # print(input[n, c, h*self.stride[0]:h*self.stride[0]+self.kernel_size[0], w*self.stride[1]:w*self.stride[1]+self.kernel_size[1]])
                        output[n, c, h, w] = np.max(padded_input[n, c, h*self.stride[0]:h*self.stride[0]+self.kernel_size[0], w*self.stride[1]:w*self.stride[1]+self.kernel_size[1]])
                        # if i == 1 and j == 0:
                        
                        # print(output[n, c, h, w])
                        # print(n, c, h, w)

        return output

    def pad_input(self, input, pad):
        
        N, C, H, W = input.shape
        padded_input = np.zeros(shape=(N, C, H+2*pad[0], W+2*pad[1]), dtype=input.dtype)
        
        for n in range(N):
            for c in range(C):
                padded_input[n, c, pad[0]:-pad[0], pad[1]:-pad[1]] = input[n, c]

        return padded_input


class AdaptiveAvgPool2d():
    def __init__(self, output_shape=(1, 1)):
        self.output_shape = output_shape if type(output_shape) != int else (output_shape, output_shape)
    
    def __call__(self, input):
        if self.output_shape[0] is None:
            self.output_shape[0] = input.shape[-2]
        if self.output_shape[1] is None:
            self.output_shape[1] = input.shape[-1]
        
        N, C, H_in, W_in = input.shape
        H, W = self.output_shape

        # if H > H_in:


        output = np.zeros(shape=(N, C, H, W))

        print(type(H_in), type(H))
        kh = (H_in + H - 1) // H 
        kw = (W_in + W - 1) // W
        kernel_size = (kh, kw)
        slices_h = [(H_in*i) // kernel_size[0] for i in range(H_in + 1)]
        slices_w = [(W_in*i) // kernel_size[1] for i in range(W_in + 1)]

        print(slices_h)
        print(slices_w)

        for n in range(N):
            for c in range(C):
                for h in range(H):
                    for w in range(W):
                        output[n, c, h, w] = np.mean(input[n, c, slices_h[h]:slices_h[h+1], slices_w[w]:slices_w[w+1]])

        return output

def verify_adaptiveavgpool2d(x=None):
    # if x is None:
    # m = nn.AdaptiveAvgPool2d((32,32))
    # input = torch.randn(1, 64, 8, 9)
    # output = m(input)
    # print("1:", output.shape)
    # # target output size of 7x7 (square)
    # m = nn.AdaptiveAvgPool2d(7)
    # input = torch.randn(1, 64, 10, 9)
    # output = m(input)
    # print("2", output.shape)
    # # target output size of 10x7
    # m = nn.AdaptiveAvgPool2d((None, 7))
    # input = torch.randn(1, 64, 10, 9)
    # output = m(input)
    # print("3", output.shape) 


    input = torch.randn(1, 1, 1, 5)
    m1 = AdaptiveAvgPool2d(output_shape=(1, 2))
    m2 = nn.AdaptiveAvgPool2d((1, 2))

    out1 = m1(input.detach().numpy())
    out2 = m2(input)
    print(input)
    print(out1)
    print(out2)
    print('MSE after AdaptiveAvgPool2d', np.mean((np.round(out1, 4) - out2.detach().numpy())**2))


class BatchNorm2d():
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True):
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats

        self.weight = np.ones(shape=self.num_features)#, dtype=int)
        self.bias = np.zeros(shape=self.num_features)#, dtype=int)
        if self.track_running_stats:
            self.mean = None
            self.var = None

    def __call__(self, input):
        assert len(input) != 4, "Input must have 4 dimensions (N, C, H, W)"

        self.mean = np.mean(input, axis=(-1, -2))
        self.var =  np.var(input, axis=(-1, -2))
        # output = np.zeros_like(input)
        # N, C, H, W = output.shape
        """
        A detailed way to do it:
        
        for n in range(N):
            for c in range(C):
                output[n, c] = ((input[n, c] - self.mean[n, c]) / np.sqrt(self.var[n, c] + self.eps)) * self.weight[c] + self.bias[c]
        """
        output = ((input - self.mean[:, :, np.newaxis, np.newaxis]) / np.sqrt(self.var[:, :, np.newaxis, np.newaxis] + self.eps)) * \
            self.weight[np.newaxis, :, np.newaxis, np.newaxis] + self.bias[np.newaxis, :, np.newaxis, np.newaxis]

        return output

# def Conv2d(input, in_channels=1, out_channels=1, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), dilation=(1, 1), weight=None, bias='random'):
    
#     if weight is None:
#         weight = np.random.randint(low=-128, high=127, size=(out_channels, in_channels, kernel_size[0], kernel_size[1])) 
#     if bias == 'random':
#         bias = np.random.randint(low=-128, high=127, size=out_channels) 
#     if bias is False or bias is None:
#         bias = np.zeros(shape=out_channels, dtype=int)
        
    
#     N, C_in, H_in, W_in = input.shape
    
#     C_out = out_channels
#     H_out = (H_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1)) // stride[0]
#     W_out = (W_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1)) // stride[1]

#     output_shape = (N, C_out, H_out, W_out)
#     output = np.zeros(shape=output_shape)#, dtype=int)  

#     padded_input = pad_input(input, padding)
#     for n in range(N):
#         for c in range(C_out):
#             for i in range(H_out):
#                 for j in range(W_out):
#                     output[n, c, i, j] = np.sum(weight[c] * padded_input[n, :, i:i+kernel_size[0], j:j+kernel_size[1]]) + bias[c]
#     return output


class Linear():
    def __init__(self, out_features=1, in_features=1, bias=True):
        self.out_features = out_features
        self.weight = np.zeros((out_features, in_features))#, dtype=int)
        if bias:
            # self.bias = np.random.rand(out_features) #, dtype=int
            self.bias = np.zeros(out_features)#, dtype=int
        else:
            self.bias = np.zeros(out_features)#, dtype=int

    def __call__(self, input):
        N = input.shape[0]
        output = np.zeros((N, self.out_features), dtype=np.int16)
        for i in range(N):
            output[i] = self.weight @ input[i] + self.bias
        return output
        

class ReLu():
    def __init__(self):
        pass

    def __call__(self, input):
        return input * (input > 0)

class Softmax():
    def __init__(self, log=False):
        self.log = log
        pass

    def __call__(self, input):
        N = input.shape[0]
        output = np.zeros(input.shape)
        for i in range(N):
            output[i] = np.exp(input[i]) / np.sum(np.exp(input[i]))
        return np.log(output) if self.log else output 

# def Linear(in_features=1, out_features=1, bias=No, weight=None):

#     if weight is None:
#         weight = np.ones(shape=(out_features, in_fetures))#, dtype=int)
    


#     return 0


def pad_input(input, padding):
    if len(padding) == 1:
        pad = (padding, padding)
    else:
        pad = padding
    N, C, H, W = input.shape
    padded_input = np.zeros(shape=(N, C, H+2*pad[0], W+2*pad[1]), dtype=input.dtype)
    
    for n in range(N):
        for c in range(C):
            padded_input[n, c, pad[0]:-pad[0], pad[1]:-pad[1]] = input[n, c]

    return padded_input


def fuse_conv_bn_eval(conv, bn):
    # assert(not (conv.training or bn.training)), "Fusion only for eval!"
    fused_conv = copy.deepcopy(conv)

    fused_conv.weight, fused_conv.bias = \
        fuse_conv_bn_weights(fused_conv.weight, fused_conv.bias,
                             bn.running_mean, bn.running_var, bn.eps, bn.weight, bn.bias)

    return fused_conv


def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
    if conv_b is None:
        conv_b = torch.zeros_like(bn_rm)
    if bn_w is None:
        bn_w = torch.ones_like(bn_rm)
    if bn_b is None:
        bn_b = torch.zeros_like(bn_rm)

    bn_var_rsqrt = torch.rsqrt(bn_rv + bn_eps)

    conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
    conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

    return torch.nn.Parameter(conv_w), torch.nn.Parameter(conv_b)


# def fuse_conv_bn_weights(conv_w, conv_b, bn_rm, bn_rv, bn_eps, bn_w, bn_b):
#     if conv_b is None:
#         conv_b = np.zeros_like(bn_rm)
#     if bn_w is None:
#         bn_w = np.ones_like(bn_rm)))
#     if bn_b is None:
#         bn_b = np.zeros_like(bn_rm)

#     bn_var_rsqrt = 1 / np.sqrt(bn_rv + bn_eps)

#     conv_w = conv_w * (bn_w * bn_var_rsqrt).reshape([-1] + [1] * (len(conv_w.shape) - 1))
#     conv_b = (conv_b - bn_rm) * bn_var_rsqrt * bn_w + bn_b

#     return conv_w, conv_b


class MNIST():
    def __init__(self):
        self.conv1 = Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=True)
        self.relu1 = ReLu()
        # self.flatten = np.flatten
        self.fc1 = Linear(128, 25088)
        self.relu2 = ReLu()
        self.fc2 = Linear(10, 128)
        self.softmax = Softmax(log=True)
        # print("Done")
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)
        x = self.relu2(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
    
    def __call__(self, x):
        return self.forward(self, x)

def verify_conv(x=None):
    if x is None:
        x = np.arange(100*3*5*5).reshape(-1, 3, 5, 5)
    _x = torch.Tensor(x)
    conv = nn.Conv2d(in_channels=x.shape[1], out_channels=1, kernel_size=(3, 3), padding=(1, 1), bias=False)
    bn = nn.BatchNorm2d(num_features=1, track_running_stats=True)
    fused_conv = fuse_conv_bn_eval(conv, bn)
    weight = fused_conv.weight.detach().numpy()
    bias = fused_conv.bias.detach().numpy() if conv.bias is not None else None
    # t1 = time.time()
    # out = Conv2d(x, in_channels=3, out_channels=2, kernel_size=(3, 3), padding=(1, 1), weight=weight, bias=bias)
    custom = Conv2d(in_channels=x.shape[1], out_channels=1, kernel_size=(3, 3), padding=(1, 1), bias=False)
    custom.weight = weight
    custom.bias = bias if bias is not None else custom.bias
    out = custom(x)
    # t2 = time.time()
    conv_out = conv(_x)
    bn.eval()
    conv_out = bn(conv_out)
    # t3 = time.time()
    # print(f"Custom took {t2 - t1} sec")
    # print(f"Torch took {t3 - t2} sec")
    print('MSE after convolution', np.mean((np.round(out, 4) - conv_out.detach().numpy())**2))


def verify_linear(x=None):
    out_features = 1000
    in_features = 10000
    bias = True
    x = np.random.rand(in_features)

    linear = nn.Linear(out_features=out_features, in_features=in_features, bias=bias)
    _linear = Linear(out_features=in_features, in_features=in_features, bias=bias)
    _linear.weight = linear.weight.detach().numpy()
    _linear.bias = linear.bias.detach().numpy() if linear.bias is not None else 0

    out1 = _linear(x)
    out2 = linear(torch.Tensor(x))
    print('MSE after linear', np.mean((np.round(out1, 4) - out2.detach().numpy())**2))

def verify_softmax(x=None):
    if x is None:
        x = np.random.rand(100)
    out1 = Softmax(log=True)(x)
    out2 = F.log_softmax(torch.Tensor(x), dim=-1)
    print('MSE after softmax', np.mean((np.round(out1, 4) - out2.detach().numpy())**2))

def verify_mnist(model, mnist, test_loader):
    mse = 0
    test_loss = 0
    correct1 = 0
    correct2 = 0
    model.eval()
    idx = 0
    for data, target in test_loader:
        # data, target = data.to(device), target.to(device)
        idx += 1
        numpy_data = data.detach().numpy()
        numpy_target = target.detach().numpy()
        output1 = mnist.forward(numpy_data)    
        output2 = model(data)
        pred1 = np.argmax(output1, axis=1)
        pred2 = output2.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct1 += np.sum(pred1 == numpy_target)
        correct2 += pred2.eq(target.view_as(pred2)).sum().item()
        # test_loss += F.nll_loss(output2, target, reduction='sum').item()
        mse += np.mean((np.round(output1, 4) - output2.detach().numpy())**2)
        if idx % 5 == 0:
            print(f"{10*idx} out of {len(test_loader.dataset)} Done")
        # if idx == 1:
        #     break
    
    # test_loss /= (10 * idx)
    print("MSE for test data: ", mse)
    print("Correct for custom model ", correct1)
    print("Correct for torch model ", correct2)
    # print("Torch Loss", test_loss)

def verify_batchnorm2d(x=None):
    if x is None:
        x = np.random.rand(1, 32, 32, 32)

    bn1 = BatchNorm2d(32)
    bn2 = nn.BatchNorm2d(32, track_running_stats=True)
    # bn2.eval()
    bn1.weight = bn2.weight.detach().numpy()
    bn1.bias = bn2.bias.detach().numpy()
    bn1.eps = bn2.eps
    bn1.mean = bn2.running_mean
    bn1.var = bn2.running_var
    # print(bn1)
    out1 = bn1(x)
    out2 = bn2(torch.Tensor(x))
    print(out2.shape)
    print('MSE after batchnorm', np.mean((np.round(out1, 4) - out2.detach().numpy())**2))


def verify_maxpool2d(x=None):
    if x is None:        
        x = torch.randn(100, 64, 8, 8)
    
    # print(x)
    stride = (1, 1)
    kernel = (3, 3)
    
    m1 = MaxPool2d(kernel_size=kernel, stride=stride, ceil_mode=False, padding=1)
    m2 = nn.MaxPool2d(kernel_size=kernel, stride=stride, ceil_mode=False, padding=1)
    # m2 = nn.MaxPool2d((3, 2), stride=(2, 1))
    out1 = m1(x.detach().numpy())
    out2 = m2(x)
    
    # print(out1)
    # print(out2)
    print('MSE after maxpool', np.mean((np.round(out1, 4) - out2.detach().numpy())**2))

def normalize(input=None, max_range=1, min_range=0):
    if input is None:
        input = np.random.rand(10, 10)
    
    max = np.max(input)
    min = np.min(input)
    
    scale = (max_range - min_range) / (max - min)
    bias = min * scale - min_range
    # print(scale, bias, min, max, max_range, min_range)

    # input1 = (input - min) / (max - min) * (max_range - min_range) + min_range
    out = input * scale - bias
    
    return scale, bias, out


def dequantize(input=None, scale=1, bias=0):
    return (input + bias) / scale

def quantize(input=None, scale=128):
    if input is None:
        input = 2 * np.random.random((2000)) - 1

    min_range = -1
    max_range = 1

    scale_input, bias_input, normalized_input1 = normalize(input, max_range=1, min_range=-1)
    scale_input, bias_input, normalized_input = normalize(normalized_input1, max_range=127, min_range=-128)

    quant_input = np.asarray(normalized_input, int)
    # dequant_input = dequantize(quant_input, scale, bias)
    # dequant_int = dequantize(quant_input, int(scale), int(bias))
    # print("UnNormalized:\n", np.round((int_input + bias)/scale, 4))

    # print(input)
    # print(quant_input)



    # print(scale)
    # output = (out - min_range) / scale + np.min(input)
    # print("Input:\n", np.round(output, 4))
    # print('MSE after Quant and Dequant', np.mean((input - dequant_input)**2))
    # print('MSE after Quant and Dequant with int scale and bias', np.mean((input - dequant_int)**2))

    n_nodes = 1000
    w = 2 * np.random.random((n_nodes, input.shape[0])) - 1
    scale_w, bias_w, normalized_w = normalize(w, 1, -1)
    # print(w)
    true_output = normalized_w @ normalized_input1
    # print(true_output)

    scale_w, bias_w, quant_w = normalize(normalized_w, 127, -128)
    quant_w = np.asarray(quant_w, dtype=int)
    # print(quant_w)
    quantized_output = quant_w @ quant_input #np.asarray(quant_w, dtype=int) @ quant_input
    dequantized_output = dequantize(quantized_output, int(scale_w))#, int(bias_w))
    pred_output = dequantize(dequantized_output, int(scale_input))#, int(bias_input))
    # print(pred_output)
    print('MSE after forward pass', np.mean((true_output - pred_output)**2))




def main():
    # verify_conv()
    # verify_linear()
    # verify_softmax()    
    mnist = MNIST()
    # print(mnist.forward(x).shape)
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    kwargs = {'batch_size': 10}
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ]
    )
    # # dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, download=True, transform=transform)
    # # train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)

    model = Model()
    model.load_state_dict(torch.load('./mnist_cnn_normalized.pt'))
    # for param_tensor in model.state_dict():
        # print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        # if param_tensor == 'conv1.weight':
        #     conv_weight =  model.state_dict()[param_tensor].detach().numpy()
        # if param_tensor == 'conv1.bias':
        #     conv_bias =  model.state_dict()[param_tensor].detach().numpy() 
    
    for idx, m in enumerate(model.children()):
        print(idx, '->', m)
        if idx == 0:
            # conv_weight =  m.weight
            # conv_bias = m.biasax
            conv = m
        if idx == 1:
            # bn_weight = m.weight
            # bn_bias = m.bias
            # bn_running_mean = m.running_mean
            # bn_running_var = m.running_var
            # bn_num_batches_tracked = m.num_batches_tracked
            # bn_epsilon = m.epsilon
            bn = m
        if idx == 2:
            fc1 = m
        if idx == 3:
            fc2 = m

    # for p in model.parameters():
    #     print(p)
    fused_conv = fuse_conv_bn_eval(conv, bn)
    # weights = [fused_conv, fc1, fc2]
    conv_weight = fused_conv.weight.detach().numpy()
    conv_bias = fused_conv.bias.detach().numpy() if conv.bias is not None else 0
    fc1_weight = fc1.weight.detach().numpy() 
    fc1_bias = fc1.bias.detach().numpy() if conv.bias is not None else 0
    fc2_weight = fc2.weight.detach().numpy()
    fc2_bias = fc2.bias.detach().numpy() if conv.bias is not None else 0

    mnist = MNIST()
    mnist.conv1.weight = conv_weight
    mnist.conv1.bias = conv_bias
    mnist.fc1.weight = fc1_weight
    mnist.fc1.bias = fc1_bias
    mnist.fc2.weight = fc2_weight
    mnist.fc2.bias = fc2_bias
    model.eval()
    x = torch.randn(14, 1, 28, 28, requires_grad=True)
    numpy_x = x.detach().numpy()
    # print(model(x))
    # print(mnist.forward(x))
    out1 = mnist.forward(numpy_x)
    out2 = model(x)
    # print('MSE after forward', np.mean((np.round(out1, 4) - out2.detach().numpy())**2))
    verify_mnist(model, mnist, test_loader)
    # return 0
    # print(out1)
    # print(out1.shape)
    # print(np.argmax(out1, axis=0, keepdims=True))
    # print(np.argmax(out1, axis=1))


if __name__ == '__main__':
    main()
    # verify_maxpool2d()
    # verify_conv()
    # verify_adaptiveavgpool2d()
    # quantize()
"""
    x = np.array([-.25, .5, .325, .8])
    w = np.array([-1.5, -2.3, 0.89, 1.2])

    print(x)
    scale, bias, x_q = normalize(x, 127, -128)
    x_q = np.asarray(x_q, dtype=int)
    print(x_q)
    # print(scale, bias)
    scale, bias, x_n = normalize(x, 1, -1)
    print(x_n)
    # print(scale, bias)


    print(w)
    scale, bias, w_q = normalize(w, 127, -128)
    w_q = np.asarray(w_q, dtype=int)
    print(w_q)
    # print(scale, bias)
    scale, bias, w_n = normalize(w, 1, -1)
    print(w_n)
    # print(scale, bias)

    scale, bias, x_nq = normalize(x_n, 127, -128) 
    x_nq = np.array(np.round(x_nq, 0), dtype=int)
    print(x_nq)

    scale, bias, w_nq = normalize(w_n, 127, -128) 
    w_nq = np.array(np.round(w_nq, 0), dtype=int)
    print(w_nq)


    print("-------------")
    answer = w * x

    norm_answer = w_n * x_n
    norm_answer = dequantize(norm_answer, 0.57142, -0.3142)
    norm_answer = dequantize(norm_answer, 1.90476, 0.52381)


    quantized_answer = (w_q * x_q)
    quantized_answer = dequantize(quantized_answer, 72.8571, -39.5714)
    quantized_answer = dequantize(quantized_answer, 242.8571, 67.2857) 

    norm_quantized_answer = w_nq * x_nq
    norm_quantized_answer = dequantize(norm_quantized_answer, 127, 0)
    norm_quantized_answer = dequantize(norm_quantized_answer, 127, 0)

    print(answer)
    print(norm_answer)
    print(quantized_answer)
    print(norm_quantized_answer)"""