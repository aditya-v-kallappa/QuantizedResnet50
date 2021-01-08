from pytorch_mnist import Model
from torchvision import datasets, transforms
import numpy as np
# from int_mnist import normalize, dequantize
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from int_mnist import ReLu, Softmax, fuse_conv_bn_eval, fuse_conv_bn_weights

def dequantize(input=None, scale=1, bias=0):
    return (input + bias) / scale

def quantize(x, max_range=127, min_range=-128, maximum=None, minimum=None):
    scale, bias, x = normalize(x, max_range=max_range, min_range=min_range, maximum=maximum, minimum=minimum)
    x = np.asarray(x, dtype= np.int8)
    return int(scale), int(bias), x

def dequantize(x, scale, bias=0):
    return (x + bias) / scale

def normalize(input, max_range=1, min_range=0, maximum=None, minimum=None):

    max = np.max(input) if maximum is None else maximum
    min = np.min(input) if minimum is None else minimum
    scale = (max_range - min_range) / (max - min)
    bias = min * scale - min_range

    print(scale, bias)

    # out = (input - min) / (max - min) * (max_range - min_range) + min_range
    out = input * scale - bias
    
    return scale, bias, out

class QuantizedLinear():
    def __init__(self, out_features=1, in_features=1, bias=True):
        self.out_features = out_features
        self.weight = np.zeros((out_features, in_features), dtype=np.int8)
        if bias:
            self.bias = np.zeros(out_features, dtype=np.int8)
        else:
            self.bias = np.zeros(out_features, dtype=np.int8)

    def __call__(self, input, scale_x, bias_x, scale_w, bias_w, scale_b=1, bias_b=0):
        N = input.shape[0]
        num_mul = input.shape[1]
        output = np.zeros((N, self.out_features), dtype=np.int32)
        for i in range(N):
            output[i] = self.weight @ input[i]
        output = self.dequantize(output, input, scale_x, bias_x, scale_w, bias_w, num_mul=num_mul)
        output = output + (self.bias+bias_b) / scale_b
        output = np.asarray(np.round(output), dtype=np.int16)
        return output
    
    def dequantize(self, quantized_value, x, scale_x, bias_x, scale_w, bias_w, num_mul=None):
        output = np.zeros_like(quantized_value)
        x_sum = np.sum(x, axis=1)
        w_sum = np.sum(self.weight, axis=1)
        N = output.shape[0]
        for n in range(N):
            output[n] = (quantized_value[n] + bias_x * w_sum + bias_w * x_sum[n] + num_mul * bias_x * bias_w) / scale_w / scale_x 
        return output

class QuantizedConv2d():
    def __init__(self, in_channels=1, out_channels=1, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), dilation=(1, 1), bias=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = tuple(kernel_size) if type(kernel_size) != int else (kernel_size, kernel_size)
        self.padding = tuple(padding) if type(padding) != int else (padding, padding)
        self.stride = stride if type(stride) != int else (stride, stride)
        self.dilation = dilation if type(dilation) != int else (dilation, dilation)

        self.weight = np.random.randint(low=-128, high=127, size=(out_channels, in_channels, kernel_size[0], kernel_size[1])) 
        
        if bias == True:
            self.bias = np.zeros(shape=out_channels, dtype=np.int8)
        elif bias == False or bias is None:
            self.bias = np.zeros(shape=out_channels, dtype=np.int8)

    def __call__(self, input, scale_x=128, bias_x=0, scale_w=128, bias_w=0, scale_b=128, bias_b=0):
        self.pad = bias_x
        N, C_in, H_in, W_in = input.shape
    
        C_out = self.out_channels
        H_out = (H_in + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1)) // self.stride[0]
        W_out = (W_in + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1)) // self.stride[1]

        output_shape = (N, C_out, H_out, W_out)
        output = np.zeros(shape=output_shape, dtype=np.int16)  
        if self.padding[0] != 0 or self.padding[1] != 0:
            padded_input = self.pad_input(input, pad=-bias_x)
        else:
            padded_input = input
        for n in range(N):
            for c in range(C_out):
                for i in range(H_out):
                    for j in range(W_out):
                        output[n, c, i, j] = \
                            np.sum(self.weight[c] * padded_input[n, :, i*self.stride[0]:i*self.stride[0]+self.kernel_size[0], j*self.stride[1]:j*self.stride[1]+self.kernel_size[1]])
        
        output = self.dequantize(output, input, scale_x, bias_x, scale_w, bias_w)
        output = output + (self.bias.reshape(1, -1, 1, 1) + bias_b) / scale_b
        output = np.asarray(np.round(output), dtype=np.int8)
        return output

    def get_k_k_sum(self, x):
        N, C, H, W = x.shape
        C_out = 1
        H_out = (H + 2 * self.padding[0] - self.dilation[0] * (self.kernel_size[0] - 1)) // self.stride[0]
        W_out = (W + 2 * self.padding[1] - self.dilation[1] * (self.kernel_size[1] - 1)) // self.stride[1]
        out_shape = (N, C_out, H_out, W_out)
        output = np.zeros(out_shape)
        for n in range(N):
            for c in range(C_out):
                for i in range(H_out):
                    for j in range(W_out):
                        output[n, c, i, j] = \
                            np.sum(x[n, :, i*self.stride[0]:i*self.stride[0]+self.kernel_size[0], j*self.stride[1]:j*self.stride[1]+self.kernel_size[1]])
        return output

    def dequantize(self, quantized_value, x, scale_x, bias_x, scale_w, bias_w):
        kernel = self.kernel_size
        stride = self.stride
        padding = self.padding
        N, C, H, W = quantized_value.shape
        x_sum = self.get_k_k_sum(x)
        output = np.zeros(quantized_value.shape)
        w_sum = np.sum(self.weight, axis=(1, 2, 3)).reshape(1, C, 1, 1)
        num_mul = kernel[0] * kernel[1] * x.shape[1]
        output = (quantized_value + (bias_x * w_sum) + (bias_w * x_sum[0]) + (num_mul * bias_x * bias_w)) / (scale_w * scale_x) 

        return output


    def pad_input(self, input, pad=0):
        
        N, C, H, W = input.shape
        padded_input = np.zeros(shape=(N, C, H+2*self.padding[0], W+2*self.padding[1]), dtype=input.dtype)
        
        for n in range(N):
            for c in range(C):
                padded_input[n, c, self.padding[0], self.padding[1]] = pad
                padded_input[n, c, -self.padding[0]:, -self.padding[1]:] = pad
                padded_input[n, c, self.padding[0]:-self.padding[0], self.padding[1]:-self.padding[1]] = input[n, c]
        
        return padded_input

def verify_conv(x=None):
    if x is None:
        x = np.random.rand(10, 1, 32, 32)
    _x = torch.Tensor(x)

    
    scale_x, bias_x, xn = normalize(x, 127, -128)
    xn = np.asarray(np.round(xn, 0), dtype=np.int)
    conv = nn.Conv2d(in_channels=x.shape[1], out_channels=32, kernel_size=(3, 3), padding=(1, 1), bias=True)
    conv.weight = fused_conv.weight
    conv.bias = fused_conv.bias
    weight = conv.weight.detach().numpy()
    bias = conv.bias.detach().numpy() if conv.bias is not None else None
    custom2 = QuantizedConv2d(in_channels=x.shape[1], out_channels=32, kernel_size=(3, 3), padding=(1, 1), bias=True)
    custom2.weight = weight
    custom2.bias = bias if bias is not None else custom2.bias
    scale_w, bias_w, custom2.weight = normalize(custom2.weight, 127, -128)
    custom2.weight = np.asarray(np.round(custom2.weight, 0), dtype=np.int8)
    scale_b, bias_b, custom2.bias = normalize(custom2.bias, 127, -128)
    custom2.bias = np.asarray(np.round(custom2.bias, 0), dtype=np.int8)

    t1 = time.time()
    out = custom2(xn, scale_x, bias_x, scale_w, bias_w, scale_b, bias_b)
    t2 = time.time()
    t3 = time.time()
    conv_out = conv(_x)
    t4 = time.time()
    print('MSE after convolution', np.mean((np.round(out, 4) - conv_out.detach().numpy())**2))
    # print("TImes:\n")
    # print("Int Conv:", t2-t1)
    # print("Int Deconv", t3-t2)
    # print("Total Int COnv", t3-t1)
    # print("tensor Conv", t4-t3)

    return out, conv_out


def verify_relu(x=None):
    if x is None:
        x = np.round(np.random.rand(10) * 5 - 2, 2)
    a = ReLu()
    rx = a(x)
    print("X\n", x)


    print("Relu(X)\n", rx)

    scale_x, bias_x, xn = normalize(x, 127, -128)
    xn = np.asarray(np.round(xn, 0), dtype=np.int)
    print("Xn:\n", xn, scale_x, bias_x)

    rxn = a(xn)
    print("Relu(Xn)\n", rxn)

    dqxn = dequantize(xn, scale=scale_x, bias=bias_x)
    print("Dequantized x\n", dqxn)

    dqrxn = dequantize(rxn, scale=scale_x, bias=bias_x)
    print("Dequantized Rleu(xn)\n", dqrxn)

    mask = xn > -bias_x
    xn_bias = xn * mask 
    xn_bias[~mask] = -bias_x

    print("Bias Xn", xn_bias)

    dqxn_bias = dequantize(xn_bias, scale=scale_x, bias=bias_x)
    print("Dequantized Rleu(xn)\n", dqxn_bias)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    kwargs = {'batch_size': 10}
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ]
    )
    # dataset1 = datasets.MNIST('../data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, download=True, transform=transform)
    # # # train_loader = torch.utils.data.DataLoader(dataset1,**kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **kwargs)
    relu = ReLu()
    softmax = Softmax(log=True)
    correct1 = 0
    correct2 = 0
    idx = 0
    for data, target in test_loader:
        idx += 1
        numpy_target = target.detach().numpy()
        numpy_conv, torch_conv = verify_conv(x=data.detach().numpy())
        numpy_conv = relu(numpy_conv) 
        numpy_conv = numpy_conv.reshape(numpy_conv.shape[0], -1)
        numpy_fc1, torch_fc1 = verify_linear(x=numpy_conv, in_features=25088, out_features=128, layer=1)
        numpy_fc1 = relu(numpy_fc1)
        numpy_fc2, torch_fc2 = verify_linear(x=numpy_fc1, in_features=128, out_features=10, layer=2)
        final_output1 = softmax(numpy_fc2)
        final_output2 = F.log_softmax(torch_fc2)
        pred1 = np.argmax(final_output1, axis=1)
        pred2 = final_output2.argmax(dim=1, keepdim=True)
        correct1 += np.sum(pred1 == numpy_target)
        correct2 += pred2.eq(target.view_as(pred2)).sum().item()
        if idx == 10:
            print(idx)
            break

    print("Accuracy:", correct1)



def verify_linear(x=None, in_features=25088, out_features=128, layer=1):
    if x is None:
        x = np.random.rand(10, in_features)
    _x = torch.Tensor(x)

    scale_x, bias_x, xn = normalize(x, 127, -128)
    xn = np.asarray(np.round(xn, 0), dtype=np.int)
    
    lin = nn.Linear(in_features, out_features, bias=True)
    if layer == 1:
        lin.weight = fc1.weight
        lin.bias = fc1.bias
    elif layer == 2:
        lin.weight = fc2.weight
        lin.bias = fc2.bias
    custom2 = QuantizedLinear(out_features, in_features, bias=True)
    weight = lin.weight.detach().numpy()
    bias = lin.bias.detach().numpy() if lin.bias is not None else None
    custom2.weight = weight
    custom2.bias = bias if bias is not None else custom2.bias
    w_maximum = None
    w_minimum = None
    if (np.max(weight) < 1) and (np.min(weight) > -1):
        w_minimum = -1
        w_maximum = 1
    scale_w, bias_w, custom2.weight = normalize(custom2.weight, 127, -128, maximum=w_maximum, minimum=w_minimum)
    custom2.weight = np.asarray(np.round(custom2.weight, 0), dtype=np.int8)
    if (np.max(bias) < 1) and (np.min(bias) > -1):
        b_minimum = -1
        b_maximum = 1
    scale_b, bias_b, custom2.bias = normalize(custom2.bias, 127, -128, maximum=b_maximum, minimum=b_minimum)
    custom2.bias = np.asarray(np.round(custom2.bias, 0), dtype=np.int8)
    tensor_out = lin(_x)
    custom_out = custom2(xn, scale_x, bias_x, scale_w, bias_w, scale_b=scale_b, bias_b=bias_b)
    print('MSE after Linear', np.mean((np.round(custom_out, 4) - tensor_out.detach().numpy())**2))

    return custom_out, tensor_out



model = Model()
model = Model()
model_file = './mnist_cnn_cpu.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == "cpu":
    model.load_state_dict(torch.load(model_file, map_location=lambda storage, loc: storage))
else:
    model.load_state_dict(torch.load(model_file))
model.eval()

for idx, m in enumerate(model.children()):
    if idx == 0:
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

fused_conv = fuse_conv_bn_eval(conv, bn)
if __name__ == '__main__':
    

    main()
