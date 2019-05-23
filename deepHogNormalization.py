import torch
from layers import Abs, Noise, MaxNorm
from deepHOG import DeepHOG

class EdgeDetector(torch.nn.Module):
    def __init__(self, channel_num, kernel_size):
        super(EdgeDetector, self).__init__()
        grad_x = torch.range(-1*(kernel_size // 2), kernel_size // 2).repeat(kernel_size, 1)
        grad_y = grad_x.t()
        grad_x = grad_x.repeat(1, channel_num, 1, 1)
        grad_y = grad_y.repeat(1, channel_num, 1, 1)
        
        conv = torch.nn.Conv2d(channel_num, 2, kernel_size=kernel_size, bias=False)
        conv.weight.data = torch.cat([grad_x, grad_y], dim=0)

        conv_add = torch.nn.Conv2d(2, 1, kernel_size=1, bias=False)
        conv_add.weight.data = torch.ones(1, 2, 1, 1)

        pad = torch.nn.ReplicationPad2d([kernel_size // 2, kernel_size // 2, 
                                            kernel_size // 2, kernel_size // 2])

        abs_layer = Abs()
        self.edge_detector = torch.nn.Sequential(pad, conv, abs_layer, conv_add)
    
    def forward(self, x):
        x = self.edge_detector(x)
        return x

class DeepHogNormalization(torch.nn.Module):
    def __init__(self, channel_num, window_size, stride):
        super(DeepHogNormalization, self).__init__()
        conv_add = torch.nn.Conv2d(channel_num, 1, kernel_size=1, bias=False)
        conv_add.weight.data = torch.ones(1, channel_num, 1, 1)

        
        abs_layer = Abs()
        noise = Noise(0.00001)  
        deep_hog = DeepHOG(window_size=window_size, stride=stride)
        max_norm = MaxNorm()

        
        self.pose_normalizer = torch.nn.Sequential(abs_layer, conv_add, noise, max_norm, deep_hog)
    
    def forward(self, x):
        _, _, _, _, theta = self.pose_normalizer(x)
        grid = torch.nn.functional.affine_grid(theta, x.size())
        x = torch.nn.functional.grid_sample(x, grid)
        return theta