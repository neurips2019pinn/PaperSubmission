import torch

class Noise(torch.nn.Module):
    def __init__(self, scale=0.01):
        super(Noise, self).__init__()
        self.scale = scale

    def forward(self, x):
        device = x.device
        noise = (torch.randn_like(x)*self.scale).to(device)
        return x + noise
    
    
class Abs(torch.nn.Module):
    def __init__(self):
        super(Abs, self).__init__()

    def forward(self, x):
        return torch.abs(x)
    
class MaxNorm(torch.nn.Module):
    def __init__(self):
        super(MaxNorm, self).__init__()

    def forward(self, x):
        batch_size, _, _, _ = x.shape
        max_values = x.view(batch_size, -1).max(dim=1)[0]
        return x / max_values.unsqueeze(1).unsqueeze(2).unsqueeze(3)