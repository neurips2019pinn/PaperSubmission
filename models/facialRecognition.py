from deepHogNormalization import DeepHogNormalization, EdgeDetector

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F 

class PINNBasic(nn.Module):
    def __init__(self, padding=10):
        super(PINNBasic, self).__init__()
        self.features1 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2), ######
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), ######
        )
        self.features2 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), ######
        )

        self.regressor1 = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.ReLU(),
            nn.Linear(8*8*32, 30),
            nn.Dropout(p=0.25),
            nn.Linear(30, 15)
        )

        edge_detector = EdgeDetector(channel_num=3, kernel_size=7)
        deep_hog_normalizer = DeepHogNormalization(channel_num=1, window_size=16, stride=8)
        self.pad_amount = padding
        self.pose_normalizer = nn.Sequential(edge_detector, deep_hog_normalizer)
        self.padder = nn.ReflectionPad2d([padding, padding, padding, padding])
        # self.pose_normalizer = nn.Sequential(deep_hog_normalizer)

    def correct_pose(self, x, padding=10):
        theta = self.pose_normalizer(x)
        x = self.padder(x)
        grid = torch.nn.functional.affine_grid(theta, x.size())
        x = torch.nn.functional.grid_sample(x, grid)
        x = x[:, :, self.pad_amount: -1*self.pad_amount, self.pad_amount: -1*self.pad_amount]
        return x

    def forward(self, x):
        x = self.correct_pose(x)
        x = self.features1(x)
        x = self.features2(x)
        

        x = x.view(x.size(0), -1)

        x = self.regressor1(x)
        return x

class CNNBasic(nn.Module):
    def __init__(self):
        super(CNNBasic, self).__init__()
        self.features1 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2), ######
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), ######
        )
        self.features2 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), ######
        )

        self.regressor1 = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.ReLU(),
            nn.Linear(8*8*32, 30),
            nn.Dropout(p=0.25),
            nn.Linear(30, 15)
        )

        self.regressor2 = nn.Sequential(
            nn.BatchNorm1d(6),
            nn.Dropout(p=0.4),
            nn.ReLU(),
            nn.Linear(6, 10),
            nn.Dropout(p=0.4),
            nn.Linear(10, 2)
        )

        self.regressor3 = nn.Sequential(
            nn.Linear(4, 10),
            nn.Linear(10, 2)
        )

        # self.pose_norm = PoseNormalization(32, img_shape=(24, 24))
        
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)

        x = x.view(x.size(0), -1)

        x = self.regressor1(x)
        return x

class STNBasic(nn.Module):
    def __init__(self):
        super(STNBasic, self).__init__()
        self.features1 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(16, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2), ######
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), ######
        )
        self.features2 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), ######
        )

        self.regressor1 = nn.Sequential(
            nn.BatchNorm1d(8*8*32),
            nn.Dropout(p=0.6),
            nn.ReLU(),
            nn.Linear(8*8*32, 30),
            nn.Dropout(p=0.6),
            nn.Linear(30, 15)
        )

        self.regressor2 = nn.Sequential(
            nn.BatchNorm1d(6),
            nn.Dropout(p=0.4),
            nn.ReLU(),
            nn.Linear(6, 10),
            nn.Dropout(p=0.4),
            nn.Linear(10, 2)
        )

        self.regressor3 = nn.Sequential(
            nn.Linear(4, 10),
            nn.Linear(10, 2)
        )

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2), ####
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.MaxPool2d(2), ####
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2) ####
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * 8 * 8, 20),
            nn.ReLU(),
            nn.Linear(20, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # self.pose_norm = PoseNormalization(32, img_shape=(24, 24))
    
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 32 * 8 * 8)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x, theta

    def forward(self, x):
        x, theta = self.stn(x)
        
        x = self.features1(x)
        x = self.features2(x)
        x = x.view(x.size(0), -1)

        x = self.regressor1(x)
        
        # x2 = self.regressor2(theta.view(x.size(0), -1))
        # x = torch.cat([x, x2], dim=1)

        # x = self.regressor3(x)
        return x

class Noise(nn.Module):
    def __init__(self, scale=0.01):
        super(Noise, self).__init__()
        self.scale = scale

    def forward(self, x):
        device = x.device
        noise = (torch.randn_like(x)*self.scale).to(device)
        return x + noise


class MultiPINNBasic(nn.Module):
    def __init__(self):
        super(MultiPINNBasic, self).__init__()
        self.weights = torch.nn.Parameter(torch.ones(2), requires_grad=True)
        self.features1 = nn.Sequential(
            nn.ReflectionPad2d((3, 3, 3, 3)),
            nn.Conv2d(3, 16, kernel_size=7),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ReflectionPad2d((3, 3, 3, 3)),
            nn.Conv2d(16, 16, kernel_size=7),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2), ######
            nn.ReflectionPad2d((3, 3, 3, 3)),
            nn.Conv2d(16, 16, kernel_size=7),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ReflectionPad2d((3, 3, 3, 3)),
            nn.Conv2d(16, 32, kernel_size=7),
            nn.BatchNorm2d(32),
            nn.Softmax2d(),
            nn.MaxPool2d(2),######
        )
        self.features2 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2), ######
        )

        self.heatmap = nn.Sequential(
            nn.ReflectionPad2d((2, 2, 2, 2)),
            nn.Conv2d(32, 10, kernel_size=5),
            nn.ReLU(),
            Noise(scale=0.001)
        )

        self.regressor1 = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.ReLU(),
            nn.Linear(8*8*32, 30),
            nn.Dropout(p=0.25),
            nn.Linear(30, 15)
        )

        self.regressor2 = nn.Sequential(
            nn.BatchNorm1d(6),
            nn.Dropout(p=0.6),
            nn.ReLU(),
            nn.Linear(6, 40),
            nn.Dropout(p=0.6),
            nn.Linear(40, 2)
        )

        self.regressor3 = nn.Sequential(
            nn.BatchNorm1d(4),
            nn.Dropout(p=0.4),
            nn.ReLU(),
            nn.Linear(4, 10),
            nn.Dropout(p=0.4),
            nn.Linear(10, 2)
        )

        self.multi_pose = WeightedMultiPoseExtraction(10)

    def extract_anti_pose(self, x):
        x_h = self.heatmap(x)
        _, _, _, _, theta, _ = self.multi_pose(x_h)
        return theta

    def extract_pose(self, x):
        x_h = self.heatmap(x)
        mu_W, U, mu_avg, T_orig, theta, (L1, L2) = self.multi_pose(x_h)
        return torch.cat([mu_avg, T_orig[:, :, :, 0]], dim=2)

    def forward(self, x):
        x = self.features1(x)
        # _, _, _, _, theta, _ = self.multi_pose(x)
        theta = self.extract_anti_pose(x)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        x = self.features2(x)
        x = x.view(x.size(0), -1)
        x = self.regressor1(x)

        # x2 = theta.view(-1, 6)
        # x2 = self.regressor2(x2)

        # x3 = (self.weights[0]*x + self.weights[1]*x2) / torch.sum(self.weights)
        return x
