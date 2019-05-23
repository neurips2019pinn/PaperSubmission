from deepHogNormalization import DeepHogNormalization, EdgeDetector 
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F 

class PreLayerResnet(nn.Module):
    def __init__(self, original_model):
        super(PreLayerResnet, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[:-5])
        
    def forward(self, x):
        x = self.features(x)
        return x
    
class PostLayerResnet(nn.Module):
    def __init__(self, original_model):
        super(PostLayerResnet, self).__init__()
        self.features = nn.Sequential(*list(original_model.children())[-5:-3])
        
    def forward(self, x):
        x = self.features(x)
        return x

class BasicResnet(nn.Module):
    def __init__(self, num_classes):
        super(BasicResnet, self).__init__()
        resnet18 = torchvision.models.resnet18()
        self.add_channels = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
        )
        self.pre_resnet = PreLayerResnet(resnet18)
        self.post_resnet = PostLayerResnet(resnet18)

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(6*6*256),
            nn.Dropout(p=0.6),
            nn.ReLU(),
            nn.Linear(6*6*256, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.6),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        x = self.add_channels(x)
        x = self.pre_resnet(x)
        # x, means, orientations, confidence, theta, (L1, L2) = self.pose_norm(x)
        x = self.post_resnet(x)

        x = x.view(x.size(0), -1)

        x = self.classifier(x)
        return x

class PINNResnet(nn.Module):
    def __init__(self, num_classes):
        super(PINNResnet, self).__init__()
        resnet18 = torchvision.models.resnet18()
        self.add_channels = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
        )
        self.pre_resnet = PreLayerResnet(resnet18)
        self.post_resnet = PostLayerResnet(resnet18)

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(6*6*256),
            nn.Dropout(p=0.6),
            nn.ReLU(),
            nn.Linear(6*6*256, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.6),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        self.pose_norm = PoseNormalization(64, img_shape=(16, 16))
        
    def forward(self, x):
        x = self.add_channels(x)
        x = self.pre_resnet(x)
        # x, means, orientations, confidence, theta, (L1, L2) = self.pose_norm(x)
        x = self.post_resnet(x)

        x = x.view(x.size(0), -1)

        x = self.classifier(x)
        return x

class PINN(torch.nn.Module):
    def __init__(self):
        super(PINN, self).__init__()
        self.features1 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(1, 16, kernel_size=3),
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
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(6*6*32),
            nn.Dropout(p=0.6),
            nn.ReLU(),
            nn.Linear(6*6*32, 20),
            nn.BatchNorm1d(20),
            nn.Dropout(p=0.6),
            nn.ReLU(),
            nn.Linear(20, 5)
        )
        
        edge_detector = EdgeDetector(channel_num=1, kernel_size=7)
        deep_hog_normalizer = DeepHogNormalization(channel_num=1, window_size=48, stride=12)
        self.pad_amount = 20
        self.padder = nn.ReflectionPad2d([self.pad_amount, self.pad_amount, 
                                         self.pad_amount, self.pad_amount])
        self.pose_normalizer = nn.Sequential(edge_detector, deep_hog_normalizer)

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

        x = self.classifier(x)
        return x        

class CNNBasic(nn.Module):
    def __init__(self, num_classes):
        super(CNNBasic, self).__init__()
        self.features1 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(1, 16, kernel_size=3),
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
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(6*6*32),
            nn.Dropout(p=0.6),
            nn.ReLU(),
            nn.Linear(6*6*32, 20),
            nn.BatchNorm1d(20),
            nn.Dropout(p=0.6),
            nn.ReLU(),
            nn.Linear(20, num_classes)
        )

        # self.pose_norm = PoseNormalization(32, img_shape=(24, 24))
        
    def forward(self, x):
        x = self.features1(x)
        x = self.features2(x)

        x = x.view(x.size(0), -1)

        x = self.classifier(x)
        return x

class STNBasic(nn.Module):
    def __init__(self, num_classes):
        super(STNBasic, self).__init__()
        self.features1 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(1, 16, kernel_size=3),
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
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(6*6*32),
            nn.Dropout(p=0.6),
            nn.ReLU(),
            nn.Linear(6*6*32, 20),
            nn.BatchNorm1d(20),
            nn.Dropout(p=0.6),
            nn.ReLU(),
            nn.Linear(20, num_classes)
        )

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(1, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2), ####
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.BatchNorm2d(32), 
            nn.ReLU(),
            nn.MaxPool2d(2), ####
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2) ####
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(64 * 12 * 12, 20),
            nn.ReLU(),
            nn.Linear(20, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        # self.pose_norm = PoseNormalization(32, img_shape=(24, 24))
    
    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 64 * 12 * 12)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        x = self.stn(x)
        x = self.features1(x)
        x = self.features2(x)
        x = x.view(x.size(0), -1)

        x = self.classifier(x)
        return x

class STNResnet(nn.Module):
    def __init__(self, num_classes):
        super(STNResnet, self).__init__()
        resnet18 = torchvision.models.resnet18()
        self.add_channels = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=1),
        )
        self.pre_resnet = PreLayerResnet(resnet18)
        self.post_resnet = PostLayerResnet(resnet18)

        self.stn_pre_resnet = PreLayerResnet(resnet18)
        self.stn_post_resnet = PostLayerResnet(resnet18)

        self.classifier = nn.Sequential(
            nn.BatchNorm1d(6*6*256),
            nn.Dropout(p=0.6),
            nn.ReLU(),
            nn.Linear(6*6*256, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(p=0.6),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

        # Spatial transformer localization-network
        self.localization = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(16, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 3 * 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))
        
    def stn(self, x):
        xs = self.stn_pre_resnet(x)
        xs = self.stn_post_resnet(xs)
        xs = xs.view(x.size(0), -1)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)

        return x

    def forward(self, x):
        x = self.add_channels(x)
        x = self.stn(x)
        x = self.pre_resnet(x)
        x = self.post_resnet(x)
        x = x.view(x.size(0), -1)
        
        x = self.classifier(x)
        return x

