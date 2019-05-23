import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import utils

def tensor_to_coord_tensor(X):
    """
    Takes in a 4D image and returns the 3*W*H coordinate representation
    """
    device = X.device
    batch_num, channel_num, height, width = X.shape
    flattened_img = X.permute((0, 1, 3, 2)).contiguous().view(batch_num, channel_num, -1)
    x_coord = torch.arange(0, width).view(width, 1)
    x_coord = x_coord.expand(channel_num, width, height).contiguous()
    x_coord = x_coord.view(channel_num, width * height).float().expand(batch_num, channel_num, -1).to(device)

    y_coord = torch.arange(height, 0, -1) - 1
    y_coord = y_coord.expand(channel_num, width, height).contiguous()
    y_coord = y_coord.view(channel_num, width * height).float().expand(batch_num, channel_num, -1).to(device)

    coord_matrix = torch.stack([x_coord, y_coord, flattened_img])

    return coord_matrix.permute((1, 2, 0, 3))

class WeightedMultiPoseExtraction(torch.nn.Module):
    def __init__(self, num_channels):
        super(WeightedMultiPoseExtraction, self).__init__()
        self.flip_y_loc = torch.Tensor([[2 , 0], [0, -2]])
        self.mu_R = torch.Tensor([0.5, 0.5])
        self.rotate_pi = torch.Tensor([[-1 ,0], [0, 1]])
        self.pi = 3.14159265358979323
        self.theta_R = (self.pi / 2)
        self.conv2d = torch.nn.Conv2d(num_channels, 1, kernel_size=(1))
        self.conv2d.weight.data = (1 / num_channels) * torch.ones(1, num_channels, 1, 1)
        self.conv2d.bias.data = torch.zeros(1)
        # Random Initialization
        # self.conv2d.weight.data = torch.rand(1, num_channels, 1, 1)
        # self.conv2d.weight.data = self.conv2d.weight.data / self.conv2d.weight.data.sum()
        
        # self.weights = torch.nn.Parameter(torch.ones(1, num_channels, 1, 1), requires_grad=True).cuda()
    
    def strength_cofficient(self, L1, L2):
        """
        Measure the relative strength of the first eigen value to the second
        """
        scaling_cofficient = 1
        eig_ratio = (L1 / L2) - 1
        if torch.nonzero(eig_ratio != eig_ratio).shape[0] > 0:
            print("BROKEN EIG VAL:", L1, L2)
            exception = Exception()
            exception.loc = torch.nonzero(eig_ratio != eig_ratio)[0][0]
            raise exception
        confidence = 2*(F.sigmoid(scaling_cofficient * eig_ratio) - 0.5)
        return confidence
    
    def left_svd_tensor(self, T):
        eps = 0.001
        batch_size, num_channels, num_coords, img_size = T.shape
        T = T.view(batch_size * num_channels, num_coords, img_size)
        M = torch.bmm(T, T.permute(0, 2, 1))
        M = M.view(batch_size, num_channels, num_coords, num_coords)
        
        D = M[:, :, 0, 0] * M[:, :, 1, 1] - M[:, :, 0, 1] * M[:, :, 1, 0]

        D = D.unsqueeze(2).unsqueeze(3)

        M = torch.mul((1 / (D + 0.001)), M)

        T = M[:, :, 0, 0] + M[:, :, 1, 1]
        D = M[:, :, 0, 0] * M[:, :, 1, 1] - M[:, :, 0, 1] * M[:, :, 1, 0]
        L1 = (T + torch.sqrt(F.relu((T**2) - 4*D))) / 2
        L2 = (T - torch.sqrt(F.relu((T**2) - 4*D))) / 2

        v1 = torch.stack([L1 - M[:, :, 1, 1], M[:, :, 1, 0]])
        v2 = torch.stack([L2 - M[:, :, 1, 1], M[:, :, 1, 0]])
        
        U = torch.stack([v1, v2]).permute(2, 3, 0, 1)
        U = torch.nn.functional.normalize(U,  dim=3)
        return U, (L1, L2)

    def rot_mat_tensor(self, thetas):
        num_tensors = thetas.shape[0]
        c = torch.cos(thetas).unsqueeze(1).unsqueeze(2)
        s = torch.sin(thetas).unsqueeze(1).unsqueeze(2)
        R = torch.stack([torch.cat([c, -1*s], dim=1),  torch.cat([s, c], dim=1)], dim=1).squeeze(3)
        return R

    def tensor_orientation_transform(self, U):
        eps = 0.0001
        rot_lambda = torch.fmod(torch.atan2(U[:, :, 0:1, 1:2], U[:, :, 0:1, 0:1] + eps) + 2 * self.pi, self.pi)
#         rot_lambda = torch.atan2(U[:, :, 0, 1], U[:, :, 0, 0])
        # rot_lambda_weighted = (rot_lambda.cuda() * self.weights[:, :, 0, 0].cuda()).sum(dim=1).unsqueeze(1) / self.weights.sum()
        rot_lambda_weighted = self.conv2d(rot_lambda)[:, 0, 0]
#         rot_lambda_weighted = torch.fmod(rot_lambda_weighted + 2 * self.pi, self.pi)
        diff_angle = (self.theta_R - rot_lambda_weighted)
        rotation = self.rot_mat_tensor(diff_angle).float()
        rotation_orig = self.rot_mat_tensor(rot_lambda_weighted).float()
        return rotation, rotation_orig

    def center_coord_tensor(self, coord_tensor):
        epsilon = 0.0001
        W = torch.abs(coord_tensor[:, :,2:3, :])
        C = coord_tensor[:, :, :2, :]
        WX = torch.mul(W, C)

        mu_W = (torch.sum(WX, dim=3) / torch.sum(W, dim=3) + epsilon).unsqueeze(3)
        
        C_tilda = (C - mu_W)
        
        W_C_tilda = torch.mul(torch.sqrt(W), C_tilda)
        return mu_W, W_C_tilda

    def extract_pose(self, img_tensor):
        coord_tensor = tensor_to_coord_tensor(img_tensor)
        mu_W , WXC = self.center_coord_tensor(coord_tensor)
        U, (L1, L2) = self.left_svd_tensor(WXC)
        return mu_W, U, (L1, L2)

    def forward(self, img_tensor):
        batch_size, channel_num, height, width = img_tensor.shape
        device = img_tensor.device
        img_tensor = torch.abs(img_tensor)

        img_scale_norm_mat = torch.Tensor([[1 / width , 0], [0, 1 / height]]).to(device)
        
        mu_W, U, (L1, L2) = self.extract_pose(img_tensor)
        T, T_orig = self.tensor_orientation_transform(U)
        T = T[:, :, :, 0]
        # mu_avg = (mu_W  * self.weights).sum(dim=1) / self.weights.sum()
        mu_avg = self.conv2d(mu_W)[:, 0, :, :]
        
        mu_T = torch.mm(img_scale_norm_mat, mu_avg[:, :, 0].t()).t()
        mu_T = mu_T - self.mu_R.to(device).expand(mu_T.shape[0], 2)
        mu_T = torch.mm(self.flip_y_loc.to(device), mu_T.t()).t().unsqueeze(2)

        theta = torch.cat([T, mu_T], dim=2)
        return mu_W, U, mu_avg, T_orig, theta, (L1, L2)


class MultiPoseExtraction(torch.nn.Module):
    def __init__(self):
        super(MultiPoseExtraction, self).__init__()
        self.flip_y_loc = torch.Tensor([[2 , 0], [0, -2]])
        self.mu_R = torch.Tensor([0.5, 0.5])
        self.rotate_pi = torch.Tensor([[-1 ,0], [0, 1]])
        self.pi = 3.14159265358979323
        self.theta_R = (self.pi / 2)
        # Random Initialization
        # self.conv2d.weight.data = torch.rand(1, num_channels, 1, 1)
        # self.conv2d.weight.data = self.conv2d.weight.data / self.conv2d.weight.data.sum()
        
        # self.weights = torch.nn.Parameter(torch.ones(1, num_channels, 1, 1), requires_grad=True).cuda()
    
    def strength_cofficient(self, L1, L2):
        """
        Measure the relative strength of the first eigen value to the second
        """
        scaling_cofficient = 1
        eig_ratio = (L1 / L2) - 1
        if torch.nonzero(eig_ratio != eig_ratio).shape[0] > 0:
            print("BROKEN EIG VAL:", L1, L2)
            exception = Exception()
            exception.loc = torch.nonzero(eig_ratio != eig_ratio)[0][0]
            raise exception
        confidence = 2*(F.sigmoid(scaling_cofficient * eig_ratio) - 0.5)
        return confidence
    
    def left_svd_tensor(self, T):
        eps = 0.001
        batch_size, num_channels, num_coords, img_size = T.shape
        T = T.view(batch_size * num_channels, num_coords, img_size)
        M = torch.bmm(T, T.permute(0, 2, 1))
        M = M.view(batch_size, num_channels, num_coords, num_coords)
        
        D = M[:, :, 0, 0] * M[:, :, 1, 1] - M[:, :, 0, 1] * M[:, :, 1, 0]

        D = D.unsqueeze(2).unsqueeze(3)

        M = torch.mul((1 / (D + 0.001)), M)

        T = M[:, :, 0, 0] + M[:, :, 1, 1]
        D = M[:, :, 0, 0] * M[:, :, 1, 1] - M[:, :, 0, 1] * M[:, :, 1, 0]
        L1 = (T + torch.sqrt(F.relu((T**2) - 4*D))) / 2
        L2 = (T - torch.sqrt(F.relu((T**2) - 4*D))) / 2

        v1 = torch.stack([L1 - M[:, :, 1, 1], M[:, :, 1, 0]])
        v2 = torch.stack([L2 - M[:, :, 1, 1], M[:, :, 1, 0]])
        
        U = torch.stack([v1, v2]).permute(2, 3, 0, 1)
        U = torch.nn.functional.normalize(U,  dim=3)
        return U, (L1, L2)

    def rot_mat_tensor(self, thetas):
        num_tensors = thetas.shape[0]
        c = torch.cos(thetas).unsqueeze(2).unsqueeze(3)
        s = torch.sin(thetas).unsqueeze(2).unsqueeze(3)
        # print("C S shape")
        # print(c.shape)
        # print(s.shape)
        R = torch.stack([torch.cat([c, -1*s], dim=2),  torch.cat([s, c], dim=2)], dim=2).squeeze(4)
        return R

    def standardize_orientation(self, U):
        orientation_sign = torch.sign(U[:, :, 0:1, 1:2])
        U = U * orientation_sign

        return U

    def tensor_orientation_transform(self, U):
        eps = 0.0001
        rot_lambda = torch.fmod(torch.atan2(U[:, :, 0:1, 1:2], U[:, :, 0:1, 0:1] + eps) + 2 * self.pi, self.pi)
        # rot_lambda = torch.atan2(U[:, :, 0, 1], U[:, :, 0, 0])
        # rot_lambda_weighted = (rot_lambda.cuda() * self.weights[:, :, 0, 0].cuda()).sum(dim=1).unsqueeze(1) / self.weights.sum()
        rot_lambda_weighted = self.conv2d(rot_lambda)[:, 0, 0]
#         rot_lambda_weighted = torch.fmod(rot_lambda_weighted + 2 * self.pi, self.pi)
        diff_angle = (self.theta_R - rot_lambda_weighted)
        rotation = self.rot_mat_tensor(diff_angle).float()
        rotation_orig = self.rot_mat_tensor(rot_lambda_weighted).float()
        return rotation, rotation_orig

    def center_coord_tensor(self, coord_tensor):
        epsilon = 0.0001
        W = torch.abs(coord_tensor[:, :,2:3, :])
        C = coord_tensor[:, :, :2, :]
        WX = torch.mul(W, C)

        mu_W = (torch.sum(WX, dim=3) / torch.sum(W, dim=3) + epsilon).unsqueeze(3)
        
        C_tilda = (C - mu_W)
        # C_tilda = C_tilda / torch.norm(C_tilda, dim=2).unsqueeze(2)
        
        W_C_tilda = torch.mul(torch.sqrt(W), C_tilda)
        return mu_W, W_C_tilda

    def extract_pose(self, img_tensor):
        coord_tensor = tensor_to_coord_tensor(img_tensor)
        mu_W , WXC = self.center_coord_tensor(coord_tensor)
        U, (L1, L2) = self.left_svd_tensor(WXC)
        U = self.standardize_orientation(U)
        return mu_W, U, (L1, L2)

    def forward(self, img_tensor):
        batch_size, channel_num, height, width = img_tensor.shape
        device = img_tensor.device
        img_tensor = torch.abs(img_tensor)

        img_scale_norm_mat = torch.Tensor([[1 / width , 0], [0, 1 / height]]).to(device)
        
        mu_W, U, (L1, L2) = self.extract_pose(img_tensor)
        return mu_W, U, (L1, L2)