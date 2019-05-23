import torch
from multiPoseExtraction import MultiPoseExtraction

class DeepHOG(torch.nn.Module):
    def __init__(self, window_size, stride=None):
        if stride is None:
            self.stride = window_size
        else:
            self.stride = stride
        super(DeepHOG, self).__init__()

        self.flip_y_loc = torch.Tensor([[2 , 0], [0, -2]])
        self.mu_R = torch.Tensor([0.5, 0.5])
        self.rotate_pi = torch.Tensor([[-1 ,0], [0, 1]])
        self.pi = 3.14159265358979323
        self.theta_R = (self.pi / 2)

        self.window_size = window_size
        self.multi_pose = MultiPoseExtraction()
        self.unfold = torch.nn.Unfold(kernel_size=self.window_size, stride=self.stride)

    def congregate_pose(self, orientations):
        M = torch.bmm(orientations.permute(0, 2, 1), orientations)
        T = M[:, 0, 0] + M[:, 1, 1]
        D = M[:, 0, 0] * M[:, 1, 1] - M[:, 0, 1] * M[:, 1, 0]
        L1 = (T + torch.sqrt(torch.nn.functional.relu((T**2) - 4*D))) / 2
        v1 = torch.stack([L1 - M[:, 1, 1], M[:, 1, 0]]).permute(1, 0)
        v1 = torch.nn.functional.normalize(v1, dim=1)
        return v1

    def rot_mat_tensor(self, thetas):
        num_tensors = thetas.shape[0]
        c = torch.cos(thetas).unsqueeze(1).unsqueeze(2)
        s = torch.sin(thetas).unsqueeze(1).unsqueeze(2)
        R = torch.stack([torch.cat([c, -1*s], dim=1),  torch.cat([s, c], dim=1)], dim=1).squeeze(3)
        return R

    def tensor_orientation_transform(self, U):
        eps = 0.0001
        rot_lambda = torch.fmod(torch.atan2(U[:, :, 1:2], U[:, :, 0:1] + eps) + 2 * self.pi, self.pi)

        diff_angle = (self.theta_R - rot_lambda)
        rotation = self.rot_mat_tensor(diff_angle).float()
        return rotation

    def anti_pose_transformation(self, mu, U, img_scale_norm_mat):
        device = mu.device
        T = self.tensor_orientation_transform(U)
        T = T[:, :, :, 0, 0]
        
        mu_T = torch.mm(img_scale_norm_mat, mu[:, 0, :, 0].t()).t()
        mu_T = mu_T - self.mu_R.to(device).expand(mu_T.shape[0], 2)
        mu_T = torch.mm(self.flip_y_loc.to(device), mu_T.t()).t().unsqueeze(2)

        theta = torch.cat([T, mu_T], dim=2)

        return theta

    def forward(self, x):
        batch_size, channel_num, height, width = x.shape
        device = x.device
        img_scale_norm_mat = torch.Tensor([[1 / width , 0], [0, 1 / height]]).to(device)

        mean = x.mean()

        blocks = self.unfold(x)
        batch_size, _, channel_num = blocks.shape
        blocks = blocks.permute([0, 2, 1]).view(-1, channel_num, self.window_size, self.window_size)
        block_means = blocks.view(-1, channel_num, self.window_size * self.window_size).mean(dim=2)

        mu_global, _, _ = self.multi_pose(x)
        mu_W, U, (L1, L2) = self.multi_pose(blocks)
        confidence = ((L1 / L2).unsqueeze(2) - 1) * (block_means.unsqueeze(2) / mean)
        confidence = 20*(torch.nn.functional.sigmoid(confidence) - 0.5)
        V = self.congregate_pose(U[:, :, 0, :] * confidence)
        
        output = torch.cat([mu_W[:, :, :, 0], U[:, :, 0, :] * confidence], dim=2)
        # weights = torch.nn.functional.softmax(confidence, dim=1)
        weights = confidence
        U_global = V.unsqueeze(1)
        theta = self.anti_pose_transformation(mu_global, U_global, img_scale_norm_mat) 
        
        return output, blocks, mu_global, U_global, theta

