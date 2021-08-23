"""
This part reuses code from https://github.com/hassony2/handobjectconsist/blob/master/meshreg/models/manobranch.py
Thanks to Yana Hasson for the excellent work.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as torch_f




class ManoBranch(nn.Module):
    def __init__(
        self,
        ncomps: int = 6,
        base_neurons=[512, 512],
        use_shape=False,
        use_pca=True,
        mano_pose_coeff=1,
        dropout=0,
    ):
        """
        Args:
            mano_root (path): dir containing mano pickle files
            center_idx: Joint idx on which to hand is centered (given joint has position
                [0, 0, 0]
            ncomps: Number of pose principal components that are predicted
        """
        super(ManoBranch, self).__init__()

        self.use_shape = use_shape
        self.use_pca = use_pca
        self.mano_pose_coeff = mano_pose_coeff

        if self.use_pca:
            # Final number of coefficients to predict for pose
            # is sum of PCA components and 3 global axis-angle params
            # for the global rotation
            mano_pose_size = ncomps + 3
        else:
            # 15 joints + 1 global rotations, 9 components per joint
            # rotation
            mano_pose_size = 48
        # Initial base layers of MANO decoder
        base_layers = []
        for inp_neurons, out_neurons in zip(base_neurons[:-1], base_neurons[1:]):
            if dropout:
                base_layers.append(nn.Dropout(p=dropout))
            base_layers.append(nn.Linear(inp_neurons, out_neurons))
            base_layers.append(nn.ReLU())
        self.base_layer = nn.Sequential(*base_layers)

        # Pose layers to predict pose parameters
        self.pose_reg = nn.Linear(base_neurons[-1], mano_pose_size)
        #if not self.use_pca:
        #    # Initialize all nondiagonal items on rotation matrix weights to 0
        #    self.pose_reg.bias.data.fill_(0)
        #    weight_mask = self.pose_reg.weight.data.new(np.identity(3)).view(9).repeat(16)
        #    self.pose_reg.weight.data = torch.abs(
        #        weight_mask.unsqueeze(1).repeat(1, 256).float() * self.pose_reg.weight.data
        #    )

        # Shape layers to predict MANO shape parameters
        if self.use_shape:
            self.shape_reg = torch.nn.Sequential(nn.Linear(base_neurons[-1], 10))

    def forward(self, inp, shape=None, pose=None):
        base_features = self.base_layer(inp)
        if pose is None:
            pose = self.pose_reg(base_features)
        if self.mano_pose_coeff != 1:
            pose = torch.cat([pose[:, :3], self.mano_pose_coeff * pose[:, 3:]], 1)



        # Get shape
        if shape is None:
            if self.use_shape:
                shape = self.shape_reg(base_features)
            else:
                shape = None
                shape_right = None
                shape_left = None

        # Get trans
        # trans = torch.Tensor([0])
        # # Get MANO vertices and joints for hands given
        # # predicted mano parameters
        # verts, joints, T_joints = self.mano_layer(
        #         pose, th_betas=shape, th_trans=trans
        #     )

        
        

        # Gather results in metric space (vs MANO millimeter outputs)
        #results = {"verts3d": verts / 1000, "joints3d": joints / 1000, "shape": shape, "pose": pose}
        #results = {"verts3d": verts, "joints3d": joints, "T_joints3d": T_joints, "shape": shape, "pose": pose, "latent":base_features}

        return pose,shape
