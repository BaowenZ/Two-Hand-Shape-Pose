import torch
import torch.nn as nn
from network.meshreg import MeshRegNet
from network.InterHand.module import BackboneNet, PoseNet

class InferenceModel(nn.Module):
    def __init__(self, backbone_net, pose_net):
        super(InferenceModel, self).__init__()

        # modules
        self.backbone_net = backbone_net
        self.pose_net = pose_net
          
    def forward(self, inputs):
        input_img = inputs
        batch_size = input_img.shape[0]
        img_feat = self.backbone_net(input_img)
        joint_heatmap_out, rel_root_depth_out, hand_type = self.pose_net(img_feat)
        return joint_heatmap_out,rel_root_depth_out, hand_type

class InterShape(nn.Module):
    def __init__(self,input_size,resnet_version,mano_use_pca,mano_neurons,\
                    cascaded_num,cascaded_input,heatmap_attention):
        super(InterShape, self).__init__()

        self.depth_downsample_factor=4
        self.spatil_downsample_factor=2
        self.mesh_reg=MeshRegNet(input_size=input_size, resnet_version=resnet_version, mano_use_pca=mano_use_pca, mano_neurons=mano_neurons,\
                                    addition_channels=42*64//self.depth_downsample_factor,\
                                    cascaded_num=cascaded_num, cascaded_input=cascaded_input)
        backbone_net = BackboneNet()
        pose_net = PoseNet(21)
        self.heatmap_predictor = InferenceModel(backbone_net, pose_net)
        self.heatmap_attention = heatmap_attention
    def forward(self,x):
        heatmap,_,_=self.heatmap_predictor(x+0.5)
        downsampled_heatmap=heatmap[:,:,::self.depth_downsample_factor,::self.spatil_downsample_factor,::self.spatil_downsample_factor]
        B,K,D,H,W=downsampled_heatmap.shape
        if self.heatmap_attention:
            attention_map=heatmap.reshape(heatmap.shape[0],-1,heatmap.shape[3],heatmap.shape[4])
            right_attention_map,_=attention_map[:,:(21*64),:,:].max(dim=1,keepdim=True)
            left_attention_map,_=attention_map[:,(21*64):,:,:].max(dim=1,keepdim=True)
            attention=torch.cat([right_attention_map,left_attention_map],dim=1)
        else:
            attention=None

        val_z, idx_z = torch.max(heatmap,2)
        val_zy, idx_zy = torch.max(val_z,2)
        val_zyx, joint_x = torch.max(val_zy,2)
        joint_x = joint_x[:,:,None]
        joint_y = torch.gather(idx_zy, 2, joint_x)
        xyc=torch.cat((joint_x, joint_y, val_zyx[:,:,None]),2).float()
        
        output=self.mesh_reg(x,downsampled_heatmap.reshape(B,K*D,H,W),xyc,attention)
        return output