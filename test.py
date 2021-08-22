import numpy as np
import torch
import cv2
from torch._C import device
import torch.nn as nn
import argparse
from tqdm import tqdm
import os
from network.full_model import InterShape

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default='0')
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--test_folder', type=str, default='')
    parser.add_argument('--render_result', type=str, default=0)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    model=InterShape(input_size=3,resnet_version=50,mano_neurons=[512, 512, 512, 512],mano_use_pca=False,\
                        cascaded_num=3,cascaded_input='double',heatmap_attention=True)
    device_run=torch.device('cuda:%d'%(args.gpu))
    para_dict=torch.load("{}".format(args.model_path), map_location=device_run)
    for k in model.state_dict().keys():
        if k in para_dict:
            model.state_dict()[k].copy_(para_dict[k])
    model.to(device_run)
    model.eval()
    print('load success')
    INPUT_SIZE=256
    right_face=model.mesh_reg.mano_layer['r'].th_faces
    left_face=model.mesh_reg.mano_layer['l'].th_faces
    for img_name in os.listdir(args.test_folder):
        img=cv2.imread(os.path.join(args.test_folder,img_name))
        if img is None:continue
        ratio=INPUT_SIZE/max(*img.shape[:2])
        M=np.array([[ratio,0,0],[0,ratio,0]],dtype=np.float32)
        img=cv2.warpAffine(img,M,(INPUT_SIZE,INPUT_SIZE),flags=cv2.INTER_LINEAR,borderValue=[0,0,0])
        img=img[:,:,::-1].astype(np.float32)/255-0.5
        input_tensor=torch.tensor(img.transpose(2,0,1),device=device_run,dtype=torch.float32).unsqueeze(0)
        right_mano_para_list,left_mano_para_list,trans_list=model(input_tensor)
        right_mano_para=right_mano_para_list[-1]
        left_mano_para=left_mano_para_list[-1]
        trans=trans_list[0]

        predict_right_length=(right_mano_para['joints3d'][:,9]-right_mano_para['joints3d'][:,0]).norm(dim=1)
        predict_left_length=(left_mano_para['joints3d'][:,9]-left_mano_para['joints3d'][:,0]).norm(dim=1)
        predict_right_joints=right_mano_para['joints3d']/predict_right_length[:,None,None]
        predict_left_joints=left_mano_para['joints3d']/predict_left_length[:,None,None]
        predict_right_verts=right_mano_para['verts3d']/predict_right_length[:,None,None]
        predict_left_verts=left_mano_para['verts3d']/predict_left_length[:,None,None]

        predict_left_joints_trans=(predict_left_joints+trans[:,1:].view(-1,1,3))*torch.exp(trans[:,0,None,None])
        predict_left_verts_trans=(predict_left_verts+trans[:,1:].view(-1,1,3))*torch.exp(trans[:,0,None,None])

        output_file_name=img_name.split('.')[0]
        
        with open(os.path.join(args.test_folder,output_file_name+'_right.obj'),'w') as file_object:
            for v in predict_right_verts[0]:
                print("v %f %f %f"%(v[0],v[1],v[2]),file=file_object)
            for f in right_face+1:
                print("f %d %d %d"%(f[0],f[1],f[2]),file=file_object)

        with open(os.path.join(args.test_folder,output_file_name+'_left.obj'),'w') as file_object:
            for v in predict_left_verts_trans[0]:
                print("v %f %f %f"%(v[0],v[1],v[2]),file=file_object)
            for f in left_face+1:
                print("f %d %d %d"%(f[0],f[1],f[2]),file=file_object)
                
    
    

        


if __name__=='__main__':
    main()
