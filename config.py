class Config:
    TRAIN_IMAGE_WIDTH=256
    TRAIN_IMAGE_HEIGHT=256

    output_hm_shape = (64, 64, 64)
    output_root_hm_shape = 64 
    
    ATTENTION_WIDTH=16
    ATTENTION_HEIGHT=16

    HEATMAP_WIDTH=32
    HEATMAP_HEIGHT=32
    
    BATCH_SIZE = 15
    
    resnet_type = 50

    input_folder='../../dataset/InterHand/InterHand2.6M.annotations.5.fps_interacting'
    input_folder2='../../dataset/InterHand/InterHand2.6M.annotations.5.fps_single'
    CMU_input_folder='../../dataset/checked_haggling'

    lambda_dict={
        'kp':10,
        'shape_reg':0.005,
        'pose_reg':5e-2,
        'pose':10,
        'length':100,
        'dis':1,
        'shape_consist':0.01,
        'attention':0.1,
        'mask':1e-3,
        'shape':0.1,
        'repulsion':100,
        'kp2d':0.1
    }


cfg=Config()