# Interacting Two-Hand 3D Pose and Shape Reconstruction from Single Color Image

### Interacting Two-Hand 3D Pose and Shape Reconstruction from Single Color Image
Baowen Zhang, Yangang Wang, Xiaoming Deng*, Yinda Zhang*, Ping Tan, Cuixia Ma and Hongan Wang

[Project page](https://baowenz.github.io/Intershape/)   &nbsp;  &nbsp;  &nbsp;  [Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Interacting_Two-Hand_3D_Pose_and_Shape_Reconstruction_From_Single_Color_ICCV_2021_paper.pdf) 

![prediction example](teaser.png)

This repository contains the model of the ICCV'2021 paper ["Interacting Two-Hand 3D Pose and Shape Reconstruction from Single Color Image"](https://openaccess.thecvf.com/content/ICCV2021/papers/Zhang_Interacting_Two-Hand_3D_Pose_and_Shape_Reconstruction_From_Single_Color_ICCV_2021_paper.pdf).

We propose a novel deep learning framework to reconstruct 3D hand poses and shapes of two interacting hands from a single color image. Previous methods designed for single hand cannot be easily applied for the two hand scenario because of the heavy inter-hand occlusion and larger solution space. In order to address the occlusion and similar appearance between hands that may confuse the network, we design a hand pose-aware attention module to extract features associated to each individual hand respectively. We then leverage the two hand context presented in interaction and propose a context-aware cascaded refinement that improves the hand pose and shape accuracy of each hand conditioned on the context between interacting hands. Extensive experiments on the main benchmark datasets demonstrate that our method predicts accurate 3D hand pose and shape from single color image, and achieves the state-of-the-art performance.


# 1.Installation
This code is tested with Cuda 11.1.
## Clone this repository.
```
git clone https://github.com/BaowenZ/Two-Hand-Shape-Pose.git
cd Two-Hand-Shape-Pose
```
In the following, `${TWO_HAND}` refers to `Two-Hand-Shape-Pose`.
## Install dependencies
```
conda create -n intershape python=3.9
conda activate intershape
pip install --upgrade pip
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```
# 2.Download models
Download pre-trained model [model.pts](https://drive.google.com/drive/folders/1cQz7uOMzhsACPb86lMrVAJnACF4-h3Q7?usp=sharing) and put it into folder `model/`.

Download the MANO model files from [MANO](https://mano.is.tue.mpg.de/). Unzip `mano_v1_2.zip` under `${TWO_HAND}` and rename the unzipped folder as `mano/`.

# 3.Running the code
```
python test.py --test_folder test_data --model_path model/model.pts
```
Our model predicts hand meshes from images in `test_data/`. The estimated meshes are saved as obj files in `test_data/`.

#### Citation
Please consider citing the paper if you use this code.
```
@inproceedings{Zhang2021twohand, 
      title={Interacting Two-Hand 3D Pose and Shape Reconstruction from Single Color Image}, 
      author={Baowen Zhang, Yangang Wang, Xiaoming Deng, Yinda Zhang, Ping Tan, Cuixia Ma and Hongan Wang}, 
      booktitle={International Conference on Computer Vision (ICCV)}, 
      year={2021} 
} 
```

# 4. Acknowledgement
We use part of the great code from [InterNet](https://mks0601.github.io/InterHand2.6M/) and [mano layer](https://github.com/hassony2/manopth).

Samples in `test_data/` are from [InterHand2.6M](https://mks0601.github.io/InterHand2.6M/). 

We thank the authors of [InterNet], [InterHand2.6M] and [mano layer] for their great work.
