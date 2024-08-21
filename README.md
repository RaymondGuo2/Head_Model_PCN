# Cascaded Refinement Network for Head Model Completion

This is an implementation of the **Cascaded Refinement Network for Point Cloud Completion** by Wang et al. (2021) (link [here](https://ieeexplore.ieee.org/document/9525242)). Extending from the original **Point Completion Network** by Yuan et al. (2018) (link [here](https://ieeexplore.ieee.org/document/8491026)), the Cascaded Refinement Network is designed to take partial point cloud inputs and reconstruct their complete point cloud representations. While the results show promise on popular datasets such as ShapeNet and ModelNet, this has to my knowledge, not been implemented in the context of reconstructing entire heads. One source of inspiration for the potential success of usng the PCN approach has been shown in the field of archaeology in a paper by Stasinakis et al. (2022) (link [here](https://www.sciencedirect.com/science/article/pii/S1296207422001054)), who managed to reconstruct archaeological objects using the approach.

This code repository reproduces the original Cascaded Refinement Network repository [here](https://github.com/xiaogangw/cascaded-point-completion). However, it adapts and contributes to the code documentation in a number of ways. Firstly, it adopts an updated PyTorch framework as opposed to the original TensorFlow implementation, allowing greater compatibility with modern CUDA versions, avoids potential dependency issues for future research, and increases the comprehensibility of the code. Secondly, rather than applying the original data, I use a partial-face mask as input data engineered by myself for the specific project use-case. The steps to this are further documented carefully to enhance reproducibility/inspire future approaches in other domains. Thirdly, I extend the original research by taking inspiration from the **Hybrid Autocompletion Approach** by Stasinakis et al. (mentioned above), providing a framework to manipulate point clouds and meshes for more accurate head model reconstruction. The autocompletion paper unfortunately does not have any open-source code, such that I code the functions from the paper by scratch.

The steps to this project are therefore discussed below.

## Step 1: Install Dependencies

To ensure that the script works, first create a conda environment using Python 3.9.

```
conda create -n myenv python=3.9
conda activate myenv
```

Next, install the necessary dependencies to run the scripts. To do so, first try: 
```
pip install -r requirements.txt
```

If this does not work however, the dependencies will need to be installed manually. A likely difficulty will occur in the installation of [PyTorch3D](https://pytorch3d.org/). I would therefore recommend first starting with building this package, the necessary documentation of which can be found [here](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md). This will therefore include the crucial PyTorch and torchvision dependencies which should be compatible with the CUDA version that will be discussed next. Outside of PyTorch3D, the rest of the dependencies should not be too difficult to install, and can be done so as required by the running scripts. 

As for the CUDA version, this script will need to be run on CUDA version 11.7.1, and can be activated in the `.bashrc` file. Furthermore, for the installation of `PyTorch3D`, the environment variable of `FORCE_CUDA=1` may need to be set for Pytorch3D to register the use of GPU. Overall, these lines can be added to the end of the `.bashrc` file. Also do not forget to `source <path_to_bashrc>` to ensure these changes are applied. You can check that the correct CUDA version has been setup by typing `which nvcc` on Linux. 

```
export CUDA_HOME=/vol/cuda/11.7.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export FORCE_CUDA=1
```

## Step 2: Data Preparation

This step is the most challenging to complete, but is documented here for reproducibility purposes. Given that the PCN is a supervised-learning approach, a set of inputs and ground truth face data are required. The data was obtained from the Liverpool-York Head Model project, the link of the website can be found [here](https://www-users.york.ac.uk/~np7/research/LYHM/). To obtain the data, you will require a research authority to sign this form [here](https://www-users.york.ac.uk/~np7/research/LYHM/) and send the completed form to nick.pears@york.ac.uk . For more information on how to do so, you can visit the website linked [above](https://www-users.york.ac.uk/~np7/research/LYHM/). To obtain the data, you will require a research authority to sign this form [here](https://www-users.york.ac.uk/~np7/research/LYHM/). The crucial datasets that you will require are called 'headspace-v02.tar.gz' which contains the complete .OBJ head models of the subjects (n=1519), and the file called 'headspacePngTka.tar.gz', which are the corresponding images used for these subjects. Note that in 'headspacePngTka.tar.gz', there are actually only 1212 complete samples, which will be discussed later.

Having downloaded this data, we first need to process the unzipped folders. For `headspacePngTka.tar.gz`, one can execute the shell script `./prepare_data.sh`. Before doing so, please edit the two lines in the file according to the directories of where the data is for `base_dir`, and where you want to extract those files in `target_dir`. Note that for `base_dir`, you must keep the `headspacePngTka/subjects` at the end. Discuss why we chose 2C.

```
base_dir="$HOME/../../vol/bitbucket/rqg23/headspacePngTka/subjects"
target_dir="$HOME/../../vol/bitbucket/rqg23/unprocessed_2c"
```

Next up, to process the ground truth data in `headspace-v02.tar.gz`, first unzip that file, which should return a folder with the name `headspaceOnline`. To extract the necessary .OBJ files, please also do something similar to the previous step for the images, by first changing these two lines in `prepare_gt.sh`, with the `base_dir` pointing to the headspaceOnline subjects folder on your computer, and `target_dir` pointing to a directory of your choosing. After those have been completed, `./prepare_gt.sh`.

```
base_dir="$HOME/../../vol/bitbucket/rqg23/headspaceOnline/subjects"
target_dir="$HOME/../../vol/bitbucket/rqg23/ground_truth_faceCompletion"
```

After the ground truth data has been prepared, I execute the python script for pre-processing the .OBJ file which removes the RGB colour coordinates. This is done initially to facilitate the ease of training the model.

```
cd scripts
python preprocess_obj.py
```

To change where this preprocessed information is, please edit the lines at the bottom of the script in the main function run. Then, taking the new folder, run:

```
python mesh_to_cloud.py
```
This is to obtain the .pcd and .npy files to help facilitate the model training.

Perhaps more complicated is the next part which involves processing the image inputs. The issue is that the supervised model requires corresponding input cloud data, which can be particularly tricky to achieve. However, one approximation approach is the use of the 3D Morphable Model fitting framework found [here](https://github.com/sicxu/Deep3DFaceRecon_pytorch), which allows one to import custom images and returns a 3D Morphable Model representation face mask (which is therefore an incomplete output). We thus use this framework as a means of generating the partial input data for the model, obtained using the images from the headspacePngTka folder discussed previously. With the correct setup, please refer to my guidance repository [ici](insert later lol). The output was stored in a folder called landmark_outputs, having recompiled the `preprocess_obj.py` folder on the partial input. Finally, this was placed inside the right folder for mesh_to_cloud.py.

Then to complete the data preparation process with the partial input and ground truth files as well as train, validation, and test datasets, enter `data_process.py`. In this function, first you want to alter the `if __name__ == '__main__' part of the script. The first step is to place the partial and grouth folders into their own separate folder. To do so, the function requires four arguments. 

```
if __name__ == '__main__':
    source_partial_dir_path = 'path_to_processed_input_data' # includes the point cloud, obj, and numpy files
    target_partial_dir_path = 'path_to_new_directory/partial' # includes the partial directory file
    source_gt_dir_path = 'path_to_processed_groundtruth_data' # includes the point cloud, obj, and numpy files
    target_gt_dir_path = 'path_to_new_directory/ground_truth' # includes the ground truth directory file

    filter_for_npy(source_partial_dir_path, target_partial_dir_path, source_gt_dir_path, target_gt_dir_path)
```

Run: `python data_process.py` 
Having completed this stage, you can finally convert the function in the script below to:


```
if __name__ == '__main__':
    unsplit_directory_path = '~/../../vol/bitbucket/rqg23/processed_pairs_data' # Path to the new directory
    split_destination_path = '~/../../vol/bitbucket/rqg23/project_data' # Path to the final completed and split dataset
    train_val_test_split(unsplit_directory_path, split_destination_path)
```
Run: `python data_process.py` 

After this, you have finally completed the data processing step, so well done!

## Step 3: Model Training and Testing

Having completed the preprocessing steps, the rest of the functionality has been directly packed into the training ands testing functions fairly self-explanatory. With the right conda environment installed and activated with CUDA enabled, one can simply type:

```
./train.sh
```

For testing, run:

```
./test.sh
```

If you would like to modify any hyperparameters, that is where to do that also.

## Step 4: Rendering the Model

To render the model

## Citations

In this repository, a number of citations are provided for the relevant use of code, data, and research material.

```
@inproceedings{Wang_2020_CVPR,
     author = {Wang, Xiaogang and , Marcelo H. Ang Jr. and Lee, Gim Hee},
     title = {Cascaded Refinement Network for Point Cloud Completion},
     booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
     month = {June},
     year = {2020},
}
```

```
Statistical Modeling of Craniofacial Shape and Texture
H. Dai, N. E. Pears, W. Smith and C. Duncan 
International Journal of Computer Vision (2019) 
[DOI][BibTeX]
```

```
@inproceedings{deng2019accurate,
    title={Accurate 3D Face Reconstruction with Weakly-Supervised Learning: From Single Image to Image Set},
    author={Yu Deng and Jiaolong Yang and Sicheng Xu and Dong Chen and Yunde Jia and Xin Tong},
    booktitle={IEEE Computer Vision and Pattern Recognition Workshops},
    year={2019}
}
```