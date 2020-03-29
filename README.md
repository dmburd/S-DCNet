# S-DCNet
This is an unofficial implementation (Pytorch) of S-DCNet.

The paper is ["From Open Set to Closed Set: Counting Objects by Spatial Divide-and-Conquer"](https://arxiv.org/abs/1908.06473) (accepted by ICCV 2019). See the exact reference at the bottom of this page.

The official repository is https://github.com/xhp-hust-2018-2011/S-DCNet. Discussions indicate that the authors do not have plans to release the training code (as of Oct 2019). 

This repository contains the code for model training and evaluation on the published datasets ShanghaiTech Part_A and Part_B. Code for inference on standalone user-provided images is also present.


## Environment
Install required packages according to `requirements.txt`.


## Datasets
Download the ShanghaiTech dataset using the links from [this repo](https://github.com/desenzhou/ShanghaiTechDataset) or [from kaggle](https://www.kaggle.com/search?q=ShanghaiTech+in%3Adatasets). After unpacking the archive, you will have the following directory structure:

```
./
└── ShanghaiTech/
    ├── part_A/
    │   ├── test_data/
    │   │   ├── ground-truth/
    │   │   │   └── GT_IMG_{1,2,3,...,182}.mat
    │   │   └── images/
    │   │       └── IMG_{1,2,3,...,182}.jpg
    │   └── train_data/
    │       ├── ground-truth/
    │       │   └── GT_IMG_{1,2,3,...,300}.mat
    │       └── images/
    │           └── IMG_{1,2,3,...,300}.jpg
    └── part_B/
        ├── test_data/
        │   ├── ground-truth/
        │   │   └── GT_IMG_{1,2,3,...,316}.mat
        │   └── images/
        │       └── IMG_{1,2,3,...,316}.jpg
        └── train_data/
            ├── ground-truth/
            │   └── GT_IMG_{1,2,3,...,400}.mat
            └── images/
                └── IMG_{1,2,3,...,400}.jpg
```


## Ground truth density maps
Generate ground truth density maps by running a command like 
```
./gen_density_maps.py \
    --dataset-rootdir ./ShanghaiTech \
    --part A
# and/or similarly for part_B
```
Files with the names `density_maps_part_{A|B}_{train,test}.npz` will appear in the current directory.

The generated density maps can be visualized and compared to the pre-calculated density maps provided by the [official repo](https://github.com/xhp-hust-2018-2011/S-DCNet) (only for the test sets of ShanghaiTech Part_A and Part_B). In order to do so, download the archive `Test_Data.zip` using the links in the `Data` section of the README in the repo. After unpacking the archive, you will have the following directory structure:
```
./
└── Test_Data/
    ├── SH_partA_Density_map/
    │   ├── test/
    │   │   ├── gtdens/
    │   │   │   └── IMG_{1,2,3,...,182}.mat
    │   │   └── images/
    │   │       └── IMG_{1,2,3,...,182}.jpg
    │   └── rgbstate.mat
    └── SH_partB_Density_map/
        ├── test/
        │   ├── gtdens/
        │   │   └── IMG_{1,2,3,...,316}.mat
        │   └── images/
        │       └── IMG_{1,2,3,...,316}.jpg
        └── rgbstate.mat
```
Next, call `./gen_density_maps.py` with the path to the `gtdens` directory for the command line argument `--xhp-dir`:
```
./gen_density_maps.py \
    --dataset-rootdir ./ShanghaiTech \
    --part A \
    --xhp-dir ./Test_Data/SH_partA_Density_map/test/gtdens/
# and/or similarly for part_B
```
Directories with the names `cmp_dmaps_part_{A|B}_test_<some_random_string>` will appear in the current directory. Each of them contains pairs of images named `IMG_<N>_my.png` / `IMG_<N>_xhp.png`.


## Training
NOTE: training requires availability of cuda.

`train.py` is the script for training a model. Launch the script by a command like this:
```
./train.py \
    --dataset-rootdir ./ShanghaiTech \
    --part A \
    --densmaps-gt-npz ./density_maps_part_A_\*.npz \
    --train-val-split 0.9 \
    --num-epochs 1000 \
    --num-intervals 22 
```
A typical command line for training on ShanghaiTech part_B is shown in `train.sh`.
The logs and checkpoints generated during training are placed to a folder named like `run_<date>_<time>`. Plots of MAE and MSE vs epoch number can be visualized by `tensorboard`:
```
tensorboard --logdir run_<date>_<time>
```

## Evaluation
`evaluate.py` is the script for evaluating a checkpoint. Select a checkpoint number `N` and run a command like this:
```
./evaluate.py --checkpoint ./run_<date>_<time>/checkpoints/epoch_<N>.pth
```
You will get an output like this for part_A:
```
Evaluating on the (whole) train data and on the test data (in ./ShanghaiTech/part_A)
Metrics on the (whole) train data: MAE: 10.78, MSE: 26.11
Metrics on the test data:          MAE: 64.59, MSE: 107.49
```
or like this for part_B:
```
Evaluating on the (whole) train data and on the test data (in ./ShanghaiTech/part_B)
Metrics on the (whole) train data: MAE: 2.17,  MSE: 7.28
Metrics on the test data:          MAE: 10.35, MSE: 18.75
```
The error values shown above were obtained after training on Google Colab, so the resources and training time were quite limited. The error values are significantly higher than that reported in the [original paper](https://arxiv.org/abs/1908.06473) (MAE = 58.3, MSE = 95.0 for part_A test set; MAE = 6.7, MSE = 10.7 for part_B test set).

|       dataset            |  MAE  | MSE    | checkpoint |
| :----------------------: | :---: | :----: | :-----------------: | 
| ShanghaiTech part A test | 64.59 | 107.49 | [Google Drive link](https://drive.google.com/file/d/12_nSmT8Lvahbq6MZ65qwEsA3EvuhZng7/view?usp=sharing) |
| ShanghaiTech part B test | 10.35 | 18.75  | [Google Drive link](https://drive.google.com/file/d/1ptWyD7CLSvYx9s64zkkbnycZZ4Icsji_/view?usp=sharing) |

You can visualize the predictions of a model on a test set by adding `--visualize yes` to the evaluation script command line. Combined images showing the ground truth counts, predicted counts, absolute / relative errors and coarse-grained density maps will be placed to a folder named `./visualized_part_{A|B}_test_set_predictions/`.


## Inference 
To perform inference on user-specified images, run a command like this:
```
./inference.py \
    --checkpoint ./run_<date>_<time>/checkpoints/epoch_<N>.pth \
    --images-dir ./dir_for_inference
```
where `./dir_for_inference` folder contains the images. The visualized predictions will be placed to the newly-created `./dir_for_inference/visualized_predictions` subdirectory. Also, the image file names and corresponding total count values will be printed to stdout.

To export the model to onnx, torch.jit.trace and torch.jit.script formats, add `--export yes` to the command line above. The generated files will be placed to the directory where the checkpoint is stored (`./run_<date>_<time>/checkpoints/`).


## References
The exact reference to the original paper is
```
@inproceedings{xhp2019SDCNet,
    title={From Open Set to Closed Set: Counting Objects by Spatial Divide-and-Conquer},
    author={Xiong, Haipeng and Lu, Hao and Liu, Chengxin and Liang, Liu and Cao, Zhiguo and Shen, Chunhua},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    year={2019}
}
```
