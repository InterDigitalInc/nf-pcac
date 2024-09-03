# NF-PCAC: Normalizing Flow Based Point Cloud Attribute Compression

## Project Information

 - Authors: [Rodrigo Borba Pinheiro<sup>1,2</sup>](https://scholar.google.com/citations?user=fwzu_toAAAAJ&hl=fr&oi=ao), [Jean-Eudes Marvie<sup>1</sup>](https://scholar.google.com/citations?hl=fr&user=eGbpfCYAAAAJ), [Giuseppe Valenzise<sup>2</sup>](https://scholar.google.com/citations?user=7ftDv4gAAAAJ), [Frederic Dufaux<sup>2</sup>](https://scholar.google.com/citations?user=ziqjbTIAAAAJ)
 - Affiliations: [<sup>1</sup>InterDigital, Inc](https://www.interdigital.com),[<sup>2</sup>Université Paris-Saclay, CNRS, CentraleSupélec, L2S 91190 Gif-sur-Yvette, France](https://l2s.centralesupelec.fr/)

## Introduction

This repository contains the implementation of [NF-PCAC](https://ieeexplore.ieee.org/document/10096294), the first end-to-end learning-based approach that makes use of a normalizing flow architecture to encode point cloud attributes. Normalizing flows are neural networks that model invertible transforms. In contrast to variational
autoencoders (VAE), these architectures do not have a low dimensional bottleneck and can achieve good reconstructions. We adapt the 2D normalizing flow architecture to take into consideration the 3D nature and the sparsity of point clouds and we add some approximations to obtain a better trade off between quality and bitrate. Experimentation shows state-of-the-art performance, while providing higher coding gains than existing learning based attribute compression approaches. It is also the first learning based approach that achieves comparable and in some cases, even better results than [G-PCC codec](https://github.com/MPEGGroup/mpeg-pcc-tmc13) for attributes.

<img src="imgs/nf-pcac.jpg" alt="drawing" width="800"/>
<!-- ![img](imgs/normalizing_flow.png) -->


## Requirements

Please refer to the requirements.txt file on the project for the necessary python packages.

* MPEG G-PCC codec [mpeg-pcc-tmc13](https://github.com/MPEGGroup/mpeg-pcc-tmc13): necessary to compare results with G-PCC and to obtain the metric config files.

* MPEG metric software [mpeg-pcc-dmetric](https://git.mpeg.expert/MPEG/3dgh/v-pcc/software/mpeg-pcc-dmetric), available on the [MPEG Gitlab](https://git.mpeg.expert), you need to register and request the permissions for `MPEG/PCC`: necessary to obtain PSNR values. (Can be replaced by other metric calculation)


## How to use

To get the help for the arguments of each file, simply use (replace "file" by the desired file to get help): 

```
python file.py --help
```

### Training

To train new models edit the config file to reflect the architecture you want.
The train_config file lets you customize the type of architecture according to their availability, choose the training dataset path and the testing dataset path. Besides you can control the number of filters of intermediate layers for the architecture.

To train all the models in the train_config file, simply run:

```
python train_all.py --config train_config.yaml
```

### Evaluating on models

Edit the eval_config.yaml file to reflect your paths:
* `MODEL_PATH`: Folder where the weights of the trained models were saved
* `MPEG_TMC13_DIR`: G-PCC folder (`mpeg-pcc-tmc13`)
* `PCERROR`: `mpeg-pcc-dmetric` folder
* `MPEG_DATASET_DIR`: MPEG PCC dataset folder
* `EXPERIMENT_DIR`: Experiment folder, all results are saved in this folder

```
python eval_all_.py --config eval_config.yaml
```
This will run the models in the config file through all the point clouds specified in the "Data" part of the .yaml

### Simple Inference for a single model

An example of command to run to perform inference in a single point cloud with the wanted model.
Make sure the model configuration reflects the checkpoint path to be loaded.

To encode:
```
python main.py --command encode --input_file input_pointcloud.ply --output_file input_pointcloud.bin --model_name model_name --arch_type NF --color_space RGB --squeeze_type avg --N_levels 3 --M 128 --enh_channels 64 --attention_channels 128 --model_path ../checkpoint.pth.tar
```

```
python main.py --command decode --input_file input_pointcloud.bin --output_file reconstructed.ply --model_name NF_128 --arch_type NF --color_space RGB --squeeze_type avg --N_levels 3 --M 128 --enh_channels 64 --attention_channels 128 --model_path ../checkpoint.pth.tar --geo input_pointcloud.ply
```

## References

[1] R. B. Pinheiro, J. -E. Marvie, G. Valenzise and F. Dufaux, "NF-PCAC: Normalizing Flow Based Point Cloud Attribute Compression," ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), Rhodes Island, Greece, 2023, pp. 1-5, doi: 10.1109/ICASSP49357.2023.10096294.

A special thanks to [@mauriceqch](https://github.com/mauriceqch) for providing a base for our code in [pcc_geo_cnn_v2](https://github.com/mauriceqch/pcc_geo_cnn_v2).

[2] M. Quach, G. Valenzise and F. Dufaux, "Improved Deep Point Cloud Geometry Compression," 2020 IEEE 22nd International Workshop on Multimedia Signal Processing (MMSP), Tampere, Finland, 2020, pp. 1-6, doi: 10.1109/MMSP48831.2020.9287077.

## Cite This Work

Please cite our work if you find it useful for your research:
```
@INPROCEEDINGS{pinheiro2022nf,
    author={Pinheiro, Rodrigo B. and Marvie, Jean-Eudes and Valenzise, Giuseppe and Dufaux, Frédéric},
    booktitle={ICASSP 2023 - 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
    title={NF-PCAC: Normalizing Flow Based Point Cloud Attribute Compression}, 
    year={2023},
    doi={10.1109/ICASSP49357.2023.10096294}}

```
