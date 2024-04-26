# 🕊️ HATO: Learning Visuotactile Skills with Two Multifingered Hands

[[Project](https://toruowo.github.io/hato/)]
[[Paper]()]
[[Video]()]

[Toru Lin](https://toruowo.github.io/),
[Yu Zhang*](),
[Qiyang Li*](https://colinqiyangli.github.io/),
[Haozhi Qi*](https://haozhi.io/),
[Sergey Levine](https://people.eecs.berkeley.edu/~svlevine/),
[Brent Yi](https://scholar.google.com/citations?user=Ecy6lXwAAAAJ&hl=en),
[Jitendra Malik](https://people.eecs.berkeley.edu/~malik/)
<br>

## Overview

This repo contains code, datasets, and instructions to support the following use cases:
- Collecting Demonstration Data
- Training and Evaluating Diffusion Policies
- Deploying Policies on Hardware

## Installation

```
conda create -n hato python=3.9
conda activate hato
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r ./requirements.txt
```

## Collecting Demonstration Data


This repo supports Meta Quest 2 as the teleoperation device. To start, install [oculus_reader](https://github.com/rail-berkeley/oculus_reader/blob/main/oculus_reader/reader.py) by following instructions in the link.

We use [ZMQ](https://zeromq.org/) to handle communication between hardwares.
Our data collection code is structured in the following way (credit to [gello_software](https://github.com/wuphilipp/gello_software) for the clean and modular template):


| Directory  / File        | Detail |
| :-------------: |:-------------:|
| agents/ | contains Agent classes that generate low-level hardware control commands from teleoperation device or diffusion policy |
| cameras/ | contains Camera classes that provide utilities to obtain real-time camera data |
| robots/ | contains Robot classes that interface between Python code and physical hardwares to read observations or send low-level control commands |
| *_node.py | contains ZMQ node classes for camera / robot / policy |
| env.py | contains environment classes that organize the teloperation and data collection process |
| launch_nodes.py | script to launch robot hardware and sensor ZMQ nodes |
| run_env.py | script to start the teleoperation and data collection process |

(Code files not mentioned contain utilities for training and evaluating diffusion policies, and deploying policies on hardware; please see the next two sections for more details.)

Currently available classes:
- `agents/quest_agent.py`, `agents/quest_agent_eef.py`: support using Meta Quest 2 for hardware teleoperation, in either joint-space control and end-effector-space control mode
- `agents/dp_agent.py`, `agents/dp_agent_zmq.py`: support using learned diffusion policies to provide control commands, either synchronously or asynchronously (see "Deploying Policies on Hardware" section for more details)
- `cameras/realsense_camera.py`: supports reading and preprocessing RGB-D data from RealSense cameras
- `robots/ur.py`, `robots/ability_gripper.py`, `robots/robotiq_gripper.py`: support hardware setup with a single UR5e arm or two UR5e arms, using Ability Hand or Robotiq gripper as the end-effector(s)

These classes can be flexibly modified or extended to support other teleoperation devices / learning pipelines / cameras / robot hardwares.

Example usage (note that `launch_node.py` and `run_env.py` should be run simultaneously in two separate windows):

```
# to collect data with two UR5e arms and Ability hands at 10Hz
python launch_nodes.py --robot bimanual_ur --hand_type ability
python run_env.py --agent quest_hand --no-show-camera-view --hz 10 --save_data

# to collect data with two UR5e arms and Ability hands at 10Hz, showing camera view during data collection
python launch_nodes.py --robot bimanual_ur --hand_type ability
python run_env.py --agent quest_hand --hz 10 --save_data
```
Node processes can be cleaned up by running `pkill -9 -f launch_nodes.py`.

## Training and Evaluating Diffusion Policies

### Download Datasets

[[Data Files](https://berkeley.app.box.com/s/379cf57zqm1akvr00vdcloxqxi3ucb9g?sortColumn=name&sortDirection=ASC)]

The linked data folder contains datasets for the four tasks featured in [the HATO paper](): `banana`, `stack`, `pour`, and `steak`. Scripts to download the datasets can be found in `workflow/download_dataset.sh`.

Full dataset files can be unzipped using the `unzip` command.
Note that the `pour` and `steak` datasets are split into part files because of the large file size. Before unzipping, the part files should be first concatenated back into single files using the following commands:
```
cat data_pour_part_0* > data_pour.zip
cat data_steak_part_0* > data_steak.zip
```

### Run Training

1. Run `python workflow/split_data.py --base_path Traj_Folder_Path --output_path Output_Folder_Path --data_name Data_Name --num_trajs N1 N2 N3` to split the data into train and validation sets. Number of trajectories used can be specified via the `num_trajs` argument.
2. Run `python ./learning/dp/pipeline.py --data_path Split_Folder_Path/Data_Name --model_save_path Model_Path` to train the model, where
    - `--data_path` is the splited trajectory folder, which is the output_path + data_name in step 1. (data_name should not include suffix like `_train` or `_train_10`)
    - `--model_save_path` is the path to save the model

Important Training Arguments
1. `--batch_size` : the batch size for training.
2. `--num_epochs` : the number of epochs for training.
3. `--representation_type`: the data representation type for the model. Format: `repr1--repr2--...`. Repr can be `eef`, `img`, `depth`, `touch`, `pos`, `hand_pos`
4. `--camera_indices`: the camera indices to use for the image data modality. Format: `01`,`012`,`02`, etc.
5. `--train_suffix`: the suffix for the training folder. This is useful when you want to train the model on different data splits and should be used with the `--num_trajs` arg of `split_data.py`. Format: `_10`, `_50`, etc.
6. `--load_img`: whether to load all the images into memory. If set to `True`, the training will be faster but will consume more memory.
7. `--use_memmap_cache`: whether to use memmap cache for the images. If set to `True`, it will create a memmap file in the training folder to accelerate the data loading.
8. `--use_wandb`: whether to use wandb for logging.
9. `--wandb_project_name`: the wandb project name.
10. `--wandb_entity_name`: the wandb entity name.
11. `--load_path`: the path to load the model. If set, the model will be loaded from the path and continue training. This should be the path of non-ema model.


### Run Evaluation

Run `python ./eval_dir.py --eval_dir Traj_Folder_Path --ckpt_path Model_Path_1 Model_Path_2` to evaluate multiple models on all trajectories in the folder.


## Deploying Policies on Hardware

A set of useful bash scripts can be generated using the following command:

```python workflow/gen_deploy_scripts.py -c [ckpt_folder]```

where `ckpt_folder` is a path that contains one or more checkpoints resulted from the training above. The generated bash scripts provide the following functionalities.

- To run policy deployment with asynchronous setup (see Section IV-D in paper for more details), first run `*_inference.sh` to launch the server, then run `*_node.sh` && `*_env_jit.sh` in separate terminal windows to deploy the robot with inference server.

- To run policy deployment without asynchronous setup, run `*_node.sh` && `*_env.sh` in separate terminal windows.

- To run policy evaluation for individual checkpoints, run `*_test.sh`. Note that path to a folder containing trajectories for evaluation needs to be specified in addition to the model checkpoint path.

- To run open-loop policy test, run `*_openloop.sh`.  Note that path to a demonstration data trajectory needs to be specified in addition to the model checkpoint path.


## Acknowledgement

This project was developed with help from the following codebases.

- [ability-hand-api](https://github.com/psyonicinc/ability-hand-api/tree/master/python)
- [diffusion_policy](https://github.com/real-stanford/diffusion_policy)
- [gello_software](https://github.com/wuphilipp/gello_software/tree/main)
- [mujoco_menagerie](https://github.com/google-deepmind/mujoco_menagerie/blob/main/universal_robots_ur5e/ur5e.xml)
- [oculus_reader](https://github.com/rail-berkeley/oculus_reader)

## Reference

If you find HATO or this codebase helpful in your research, please consider citing:

```
@article{lin2024learning,
  author={Lin, Toru and Zhang, Yu and Li, Qiyang and Qi, Haozhi and Yi, Brent and Levine, Sergey and Malik, Jitendra},
  title={Learning Visuotactile Skills with Two Multifingered Hands},
  journal={arXiv:2404.16823},
  year={2024}
}
```
