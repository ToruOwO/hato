import argparse
import collections
import concurrent
import os
import pickle
import sys
import time

mypath = os.path.dirname(os.path.realpath(__file__))

print("adding", mypath, "to the sys path")

sys.path.append(mypath)

import data_processing
import numpy as np
import torch
from dataset import Dataset
from learner import DiffusionPolicy
from models import GaussianNoise, ImageEncoder, StateEncoder
from torch import nn
from torch.nn import ModuleList
from torchvision import transforms
from utils import WandBLogger, generate_random_string, get_eef_delta, save_args

LEFT_UR_IDX = list(range(0, 6))
RIGHT_UR_IDX = list(range(12, 18))
LEFT_HAND_IDX = list(range(6, 12))
RIGHT_HAND_IDX = list(range(18, 24))
HAND_IDX = LEFT_HAND_IDX + RIGHT_HAND_IDX
RT_DIM = {
    "eef": 12,
    "hand_pos": 12,
    "pos": 24,
    "touch": 60,
    "action": 24,
}
TEST_INPUT = {
    "joint_positions": torch.zeros(24),
    "ee_pos_quat": torch.zeros(12),
    "base_rgb": torch.zeros(3, 480, 640, 3),
    "base_depth": torch.zeros(3, 480, 640),
    "control": torch.zeros(24),
    "touch": torch.zeros(60),
    "hand_pos": torch.zeros(12),
}


class Agent:
    def __init__(
        self,
        output_sizes={
            "eef": 64,
            "hand_pos": 64,
            "img": 128,
            "pos": 128,
            "touch": 64,
        },
        dropout={
            "eef": 0.0,
            "hand_pos": 0.0,
            "img": 0.0,
            "pos": 0.0,
            "touch": 0.0,
        },
        action_dim=24,
        camera_indices=[0, 1, 2],
        representation_type=["eef", "hand_pos", "img", "touch", "depth"],
        pred_horizon=4,
        obs_horizon=1,
        action_horizon=2,
        identity_encoder=False,
        without_sampling=False,
        predict_eef_delta=False,
        predict_pos_delta=False,
        clip_far=False,
        color_jitter=False,
        num_diffusion_iters=100,
        load_img=False,
        weight_decay=1e-6,
        num_workers=64,
        use_ddim=False,
        binarize_touch=False,
        policy_dropout_rate=0.0,
        state_noise=0.0,
        img_gaussian_noise=0.0,
        img_masking_prob=0.0,
        img_patch_size=16,
        compile_train=False,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.num_workers = num_workers
        self.binarize_touch = binarize_touch

        # Merge depth and rgb image
        if "depth" in representation_type:
            self.image_channel = 4
            representation_type.remove("depth")
        else:
            self.image_channel = 3
        self.representation_type = representation_type

        self.camera_indices = camera_indices
        self.predict_pos_delta = predict_pos_delta
        self.image_num = len(camera_indices)
        self.cpu = torch.device("cpu")
        image_encoder, touch_encoder, pos_encoder = None, None, None
        eef_dim, hand_pos_dim, image_dim, touch_dim, pos_dim = 0, 0, 0, 0, 0
        self.clip_far = clip_far
        self.load_img = load_img
        self.color_jitter = color_jitter
        self.epi_dir = []
        self.compile_train = compile_train

        # Use color jitter to augment the image
        if self.color_jitter:
            if self.image_channel == 3:
                # no depth
                self.downsample = nn.Sequential(
                    transforms.Resize(
                        (240, 320),
                        interpolation=transforms.InterpolationMode.BILINEAR,
                        antialias=True,
                    ),
                    transforms.ColorJitter(brightness=0.1),
                )
            else:
                # with depth, only jitter the rgb part
                self.downsample = lambda x: transforms.Resize(
                    (240, 320),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                    antialias=True,
                )(
                    torch.concat(
                        [transforms.ColorJitter(brightness=0.1)(x[:, :3]), x[:, 3:]],
                        axis=1,
                    )
                )

        # Not using color jitter, only downsample the image
        else:
            self.downsample = nn.Sequential(
                transforms.Resize(
                    (240, 320),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
            )

        mean_vec = [128.0] * 3 + [35767.0] * (self.image_channel - 3)
        std_vec = [128.0] * 3 + [35767.0] * (self.image_channel - 3)

        # Crop randomization, normalization
        self.transform = nn.Sequential(
            transforms.RandomCrop((216, 288)),
            transforms.Normalize(
                mean=mean_vec,
                std=std_vec,
            ),
        )

        # Add gaussian noise to the image
        if img_gaussian_noise > 0.0:
            self.transform = nn.Sequential(
                self.transform,
                GaussianNoise(img_gaussian_noise),
            )

        def mask_img(x):
            # Divide the image into patches and randomly mask some of them
            img_patch = x.unfold(2, img_patch_size, img_patch_size).unfold(
                3, img_patch_size, img_patch_size
            )
            mask = (
                torch.rand(
                    (
                        x.shape[0],
                        x.shape[-2] // img_patch_size,
                        x.shape[-1] // img_patch_size,
                    )
                )
                < img_masking_prob
            )
            mask = mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1).expand_as(img_patch)
            x = x.clone()
            x.unfold(2, img_patch_size, img_patch_size).unfold(
                3, img_patch_size, img_patch_size
            )[mask] = 0
            return x

        if img_masking_prob > 0.0:
            self.transform = lambda x: mask_img(
                nn.Sequential(
                    transforms.RandomCrop((216, 288)),
                    transforms.Normalize(mean=mean_vec, std=std_vec),
                )(x)
            )
        # For evaluation, only center crop and normalize
        self.eval_transform = nn.Sequential(
            transforms.CenterCrop((216, 288)),
            transforms.Normalize(
                mean=mean_vec,
                std=std_vec,
            ),
        )

        self.stats = None
        obs_dim = 0
        encoders = {}
        if "eef" in self.representation_type:
            eef_dim = RT_DIM["eef"]  # pos, rot (axis-angle) for two arms
            if identity_encoder:
                eef_encoder = nn.Identity(eef_dim)
            else:
                eef_encoder = StateEncoder(
                    input_size=eef_dim,
                    output_size=output_sizes["eef"],
                    hidden_size=128,
                    dropout=dropout["eef"],
                )
                eef_dim = output_sizes["eef"]
            encoders["eef"] = eef_encoder
            obs_dim += eef_dim
        if "hand_pos" in self.representation_type:
            hand_pos_dim = RT_DIM["hand_pos"]  # 6 on each hand
            if identity_encoder:
                hand_pos_encoder = nn.Identity(hand_pos_dim)
            else:
                hand_pos_encoder = StateEncoder(
                    input_size=hand_pos_dim,
                    output_size=output_sizes["hand_pos"],
                    hidden_size=128,
                    dropout=dropout["hand_pos"],
                )
                hand_pos_dim = output_sizes["hand_pos"]
            encoders["hand_pos"] = hand_pos_encoder
            obs_dim += hand_pos_dim
        if "img" in self.representation_type:
            image_encoder = ModuleList(
                [
                    # Use different image encoders for each camera
                    ImageEncoder(
                        output_sizes["img"], self.image_channel, dropout["img"]
                    )
                    for i in range(self.image_num)
                ]
            )
            image_dim = output_sizes["img"] * self.image_num
            encoders["img"] = image_encoder
            obs_dim += image_dim
        if "pos" in self.representation_type:
            pos_dim = RT_DIM["pos"]
            if identity_encoder:
                pos_encoder = nn.Identity(pos_dim)
            else:
                pos_encoder = StateEncoder(
                    pos_dim, output_sizes["pos"], dropout=dropout["pos"]
                )
                pos_dim = output_sizes["pos"]
            encoders["pos"] = pos_encoder
            obs_dim += pos_dim
        if "touch" in self.representation_type:
            touch_dim = RT_DIM["touch"]  # 6 on each finger
            if identity_encoder:
                touch_encoder = nn.Identity(touch_dim)
            else:
                touch_encoder = StateEncoder(
                    touch_dim,
                    output_sizes["touch"],
                    dropout=dropout["touch"],
                    binarize_touch=binarize_touch,
                )
                touch_dim = output_sizes["touch"]
            encoders["touch"] = touch_encoder
            obs_dim += touch_dim

        self.policy = DiffusionPolicy(
            obs_horizon=obs_horizon,
            obs_dim=obs_dim,
            pred_horizon=pred_horizon,
            action_horizon=action_horizon,
            action_dim=action_dim,
            representation_type=representation_type,
            encoders=encoders,
            num_diffusion_iters=num_diffusion_iters,
            without_sampling=without_sampling,
            weight_decay=weight_decay,
            use_ddim=use_ddim,
            binarize_touch=self.binarize_touch,
            policy_dropout_rate=policy_dropout_rate,
        )

        # Compile the forward function to accelerate deployment inference
        if self.compile_train:
            self.policy.nets["noise_pred_net"].forward = torch.compile(
                self.policy.nets["noise_pred_net"].forward
            )

        self.policy.to(self.device)
        self.iter = 0
        self.obs_deque = None
        self.threshold = 8000
        self.state_noise = state_noise

        self.predict_eef_delta = predict_eef_delta

    def _get_image_observation(self, data):
        # allocate memory for the image
        img = torch.zeros(
            (len(data), len(self.camera_indices), self.image_channel, 240, 320),
            dtype=torch.float32,
        )

        image_size = data[0]["base_rgb"].shape
        H, W = image_size[1], image_size[2]

        if self.image_channel == 4:
            # Use depth
            def process_rgbd(d):
                rgbd = np.concatenate(
                    [d["base_rgb"], d["base_depth"][..., None]], axis=-1
                ).reshape(-1, H, W, self.image_channel)  # [camera_num, 480, 640, 4]
                if self.clip_far:
                    # Crop image based on depth
                    clip_back_view = d["base_depth"][0] > (self.threshold / 10)
                    clip_wrist = d["base_depth"][1:] > (self.threshold)
                    clip = np.concatenate([clip_back_view, clip_wrist], axis=0)
                    clip = np.concatenate(
                        [clip[..., None]] * self.image_channel, axis=-1
                    )
                    rgbd = rgbd * clip
                rgbd = rgbd[:, self.camera_indices].astype(np.float32)
                rgbd = np.moveaxis(rgbd, -1, 1)  # [camera_num, 4, 480, 640]
                if H == 480 and W == 640 and self.color_jitter == False:
                    rgbd = self.downsample(
                        torch.tensor(rgbd)
                    )  # [camera_num, 4, 240, 320]
                else:
                    rgbd = torch.tensor(rgbd)
                return rgbd

            fn = process_rgbd

        else:
            # Only use rgb
            def process_rgb(d):
                rgb = d["base_rgb"].reshape(
                    -1, H, W, self.image_channel
                )  # [camera_num, 480, 640, 3]
                rgb = rgb[self.camera_indices].astype(np.float32)
                rgb = np.moveaxis(rgb, -1, 1)  # [camera_num, 3, 480, 640]
                if H == 480 and W == 640 and self.color_jitter == False:
                    rgb = self.downsample(
                        torch.tensor(rgb)
                    )  # [camera_num, 3, 240, 320]
                else:
                    rgb = torch.tensor(rgb)
                return rgb

            fn = process_rgb

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.num_workers
        ) as executor:
            future_to_data = {
                executor.submit(fn, d): (d, i) for i, d in enumerate(data)
            }
            for future in concurrent.futures.as_completed(future_to_data):
                d, i = future_to_data[future]
                try:
                    img[i] = future.result()
                except Exception as exc:
                    print(f"loading image failed: {exc}")

        return img

    def get_observation(self, data, load_img=False):
        input_data = {}
        for rt in self.representation_type:
            if rt == "img":
                if load_img:
                    # Load all the image to the memory
                    input_data[rt] = self._get_image_observation(data)
                else:
                    # Only keep the file path, and load the image while training
                    input_data[rt] = np.stack([d["file_path"] for d in data])
            elif rt == "eef":
                input_data[rt] = np.stack([d["ee_pos_quat"] for d in data])
            elif rt == "hand_pos":
                input_data[rt] = np.stack(
                    [d["joint_positions"][HAND_IDX] for d in data]
                )
            elif rt == "pos":
                input_data[rt] = np.stack([d["joint_positions"] for d in data])
            else:
                input_data[rt] = np.stack([d[rt] for d in data])
        return input_data

    def predict(self, obs_deque: collections.deque, num_diffusion_iters=15):
        """
        data: dict
            data['image']: torch.tensor (1,5,224,224)
            data['touch']: torch.tensor (1,6)
            data['pos']: torch.tensor (1,24)
        """
        pred = self.policy.forward(
            self.stats, obs_deque, num_diffusion_iters=num_diffusion_iters
        )
        return pred

    def _get_init_train_data(self, total_data_points, memmap_loader_path=""):
        init_data = {}
        for rt in self.representation_type + ["action"]:
            if rt == "img":
                if memmap_loader_path == "":
                    # Not using memmap
                    if self.load_img:
                        # Declear the memory to store the image
                        init_data[rt] = np.empty(
                            (
                                total_data_points,
                                self.image_num,
                                self.image_channel,
                                240,
                                320,
                            ),
                            dtype=np.uint16,
                        )
                    else:
                        # Only store the file path
                        init_data[rt] = np.empty((total_data_points,), dtype=object)
            else:
                # Other representation types
                init_data[rt] = np.zeros(
                    (total_data_points, RT_DIM[rt]), dtype=np.float32
                )
        return init_data

    def get_train_loader(self, batch_size, memmap_loader_path="", eval=False):
        current_epi_dir = self.epi_dir

        total_data_points = sum([len(os.listdir(epi)) for epi in current_epi_dir])
        train_data = {"data": {}, "meta": {}}
        train_data["data"] = self._get_init_train_data(
            total_data_points, memmap_loader_path=memmap_loader_path
        )
        train_data["meta"] = {"episode_ends": []}

        cache_memmap = False
        img_shape = (total_data_points, self.image_num, self.image_channel, 240, 320)
        if memmap_loader_path != "":
            # Use memmap to load imgs
            if os.path.exists(memmap_loader_path):
                # Load the existing memmap file
                fp = np.memmap(
                    memmap_loader_path, dtype=np.uint16, mode="r", shape=img_shape
                )
                train_data["data"]["img"] = fp
            else:
                # Create a new memmap file
                cache_memmap = True
                fp = np.memmap(
                    memmap_loader_path, dtype=np.uint16, mode="w+", shape=img_shape
                )
        else:
            pass

        data_index = 0

        print("Loading training data    ")
        for i, epi in enumerate(current_epi_dir):
            print("loading {}-th data from {}\r".format(i, epi), end="")
            data = data_processing.iterate(epi, load_img=self.load_img)
            if len(data) == 0:
                continue

            data_length = len(data)

            # images - (N, num_cams, self.image_channel, 240, 320)
            obs = self.get_observation(data, self.load_img or cache_memmap)

            # obs space
            for rt in self.representation_type:
                if rt == "img":
                    if cache_memmap:
                        fp[data_index : data_index + data_length] = obs[rt]
                        fp.flush()
                    elif memmap_loader_path == "":
                        train_data["data"][rt][
                            data_index : data_index + data_length
                        ] = obs[rt]
                else:
                    train_data["data"][rt][data_index : data_index + data_length] = obs[
                        rt
                    ]

            # action space
            train_data["data"]["action"][data_index : data_index + data_length] = (
                self.get_train_action(data)
            )

            if len(train_data["meta"]["episode_ends"]) == 0:
                train_data["meta"]["episode_ends"].append(data_length)
            else:
                train_data["meta"]["episode_ends"].append(
                    data_length + train_data["meta"]["episode_ends"][-1]
                )
            data_index += data_length

        if cache_memmap:
            fp = np.memmap(
                memmap_loader_path, dtype=np.uint16, mode="r", shape=img_shape
            )
            train_data["data"]["img"] = fp

        print("Train data loaded")
        for k, v in train_data["data"].items():
            print(k, v.shape)

        train_dataset = Dataset(
            data=train_data,
            representation_type=self.representation_type,
            pred_horizon=self.pred_horizon,
            obs_horizon=self.obs_horizon,
            action_horizon=self.action_horizon,
            stats=self.stats,
            load_img=self.load_img or memmap_loader_path != "",
            transform=self.transform if not eval else self.eval_transform,
            get_img=self._get_image_observation,
            binarize_touch=self.binarize_touch,
            state_noise=self.state_noise if not eval else 0.0,
        )
        dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=self.num_workers,
            shuffle=not eval,
            pin_memory=True,
            persistent_workers=True,
        )
        self.policy.data_stat = train_dataset.stats
        return dataloader

    def train(
        self,
        path_list,
        batch_size=4,
        epochs=300,
        traj_type="all",
        prefix="",
        save_path=None,
        save_freq=10,
        eval_freq=10,
        wandb_logger=None,
        memmap_loader_path="",
        train_path=None,
        test_path=None,
    ):
        torch.cuda.empty_cache()

        if train_path is not None and test_path is not None:
            print(train_path)
            print(test_path)
            self.epi_dir = data_processing.get_epi_dir(
                train_path, traj_type=traj_type, prefix=prefix
            )
            print(self.epi_dir)
            eval_trajs = data_processing.get_epi_dir(
                test_path, traj_type=traj_type, prefix=prefix
            )
            assert len(eval_trajs) == 1
            eval_traj = eval_trajs[0]
            print("eval traj:", eval_traj)
        else:
            if type(path_list) != list:
                path_list = [path_list]
            for path in path_list:
                self.epi_dir += data_processing.get_epi_dir(
                    path, traj_type=traj_type, prefix=prefix
                )
            eval_traj = self.epi_dir[-1]
            self.epi_dir.remove(eval_traj)
            print("eval traj:", eval_traj)
        eval_data = self.get_eval_data(eval_traj)

        train_loader = self.get_train_loader(batch_size, memmap_loader_path)
        self.policy.set_lr_scheduler(len(train_loader) * epochs)
        if self.stats is None:
            self.stats = train_loader.dataset.stats
            self.save_stats(save_path)
        if self.compile_train:
            train_loader.dataset.__getitem__ = torch.compile(
                train_loader.dataset.__getitem__
            )

        self.policy.train(
            epochs,
            train_loader,
            save_path=save_path,
            eval_data=eval_data,
            eval_freq=eval_freq,
            save_freq=save_freq,
            wandb_logger=wandb_logger,
        )

        self.policy.to_ema()
        self.eval(eval_traj)

    def get_train_action(self, data):
        act = self.get_eval_action(data)
        return np.stack(act)

    def get_eval_action(self, data):
        if self.predict_eef_delta:
            # TODO: make sure this is only used when "control" is eef pose
            act = []
            for d in data:
                left_arm_act = get_eef_delta(
                    d["ee_pos_quat"][:6], d["control"][LEFT_UR_IDX]
                )
                left_hand_act = d["control"][LEFT_HAND_IDX]
                right_arm_act = get_eef_delta(
                    d["ee_pos_quat"][6:], d["control"][RIGHT_UR_IDX]
                )
                right_hand_act = d["control"][RIGHT_HAND_IDX]
                act.append(
                    np.concatenate(
                        [left_arm_act, left_hand_act, right_arm_act, right_hand_act],
                        axis=-1,
                    )
                )
            return act
        elif self.predict_pos_delta:
            # TODO: make sure this is only used when "control" is joint pos
            act = [d["control"] for d in data]
            act = np.diff(act, axis=0, append=act[-1:])
            return act
        else:
            return [d["control"] for d in data]

    def get_eval_data(self, data_path):
        print("GETTING EVAL DATA", end="\r")
        data = data_processing.iterate(data_path)

        action = self.get_eval_action(data)
        B = len(data)

        print("GETTING EVAL OBSERVATION", end="\r")
        obs = self.get_observation(data, load_img=True)
        obs_list = []

        if "img" in self.representation_type:
            # transfer image type to float32
            obs["img"] = obs["img"].float()
            obs["img"] = self.eval_transform(obs["img"])
        for i in range(B):
            obs_list.append({rt: obs[rt][i] for rt in self.representation_type})
        return obs_list, action

    def eval(self, data_path, save_path=None):
        print("GETTING EVAL DATA")
        eval_data = self.get_eval_data(data_path)
        obs, action = eval_data
        print("EVALUATING")
        action, mse, norm_mse = self.policy.eval(obs, action)
        print("ACTION_MSE: {}, NORM_MSE: {}".format(mse, norm_mse))
        if save_path is None:
            save_path = "./eval/{}".format(
                data_path.split("/")[-1]
                + "_"
                + time.strftime("%m%d_%H%M%S", time.localtime())
            )
        os.makedirs(save_path, exist_ok=True)
        for i in range(len(action)):
            if save_path is not None:
                with open(os.path.join(save_path, str(i) + ".pkl"), "wb") as f:
                    pickle.dump(
                        {
                            "control": action[i],
                            "joint_positions": eval_data[0][i]["pos"],
                        },
                        f,
                    )

    def get_eval_loader(
        self, dir_path, traj_type="plain", prefix="0", batch_size=32, num_workers=16
    ):
        self.num_workers = num_workers
        print(f"GETTING EVAL DATA FROM {dir_path}")
        self.epi_dir = data_processing.get_epi_dir(dir_path, traj_type, prefix)
        eval_loader = self.get_train_loader(batch_size, "", eval=True)
        return eval_loader

    def eval_dir(self, eval_loader, num_diffusion_iters=15):
        self.policy.num_diffusion_iters = num_diffusion_iters
        with torch.no_grad():
            mse, action_mse = self.policy.eval_loader(eval_loader)
        print(f"MSE: {mse}", f"ACTION_MSE: {action_mse}")
        return mse, action_mse

    def load(self, path):
        model_path = os.path.join(path)
        dir_path = os.path.dirname(path)
        stat_path = os.path.join(dir_path, "stats.pkl")
        self.stats = pickle.load(open(stat_path, "rb"))
        self.policy.data_stat = self.stats
        self.policy.load(model_path)
        print("model loaded")

    def save_stats(self, path):
        os.makedirs(path, exist_ok=True)
        stat_path = os.path.join(path, "stats.pkl")
        if not os.path.exists(stat_path):
            with open(stat_path, "wb") as f:
                pickle.dump(self.stats, f)
        print("stats saved")

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        model_path = os.path.join(path, "model.ckpt")
        stat_path = os.path.join(path, "stats.pkl")
        self.policy.save(model_path)

        # if stat not exist, create one
        if not os.path.exists(stat_path):
            with open(stat_path, "wb") as f:
                pickle.dump(self.stats, f)
        print("model saved")


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


if __name__ == "__main__":
    # TODO: better config management
    args = argparse.ArgumentParser()
    # train config
    args.add_argument("--batch_size", type=int, default=32)
    args.add_argument("--obs_horizon", type=int, default=1)
    args.add_argument("--action_horizon", type=int, default=8)
    args.add_argument("--pred_horizon", type=int, default=16)
    args.add_argument("--epochs", type=int, default=300)

    # input config
    args.add_argument("--traj_type", type=str, default="plain")
    args.add_argument("--prefix", type=str, default=None)
    args.add_argument("--save_path", type=str, default=None)
    args.add_argument("--load_path", type=str, default=None)

    args.add_argument("--eval", type=boolean_string, default=False)
    args.add_argument(
        "--representation_type", type=str, default="img-depth-eef-hand_pos-touch"
    )

    args.add_argument("--base_path", type=str, default="/shared")
    args.add_argument("--data_name", type=str, default="test_data")
    args.add_argument("--data_path", type=str, default=None)
    args.add_argument("--data_prefix", type=str, default=None)
    args.add_argument("--model_save_path", type=str, default=None)

    args.add_argument("--clip_far", type=boolean_string, default=False)
    args.add_argument("--color_jitter", type=boolean_string, default=False)
    args.add_argument("--predict_eef_delta", type=boolean_string, default=False)
    args.add_argument("--predict_pos_delta", type=boolean_string, default=False)
    args.add_argument("--use_ddim", type=boolean_string, default=False)

    args.add_argument("--policy_dropout_rate", type=float, default=0.0)  # For simple BC
    args.add_argument(
        "--dropout_rate", type=float, default=0.0
    )  # For the state encoder of Diffusion Policy
    args.add_argument(
        "--img_dropout_rate", type=float, default=0.0
    )  # For the image encoder of Diffusion Policy
    args.add_argument("--weight_decay", type=float, default=1e-5)
    args.add_argument("--image_output_size", type=int, default=32)
    args.add_argument("--state_noise", type=float, default=0.0)
    args.add_argument("--img_gaussian_noise", type=float, default=0.0)
    args.add_argument("--img_masking_prob", type=float, default=0.0)
    args.add_argument("--img_patch_size", type=int, default=16)
    args.add_argument("--num_workers", type=int, default=16)

    args.add_argument("--eval_path", type=str, default=None)
    args.add_argument("--identity_encoder", type=boolean_string, default=False)
    args.add_argument("--gpu", type=int, default=0)

    args.add_argument("--camera_indices", type=str, default="012")
    args.add_argument("--save_freq", type=int, default=10)
    args.add_argument("--eval_freq", type=int, default=10)

    args.add_argument("--add_model_save_path_suffix", type=boolean_string, default=True)
    args.add_argument("--use_wandb", type=boolean_string, default=False)
    args.add_argument("--without_sampling", type=boolean_string, default=False)
    args.add_argument("--binarize_touch", type=boolean_string, default=False)

    # model config
    args.add_argument("--num_diffusion_iters", type=int, default=100)
    args.add_argument("--wandb_exp_name", type=str, default=None)
    args.add_argument("--load_img", type=boolean_string, default=False)
    args.add_argument("--train_suffix", type=str, default="")

    args.add_argument("--use_train_test_split", type=boolean_string, default=True)
    args.add_argument("--use_memmap_cache", type=boolean_string, default=False)
    args.add_argument("--memmap_loader_path", type=str, default=None)
    args.add_argument("--compile_train", type=boolean_string, default=False)

    # wandb config
    args.add_argument("--wandb_entity_name", type=str, default=None)
    args.add_argument("--wandb_project_name", type=str, default=None)
    args = args.parse_args()

    if args.gpu is not None:
        torch.cuda.set_device("cuda:{}".format(args.gpu))
        print("Using gpu: {}".format(args.gpu))

    # automatic naming
    curr_time = time.strftime("%m%d_%H%M%S", time.localtime())
    random_tag = generate_random_string()
    model_id = f"{curr_time}_{random_tag}"

    if args.use_wandb:
        wandb_config = WandBLogger.get_default_config()
        wandb_config.entity = args.wandb_entity_name
        wandb_config.project = args.wandb_project_name
        if args.wandb_exp_name is not None:
            wandb_config.exp_name = args.wandb_exp_name + "_" + model_id
        else:
            wandb_config.exp_name = model_id
        wandb_logger = WandBLogger(
            config=wandb_config, variant=vars(args), prefix="logging"
        )
    else:
        wandb_logger = None

    agent = Agent(
        dropout={
            "eef": args.dropout_rate,
            "hand_pos": args.dropout_rate,
            "img": args.img_dropout_rate,
            "pos": args.dropout_rate,
            "touch": args.dropout_rate,
        },
        output_sizes={
            "eef": 64,
            "hand_pos": 64,
            "img": args.image_output_size,
            "pos": 128,
            "touch": 64,
        },
        representation_type=args.representation_type.split("-"),
        identity_encoder=args.identity_encoder,
        camera_indices=list(map(int, args.camera_indices)),
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        action_horizon=args.action_horizon,
        without_sampling=args.without_sampling,
        predict_eef_delta=args.predict_eef_delta,
        predict_pos_delta=args.predict_pos_delta,
        clip_far=args.clip_far,
        color_jitter=args.color_jitter,
        num_diffusion_iters=args.num_diffusion_iters,
        load_img=args.load_img,
        num_workers=args.num_workers,
        weight_decay=args.weight_decay,
        use_ddim=args.use_ddim,
        binarize_touch=args.binarize_touch,
        policy_dropout_rate=args.policy_dropout_rate,
        state_noise=args.state_noise,
        img_gaussian_noise=args.img_gaussian_noise,
        img_masking_prob=args.img_masking_prob,
        img_patch_size=args.img_patch_size,
        compile_train=args.compile_train,
    )
    if args.load_path is not None:
        agent.load(args.load_path)

    use_depth = "depth" in args.representation_type.split("-")

    if not args.eval:
        if args.model_save_path is None:
            data_name = args.data_name.replace("/", "-")
            model_path = os.path.join(args.base_path, "ckpts", data_name)
        else:
            model_path = args.model_save_path

        model_path_suffix = f"{model_id}"

        if args.add_model_save_path_suffix:
            args_ = [
                ("camera", args.camera_indices),
                ("identity", args.identity_encoder),
                (
                    "repr",
                    "".join(
                        [(x[0]).upper() for x in args.representation_type.split("-")]
                    ),
                ),
                ("oh", args.obs_horizon),
                ("ah", args.action_horizon),
                ("ph", args.pred_horizon),
                ("prefix", args.prefix),
                ("do", args.dropout_rate),
                ("imgos", args.image_output_size),
                ("wd", args.weight_decay),
                ("use_ddim", args.use_ddim),
                ("binarize_touch", args.binarize_touch),
            ]
            args_str = "-".join([f"{k}={v}" for k, v in args_])
            if args.without_sampling:
                args_str += "-ws"
            if args.predict_pos_delta:
                args_str += "-posdelta"
            if args.predict_eef_delta:
                args_str += "-eefdelta"
            model_path_suffix += "-" + args_str
        model_path = os.path.join(model_path, model_path_suffix)

        print(f"Saving to model path {model_path}")

        if not os.path.exists(model_path):
            os.makedirs(model_path, exist_ok=True)

        save_args(args, model_path)

        if agent.stats is not None:
            agent.save_stats(model_path)

        if args.data_path is not None:
            data_path = args.data_path
        else:
            if args.data_prefix is not None:
                data_path = os.path.join(
                    args.base_path, args.data_prefix, args.data_name
                )
            else:
                data_path = os.path.join(args.base_path, args.data_name)

        if args.use_train_test_split:
            train_path = data_path + "_train" + args.train_suffix
            test_path = data_path + "_test"
        else:
            train_path = test_path = None
        print(f"using data path {data_path}")
        if args.use_memmap_cache:
            if args.memmap_loader_path is not None:
                memmap_loader_path = args.memmap_loader_path
            else:
                memmap_base_path = train_path if train_path is not None else data_path
                memmap_loader_path = os.path.join(
                    memmap_base_path, f"{args.camera_indices}-{use_depth}-mem.dat"
                )
        else:
            memmap_loader_path = ""

        print("using memmap loader path:", memmap_loader_path)
        agent.train(
            data_path,
            batch_size=args.batch_size,
            epochs=args.epochs,
            traj_type=args.traj_type,
            prefix=args.prefix,
            save_path=model_path,
            save_freq=args.save_freq,
            eval_freq=args.eval_freq,
            wandb_logger=wandb_logger,
            train_path=train_path,
            test_path=test_path,
            memmap_loader_path=memmap_loader_path,
        )

    else:
        agent.eval(args.eval_path, save_path=args.save_path)
