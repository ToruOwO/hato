import os
import pickle

import numpy as np
import torch


def create_sample_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append(
                [buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx]
            )
    indices = np.array(indices)
    return indices


def sample_sequence(
    train_data,
    sequence_length,
    buffer_start_idx,
    buffer_end_idx,
    sample_start_idx,
    sample_end_idx,
):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype
            )
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


# normalize data
def get_data_stats(data):
    data = data.reshape(-1, data.shape[-1])
    stats = {"min": np.min(data, axis=0), "max": np.max(data, axis=0)}
    if np.any(stats["max"] > 1e5) or np.any(stats["min"] < -1e5):
        raise ValueError("data out of range")
    return stats


def normalize_data(data, stats):
    # nomalize to [0,1]
    ndata = (data - stats["min"]) / (stats["max"] - stats["min"] + 1e-8)
    # normalize to [-1, 1]
    ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats["max"] - stats["min"] + 1e-8) + stats["min"]
    return data


class MemmapLoader:
    def __init__(self, path):
        with open(os.path.join(path, "metadata.pkl"), "rb") as f:
            meta_data = pickle.load(f)

        print("Meta Data:", meta_data)
        self.fps = {}

        self.length = None
        for key, (shape, dtype) in meta_data.items():
            self.fps[key] = np.memmap(
                os.path.join(path, key + ".dat"), dtype=dtype, shape=shape, mode="r"
            )
            if self.length is None:
                self.length = shape[0]
            else:
                assert self.length == shape[0]

    def __getitem__(self, index):
        rets = {}
        for key in self.fps.keys():
            value = self.fps[key]
            value = value[index]
            value_cp = np.empty(dtype=value.dtype, shape=value.shape)
            value_cp[:] = value
            rets[key] = value_cp
        return rets

    def __length__(self):
        return self.length


# dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: dict,
        representation_type: list,
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
        stats: dict = None,
        transform=None,
        get_img=None,
        load_img: bool = False,
        hand_grip_range: int = 110,
        binarize_touch: bool = False,
        state_noise: float = 0.0,
    ):
        self.state_noise = state_noise
        self.memmap_loader = None
        if "memmap_loader_path" in data.keys():
            self.memmap_loader = MemmapLoader(data["memmap_loader_path"])
        self.representation_type = representation_type
        self.transform = transform
        self.get_img = get_img
        self.load_img = load_img

        print("Representation type: ", representation_type)
        except_img_representation_type = representation_type.copy()

        if "img" in representation_type:
            train_image_data = data["data"]["img"][:]
            except_img_representation_type.remove("img")

        train_data = {
            rt: data["data"][rt][:, :] for rt in except_img_representation_type
        }
        train_data["action"] = data["data"]["action"][:]
        episode_ends = data["meta"]["episode_ends"][:]

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
        )

        normalized_train_data = dict()

        # compute statistics and normalized data to [-1,1]
        if stats is None:
            stats = dict()
            for key, data in train_data.items():
                stats[key] = get_data_stats(data)

        # overwrite the min max of hand info
        left_hand_indices = np.array([6, 7, 8, 9, 10, 11])
        right_hand_indices = np.array([18, 19, 20, 21, 22, 23])

        # hand joint_position range
        hand_upper_ranges = np.array(
            [hand_grip_range] * 4 + [90, 120], dtype=np.float32
        )
        hand_lower_ranges = np.array([5, 5, 5, 5, 5, 5], dtype=np.float32)

        if "pos" in representation_type:
            stats["pos"]["min"][left_hand_indices] = hand_lower_ranges
            stats["pos"]["min"][right_hand_indices] = hand_lower_ranges
            stats["pos"]["max"][left_hand_indices] = hand_upper_ranges
            stats["pos"]["max"][right_hand_indices] = hand_upper_ranges
        elif "hand_pos" in representation_type:
            stats["hand_pos"]["min"][range(6)] = hand_lower_ranges
            stats["hand_pos"]["min"][range(6, 12)] = hand_lower_ranges
            stats["hand_pos"]["max"][range(6)] = hand_upper_ranges
            stats["hand_pos"]["max"][range(6, 12)] = hand_upper_ranges

        # hand action is normalized to [0,1]
        stats["action"]["min"][left_hand_indices] = 0.0
        stats["action"]["max"][left_hand_indices] = 1.0
        stats["action"]["min"][right_hand_indices] = 0.0
        stats["action"]["max"][right_hand_indices] = 1.0

        for key, data in train_data.items():
            if key == "touch" and binarize_touch:
                normalized_train_data[key] = (
                    data  # don't normalize if binarize touch in model
                )
            else:
                normalized_train_data[key] = normalize_data(data, stats[key])

        # images are already normalized
        if "img" in representation_type:
            normalized_train_data["img"] = train_image_data

        self.indices = indices
        self.stats = stats
        self.normalized_train_data = normalized_train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.binarize_touch = binarize_touch

    def __len__(self):
        return len(self.indices)

    def read_img(self, image_pathes, idx):
        if self.memmap_loader is not None:
            # using memmap loader
            indices = range(idx, idx + self.obs_horizon)
            data = self.memmap_loader[indices]
            data = [
                {"base_rgb": data["base_rgb"][i], "base_depth": data["base_depth"][i]}
                for i in range(data["base_rgb"].shape[0])
            ]
        else:
            # not using memmap loader and loading images while training
            data = [pickle.load(open(image_path, "rb")) for image_path in image_pathes]
        imgs = self.get_img(data)
        return imgs

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        (
            buffer_start_idx,
            buffer_end_idx,
            sample_start_idx,
            sample_end_idx,
        ) = self.indices[idx]

        # get nomralized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )

        for k in self.representation_type:
            # discard unused observations
            nsample[k] = nsample[k][: self.obs_horizon]
            if k == "img":
                if not self.load_img:
                    nsample["img"] = self.read_img(nsample["img"], idx)
                else:
                    nsample["img"] = torch.tensor(
                        nsample["img"].astype(np.float32), dtype=torch.float32
                    )
                nsample_shape = nsample["img"].shape
                # transform the img
                nsample["img"] = nsample["img"].reshape(
                    nsample_shape[0] * nsample_shape[1], *nsample_shape[2:]
                )                                                                           # (Batch * num_cam, Channel, Height, Width)
                nsample["img"] = self.transform(nsample["img"])
                nsample["img"] = nsample["img"].reshape(nsample_shape[:3] + (216, 288))     # (Batch, num_cam, Channel, Height, Width)

            else:
                nsample[k] = torch.tensor(nsample[k], dtype=torch.float32)
                if self.state_noise > 0.0:
                    # add noise to the state
                    nsample[k] = nsample[k] + torch.randn_like(nsample[k]) * self.state_noise
        nsample["action"] = torch.tensor(nsample["action"], dtype=torch.float32)
        return nsample
