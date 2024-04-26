import concurrent.futures
import os
import pickle

import cv2
import natsort
import numpy as np


def from_pickle(path, load_img = True, num_cam = 3):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if "base_rgb" not in data and load_img:
        rgb = []
        for i in range(num_cam):
            rgb_path = path.replace(".pkl", f"-{i}.png")
            if os.path.exists(rgb_path):
                rgb.append(cv2.imread(rgb_path))
        data["base_rgb"] = np.stack(rgb, axis=0)

    return data

# Get the trajectory data from the given directory
def iterate(path, workers=32, load_img=True, num_cam=3):
    dir = os.listdir(path)
    dir = [d for d in dir if d.endswith(".pkl")]
    dir = natsort.natsorted(dir)
    dirname = os.path.basename(path)
    root_path = "./mask_cache"
    data = []
    with concurrent.futures.ThreadPoolExecutor(max_workers = workers) as executor:
        futures = {executor.submit(from_pickle, os.path.join(path, file), load_img, num_cam): (i, file) for i, file in enumerate(dir)}
        for future in futures:
            try:
                i, file = futures[future]
                d = future.result()
                if not d["activated"]["l"] and not d["activated"]["r"]:
                    continue
                basedirfile = os.path.join(dirname, file)
                maskfile = os.path.join(root_path, basedirfile)
                if os.path.exists(maskfile):
                    d["mask"] = from_pickle(maskfile)
                d["mask_path"] = maskfile
                d["file_path"] = os.path.join(path, file)
                data.append(d)
            except:
                print(f"Failed to load {file}")
                pass
    return data


def get_latest(path):
    dir = os.listdir(path)
    dir = natsort.natsorted(dir)
    return from_pickle(os.path.join(path, dir[-1]))

# Get all trajectory directories from the given path
def get_epi_dir(path, traj_type, prefix=None):
    dir = natsort.natsorted(os.listdir(path))
    if prefix is not None:
        prefixs = prefix.split("-")

    new_dir = []
    for d in dir:
        if os.path.isdir(os.path.join(path, d)):
            matched = False
            if prefix is None:
                matched = True
            else:
                for prefix in prefixs:
                    if d.startswith(prefix):
                        matched = True
            if matched:
                new_dir.append(d)

    print("All Directories")
    print(new_dir)
    print("==========")
    dir = new_dir
    if traj_type == "plain":
        dir = [
            d
            for d in dir
            if not d.endswith("failed")
            and not d.endswith("ood")
            and not d.endswith("ikbad")
            and not d.endswith("heated")
            and not d.endswith("stop")
            and not d.endswith("hard")
        ]
    elif traj_type == "all":
        dir = dir
    else:
        raise NotImplementedError
    dir_list = [
        os.path.join(path, d) for d in dir if os.path.isdir(os.path.join(path, d))
    ]
    return dir_list

