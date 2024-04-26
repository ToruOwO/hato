import argparse
import os
import random
import shutil

random.seed(0)


def create_eval(origin_path, target_path, num=20):
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    data_dir = os.listdir(origin_path)
    assert (
        len(data_dir) >= num
    ), "The number of data is less than the number of data to move"
    assert len(data_dir) > 100, "at least 100 traj"
    # sample without replacement
    eval_dir = random.sample(data_dir, num)
    print(eval_dir)
    for file in eval_dir:
        shutil.move(os.path.join(origin_path, file), target_path)
        print("Move {} to {}".format(file, target_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--origin_path", type=str, default=None, help="path to the original data"
    )
    parser.add_argument(
        "--target_path", type=str, default=None, help="path to the target data"
    )
    parser.add_argument("--num", type=int, default=20, help="number of data to move")
    args = parser.parse_args()
    create_eval(args.origin_path, args.target_path, args.num)
