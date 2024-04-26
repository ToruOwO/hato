import argparse
import glob
import os
import random


def split_symlink_train_test_merge_dataset(root_list, t_root, num_traj_per_task):
    target_train_root = t_root + f"_train"
    target_test_root = t_root + f"_test"
    print(
        f"split data points from",
        root_list,
        f"to {target_train_root}: train and {target_test_root}: test",
    )
    if os.path.exists(target_train_root):
        return
    assert not os.path.exists(target_test_root)

    os.makedirs(target_train_root)
    os.makedirs(target_test_root)

    all_paths = []
    for root in sorted(root_list):
        for path in sorted(os.listdir(root))[:num_traj_per_task]:
            if not os.path.isfile(os.path.join(root, path)):
                d = path
                if (
                    d.endswith("failed")
                    or d.endswith("ood")
                    or d.endswith("ikbad")
                    or d.endswith("heated")
                    or d.endswith("stop")
                    or d.endswith("hard")
                ):
                    continue
                all_paths.append((root, path))

    train_paths = all_paths[:-1]
    test_paths = all_paths[-1:]

    for root, sl_path in train_paths:
        src_path = os.path.abspath(os.path.join(root, sl_path))
        tgt_path = os.path.abspath(os.path.join(target_train_root, sl_path))
        print("\rlinking", src_path, tgt_path, end="")
        os.symlink(src_path, tgt_path)

    for root, sl_path in test_paths:
        src_path = os.path.abspath(os.path.join(root, sl_path))
        tgt_path = os.path.abspath(os.path.join(target_test_root, sl_path))
        print("\rlinking", src_path, tgt_path, end="")
        os.symlink(src_path, tgt_path)

    print()
    print("Done!!")


def split_symlink_train_test_dataset(root, t_root):
    target_train_root = t_root + f"_train"
    target_test_root = t_root + f"_test"
    print(
        f"split data points from {root} to {target_train_root}: train and {target_test_root}: test"
    )

    if os.path.exists(target_train_root):
        return
    assert not os.path.exists(target_test_root)

    os.makedirs(target_train_root)
    os.makedirs(target_test_root)

    all_paths = []
    for path in sorted(os.listdir(root)):
        if not os.path.isfile(os.path.join(root, path)):
            d = path
            if (
                d.endswith("failed")
                or d.endswith("ood")
                or d.endswith("ikbad")
                or d.endswith("heated")
                or d.endswith("stop")
                or d.endswith("hard")
            ):
                continue
            all_paths.append(path)

    train_paths = all_paths[:-1]
    test_paths = all_paths[-1:]

    for sl_path in train_paths:
        src_path = os.path.abspath(os.path.join(root, sl_path))
        tgt_path = os.path.abspath(os.path.join(target_train_root, sl_path))
        print("\rlinking", src_path, tgt_path, end="")
        os.symlink(src_path, tgt_path)
    for sl_path in test_paths:
        src_path = os.path.abspath(os.path.join(root, sl_path))
        tgt_path = os.path.abspath(os.path.join(target_test_root, sl_path))
        print("\rlinking", src_path, tgt_path, end="")
        os.symlink(src_path, tgt_path)
    print()
    print("Done!!")


def split_symlink_dataset(root, num_trajs):
    target_root = root + f"_{num_trajs}"
    print(f"split {num_trajs} data points from {root} to {target_root}")

    if os.path.exists(target_root):
        return
    assert not os.path.exists(target_root)

    os.makedirs(target_root)

    all_paths = []
    for path in os.listdir(root):
        if not os.path.isfile(os.path.join(root, path)):
            all_paths.append(path)

    random.seed(0)
    random.shuffle(all_paths)

    sl_paths = all_paths[:num_trajs]

    assert len(all_paths) >= num_trajs
    for sl_path in sl_paths:
        src_path = os.path.abspath(os.path.join(root, sl_path))
        tgt_path = os.path.abspath(os.path.join(target_root, sl_path))
        print("\rlinking", src_path, tgt_path, end="")
        os.symlink(src_path, tgt_path)
    print()
    print("Done!!")


if __name__ == "__main__":
    arg = argparse.ArgumentParser()
    arg.add_argument("--base_path", type=str, default="/hato/")
    arg.add_argument("--output_path", type=str, default="/split_data")
    arg.add_argument(
        "--data_name",
        nargs="+",
        type=str,
        default=[
            "data_banana",
        ],
    )
    arg.add_argument("--num_trajs", nargs="+", type=int, default=[10, 25, 50, 75])
    arg.add_argument("--merge", action="store_true")
    arg.add_argument("--merge_name", type=str, default="data_banana_all")
    arg.add_argument("--num_traj_per_task", type=int, default=20)
    args = arg.parse_args()

    if not args.merge:
        for data_name in args.data_name:
            split_symlink_train_test_dataset(
                os.path.join(args.base_path, data_name),
                os.path.join(args.output_path, data_name),
            )
            for num_trajs in args.num_trajs:
                split_symlink_dataset(
                    os.path.join(args.output_path, data_name) + "_train", num_trajs
                )

    else:
        data_name_list = [
            os.path.join(args.base_path, data_name) for data_name in args.data_name
        ]
        split_symlink_train_test_merge_dataset(
            data_name_list,
            os.path.join(args.output_path, args.merge_name),
            args.num_traj_per_task,
        )
