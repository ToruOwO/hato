import argparse
import os
import pickle

from agents.dp_agent import BimanualDPAgent


def eval_ckpts(ckpt_paths, eval_dir, save_path):
    mse_dict = {}

    if os.path.exists(save_path):
        with open(save_path, "rb") as f:
            mse_dict = pickle.load(f)
        print(f"Loaded previous MSE dict from {save_path}")

    eval_loader = None
    last_arg = None

    for ckpt_path in ckpt_paths:
        ckpt_name = os.path.basename(os.path.dirname(ckpt_path))
        ckpt_num = os.path.basename(ckpt_path)
        agent = BimanualDPAgent(ckpt_path)
        if eval_loader is None:
            eval_loader = agent.dp.get_eval_loader(eval_dir)
        elif (
            agent.dp_args["representation_type"] != last_arg["representation_type"]
            or agent.dp_args["camera_indices"] != last_arg["camera_indices"]
        ):
            eval_loader = agent.dp.get_eval_loader(eval_dir)
        last_arg = agent.dp_args
        mse, action_mse = agent.dp.eval_dir(eval_loader)
        if mse_dict.get(ckpt_name) is None:
            mse_dict[ckpt_name] = {}
        mse_dict[ckpt_name]["config"] = agent.dp_args

        mse_dict[ckpt_name][ckpt_num] = {}
        mse_dict[ckpt_name][ckpt_num]["mse"] = mse
        mse_dict[ckpt_name][ckpt_num]["action_mse"] = action_mse

        print(f"MSE for {ckpt_name}: {mse}")

    with open(save_path, "wb") as f:
        pickle.dump(mse_dict, f)
    print(f"Saved MSE dict to {save_path}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument(
        "--ckpt_path",
        nargs="+",
        type=str,
        default=[
            "model_epoch_100.ckpt",
            "model_epoch_200.ckpt",
            "model_epoch_300.ckpt",
        ],
    )
    args.add_argument(
        "--eval_dir",
        type=str,
        default="/split_data/data_pour_train_10",
    )
    args.add_argument("--save_path", type=str, default=None)

    args = args.parse_args()

    if args.save_path is None:
        data_name = os.path.basename(args.eval_dir)
        args.save_path = "./eval_results/eval_{}.pkl".format(data_name)
    eval_ckpts(args.ckpt_path, args.eval_dir, args.save_path)
