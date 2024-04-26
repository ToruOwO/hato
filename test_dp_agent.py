import argparse
import os
import pickle
import time

import numpy as np
import torch

from agents.dp_agent import BimanualDPAgent
from learning.dp.data_processing import iterate

torch.cuda.set_device(0)


def main(args):
    hand_uppers = np.array([110.0, 110.0, 110.0, 110.0, 90.0, 120.0])
    hand_lowers = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0])

    dp_args = BimanualDPAgent.get_default_dp_args()
    data = iterate(args.data_dir)
    for num_diffusion_iters in args.num_diffusion_iters:
        dp_args["num_diffusion_iters"] = num_diffusion_iters
        dp_agent = BimanualDPAgent(ckpt_path=args.ckpt_path, dp_args=dp_args)

        controls = []
        pred_actions = []
        delta_action = []
        last_action = data[0]["control"]

        start = time.time()
        dp_agent.compile_inference(data[0], num_inference_iters=num_diffusion_iters)
        end = time.time()
        ctime = end - start
        print(f"compilation time: {ctime}")

        start = time.time()
        max_infer_time = 0
        for i, obs in enumerate(data):
            control = obs["control"]
            delta_action.append(control - last_action)
            last_action = control
            controls.append(control)
            if i != 0:
                obs["joint_positions"][list(range(6)) + list(range(12, 18))] = (
                    pred_actions[-1][list(range(6)) + list(range(12, 18))]
                )
                obs["joint_positions"][6:12] = (
                    pred_actions[-1][6:12] * (hand_uppers - hand_lowers) + hand_lowers
                )
                obs["joint_positions"][18:24] = (
                    pred_actions[-1][18:24] * (hand_uppers - hand_lowers) + hand_lowers
                )
            infer_start = time.time()
            pred_actions.append(dp_agent.act(obs))
            infer_time = time.time() - infer_start
            max_infer_time = max(max_infer_time, infer_time)

        print("num_diffusion_iters:", num_diffusion_iters)
        print("time:", time.time() - start)
        print("Hz:", len(data) / (time.time() - start))
        print("max_infer_time:", max_infer_time)
        print("lowest_freq:", 1 / max_infer_time)

        pred_actions = np.array(pred_actions)
        controls = np.array(controls)

        mse = np.mean(np.abs((pred_actions - controls)), axis=0)
        mean_delta_action = np.mean(np.abs(delta_action), axis=0)

        print_str = "\n".join(
            [
                "mse:",
                str(mse.tolist()),
                "\n",
                "mean_delta_action:",
                str(mean_delta_action.tolist()),
                "\nfinal_diff:",
                str((pred_actions[-1] - controls[-1]).tolist()),
            ]
        )
        print_str += "\n"
        print_str += (
            "mse: "
            + str(mse.mean())
            + " mean_delta_action: "
            + str(mean_delta_action.mean())
        )
        print(print_str)

        # save print_str as txt in ckpt dir
        ckpt_dir = os.path.dirname(args.ckpt_path)
        with open(
            os.path.join(ckpt_dir, f"eval_stats_{num_diffusion_iters}.txt"), "w"
        ) as f:
            f.write(print_str)

        traj_name = os.path.basename(args.data_dir)
        save_path = os.path.join(
            ckpt_dir, f"openloop_{traj_name}_{num_diffusion_iters}"
        )
        os.makedirs(save_path, exist_ok=True)
        for i in range(len(pred_actions)):
            with open(os.path.join(save_path, str(i) + ".pkl"), "wb") as f:
                pickle.dump(
                    {
                        "control": pred_actions[i],
                        "joint_positions": data[i]["joint_positions"],
                    },
                    f,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="best.ckpt",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/",
    )
    parser.add_argument(
        "--num_diffusion_iters", default=[5, 15, 25, 50], type=int, nargs="+"
    )
    args = parser.parse_args()
    main(args)
