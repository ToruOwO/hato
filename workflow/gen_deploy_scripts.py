import os
from pathlib import Path


def find_ckpts(ckpt_dirs):
    """
    Find all checkpoint paths given a parent dir or a list of checkpoint directory names.
    """
    ckpt_dict = {}
    for ckpt_dir in ckpt_dirs:
        for root, _, files in os.walk(ckpt_dir):
            for file in files:
                if file.endswith(".ckpt"):
                    run = root.split("/")[-1]
                    ckpt_idx = int(file.split(".")[-2].split("_")[-1])
                    if (run not in ckpt_dict) or (
                        run in ckpt_dict and ckpt_dict[run]["idx"] < ckpt_idx
                    ):
                        # update the ckpt path
                        if os.path.exists(os.path.join(root, "args_log.txt")):
                            args_path = os.path.join(root, "args_log.txt")
                        else:
                            args_path = None
                        ckpt_dict[run] = {
                            "root": root,
                            "ckpt": os.path.join(root, file),
                            "args": args_path,
                            "idx": ckpt_idx,
                        }
    return ckpt_dict


def gen_deploy_scripts(
    ckpt_dirs, agent_type="dp", test_data_path="", conda_env_name="hato"
):
    ckpt_dict = find_ckpts(ckpt_dirs)

    for run, ckpt_args in ckpt_dict.items():
        ckpt = ckpt_args["ckpt"]

        # generate env script
        env_script = f"python run_env.py --agent {agent_type} --no-show-camera-view --save_data --dp_ckpt_path {ckpt}"

        if test_data_path is None or test_data_path == "":
            print("No test traj for this ckpt.")
        openloop_script = f"python run_openloop.py --agent {agent_type} --no-show-camera-view --save_data --dp_ckpt_path {ckpt} --traj_path {test_data_path}"

        # generate node script
        node_script = (
            'python launch_nodes.py --hand_type ability --faster --cam_names "435"'
        )
        env_script += " --hz 10"
        openloop_script += " --hz 10"

        if "pour" in ckpt or "steak" in ckpt:
            node_script += " --ability_gripper_grip_range 75"

        test_script = (
            f"python test_dp_agent.py --ckpt_path {ckpt} --data_dir {test_data_path}"
        )
        env_jit_script = (
            env_script
            + " --use_jit_agent --inference_agent_port=1325 --num_diffusion_iters_compile=50 --jit_compile"
        )

        inference_script = f"'conda activate {conda_env_name} && python launch_inference_nodes.py --dp_ckpt_path {ckpt} --port=1325'"
        # save the scripts at the ckpt directory
        save_dir = ckpt_args["root"]
        with open(os.path.join(save_dir, f"{run}_node.sh"), "w") as f:
            f.write(node_script)
        with open(os.path.join(save_dir, f"{run}_env.sh"), "w") as f:
            f.write(env_script)
        with open(os.path.join(save_dir, f"{run}_env_jit.sh"), "w") as f:
            f.write(env_jit_script)
        with open(os.path.join(save_dir, f"{run}_openloop.sh"), "w") as f:
            f.write(openloop_script)
        with open(os.path.join(save_dir, f"{run}_test.sh"), "w") as f:
            f.write(test_script)
        with open(os.path.join(save_dir, f"{run}_inference.sh"), "w") as f:
            f.write(inference_script)

    return ckpt_dict


def run_test_scripts(ckpt_dirs, filter="", run_all=True):
    for checkpoint_dir in ckpt_dirs:
        for root, _, files in os.walk(checkpoint_dir):
            for file in files:
                if file.endswith("_test.sh") and filter in root:
                    print("Running test script: ", file)
                    if run_all:
                        os.system(f"bash {os.path.join(root, file)}")
                    else:
                        # run the test script if enter yes
                        answer = input(f"\nRun {os.path.join(root, file)}? (y/n)")
                        if answer == "y":
                            os.system(f"bash {os.path.join(root, file)}")
                        else:
                            print("Skip.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--ckpt_dirs",
        nargs="+",
        default=[
            "/shared/ckpts/data_banana",
        ],
    )
    parser.add_argument("-m", "--mode", type=str, default="gen")
    parser.add_argument("-f", "--filter", type=str, default="")
    args = parser.parse_args()

    if args.mode == "gen":
        checkpoint_paths = gen_deploy_scripts(args.ckpt_dirs)
        print(checkpoint_paths)
        print(len(checkpoint_paths))
        print("Done.")
    elif args.mode == "test":
        run_test_scripts(args.ckpt_dirs, filter=args.filter)
    elif args.mode == "all":
        checkpoint_paths = gen_deploy_scripts(args.ckpt_dirs)
        print(checkpoint_paths)
        print(len(checkpoint_paths))
        run_test_scripts(args.ckpt_dirs, filter=args.filter)
