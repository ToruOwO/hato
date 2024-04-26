from dataclasses import dataclass

import tyro

from agents.dp_agent_zmq import BimanualDPAgentServer


def boolean_string(s):
    if s not in {"False", "True"}:
        raise ValueError("Not a valid boolean string")
    return s == "True"


@dataclass
class Args:
    port: str = "4321"
    dp_ckpt_path: str = "best.ckpt"


def launch_server(args: Args):
    server = BimanualDPAgentServer(ckpt_path=args.dp_ckpt_path, port=args.port)
    print(f"Starting inference server on {args.port}")

    print("Compiling inference")
    server.compile_inference()
    print("Done. Inference available.")
    server.serve()


def main(args):
    launch_server(args)


if __name__ == "__main__":
    main(tyro.cli(Args))
