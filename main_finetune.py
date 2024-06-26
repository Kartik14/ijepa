import argparse
import multiprocessing as mp
import pprint
import yaml

from src.finetune import main as app_main
from src.utils.distributed import init_distributed
from src.finetune import main as app_main

parser = argparse.ArgumentParser()
# parser.add_argument(
#     "--fname", type=str, help="name of config file to load", default="configs.yaml"
# )
parser.add_argument(
    "--devices",
    type=str,
    nargs="+",
    default=["cuda:0"],
    help="which devices to use on local machine",
)
parser.add_argument(
    "--config", type=str, default="config.txt", help="Path to the config file"
)


def process_main(rank, world_size, devices, fname):
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(devices[rank].split(":")[-1])

    import logging

    logging.basicConfig()
    logger = logging.getLogger()
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f"called-params {fname}")

    # -- load script params
    params = None
    with open(fname, "r") as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info("loaded params...")
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(params)

    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f"Running... (rank: {rank}/{world_size})")
    app_main(params)


if __name__ == "__main__":
    args = parser.parse_args()

    num_gpus = len(args.devices)
    mp.set_start_method("spawn")

    for rank in range(num_gpus):
        mp.Process(
            target=process_main, args=(rank, num_gpus, args.devices, args.config)
        ).start()
