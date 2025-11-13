"""
Teleoperation Example

Run this example on UPC to which the master arm and hands are connected
"""
import time
import logging
import argparse
import signal
from typing import *
import threading


from shared import Shared
from data_collector import DataCollector
from rby1_controller import RBY1Controller

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

def main(address, model, power, servo, control_mode, out_dir, target_episode_num, task):

    # ===== SETUP SHARED =====
    shared = Shared()
    # ===== SETUP RBY1CONTROLLER =====

    rby1_controller = RBY1Controller(
        address=address,
        teleop=True,
        control_freq=10,
        shared=shared
    )
    rby1_controller.start_control()

    # ===== SETUP DATA COLLECTOR =====
    data_collector = DataCollector(
        out_dir, 
        target_episode_num, 
        task,
        shared
    )
    data_collector.start()
    data_collector.reset()

    logging.info("Ready to teleop!")
    
    # ===== SETUP SIGNAL =====
    def handler(signum, frame):

        data_collector.stop()
        exit(1)

    signal.signal(signal.SIGINT, handler)

    while True:
        time.sleep(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="teleop data collect")
    parser.add_argument("--address", type=str, default="192.168.30.1:50051", help="Robot address")
    parser.add_argument(
        "--model", type=str, default="a", help="Robot Model Name (default: 'a')"
    )
    parser.add_argument(
        "--power",
        type=str,
        default=".*",
        help="Regex pattern for power device names (default: '.*')",
    )
    parser.add_argument(
        "--servo",
        type=str,
        default="torso_.*|right_arm_.*|left_arm_.*",
        help="Regex pattern for servo names (default: 'torso_.*|right_arm_.*|left_arm_.*')",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="position",
        choices=["position", "impedance"],
        help="Control mode to use: 'position' or 'impedance' (default: 'position')",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        default="/home/nvidia/ws/angad/rby1-new-data-collection/data",
        help="Root path of saved raw data.",
    )

    parser.add_argument(
        "--target_episode_num",
        type=int,
        default=50,
        help="number of episodes to collect",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="put the sushi into drawer",
        help="Language description of task",
    )
    args = parser.parse_args()

    main(args.address, args.model, args.power, args.servo, args.mode, args.out_dir, args.target_episode_num, args.task)
