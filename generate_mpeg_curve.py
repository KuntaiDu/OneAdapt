import argparse
import logging
import os
import subprocess
from pathlib import Path

from munch import Munch
from pdb import set_trace

from utils.results import read_results
from itertools import product

# gt_qp = 24
# qp_list = [24]
# qp_list = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]

# qp_list = [24]
# fr_list = [30]
# res_list = {720:"1280:720"}

qp_list = [24, 26, 30, 32, 34, 36, 40, 44, 50]
fr_list = [30, 10, 5, 3]
res_list = {240:"352:240", 360:"480:360", 480:"858:480", 720:"1280:720"}


gt = 'qp_24_fr_30_res_720'


def main(args):

    logger = logging.getLogger("mpeg_curve")

    for video_name in args.inputs:
        video_name = Path(video_name)
        
        # generate mpeg curve
        for qp, fr, res in product(qp_list, fr_list, sorted(list(res_list.keys()))[::-1]):
            
            input_name = f"{video_name}.mp4"
            output_name = f"{video_name}_qp_{qp}_fr_{fr}_res_{res}.mp4"
            print(f"Generate video for {output_name}")
            # encode_with_qp(input_name, output_name, qp, args)

            if args.force or not os.path.exists(output_name):

                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        input_name,
                        # "-start_number",
                        # "0",
                        "-qp",
                        f"{qp}",
                        "-s",
                        f"{res_list[res]}",
                        "-filter:v",
                        f"fps={fr}",
                        output_name,
                    ]
                )
                    
                subprocess.run(
                    [
                        "python",
                        "inference.py",
                        "-i",
                        output_name,
                        "--app",
                        args.app,
                        "--visualize_step_size",
                        '10',
                        '--confidence_threshold',
                        '0.8',
                        '--gt_confidence_threshold',
                        '0.8'
                    ]
                )

            subprocess.run(
                [
                    "python",
                    "examine.py",
                    "-i",
                    output_name,
                    "-g",
                    f"{video_name}_{gt}.mp4",
                    "--app",
                    args.app,
                    "--stats",
                    args.stats,
                    '--confidence_threshold',
                    '0.8',
                    '--gt_confidence_threshold',
                    '0.8'
                ]
            )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--inputs",
        nargs="+",
        help="The video file names. The largest video file will be the ground truth.",
        # default=['youtube_driving/chicago/chicago_%d/video' % i for i in range(20)],
        default = ['youtube/la_driving/%d/video' % i for i in range(30, 60)]
    )
    parser.add_argument(
        "-f",
        "--force",
        type=bool,
        help="Force the program to regenerate all the outputs or not.",
        default=False,
    )
    parser.add_argument(
        '--app',
        type=str,
        default='COCO-Detection/faster_rcnn_R_101_FPN_3x.yaml'
    )
    parser.add_argument(
        '--stats',
        type=str,
        default='stats_la_driving'
    )

    

    args = parser.parse_args()
    main(args)
