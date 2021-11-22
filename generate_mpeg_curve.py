import argparse
import logging
import os
import subprocess
from pathlib import Path

from munch import Munch
from pdb import set_trace
from utils.results import read_results
from itertools import product
os.environ['MKL_THREADING_LAYER'] = 'GNU'
# gt_qp = 24
# qp_list = [24]
# qp_list = [24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]

# qp_list = [24]
# fr_list = [30]
# res_list = {720:"1280:720"}

res_list = {360: "512:360", 480: "768:480", 720: "1280:720"}
qp_list = [24, 27, 33, 42 ]
fr_list = [10, 7, 5, 3]
# res_list = {360: "512:360",}

# qp_list = [24]
# fr_list = [10, 7, 5, 3]
gt = 'qp_24_fr_30_res_720'


def main(args):

    logger = logging.getLogger("mpeg_curve")
    print(args.app)
    for video_name in args.inputs:
        print(video_name)
        second = (video_name.split("part")[1])
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
                    "examine.py",
                    "-i",
                    output_name,
                    "-g",
                    f"{video_name}_{gt}.mp4",
                    "--app",
                    args.app,
                    "--stats",
                    args.stats,
                    "--qp",
                    str(qp),
                    "--fr",
                    str(fr),
                    "--res",
                    str(res_list[res]),
                    "--second",
                    second,
                    '--image_idx',
                    args.image_idx
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
        default = ['videos/dashcam/dashcam_8/part%d' % i for i in range(0, 30)]
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
        default='EfficientDet'
    )
    parser.add_argument(
        '--stats',
        type=str,
        default='stats_la_driving'
    )
    parser.add_argument(
        '--image_idx',
        type=str,
        default='0'
    )

    

    args = parser.parse_args()
    main(args)
