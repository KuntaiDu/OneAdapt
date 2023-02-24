
import av
from pathlib import Path
from config import settings
from PIL import Image
import numpy as np
import logging
from tqdm import tqdm
import ImageDraw
import ImageFont
from detectron2.structures.instances import Instances
import torchvision.transforms as T

text = ""
visualize_folder = None
results_vis = None
errors_vis = None



logger = logging.getLogger("visualize")

class VideoVisualizer:

    def __init__(self, output_path):
        """Init the video visualizer

        Args:
            output_path (str): the output path for the visualized video
        """        

        # remove file if exists
        if Path(output_path).exists():
            Path(output_path).unlink()

        # open the video writer
        self.container = av.open(output_path, mode="w")
        self.stream = self.container.add_stream("h264", 10) # make it a bit slower
        height, width = settings.input_shape
        self.stream.width = width
        self.stream.height = height
        # self.stream.pix_fmt = 'yuv420p'
        self.stream.options = {'crf': '20'}

    def add_frame(self, image):
        """add a PIL frame to the visualizer

        Args:
            image (PIL image): the image to add
        """

        image = np.array(image)
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        packet = self.stream.encode(frame)
        self.container.mux(packet)
    

    def __del__(self):
        packet = self.stream.encode(None)
        self.container.mux(packet)

        self.container.close()



def init(command_line_args):
    global visualize_folder, results_vis, errors_vis
    visualize_folder = Path('debug/' + command_line_args.input).parent
    visualize_folder.mkdir(exist_ok=True, parents=True)
    results_vis = VideoVisualizer(f"{visualize_folder}/{command_line_args.approach}_results.mp4")
    errors_vis = VideoVisualizer(f"{visualize_folder}/{command_line_args.approach}_errors.mp4")




def visualize(stat, app, video, my_results, gt_results):

    global text
    
    # visualize
    my_video_config = stat["my_video_config"]
    logger.info("Actual compute: %d" % my_video_config["compute"])
    text += ("Comp: %d\n" "Acc : %.3f\n" "Bw  : %.3f\n") % (
        my_video_config["compute"],
        stat["acc"],
        stat["norm_bw"],
    )
    
    if settings.backprop.visualize:
        for idx, frame in enumerate(tqdm(video, desc="visualize", unit="frame")):

            image = T.ToPILImage()(frame.clamp(0, 1))

            draw = ImageDraw.Draw(image)
            font = ImageFont.truetype("Pillow/Tests/fonts/FreeMono.ttf", 24)
            # manually bold the text :-P
            draw.multiline_text((10, 10), text, fill=(255, 0, 0), font=font)
            draw.multiline_text((11, 10), text, fill=(255, 0, 0), font=font)
            draw.multiline_text((12, 10), text, fill=(255, 0, 0), font=font)

            my_result = app.filter_result(my_results[idx])
            gt_result = app.filter_result(gt_results[idx], gt=True)

            (
                gt_ind,
                my_ind,
                gt_filtered,
                my_filtered,
            ) = app.get_undetected_ground_truth_index(my_result, gt_result)

            image_error = app.visualize(
                image,
                {
                    "instances": Instances.cat(
                        [gt_filtered[gt_ind], my_filtered[my_ind]]
                    )
                },
            )
            image_inference = app.visualize(image, my_result)
            errors_vis.add_frame(image_error)
            results_vis.add_frame(image_inference)

        logger.info("Visualize text:\n%s", text)