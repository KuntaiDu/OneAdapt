
import av
from pathlib import Path
from config import settings
from PIL import Image
import numpy as np

__all__ = ['VideoVisualizer']

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
