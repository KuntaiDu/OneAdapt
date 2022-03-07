import io
from pathlib import Path

import matplotlib.pyplot as plt
import PIL
import seaborn as sns
import torchvision.transforms as T






def visualize_heat_by_summarywriter(
    image, heat, tag, writer, fid, tile=True, alpha=0.5
):

    fig, ax = plt.subplots(1, 1, figsize=(11, 5), dpi=200)
    ax = sns.heatmap(
        heat.cpu().detach().numpy(),
        zorder=3,
        alpha=alpha,
        ax=ax,
        xticklabels=False,
        yticklabels=False,
    )
    ax.imshow(image, zorder=3, alpha=(1 - alpha))
    ax.tick_params(left=False, bottom=False)
    buf = io.BytesIO()
    fig.savefig(buf, bbox_inches="tight")
    buf.seek(0)
    result = PIL.Image.open(buf)
    result = result.convert('RGB')
    result.save(tag+'.jpg')
    writer.add_image(tag, T.ToTensor()(result), fid)
    plt.close(fig)
