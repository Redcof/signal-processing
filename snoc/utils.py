import io
import pathlib
from PIL import Image


def save_fig(fig, save_path, bw=False):
    bw = False
    if bw:
        buf = io.BytesIO()
        fig.savefig(buf, dpi=400, bbox_inches="tight")
        buf.seek(0)
        bw_image = Image.open(buf).convert('L')
        bw_image.save(save_path)
        buf.close()
    else:
        fig.savefig(save_path, dpi=400, bbox_inches="tight")

def copy_figure(src_path, dest_name):
    dest_path = pathlib.Path(__file__).parent / "figures" / dest_name
    with Image.open(src_path) as img:
        img.save(dest_path)
        print(f"{src_path} -> {dest_path}")