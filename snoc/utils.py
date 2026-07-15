import io
import pathlib
import shutil
from PIL import Image


def save_fig(fig, save_path, bw=False):
    dpi = 500
    bw = False
    if bw:
        # Convert the figure to black and white before saving
        buf = io.BytesIO()
        fig.savefig(buf, dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        bw_image = Image.open(buf).convert('L')
        bw_image.save(save_path)
        buf.close()
    else:
        # Save the figure as is
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        # save the figure as TIFF format
        tiff_path = pathlib.Path(save_path).with_suffix('.tiff')
        fig.savefig(tiff_path, dpi=dpi, bbox_inches="tight", format='tiff')

def copy_figure(src_path, dest_name):
    dest_path = (pathlib.Path(__file__).parent / "figures" / dest_name)
    shutil.copy(pathlib.Path(src_path).with_suffix('.tiff'), dest_path.with_suffix('.tiff'))
    shutil.copy(pathlib.Path(src_path).with_suffix('.png'), dest_path.with_suffix('.png'))
    print(f"Copied : {src_path} -> {dest_path}")
