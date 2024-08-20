import imageio
import numpy as np


def create_video(image_list, output_path, fps=25):
    writer = imageio.get_writer(output_path, fps=fps)
    for pil_img in image_list:
        img = np.array(pil_img, dtype=np.uint8)
        writer.append_data(img)
    writer.close()
