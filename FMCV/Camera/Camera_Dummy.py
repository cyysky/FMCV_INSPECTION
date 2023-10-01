import numpy as np
def get_image(name=None):
    image_height = 224
    image_width = 224
    number_of_color_channels = 3
    color = (255,255,255)
    pixel_array = np.full((image_height, image_width, number_of_color_channels), color, dtype=np.uint8)
    frames = {"1":pixel_array}

    if name is not None:
       return frames[name]
    return frames