import numpy as np
from PIL import Image

def read_image_L(location):
    img = Image.open(location)
    assert(img.mode == 'L')
    a = np.array(img)
    assert(a.shape[0] > 0 and a.shape[1] > 0)
    return a / 256.

def write_image_L(location, a):
    b = a * 256.
    img = Image.fromarray(np.uint8(b))
    img.save(location)
