import cv2
import numpy as np


def resize_image(image, width=None, height=None):
    """Resize an image to given width/height while keeping aspect ratio."""
    h, w = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        scale = height / float(h)
        dim = (int(w * scale), height)
    else:
        scale = width / float(w)
        dim = (width, int(h * scale))
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)


def concat_images(images, axis=1):
    """Concatenate a list of images horizontally (axis=1) or vertically (axis=0)."""
    return np.concatenate(images, axis=axis)
