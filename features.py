import numpy as np


def rgb_to_bv(image_rgb):
    """Convert an RGB image or pixel array to Brightness Value (BV).

    Accepts numpy arrays in HxWx3 or a single RGB tuple.
    Returns an array of the same HxW shape (or scalar for a single pixel).
    """
    arr = np.asarray(image_rgb, dtype=np.float32)
    if arr.ndim == 1 or (arr.ndim == 3 and arr.shape[2] == 3 and arr.shape[0] != 3):
        # HxWx3
        R = arr[..., 0]
        G = arr[..., 1]
        B = arr[..., 2]
    else:
        # single pixel as (R,G,B)
        R, G, B = arr[0], arr[1], arr[2]
    BV = 0.299 * R + 0.587 * G + 0.114 * B
    return BV


def luminance_from_bv(bv, camera_constant=0.0929):
    """Convert BV to luminance (lumens) given a camera constant.

    Uses: luminance = 2**BV / camera_constant
    """
    bv_arr = np.asarray(bv, dtype=np.float64)
    return np.power(2.0, bv_arr) / float(camera_constant)


if __name__ == "__main__":
    import numpy as _np
    px = _np.array([128, 128, 128])
    print("BV of mid-gray:", rgb_to_bv(px))
    print("Luminance:", luminance_from_bv(rgb_to_bv(px)))

import numpy as np


def rgb_to_bv(rgb_array):
    """Convert an RGB NumPy array (H,W,3) with 0-255 values to Brightness Value (BV).

    BV = 0.299*R + 0.587*G + 0.114*B
    Returns a float array same shape HxW.
    """
    arr = rgb_array.astype(np.float32)
    r = arr[..., 2]
    g = arr[..., 1]
    b = arr[..., 0]
    bv = 0.299 * r + 0.587 * g + 0.114 * b
    return bv


def luminance_from_bv(bv_array, camera_constant=0.0929):
    """Estimate luminance from BV and a camera constant.

    Example given: luminance = 2**BV / 0.0929 (for an iPhone X example)
    """
    # Use elementwise power; clip BV to avoid overflow
    bv_clipped = np.clip(bv_array, -100, 100)
    lum = np.power(2.0, bv_clipped) / float(camera_constant)
    return lum


def image_luminance(image_bgr, camera_constant=0.0929):
    """Convenience: compute BV per-pixel and aggregate luminance map.

    Returns (bv_map, lum_map)
    """
    bv = rgb_to_bv(image_bgr)
    lum = luminance_from_bv(bv, camera_constant=camera_constant)
    return bv, lum


if __name__ == "__main__":
    import cv2
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image")
    args = parser.parse_args()
    im = cv2.imread(args.image)
    bv, lum = image_luminance(im)
    print("BV stats:", bv.min(), bv.max(), bv.mean())
    print("Luminance stats:", lum.min(), lum.max(), lum.mean())
