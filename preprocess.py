import os
import cv2
import numpy as np
from skimage import morphology, filters, segmentation


def standardize_eye_image(image_path, target_size=(224, 224)):
    """
    Reads, crops, standardizes lighting, and resizes an eye image.
    Removes black space, applies CLAHE for lighting normalization.
    Returns the preprocessed image or None on error.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading {image_path}")
        return None
    
    # Convert to grayscale for boundary detection
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Enhance contrast to highlight eye structure against background
    clahe_gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe_gray.apply(gray)
    
    # Thresholding and Contours: separate bright eye from dark background
    _, thresh = cv2.threshold(enhanced_gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find largest contour (the eye)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        # Crop original image to bounding box
        img = img[y:y+h, x:x+w]
    
    # Resize to target architecture size
    img_resized = cv2.resize(img, target_size)
    
    # Normalize lighting: convert to LAB, apply CLAHE to L channel, convert back
    lab = cv2.cvtColor(img_resized, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    clahe_color = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe_color.apply(l)
    
    limg = cv2.merge((cl, a, b))
    final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    return final_img


def extract_frames(video_path, out_dir, target_fps=60):
    """Extract frames from a video at target fps rate."""
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(src_fps / target_fps)))
    idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % step == 0:
            out_path = os.path.join(out_dir, f"frame_{saved:06d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
        idx += 1
    cap.release()
    return saved


def circular_roi(image, center=None, radius=None):
    h, w = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    if radius is None:
        radius = min(center[0], center[1], w - center[0], h - center[1])
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    if image.ndim == 3:
        masked = cv2.bitwise_and(image, image, mask=mask)
    else:
        masked = cv2.bitwise_and(image, image, mask=mask)
    return masked, mask


def denoise_median(image, ksize=5):
    return cv2.medianBlur(image, ksize)


def watershed_segmentation(image_rgb):
    # Convert to gray and remove small artifacts
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2GRAY)
    # Use morphological opening to remove small bright spots
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=2)
    # Compute sure background and sure foreground
    ret, thresh = cv2.threshold(opened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # Sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret2, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    # Marker labeling
    ret3, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    markers = cv2.watershed(image_rgb, markers)
    # markers == -1 are boundaries
    seg = np.zeros_like(image_rgb)
    seg[markers > 1] = image_rgb[markers > 1]
    return seg, markers


def enhance_contrast(image_bgr):
    # Histogram equalization on the luminance channel (Y in YCrCb)
    ycrcb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge([y_eq, cr, cb])
    return cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)


def preprocess_image(image_bgr, do_watershed=True, median_ksize=5):
    roi_img, mask = circular_roi(image_bgr)
    den = denoise_median(roi_img, ksize=median_ksize)
    if do_watershed:
        seg, markers = watershed_segmentation(den)
        # Fill with segmented region where available, else fallback to den
        result = np.where(seg.sum(axis=2, keepdims=True) != 0, seg, den)
    else:
        result = den
    enhanced = enhance_contrast(result)
    return enhanced


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", help="Path to image for demo")
    parser.add_argument("--out", default="preprocessed.jpg")
    args = parser.parse_args()
    img = cv2.imread(args.image)
    if img is None:
        print("Failed to read image")
        raise SystemExit(1)
    out = preprocess_image(img)
    cv2.imwrite(args.out, out)
    print(f"Wrote: {args.out}")
