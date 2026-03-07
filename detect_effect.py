import cv2
import numpy as np


def detect_icon(image_path, mode, save_path):

    img = cv2.imread(image_path)
    h, w = img.shape[:2]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0

    for c in contours:

        x, y, wc, hc = cv2.boundingRect(c)
        area = wc * hc

        if area < 2000:
            continue

        if mode == "nut":

            if y > h * 0.6 and x > w * 0.5:

                ratio = wc / hc

                if 0.7 < ratio < 1.3:

                    if area > best_area:
                        best = (x, y, wc, hc)
                        best_area = area

        if mode == "tbha":

            if y < h * 0.3 and x > w * 0.5:

                if wc > hc * 2:

                    if area > best_area:
                        best = (x, y, wc, hc)
                        best_area = area

    if best:

        x, y, wc, hc = best

        # nới rộng vùng cắt để lấy hết nút
        pad_left = int(wc * 0.8)
        pad_right = int(wc * 0.35)

        pad_top = int(hc * 0.45)
        pad_bottom = int(hc * 0.1)   # 👈 phần dưới nhỏ lại

        x1 = max(0, x - pad_left)
        y1 = max(0, y - pad_top)
        x2 = min(img.shape[1], x + wc + pad_right)
        y2 = min(img.shape[0], y + hc + pad_bottom)

        cut = img[y1:y2, x1:x2]

        cv2.imwrite(save_path, cut)

        return True

    return False