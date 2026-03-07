import cv2
def auto_cut(img_path, template_path, save_path):

    img = cv2.imread(img_path)
    template = cv2.imread(template_path)

    if img is None:
        print("Không đọc được ảnh gốc")
        return False

    if template is None:
        print("Không đọc được template:", template_path)
        return False

    result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < 0.7:
        return False

    h, w = template.shape[:2]

    x, y = max_loc

    crop = img[y:y+h, x:x+w]

    cv2.imwrite(save_path, crop)

    return True