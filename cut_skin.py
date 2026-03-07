import cv2
import numpy as np
import os
import uuid

def cut_skin_process(shop_paths, template_path, user_skin, session_uid):

    skins = []

    template = cv2.imread(template_path)

    if template is None:
        return []

    th, tw = template.shape[:2]

    skin_width = 330
    skin_height = 522

    xs = [699,1049,1399,1749,2099]

    for shop_path in shop_paths:

        if not os.path.exists(shop_path):
            print("File không tồn tại:", shop_path)
            continue

        img = cv2.imread(shop_path)

        if img is None:
            print("Không đọc được:", shop_path)
            continue

        h, w = img.shape[:2]

        # chỉ scan nửa dưới để tránh nhầm chữ phía trên
        roi = img[int(h*0.45):h, :]

        result = cv2.matchTemplate(roi, template, cv2.TM_CCOEFF_NORMED)

        loc = np.where(result >= 0.55)

        points = list(zip(loc[1], loc[0]))

        filtered = []

        for (x,y) in points:

            duplicate = False

            for (fx,fy) in filtered:
                if abs(x-fx) < 80 and abs(y-fy) < 40:
                    duplicate = True
                    break

            if not duplicate:
                filtered.append((x,y))

        points = filtered

        if len(points) == 0:
            continue

        # sort theo chiều dọc
        points = sorted(points, key=lambda p: p[1])

        # lấy điểm giữa
        x_mid, y_mid = points[len(points)//2]

        # cộng lại offset roi
        y_mid += int(h*0.45)

        offset = 604 - th

        top = y_mid - offset

        for x in xs:

            check = img[top+skin_height-20:top+skin_height+140, x:x+skin_width]

            if check.size == 0:
                continue

            res = cv2.matchTemplate(check, template, cv2.TM_CCOEFF_NORMED)

            score = np.max(res)

            if score > 0.45:

                crop = img[top:top+skin_height, x:x+skin_width]

                if crop.shape[0] != skin_height:
                    continue

                name = f"{uuid.uuid4()}.png"

                path = os.path.join(user_skin, name)

                cv2.imwrite(path, crop)

                skins.append({
                    "name": name,
                    "url": f"/skins/{session_uid}/{name}"
                })

    return skins