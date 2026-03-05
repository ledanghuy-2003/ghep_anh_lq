from tempfile import template
from flask import Flask, render_template, request, send_file, jsonify
from flask import send_from_directory
import cv2
import numpy as np
import os
import uuid
import tempfile

app = Flask(__name__)

os.makedirs("uploads", exist_ok=True)
os.makedirs("skins", exist_ok=True)
os.makedirs("result", exist_ok=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD = os.path.join(BASE_DIR, "uploads")
SKIN = os.path.join(BASE_DIR, "skins")
RATE = os.path.join(BASE_DIR, "rate")
RESULT = os.path.join(BASE_DIR, "result")
temp_dir = tempfile.mkdtemp()

for f in [UPLOAD, SKIN, RATE, RESULT]:
    os.makedirs(f, exist_ok=True)

bg_path = None
skins = []


@app.route("/")
def index():
    return render_template("index.html")


# =============================
# Upload background
# =============================
@app.route("/upload_bg", methods=["POST"])
def upload_bg():

    global bg_path

    file = request.files["file"]

    filename = str(uuid.uuid4()) + ".png"

    path = os.path.join(UPLOAD, filename)

    file.save(path)

    bg_path = path

    return "ok"


# =============================
# Upload rate
# =============================
@app.route("/upload_rate", methods=["POST"])
def upload_rate():

    file = request.files["file"]

    filename = str(uuid.uuid4()) + ".png"

    path = os.path.join(UPLOAD, filename)

    file.save(path)

    img = cv2.imread(path)

    if img is None:
        return "Không đọc được ảnh"

    left = 239
    top = 155
    width = 617
    height = 1106

    crop = img[top:top+height, left:left+width]

    save = os.path.join(RATE, "rate.png")

    cv2.imwrite(save, crop)

    return "ok"


# =============================
# Upload shop
# =============================
@app.route("/upload_shop", methods=["POST"])
def upload_shop():

    files = request.files.getlist("files")

    paths = []

    for f in files:

        filename = str(uuid.uuid4()) + ".png"
        path = os.path.join(UPLOAD, filename)

        f.save(path)

        print("Saved shop:", path)

        paths.append(path)

    return jsonify({"paths": paths})


# =============================
# Cắt skin
# =============================
@app.route("/cut_skin", methods=["POST"])
def cut_skin():
    global skins
    skins.clear()
    
    template_path = os.path.join(BASE_DIR, "sohuu.png")
    template = cv2.imread(template_path)
    
    if template is None:
        return jsonify({"error": f"Không tìm thấy {template_path}"})
    
    th, tw = template.shape[:2]
    skin_width = 330
    skin_height = 522
    xs = [699,1049,1399,1749,2099]
    
    shop_paths = request.json["paths"]
    
    for shop_path in shop_paths:
        # Kiểm tra file tồn tại
        if not os.path.exists(shop_path):
            print(f"File không tồn tại: {shop_path}")
            continue
            
        img = cv2.imread(shop_path)
        if img is None:
            print(f"Không đọc được ảnh: {shop_path}")
            continue

        try:
            result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            loc = np.where(result >= 0.75)
            points = list(zip(loc[1], loc[0]))
            
            if len(points) == 0:
                print(f"Không tìm thấy template trong ảnh: {shop_path}")
                continue
                
        except Exception as e:
            print(f"Lỗi khi xử lý ảnh {shop_path}: {str(e)}")
            continue

        points = sorted(points, key=lambda p: p[1])
        x_mid, y_mid = points[len(points)//2]
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
                name = f"{uuid.uuid4()}.png"
                path = os.path.join(SKIN, name)
                cv2.imwrite(path, crop)
                skins.append(name)
    
    print("Shop paths:", shop_paths)
    print("Template shape:", template.shape)
    print("Skins found:", len(skins))

    return jsonify({"skins": skins})


# =============================
# Ghép ảnh
# =============================
@app.route("/merge", methods=["POST"])
def merge():

    global bg_path

    if bg_path is None:
        return jsonify({"error":"Chưa có background"})

    skins = request.json["skins"]

    if len(skins) == 0:
        return jsonify({"error":"Chưa có skin"})

    bg = cv2.imread(bg_path)

    if bg is None:
        return jsonify({"error":"Không đọc được background"})

    bg = cv2.resize(bg,(2796,1290))

    rate_path = os.path.join(RATE,"rate.png")

    if not os.path.exists(rate_path):
        return jsonify({"error":"Chưa có rate"})

    rate = cv2.imread(rate_path)

    rate = cv2.copyMakeBorder(rate,5,5,5,5,cv2.BORDER_CONSTANT,value=[255,255,255])

    bg[155:155+rate.shape[0],239:239+rate.shape[1]] = rate

    left_margin = 890
    right_margin = 100
    bottom_margin = 29

    available_width = bg.shape[1] - left_margin - right_margin

    skin_count = len(skins)

    default_w = 330
    default_h = 522

    new_w = min(default_w, int(available_width / skin_count))

    scale = new_w / default_w

    new_h = int(default_h * scale)

    start_x = left_margin

    for i, name in enumerate(skins):

        path = os.path.join(SKIN, name)

        skin = cv2.imread(path)

        if skin is None:
            continue

        skin = cv2.resize(skin,(new_w,new_h))

        skin = cv2.copyMakeBorder(skin,5,5,5,5,cv2.BORDER_CONSTANT,value=[255,255,255])

        h, w = skin.shape[:2]

        x = start_x + i * w

        y = bg.shape[0] - h - bottom_margin

        bg[y:y+h, x:x+w] = skin

    save = os.path.join(RESULT,"final.png")

    cv2.imwrite(save,bg)

    return jsonify({
        "status":"success",
        "image":"/result/final.png",
        "download":"/download"
    })


# =============================
# Download
# =============================
@app.route("/download")
def download():

    path = os.path.join(RESULT,"final.png")

    if os.path.exists(path):
        return send_file(path, as_attachment=True)

    return "Chưa có ảnh"


@app.route("/skins/<filename>")
def show_skin(filename):
    return send_from_directory(SKIN, filename)


@app.route("/result/<filename>")
def show_result(filename):
    return send_from_directory(RESULT, filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)