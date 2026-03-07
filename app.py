from flask import Flask, render_template, request, send_file, jsonify, session
from cut_effect import auto_cut
from detect_effect import detect_icon
from flask import after_this_request
import shutil
from flask import send_from_directory
import cv2
cv2.setNumThreads(1)
cv2.setUseOptimized(True)
import numpy as np
import os
import uuid
import tempfile
import glob

app = Flask(__name__)
app.secret_key = "secret123"

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


def get_user_id():
    if "uid" not in session:
        session["uid"] = str(uuid.uuid4())
    return session["uid"]

def get_user_dir(folder):
    uid = get_user_id()
    path = os.path.join(folder, uid)
    os.makedirs(path, exist_ok=True)
    return path

@app.route("/")
def index():
    return render_template("index.html")


# =============================
# Upload background
# =============================
@app.route("/upload_bg", methods=["POST"])
def upload_bg():

    file = request.files.get("file")

    if not file:
        return "Không có file"

    user_upload = get_user_dir(UPLOAD)

    filename = str(uuid.uuid4()) + ".png"
    path = os.path.join(user_upload, filename)

    file.save(path)

    session["bg_path"] = path

    return "ok"


# =============================
# Upload rate
# =============================
@app.route("/upload_rate", methods=["POST"])
def upload_rate():

    file = request.files.get("file")

    if not file:
        return "Không có file"

    filename = str(uuid.uuid4()) + ".png"
    user_upload = get_user_dir(UPLOAD)
    path = os.path.join(user_upload, filename)

    file.save(path)

    img = cv2.imread(path)

    if img is None:
        return "Không đọc được ảnh"

    template_path = os.path.join(BASE_DIR, "profile.png")
    template = cv2.imread(template_path)

    if template is None:
        return "Không tìm thấy profile.png"

    # ======================
    # ORB detect
    # ======================

    orb = cv2.ORB_create(2000)

    kp1, des1 = orb.detectAndCompute(template, None)
    kp2, des2 = orb.detectAndCompute(img, None)

    if des1 is None or des2 is None:
        return "Không detect được feature"

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = bf.knnMatch(des1, des2, k=2)

    good = []

    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    if len(good) < 10:
        return "Không tìm thấy profile"

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if M is None:
        return "Không tìm thấy profile"

    h, w = template.shape[:2]

    pts = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts, M)

    x1 = int(min(p[0][0] for p in dst))
    y1 = int(min(p[0][1] for p in dst))
    x2 = int(max(p[0][0] for p in dst))
    y2 = int(max(p[0][1] for p in dst))

    # ======================
    # xác định vùng rate
    # ======================

    margin = 40

    rate_x1 = max(0, x1 - margin)
    rate_y1 = max(0, y1 - margin)
    rate_x2 = min(img.shape[1], x2 + margin)
    rate_y2 = min(img.shape[0], y2 + margin)

    crop = img[rate_y1:rate_y2, rate_x1:rate_x2]

    user_rate = get_user_dir(RATE)

    save = os.path.join(user_rate, "rate.png")

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
        user_upload = get_user_dir(UPLOAD)
        path = os.path.join(user_upload, filename)

        f.save(path)

        print("Saved shop:", path)

        paths.append(path)

    return jsonify({"paths": paths})


# =============================
# Cắt skin
# =============================
@app.route("/cut_skin", methods=["POST"])
def cut_skin():
    skins = []
    user_skin = get_user_dir(SKIN)
     # XÓA toàn bộ skin cũ
    for f in os.listdir(user_skin):
        os.remove(os.path.join(user_skin, f))

    template_path = os.path.join(BASE_DIR, "sohuu.png")
    template = cv2.imread(template_path)
    
    if template is None:
        return jsonify({"error": f"Không tìm thấy {template_path}"})
    
    th, tw = template.shape[:2]
    skin_width = 330
    skin_height = 522
    xs = [699,1049,1399,1749,2099]
    
    data = request.json or {}
    shop_paths = data.get("paths", [])
    
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
            loc = np.where(result >= 0.6)
            points = list(zip(loc[1], loc[0]))

            # LỌC TEMPLATE TRÙNG
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
                user_skin = get_user_dir(SKIN)
                path = os.path.join(user_skin, name)
                cv2.imwrite(path, crop)
                skins.append({
                    "name": name,
                    "url": f"/skins/{session['uid']}/{name}"
                })
           
    
    print("Shop paths:", shop_paths)
    print("Template shape:", template.shape)
    print("Skins found:", len(skins))

    return jsonify({"skins": skins})

def find_profile(bg):

    template_path = os.path.join(BASE_DIR, "profile.png")
    template = cv2.imread(template_path)

    orb = cv2.ORB_create(2000)

    kp1, des1 = orb.detectAndCompute(template, None)
    kp2, des2 = orb.detectAndCompute(bg, None)

    if des1 is None or des2 is None:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append(m)

    if len(good) < 10:
        return None

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)

    if M is None:
        return None

    h,w = template.shape[:2]

    pts = np.float32([[0,0],[0,h],[w,h],[w,0]]).reshape(-1,1,2)

    dst = cv2.perspectiveTransform(pts,M)

    x1 = int(min(p[0][0] for p in dst))
    y1 = int(min(p[0][1] for p in dst))
    x2 = int(max(p[0][0] for p in dst))
    y2 = int(max(p[0][1] for p in dst))

    return x1,y1,x2,y2



@app.route("/cut_tbha", methods=["POST"])
def cut_tbha():

    file = request.files["image"]

    uid = str(uuid.uuid4())

    upload_path = f"uploads/{uid}.png"
    save_path = f"result/{uid}_tbha.png"

    file.save(upload_path)

    template_path = os.path.join(BASE_DIR, "tbha_template.png")

    success = auto_cut(
        upload_path,
        template_path,
        save_path
    )

    if not success:
        return jsonify({"error": "Không tìm thấy thông báo hạ"}), 400

    return send_file(save_path, mimetype="image/png")

@app.route("/cut_nut", methods=["POST"])
def cut_nut():

    file = request.files["image"]

    uid = str(uuid.uuid4())

    upload_path = f"uploads/{uid}.png"
    save_path = f"result/{uid}_nut.png"

    file.save(upload_path)

    success = detect_icon(upload_path, "nut", save_path)

    if not success:
        return jsonify({"error": "Không tìm thấy nút"}), 400

    return send_file(save_path, mimetype="image/png")


@app.route("/merge_icon", methods=["POST"])
def merge_icon():

    skin = request.files["skin"]
    icon_file = request.files["icon"]

    uid = str(uuid.uuid4())

    skin_path = f"uploads/{uid}_skin.png"
    icon_path = f"uploads/{uid}_icon.png"
    save_path = f"result/{uid}_merged.png"

    skin.save(skin_path)
    icon_file.save(icon_path)

    skin_img = cv2.imread(skin_path)

    # đọc icon
    icon = cv2.imread(icon_path, cv2.IMREAD_UNCHANGED)

    # resize về 100x100
    icon = cv2.resize(icon, (100,100))

    # vẽ viền vàng
    icon = cv2.copyMakeBorder(
        icon,
        3,3,3,3,
        cv2.BORDER_CONSTANT,
        value=(0,255,255)
    )

    h, w = skin_img.shape[:2]

    # vị trí ghép
    x = w - 120
    y = h - 120

    # ghép icon
    skin_img[y:y+100, x:x+100] = icon

    cv2.imwrite(save_path, skin_img)

    return send_file(save_path, mimetype="image/png")

@app.route("/get_skins")
def get_skins():

    uid = session.get("uid")

    user_skin = os.path.join("skins", uid)

    if not os.path.exists(user_skin):
        return jsonify([])

    files = os.listdir(user_skin)

    skins = []

    for f in files:
        skins.append({
            "name": f,
            "url": f"/skins/{uid}/{f}"
        })

    return jsonify(skins)

@app.route("/skins/<uid>/<filename>")
def serve_skin(uid, filename):
    return send_from_directory(os.path.join("skins", uid), filename)

@app.route("/merge_skin", methods=["POST"])
def merge_skin():

    data = request.json or {}
    skins = data.get("skins", [])

    user_skin = get_user_dir(SKIN)

    # tìm icon mới nhất
    nut_files = glob.glob("result/*_nut.png")
    tbha_files = glob.glob("result/*_tbha.png")

    if not nut_files or not tbha_files:
        return jsonify({"error":"Chưa cắt nút hoặc thông báo hạ"})

    nut_path = sorted(nut_files)[-1]
    tbha_path = sorted(tbha_files)[-1]

    print("Nut:", nut_path)
    print("TBHA:", tbha_path)

    nut = cv2.imread(nut_path, cv2.IMREAD_UNCHANGED)
    tbha = cv2.imread(tbha_path, cv2.IMREAD_UNCHANGED)

    if nut is None or tbha is None:
        return jsonify({"error":"Không đọc được icon"})

    # resize cố định
    nut = cv2.resize(nut,(100,75))
    tbha = cv2.resize(tbha,(200,75))

    # viền vàng
    nut = cv2.copyMakeBorder(nut,3,3,3,3,cv2.BORDER_CONSTANT,value=(255,255,255))
    tbha = cv2.copyMakeBorder(tbha,3,3,3,3,cv2.BORDER_CONSTANT,value=(255,255,255))

    results = []

    for skin_name in skins:

        skin_path = os.path.join(user_skin, skin_name)

        if not os.path.exists(skin_path):
            continue

        skin = cv2.imread(skin_path)

        if skin is None:
            continue

        # vị trí ghép
        h_skin, w_skin = skin.shape[:2]

        # vị trí
        x1 = 18
        y1 = 290

        x2 = 108
        y2 = 290

    # resize nếu vượt skin
        if x1 + nut.shape[1] > w_skin:
            nut = cv2.resize(nut, (w_skin - x1 - 5, nut.shape[0]))

        if x2 + tbha.shape[1] > w_skin:
            tbha = cv2.resize(tbha, (w_skin - x2 - 5, tbha.shape[0]))

        # ghép
        skin[y1:y1+nut.shape[0], x1:x1+nut.shape[1]] = nut
        skin[y2:y2+tbha.shape[0], x2:x2+tbha.shape[1]] = tbha

        new_name = f"{uuid.uuid4()}.png"
        save = os.path.join(user_skin, new_name)

        cv2.imwrite(save, skin)

        # lưu mapping skin bị thay
        merged = session.get("merged_skin", {})
        merged[skin_name] = new_name
        session["merged_skin"] = merged

        results.append(save)

    if len(results) == 0:
        return jsonify({"error":"Không merge được skin"})

    return send_file(results[0], mimetype="image/png")
# =============================
# Ghép ảnh
# =============================
@app.route("/merge", methods=["POST"])
def merge():

    bg_path = session.get("bg_path")
    try:
        paper_count = int(request.json.get("giay_ts", 0))
    except:
        paper_count = 0
    if bg_path is None:
        return jsonify({"error":"Chưa có background"})

    data = request.json or {}
    skins = data.get("skins", [])
    use_email = data.get("email", False)
    use_chest = data.get("ruong_cs", False)
    if len(skins) == 0:
        return jsonify({"error":"Chưa có skin"})

    bg = cv2.imread(bg_path)

    if bg is None:
        return jsonify({"error":"Không đọc được background"})

    bg = cv2.resize(bg,(2796,1290))

    user_rate = get_user_dir(RATE)
    rate_path = os.path.join(user_rate,"rate.png")

    if not os.path.exists(rate_path):
        return jsonify({"error":"Chưa có rate"})

    rate = cv2.imread(rate_path)

    rate = cv2.copyMakeBorder(rate,5,5,5,5,cv2.BORDER_CONSTANT,value=[255,255,255])

    pos = find_profile(bg)

    if pos is None:
        return jsonify({"error":"Không tìm thấy profile trong background"})

    x1,y1,x2,y2 = pos

    profile_w = x2 - x1
    profile_h = y2 - y1

    # resize rate đúng kích thước profile
    rate = cv2.resize(rate,(profile_w,profile_h))

    # đảm bảo không vượt biên
    x1 = max(0,x1)
    y1 = max(0,y1)
    x2 = min(bg.shape[1],x1+profile_w)
    y2 = min(bg.shape[0],y1+profile_h)

    bg[y1:y2, x1:x2] = rate[0:(y2-y1),0:(x2-x1)]

    

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

    gap = 5
    border = 5
    start_x = left_margin

    email_img = None
    # load email

    if use_email:

        email_path = os.path.join(BASE_DIR,"email.png")

        if os.path.exists(email_path):

            email_img = cv2.imread(email_path)

            email_img = cv2.resize(email_img,(800,130))

            email_img = cv2.copyMakeBorder(
                email_img,
                5,5,5,5,
                cv2.BORDER_CONSTANT,
                value=(255,255,255)
            )
    # ===== load rương cs =====
    chest = None

    if use_chest:
        chest_path = os.path.join(BASE_DIR, "ruong_cs.png")

        if os.path.exists(chest_path):
            chest = cv2.imread(chest_path)
            chest = cv2.resize(chest, (350,300))
            # thêm viền trắng
            chest = cv2.copyMakeBorder(
                chest,
                5,5,5,5,
                cv2.BORDER_CONSTANT,
                value=(255,255,255)
            )

    skins_data = []
    # ===== load gts =====
    paper = None

    if paper_count > 0:

        paper_path = os.path.join(BASE_DIR,"giay_ts.png")

        if os.path.exists(paper_path):

            paper = cv2.imread(paper_path)

            paper = cv2.resize(paper,(300,300))

            # viền trắng
            paper = cv2.copyMakeBorder(
                paper,
                5,5,5,5,
                cv2.BORDER_CONSTANT,
                value=(255,255,255)
            )
    if paper is not None:

        text = str(paper_count)

        cv2.putText(
            paper,
            text,
            (180,220),   # tọa độ chữ
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255,255,255),
            3,
            cv2.LINE_AA
        )
    # ===== load skin =====
    merged = session.get("merged_skin", {})

    for name in skins:

        user_skin = get_user_dir(SKIN)

        # nếu skin đã ghép icon thì dùng ảnh mới
        if name in merged:
            path = os.path.join(user_skin, merged[name])
        else:
            path = os.path.join(user_skin, name)

        skin = cv2.imread(path)

        if skin is None:
            continue

        skin = cv2.resize(skin,(new_w,new_h))

        skin = cv2.copyMakeBorder(
            skin,
            border, border, border, border,
            cv2.BORDER_CONSTANT,
            value=(255,255,255)
        )

        skins_data.append(skin)

    # ===== ghép skin vào background =====
    for i, skin in enumerate(skins_data):

        h, w = skin.shape[:2]

        x = start_x + i * (w - border*2 + gap)
        y = bg.shape[0] - h - bottom_margin

        bg[y:y+h, x:x+w] = skin


    # ===== ghép email =====
    if email_img is not None and len(skins_data) > 0:

        first_skin = skins_data[0]

        h, w = first_skin.shape[:2]

        first_x = start_x
        first_y = bg.shape[0] - h - bottom_margin

        email_x = first_x

        # nếu có rương hoặc giấy
        if chest is not None or paper is not None:

            # vị trí rương
            if chest is not None:
                chest_x = first_x
                chest_y = first_y - chest.shape[0] - 15

            # vị trí giấy
            if paper is not None:

                if chest is not None:
                    paper_x = chest_x + chest.shape[1] + 10
                    paper_y = chest_y
                else:
                    paper_x = first_x
                    paper_y = first_y - paper.shape[0] - 15

            # ===== email nằm dưới cùng của khối =====
            bottom_block = first_y

            if chest is not None:
                bottom_block = max(bottom_block, chest_y + chest.shape[0])

            if paper is not None:
                bottom_block = max(bottom_block, paper_y + paper.shape[0])

            email_y = first_y - email_img.shape[0] - 15

            # đặt email bên phải khối
            if paper is not None:
                email_x = paper_x + paper.shape[1] + 10
            elif chest is not None:
                email_x = chest_x + chest.shape[1] + 10

        else:

            email_y = first_y - email_img.shape[0] - 15


        bg[
            email_y:email_y+email_img.shape[0],
            email_x:email_x+email_img.shape[1]
        ] = email_img

    # ===== ghép rương chung sức =====
    if chest is not None and len(skins_data) > 0:

        first_skin = skins_data[0]

        h, w = first_skin.shape[:2]

        first_x = start_x
        first_y = bg.shape[0] - h - bottom_margin

        chest_x = first_x
        chest_y = first_y - chest.shape[0] - 15

        bg[chest_y:chest_y+chest.shape[0], chest_x:chest_x+chest.shape[1]] = chest


    # ===== ghép giấy tuyệt sắc =====

    if paper is not None and len(skins_data) > 0:

        first_skin = skins_data[0]

        h, w = first_skin.shape[:2]

        first_x = start_x
        first_y = bg.shape[0] - h - bottom_margin

        if chest is not None:

            chest_x = first_x
            chest_y = first_y - chest.shape[0] - 15

            paper_x = chest_x + chest.shape[1] + 10
            paper_y = chest_y

        else:

            paper_x = first_x
            paper_y = first_y - paper.shape[0] - 15


        bg[
            paper_y:paper_y+paper.shape[0],
            paper_x:paper_x+paper.shape[1]
        ] = paper

    name = str(uuid.uuid4()) + ".png"
    user_result = get_user_dir(RESULT)

    save = os.path.join(user_result, name)

    cv2.imwrite(save,bg)
    session.pop("merged_skin", None)
    return jsonify({
    "status":"success",
    "image":f"/result/{session['uid']}/{name}",
    "download":f"/download/{session['uid']}/{name}"
})


# =============================
# Download
# =============================
@app.route("/download/<uid>/<filename>")
def download(uid, filename):

    path = os.path.join(RESULT, uid, filename)

    if not os.path.exists(path):
        return "Không có ảnh"

    @after_this_request
    def cleanup(response):
        try:
            # xoá toàn bộ dữ liệu của user
            shutil.rmtree(os.path.join(UPLOAD, uid), ignore_errors=True)
            shutil.rmtree(os.path.join(SKIN, uid), ignore_errors=True)
            shutil.rmtree(os.path.join(RATE, uid), ignore_errors=True)
            shutil.rmtree(os.path.join(RESULT, uid), ignore_errors=True)
        except Exception as e:
            print("Cleanup error:", e)

        return response

    return send_file(path, as_attachment=True)
@app.route("/skins/<uid>/<filename>")
def show_skin(uid, filename):
    return send_from_directory(os.path.join(SKIN, uid), filename)


@app.route("/result/<uid>/<filename>")
def show_result(uid, filename):
    return send_from_directory(os.path.join(RESULT, uid), filename)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)