import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from skimage import data

st.set_page_config(
    page_title="A2 图像滤波交互实验平台",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_TITLE = "Vibe Coding 图像滤波交互实验平台"


def to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr)
    arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)


def normalize_float_image(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    mn, mx = float(np.min(img)), float(np.max(img))
    if mx - mn < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img - mn) / (mx - mn) * 255).astype(np.uint8)


def ensure_odd(k: int) -> int:
    k = int(k)
    return k if k % 2 == 1 else k + 1


def rgb_to_gray(img_rgb: np.ndarray) -> np.ndarray:
    if img_rgb.ndim == 2:
        return to_uint8(img_rgb)
    return cv2.cvtColor(to_uint8(img_rgb), cv2.COLOR_RGB2GRAY)


def load_example_image(name: str) -> np.ndarray:
    examples = {
        "camera": data.camera(),
        "coins": data.coins(),
        "moon": data.moon(),
        "astronaut": data.astronaut(),
    }
    arr = examples.get(name, data.camera())
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    return to_uint8(arr)


def pil_or_bytes_to_rgb(uploaded_file) -> np.ndarray | None:
    if uploaded_file is None:
        return None
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image)


def spectrum_magnitude(gray: np.ndarray) -> np.ndarray:
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))
    return normalize_float_image(magnitude)


def spatial_filter_compare(img_rgb: np.ndarray, box_k: int, gauss_k: int, sigma: float, median_k: int):
    gray = rgb_to_gray(img_rgb)
    box_k = ensure_odd(box_k)
    gauss_k = ensure_odd(gauss_k)
    median_k = ensure_odd(median_k)

    mean_img = cv2.blur(gray, (box_k, box_k))
    gauss_img = cv2.GaussianBlur(gray, (gauss_k, gauss_k), sigmaX=float(sigma))
    median_img = cv2.medianBlur(gray, median_k)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobel_x.astype(np.float32), sobel_y.astype(np.float32))

    return {
        "原灰度图": gray,
        "均值滤波": mean_img,
        "高斯滤波": gauss_img,
        "中值滤波": median_img,
        "Sobel梯度幅值": normalize_float_image(sobel_mag),
        "stats": {
            "原图均值": float(np.mean(gray)),
            "原图标准差": float(np.std(gray)),
            "均值滤波标准差": float(np.std(mean_img)),
            "高斯滤波标准差": float(np.std(gauss_img)),
            "中值滤波标准差": float(np.std(median_img)),
            "Sobel最大响应": float(np.max(sobel_mag)),
        },
    }


def gradient_analysis(img_rgb: np.ndarray, x1: int, y1: int, x2: int, y2: int):
    gray = rgb_to_gray(img_rgb)
    h, w = gray.shape
    x1, x2 = sorted([max(0, int(x1)), min(w - 1, int(x2))])
    y1, y2 = sorted([max(0, int(y1)), min(h - 1, int(y2))])

    if x2 <= x1 + 2:
        x2 = min(w - 1, x1 + 50)
    if y2 <= y1 + 2:
        y2 = min(h - 1, y1 + 50)

    overlay = Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle([x1, y1, x2, y2], outline=(255, 80, 80), width=3)

    roi = gray[y1:y2, x1:x2]
    gx = cv2.Sobel(roi, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(roi, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    ang = (np.rad2deg(np.arctan2(gy, gx)) + 180) % 180
    mask = mag > np.percentile(mag, 60) if np.any(mag > 1e-6) else np.zeros_like(mag, dtype=bool)
    dominant = float(np.mean(ang[mask])) if np.any(mask) else 0.0

    angle_vis = cv2.applyColorMap(normalize_float_image(ang / 180.0 * 255), cv2.COLORMAP_HSV)
    angle_vis = cv2.cvtColor(angle_vis, cv2.COLOR_BGR2RGB)

    return {
        "ROI框选位置": np.array(overlay),
        "ROI灰度图": roi,
        "梯度幅值图": normalize_float_image(mag),
        "梯度方向图": angle_vis,
        "梯度强度均值": float(np.mean(mag)),
        "梯度强度最大值": float(np.max(mag)),
        "主导梯度方向": dominant,
        "roi_text": f"ROI: x=({x1}, {x2}), y=({y1}, {y2})",
    }


def make_frequency_mask(shape, mode: str, cutoff_low: int, cutoff_high: int):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    dist = np.sqrt((y - crow) ** 2 + (x - ccol) ** 2)
    cutoff_low = max(1, int(cutoff_low))
    cutoff_high = max(cutoff_low + 1, int(cutoff_high))

    if mode == "低通":
        mask = (dist <= cutoff_low).astype(np.float32)
    elif mode == "高通":
        mask = (dist >= cutoff_low).astype(np.float32)
    elif mode == "带通":
        mask = ((dist >= cutoff_low) & (dist <= cutoff_high)).astype(np.float32)
    elif mode == "高斯低通":
        mask = np.exp(-(dist ** 2) / (2 * float(cutoff_low) ** 2)).astype(np.float32)
    else:
        mask = (1 - np.exp(-(dist ** 2) / (2 * float(cutoff_low) ** 2))).astype(np.float32)
    return mask


def apply_transform(gray: np.ndarray, shift_x: int, shift_y: int, rotate_deg: float, scale: float):
    h, w = gray.shape
    translated = np.roll(gray, shift=(int(shift_y), int(shift_x)), axis=(0, 1))
    center = (w / 2, h / 2)
    mat = cv2.getRotationMatrix2D(center, float(rotate_deg), float(scale))
    transformed = cv2.warpAffine(
        translated,
        mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )
    return transformed


def frequency_demo(img_rgb: np.ndarray, filter_mode: str, cutoff_low: int, cutoff_high: int, shift_x: int, shift_y: int, rotate_deg: float, scale: float):
    gray = rgb_to_gray(img_rgb)
    transformed = apply_transform(gray, shift_x, shift_y, rotate_deg, scale)

    f = np.fft.fft2(transformed)
    fshift = np.fft.fftshift(f)
    mask = make_frequency_mask(gray.shape, filter_mode, cutoff_low, cutoff_high)
    filtered_shift = fshift * mask
    inv = np.fft.ifft2(np.fft.ifftshift(filtered_shift))
    recon = np.abs(inv)

    return {
        "原图": gray,
        "变换后图像": transformed,
        "原图频谱": spectrum_magnitude(gray),
        "变换后频谱": spectrum_magnitude(transformed),
        "滤波器掩膜": normalize_float_image(mask * 255),
        "滤波后频谱": normalize_float_image(np.log1p(np.abs(filtered_shift))),
        "反变换重建图": normalize_float_image(recon),
        "重建均值": float(np.mean(recon)),
        "重建标准差": float(np.std(recon)),
    }


st.markdown(
    """
    <style>
    .main-title {font-size: 2.1rem; font-weight: 800; margin-bottom: 0.1rem;}
    .sub-note {color: #4b5563; margin-bottom: 1rem;}
    .metric-card {padding: 0.6rem 0.8rem; border-radius: 12px; background: #f7f7fb;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(f"<div class='main-title'>{APP_TITLE}</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-note'>覆盖空间域滤波、局部梯度分析、频率域滤波三大模块，适合直接用于课程 A2 的交互式展示与在线部署。</div>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("输入图像")
    source = st.radio("图像来源", ["上传图片", "使用示例图"], index=1)
    if source == "上传图片":
        uploaded = st.file_uploader("上传 JPG / PNG / JPEG", type=["jpg", "jpeg", "png"])
        image_rgb = pil_or_bytes_to_rgb(uploaded)
        if image_rgb is None:
            st.info("还没有上传图片，暂时先使用 camera 示例图。")
            image_rgb = load_example_image("camera")
    else:
        example_name = st.selectbox("选择示例", ["camera", "coins", "moon", "astronaut"], index=0)
        image_rgb = load_example_image(example_name)

    st.image(image_rgb, caption="当前输入图像", use_container_width=True)

    st.divider()
    st.caption("建议：上传灰度纹理图时，频域效果通常更明显；上传边缘清晰的人像图时，梯度分析更直观。")

img_h, img_w = image_rgb.shape[:2]

with st.expander("实验说明与评分对应关系", expanded=False):
    st.write(
        "1. 空间域模块实现 Box / Gaussian / Median / Sobel；"
        "2. 梯度模块支持 ROI 交互与梯度方向分析；"
        "3. 频率域模块实现 FFT、频谱图、低通/高通/带通/高斯滤波；"
        "4. 页面可在线交互，适合作为课程展示网站。"
    )


tab1, tab2, tab3 = st.tabs(["空间域滤波比较", "灰度梯度与 ROI 分析", "频率域滤波与频谱变化"])

with tab1:
    colp1, colp2, colp3, colp4 = st.columns(4)
    with colp1:
        box_k = st.slider("均值滤波核大小", 3, 15, 5, 2)
    with colp2:
        gauss_k = st.slider("高斯滤波核大小", 3, 15, 5, 2)
    with colp3:
        sigma = st.slider("高斯 σ", 0.1, 5.0, 1.2, 0.1)
    with colp4:
        median_k = st.slider("中值滤波核大小", 3, 15, 5, 2)

    result1 = spatial_filter_compare(image_rgb, box_k, gauss_k, sigma, median_k)
    c1, c2, c3 = st.columns(3)
    c1.image(result1["原灰度图"], caption="原灰度图", use_container_width=True)
    c2.image(result1["均值滤波"], caption="均值滤波", use_container_width=True)
    c3.image(result1["高斯滤波"], caption="高斯滤波", use_container_width=True)
    c4, c5 = st.columns(2)
    c4.image(result1["中值滤波"], caption="中值滤波", use_container_width=True)
    c5.image(result1["Sobel梯度幅值"], caption="Sobel 梯度幅值", use_container_width=True)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("原图均值", f"{result1['stats']['原图均值']:.2f}")
    m2.metric("原图标准差", f"{result1['stats']['原图标准差']:.2f}")
    m3.metric("高斯后标准差", f"{result1['stats']['高斯滤波标准差']:.2f}")
    m4.metric("Sobel最大响应", f"{result1['stats']['Sobel最大响应']:.2f}")

    st.markdown(
        "均值滤波更偏向整体平滑；高斯滤波通常能在去噪和细节保留之间取得较平衡效果；中值滤波对椒盐噪声特别有效；Sobel 输出的是边缘强度图，因此纹理和轮廓会被明显强化。"
    )

with tab2:
    st.write("通过调整局部区域坐标，分析某一块区域内的梯度幅值与主导方向。")
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        x1 = st.number_input("x1", min_value=0, max_value=max(0, img_w - 1), value=min(120, max(0, img_w - 2)), step=1)
    with g2:
        y1 = st.number_input("y1", min_value=0, max_value=max(0, img_h - 1), value=min(120, max(0, img_h - 2)), step=1)
    with g3:
        x2 = st.number_input("x2", min_value=0, max_value=max(0, img_w - 1), value=min(320, max(1, img_w - 1)), step=1)
    with g4:
        y2 = st.number_input("y2", min_value=0, max_value=max(0, img_h - 1), value=min(320, max(1, img_h - 1)), step=1)

    result2 = gradient_analysis(image_rgb, x1, y1, x2, y2)
    c1, c2 = st.columns(2)
    c1.image(result2["ROI框选位置"], caption="ROI 框选位置", use_container_width=True)
    c2.image(result2["ROI灰度图"], caption="ROI 灰度图", use_container_width=True)
    c3, c4 = st.columns(2)
    c3.image(result2["梯度幅值图"], caption="梯度幅值图", use_container_width=True)
    c4.image(result2["梯度方向图"], caption="梯度方向图", use_container_width=True)

    m1, m2, m3 = st.columns(3)
    m1.metric("梯度强度均值", f"{result2['梯度强度均值']:.2f}")
    m2.metric("梯度强度最大值", f"{result2['梯度强度最大值']:.2f}")
    m3.metric("主导梯度方向", f"{result2['主导梯度方向']:.2f}°")

    st.markdown(
        f"{result2['roi_text']}。梯度方向表示灰度变化最快的方向，而视觉上看到的边缘方向通常与梯度方向近似垂直。"
    )

with tab3:
    c1, c2, c3 = st.columns(3)
    with c1:
        filter_mode = st.selectbox("频域滤波器", ["低通", "高通", "带通", "高斯低通", "高斯高通"], index=0)
    with c2:
        cutoff_low = st.slider("截止半径 1", 5, 120, 35, 1)
    with c3:
        cutoff_high = st.slider("截止半径 2（带通用）", 10, 200, 80, 1)

    t1, t2, t3, t4 = st.columns(4)
    with t1:
        shift_x = st.slider("平移 x", -120, 120, 0, 1)
    with t2:
        shift_y = st.slider("平移 y", -120, 120, 0, 1)
    with t3:
        rotate_deg = st.slider("旋转角度", -180, 180, 0, 1)
    with t4:
        scale = st.slider("缩放倍数", 0.5, 1.8, 1.0, 0.05)

    result3 = frequency_demo(image_rgb, filter_mode, cutoff_low, cutoff_high, shift_x, shift_y, rotate_deg, scale)

    r1, r2, r3 = st.columns(3)
    r1.image(result3["原图"], caption="原图", use_container_width=True)
    r2.image(result3["变换后图像"], caption="变换后图像", use_container_width=True)
    r3.image(result3["原图频谱"], caption="原图频谱", use_container_width=True)
    r4, r5, r6 = st.columns(3)
    r4.image(result3["变换后频谱"], caption="变换后频谱", use_container_width=True)
    r5.image(result3["滤波器掩膜"], caption="滤波器掩膜", use_container_width=True)
    r6.image(result3["滤波后频谱"], caption="滤波后频谱", use_container_width=True)
    st.image(result3["反变换重建图"], caption="反变换重建图", use_container_width=True)

    m1, m2 = st.columns(2)
    m1.metric("重建均值", f"{result3['重建均值']:.2f}")
    m2.metric("重建标准差", f"{result3['重建标准差']:.2f}")

    st.markdown(
        "平移主要改变图像在空间域中的位置，对幅度谱整体结构影响较小；旋转会带来频谱的同步旋转；缩放会改变频谱能量的密集程度。低通更突出轮廓，高通更突出边缘，带通更适合观察中频纹理。"
    )

st.divider()
st.caption("部署提示：Streamlit Cloud 启动命令通常是 `streamlit run app.py`；Hugging Face Spaces 若使用 Gradio 版，则使用单独的 Gradio 项目文件更稳。")
