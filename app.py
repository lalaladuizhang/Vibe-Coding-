import math
import os
from dataclasses import dataclass

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from skimage import data

APP_TITLE = "Vibe Coding 图像滤波实验平台"


def to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr)
    if arr.ndim == 2:
        arr = np.clip(arr, 0, 255)
        return arr.astype(np.uint8)
    arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)


def pil_to_rgb_array(img) -> np.ndarray:
    if img is None:
        sample = data.astronaut()
        return sample
    if isinstance(img, dict) and "composite" in img and img["composite"] is not None:
        img = img["composite"]
    if isinstance(img, Image.Image):
        return np.array(img.convert("RGB"))
    if isinstance(img, np.ndarray):
        if img.ndim == 2:
            return cv2.cvtColor(to_uint8(img), cv2.COLOR_GRAY2RGB)
        return to_uint8(img)
    return np.array(Image.open(img).convert("RGB"))


def rgb_to_gray(img_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(to_uint8(img_rgb), cv2.COLOR_RGB2GRAY)


def ensure_odd(k: int) -> int:
    return int(k) if int(k) % 2 == 1 else int(k) + 1


def normalize_float_image(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)
    return ((img - mn) / (mx - mn) * 255).astype(np.uint8)


def spectrum_magnitude(gray: np.ndarray) -> np.ndarray:
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.log1p(np.abs(fshift))
    return normalize_float_image(magnitude)


def spatial_filter_compare(img, box_k, gauss_k, sigma, median_k):
    img_rgb = pil_to_rgb_array(img)
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

    stats = []
    for name, arr in [
        ("原图", gray),
        ("均值滤波", mean_img),
        ("高斯滤波", gauss_img),
        ("中值滤波", median_img),
        ("Sobel边缘强度", sobel_mag),
    ]:
        stats.append(
            f"- **{name}**：均值={np.mean(arr):.2f}，标准差={np.std(arr):.2f}，最小值={np.min(arr):.0f}，最大值={np.max(arr):.0f}"
        )
    summary = (
        "### 空间域滤波结果说明\n"
        "均值滤波更适合整体平滑；高斯滤波在去噪和边缘保留之间更均衡；中值滤波对椒盐噪声尤其有效；Sobel展示的是梯度响应而不是去噪图像。\n\n"
        + "\n".join(stats)
    )
    return (
        Image.fromarray(gray),
        Image.fromarray(mean_img),
        Image.fromarray(gauss_img),
        Image.fromarray(median_img),
        Image.fromarray(normalize_float_image(sobel_mag)),
        summary,
    )


def gradient_analysis(img, x1, y1, x2, y2):
    img_rgb = pil_to_rgb_array(img)
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
    mag = np.sqrt(gx**2 + gy**2)
    ang = (np.rad2deg(np.arctan2(gy, gx)) + 180) % 180
    dominant = float(np.mean(ang[mag > np.percentile(mag, 60)])) if np.any(mag > 1e-6) else 0.0

    angle_vis = cv2.applyColorMap(normalize_float_image(ang / 180.0 * 255), cv2.COLORMAP_HSV)
    angle_vis = cv2.cvtColor(angle_vis, cv2.COLOR_BGR2RGB)

    direction_text = f"### 梯度分析\n- ROI范围：x=({x1}, {x2}), y=({y1}, {y2})\n- 梯度强度均值：{np.mean(mag):.2f}\n- 梯度强度最大值：{np.max(mag):.2f}\n- 主导梯度方向：{dominant:.2f}°\n\n说明：梯度方向表示灰度变化最快的方向，边缘方向与梯度方向近似垂直。"

    return (
        overlay,
        Image.fromarray(roi),
        Image.fromarray(normalize_float_image(mag)),
        Image.fromarray(angle_vis),
        direction_text,
    )


def make_frequency_mask(shape, mode, cutoff_low, cutoff_high):
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
        mask = np.exp(-(dist**2) / (2 * float(cutoff_low) ** 2)).astype(np.float32)
    else:  # 高斯高通
        mask = (1 - np.exp(-(dist**2) / (2 * float(cutoff_low) ** 2))).astype(np.float32)
    return mask


def apply_transform(gray, shift_x, shift_y, rotate_deg, scale):
    h, w = gray.shape
    translated = np.roll(gray, shift=(int(shift_y), int(shift_x)), axis=(0, 1))
    center = (w / 2, h / 2)
    mat = cv2.getRotationMatrix2D(center, float(rotate_deg), float(scale))
    transformed = cv2.warpAffine(translated, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return transformed


def frequency_demo(img, filter_mode, cutoff_low, cutoff_high, shift_x, shift_y, rotate_deg, scale):
    img_rgb = pil_to_rgb_array(img)
    gray = rgb_to_gray(img_rgb)
    transformed = apply_transform(gray, shift_x, shift_y, rotate_deg, scale)

    f = np.fft.fft2(transformed)
    fshift = np.fft.fftshift(f)
    mask = make_frequency_mask(gray.shape, filter_mode, cutoff_low, cutoff_high)
    filtered_shift = fshift * mask
    inv = np.fft.ifft2(np.fft.ifftshift(filtered_shift))
    recon = np.abs(inv)

    orig_spec = spectrum_magnitude(gray)
    trans_spec = spectrum_magnitude(transformed)
    filt_spec = normalize_float_image(np.log1p(np.abs(filtered_shift)))
    mask_img = normalize_float_image(mask * 255)

    text = (
        "### 频域分析结论\n"
        f"- 当前滤波器：**{filter_mode}**\n"
        f"- 平移：({int(shift_x)}, {int(shift_y)}) 像素；旋转：{float(rotate_deg):.1f}°；缩放：{float(scale):.2f}\n"
        "- 平移主要改变空间位置，对幅度谱影响较小；旋转会使频谱同步旋转；缩放会改变频谱能量分布的疏密。\n"
        "- 低通滤波保留低频轮廓，高通滤波强化细节与边缘，带通滤波适合观察中频纹理。"
    )

    return (
        Image.fromarray(gray),
        Image.fromarray(transformed),
        Image.fromarray(orig_spec),
        Image.fromarray(trans_spec),
        Image.fromarray(mask_img),
        Image.fromarray(filt_spec),
        Image.fromarray(normalize_float_image(recon)),
        text,
    )


def load_example(name):
    examples = {
        "camera": data.camera(),
        "coins": data.coins(),
        "moon": data.moon(),
        "astronaut": cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2GRAY),
    }
    arr = examples.get(name, data.camera())
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(arr)


def build_app():
    with gr.Blocks(title=APP_TITLE, theme=gr.themes.Soft(), css="""
        .gradio-container {max-width: 1280px !important;}
        .hero {padding: 12px 0 2px 0;}
        .hero h1 {font-size: 34px; margin-bottom: 8px;}
        .hero p {font-size: 16px; color: #334155;}
    """) as demo:
        gr.HTML(
            """
            <div class='hero'>
              <h1>Vibe Coding 图像滤波实验平台</h1>
              <p>覆盖空间域滤波、灰度梯度分析、频率域滤波与频谱变换对比，满足课程 A2 的交互式实验要求。</p>
            </div>
            """
        )

        with gr.Row():
            example_name = gr.Dropdown(["camera", "coins", "moon", "astronaut"], value="camera", label="快速载入示例图像")
            example_btn = gr.Button("载入示例")

        shared_input = gr.Image(type="pil", label="输入图像（支持上传自己的图片）", height=320)
        example_btn.click(load_example, inputs=example_name, outputs=shared_input)

        with gr.Tabs():
            with gr.Tab("1. 空间域滤波比较"):
                with gr.Row():
                    box_k = gr.Slider(3, 15, value=5, step=2, label="均值滤波核大小")
                    gauss_k = gr.Slider(3, 15, value=5, step=2, label="高斯滤波核大小")
                    sigma = gr.Slider(0.1, 5.0, value=1.2, step=0.1, label="高斯σ")
                    median_k = gr.Slider(3, 15, value=5, step=2, label="中值滤波核大小")
                run1 = gr.Button("运行空间域滤波")
                with gr.Row():
                    out1 = gr.Image(label="原灰度图")
                    out2 = gr.Image(label="均值滤波")
                    out3 = gr.Image(label="高斯滤波")
                with gr.Row():
                    out4 = gr.Image(label="中值滤波")
                    out5 = gr.Image(label="Sobel梯度幅值")
                    txt1 = gr.Markdown()
                run1.click(spatial_filter_compare, inputs=[shared_input, box_k, gauss_k, sigma, median_k], outputs=[out1, out2, out3, out4, out5, txt1])

            with gr.Tab("2. 灰度梯度与局部区域分析"):
                gr.Markdown("通过调整 ROI 坐标，观察局部区域的梯度强度与方向变化。")
                with gr.Row():
                    x1 = gr.Slider(0, 512, value=120, step=1, label="x1")
                    y1 = gr.Slider(0, 512, value=120, step=1, label="y1")
                    x2 = gr.Slider(0, 512, value=320, step=1, label="x2")
                    y2 = gr.Slider(0, 512, value=320, step=1, label="y2")
                run2 = gr.Button("分析梯度")
                with gr.Row():
                    g1 = gr.Image(label="ROI框选位置")
                    g2 = gr.Image(label="ROI灰度图")
                with gr.Row():
                    g3 = gr.Image(label="梯度幅值图")
                    g4 = gr.Image(label="梯度方向图")
                gtxt = gr.Markdown()
                run2.click(gradient_analysis, inputs=[shared_input, x1, y1, x2, y2], outputs=[g1, g2, g3, g4, gtxt])

            with gr.Tab("3. 频率域滤波与频谱变化"):
                with gr.Row():
                    filter_mode = gr.Dropdown(["低通", "高通", "带通", "高斯低通", "高斯高通"], value="低通", label="频域滤波器")
                    cutoff_low = gr.Slider(5, 120, value=35, step=1, label="截止半径1")
                    cutoff_high = gr.Slider(10, 200, value=80, step=1, label="截止半径2（带通用）")
                with gr.Row():
                    shift_x = gr.Slider(-120, 120, value=0, step=1, label="平移 x")
                    shift_y = gr.Slider(-120, 120, value=0, step=1, label="平移 y")
                    rotate_deg = gr.Slider(-180, 180, value=0, step=1, label="旋转角度")
                    scale = gr.Slider(0.5, 1.8, value=1.0, step=0.05, label="缩放倍数")
                run3 = gr.Button("执行频域实验")
                with gr.Row():
                    f1 = gr.Image(label="原图")
                    f2 = gr.Image(label="变换后图像")
                    f3 = gr.Image(label="原图频谱")
                with gr.Row():
                    f4 = gr.Image(label="变换后频谱")
                    f5 = gr.Image(label="滤波器掩膜")
                    f6 = gr.Image(label="滤波后频谱")
                with gr.Row():
                    f7 = gr.Image(label="反变换重建图")
                    ftxt = gr.Markdown()
                run3.click(
                    frequency_demo,
                    inputs=[shared_input, filter_mode, cutoff_low, cutoff_high, shift_x, shift_y, rotate_deg, scale],
                    outputs=[f1, f2, f3, f4, f5, f6, f7, ftxt],
                )

        gr.Markdown(
            """
            **建议运行方式**：`python app.py` 后在浏览器中打开本地链接。  
            **实验说明**：本程序基于 OpenCV + NumPy + SciPy + Gradio 开发，适合作为课程作业提交的交互式 web 应用。
            """
        )
    return demo


if __name__ == "__main__":
    demo = build_app()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
