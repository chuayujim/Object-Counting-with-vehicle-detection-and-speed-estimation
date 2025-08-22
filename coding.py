import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from ultralytics import YOLO
import os

# 初始化模型（加载你训练好的 best.pt）
model = YOLO("models/yolov8n_run12/weights/best.pt")

def predict_image(img_path):
    # 自动 resize 成 640x640，并保存为中间文件
    resized_path = "resized_input.jpg"
    img = Image.open(img_path).convert("RGB")
    img = img.resize((640, 640))
    img.save(resized_path)

    # 推理并保存结果图
    results = model(resized_path, save=True, show=False)

    # 获取保存的图片路径
    result_dir = results[0].save_dir
    result_img = os.path.join(result_dir, os.path.basename(resized_path))
    return result_img

def select_and_predict():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        return

    # 显示原图
    original = Image.open(file_path).resize((300, 300))
    original_tk = ImageTk.PhotoImage(original)
    panel_original.configure(image=original_tk)
    panel_original.image = original_tk
    panel_original.config(text="原图")

    # 模型预测图
    result_path = predict_image(file_path)
    result = Image.open(result_path).resize((300, 300))
    result_tk = ImageTk.PhotoImage(result)
    panel_result.configure(image=result_tk)
    panel_result.image = result_tk
    panel_result.config(text="识别后")

# 创建窗口
root = tk.Tk()
root.title("车辆识别系统")

btn = tk.Button(root, text="选择图片进行识别", command=select_and_predict)
btn.pack(pady=10)

frame = tk.Frame(root)
frame.pack()

panel_original = tk.Label(frame, text="原图")
panel_original.pack(side="left", padx=10)

panel_result = tk.Label(frame, text="识别后")
panel_result.pack(side="right", padx=10)

root.mainloop()
