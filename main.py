import cv2
import torch
import pandas as pd
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import os
from tkinter import Tk, Button, filedialog, Label, Frame

# === 模型路径 ===
MODEL_PATH = r"C:\Users\User\vehicle counting ass\runs\detect\yolov8x_merged_dataset23\weights\best.pt"  # 使用新训练的模型
TARGET_CLASSES = {'bus', 'car', 'lorry', 'motorcycle', 'van'}

# === 加载模型 ===
model = YOLO(MODEL_PATH)  # 加载新训练的模型
class_names = model.names  # 获取类别名称

# === 速度计算参数 ===
MIN_SPEED_THRESHOLD = 5  # 最小有效速度(km/h)
MAX_SPEED_THRESHOLD = 200  # 最大有效速度(km/h)
SPEED_SMOOTHING_WINDOW = 5  # 速度平滑窗口大小
CONFIDENCE_THRESHOLD = 0.5  # 设置置信度阈值，低于此值的目标不显示

# 已知的物体实际高度（例如，高速公路的高度为5.4米）
ACTUAL_OBJECT_HEIGHT = 5.4  # 高速公路高度（米）

# 默认的 PPM 值，需根据实际情况动态更新
PIXELS_PER_METER = 20  # 默认值，根据需要修改

class VehicleTracker:
    def __init__(self):
        self.next_id = 0
        self.vehicles = {}  # {id: VehicleObject}
        self.frame_count = 0
        self.disappeared = {}  # 记录车辆消失帧数

    def update(self, detections, frame):
        self.frame_count += 1
        current_ids = []

        # 如果没有检测到车辆，增加所有跟踪车辆的消失计数
        if len(detections) == 0:
            for vehicle_id in list(self.vehicles.keys()):
                self._register_disappearance(vehicle_id)
            return self.vehicles

        # 获取当前帧所有检测框的中心点
        detections_centers = []
        for *box, conf, cls in detections:
            if conf < CONFIDENCE_THRESHOLD:  # 忽略低于置信度阈值的目标
                continue
            x1, y1, x2, y2 = map(int, box)
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            detections_centers.append((center_x, center_y, (x1, y1, x2, y2)))

        # 如果当前没有跟踪任何车辆，初始化所有检测到的车辆
        if len(self.vehicles) == 0:
            for center in detections_centers:
                self._register_new_vehicle(center)

        else:
            # 匹配现有车辆与新检测到的车辆
            matched_detections = set()
            matched_vehicles = set()

            # 简单最近邻匹配（实际项目可用匈牙利算法）
            for vehicle_id, vehicle in self.vehicles.items():
                last_center = vehicle.positions[-1][1:3] if vehicle.positions else (0, 0)

                min_dist = float('inf')
                best_match = None

                for i, (cx, cy, bbox) in enumerate(detections_centers):
                    if i in matched_detections:
                        continue

                    dist = np.sqrt((cx - last_center[0])**2 + (cy - last_center[1])**2)

                    # 只考虑距离小于50像素的匹配
                    if dist < 50 and dist < min_dist:
                        min_dist = dist
                        best_match = i
                if best_match is not None:
                    cx, cy, bbox = detections_centers[best_match]
                    vehicle.update((cx, cy), self.frame_count, bbox)
                    matched_detections.add(best_match)
                    matched_vehicles.add(vehicle_id)
                    current_ids.append(vehicle_id)

            # 处理未匹配的检测（新车辆）
            for i, (cx, cy, bbox) in enumerate(detections_centers):
                if i not in matched_detections:
                    self._register_new_vehicle((cx, cy, bbox))

            # 处理消失的车辆
            for vehicle_id in set(self.vehicles.keys()) - matched_vehicles:
                self._register_disappearance(vehicle_id)

        # 清理长时间消失的车辆
        self._cleanup_vehicles()
        return self.vehicles

    def _register_new_vehicle(self, center):
        vehicle_id = self.next_id
        self.next_id += 1
        cx, cy, bbox = center
        self.vehicles[vehicle_id] = Vehicle(vehicle_id, cx, cy, self.frame_count, bbox)

    def _register_disappearance(self, vehicle_id):
        if vehicle_id in self.disappeared:
            self.disappeared[vehicle_id] += 1
        else:
            self.disappeared[vehicle_id] = 1

    def _cleanup_vehicles(self):
        # 移除消失超过5帧的车辆
        for vehicle_id in list(self.disappeared.keys()):
            if self.disappeared[vehicle_id] > 5:
                if vehicle_id in self.vehicles:
                    del self.vehicles[vehicle_id]
                del self.disappeared[vehicle_id]

class Vehicle:
    def __init__(self, id, x, y, frame_num, bbox):
        self.id = id
        self.positions = deque(maxlen=30)  # 保存位置历史 (frame_num, x, y)
        self.positions.append((frame_num, x, y))
        self.bbox_history = deque(maxlen=30)  # 保存bbox历史
        self.bbox_history.append(bbox)
        self.speeds = deque(maxlen=SPEED_SMOOTHING_WINDOW)
        self.last_speed = 0

    def update(self, center, frame_num, bbox):
        self.positions.append((frame_num, center[0], center[1]))
        self.bbox_history.append(bbox)

    def calculate_speed(self, fps, ppm):
        if len(self.positions) < 2:
            return 0

        # 获取最近两个位置点
        (frame1, x1, y1), (frame2, x2, y2) = self.positions[-2], self.positions[-1]

        # 计算像素距离
        pixel_dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        # 转换成实际距离（米）
        real_dist = pixel_dist / ppm

        # 计算时间差（秒）
        time_elapsed = (frame2 - frame1) / fps

        # 避免除以零
        if time_elapsed == 0:
            return 0

        # 计算速度（m/s → km/h）
        speed_mps = real_dist / time_elapsed
        speed_kmh = speed_mps * 3.6

        # 过滤异常值
        if speed_kmh < MIN_SPEED_THRESHOLD or speed_kmh > MAX_SPEED_THRESHOLD:
            return self.last_speed

        # 存储速度用于平滑
        self.speeds.append(speed_kmh)
        self.last_speed = np.mean(self.speeds) if self.speeds else 0
        return self.last_speed

# === 计算 PPM ===
def calculate_ppm(object_pixel_height, actual_height_in_meters):
    return object_pixel_height / actual_height_in_meters

# === 图像预处理 ===
def preprocess_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return blur

# === 标注图像 + 统计数量 ===
def annotate_image(img, detections, class_names, tracker=None, fps=30):
    counts = defaultdict(int)
    speed_info = {}

    for *box, conf, cls in detections:
        class_id = int(cls.item())
        label = class_names[class_id]
        if label in TARGET_CLASSES and conf >= CONFIDENCE_THRESHOLD:  # 只显示高于阈值的检测
            counts[label] += 1
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 绘制绿色框
            cv2.putText(img, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)  # 在框上方添加标签

    # 如果有跟踪器，显示速度信息
    if tracker:
        for vehicle_id, vehicle in tracker.vehicles.items():
            if len(vehicle.bbox_history) > 0:
                bbox = vehicle.bbox_history[-1]
                speed = vehicle.calculate_speed(fps, PIXELS_PER_METER)
                if speed > 0:  # 只显示有效速度
                    x1, y1, x2, y2 = bbox
                    speed_text = f"{speed:.1f} km/h"
                    cv2.putText(img, speed_text, (x1, y2 + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)  # 红色显示速度
                    speed_info[vehicle_id] = speed

    summary_text = ', '.join([f"{k}: {v}" for k, v in counts.items()])
    cv2.putText(img, summary_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)  # 显示总计信息

    return img, counts, speed_info

# === 图像检测 ===
def detect_vehicles_image(image_path, output_csv_path=None):
    img = cv2.imread(image_path)
    preprocessed_img = preprocess_image(img)
    results = model(img)
    img_annotated, count_dict, _ = annotate_image(img, results[0].boxes.data, class_names)

    if output_csv_path:
        df = pd.DataFrame([count_dict])
        df.to_csv(output_csv_path, index=False)

    cv2.imshow("Image Result", img_annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# === 视频检测 ===
def detect_vehicles_video(video_path, output_path="output_video.avi"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 初始化跟踪器
    tracker = VehicleTracker()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 检测车辆
        results = model(frame)
        detections = results[0].boxes.data
        
        # 更新跟踪器
        tracker.update(detections, frame)
        
        # 标注图像（含速度信息）
        annotated_frame, count_dict, speed_info = annotate_image(
            frame, detections, class_names, tracker, fps)
        
        # 显示统计信息
        vehicle_count = len(tracker.vehicles)
        cv2.putText(annotated_frame, f"Vehicles: {vehicle_count}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 写入输出视频
        out.write(annotated_frame)
        
        # 显示结果
        cv2.imshow("Video Result", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# === UI ===
def choose_image():
    path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.png *.jpeg")])
    if path:
        output_csv = os.path.splitext(path)[0] + "_count.csv"
        detect_vehicles_image(path, output_csv)

def choose_video():
    path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4 *.avi *.mov")])
    if path:
        output_video = os.path.splitext(path)[0] + "_output.avi"
        detect_vehicles_video(path, output_video)

def main_ui():
    root = Tk()
    root.title("Vehicle Detection & Speed Estimation")
    root.geometry("400x300")
    root.config(bg="#f0f0f0")

    # 创建Frame来包含标题和按钮
    frame = Frame(root, bg="#f0f0f0")
    frame.pack(pady=20)

    # 设置标题
    title = Label(frame, text="Vehicle Counting With Speed Estimation", font=("Helvetica", 16, "bold"), bg="#f0f0f0")
    title.pack(pady=10)

    # 设置按钮
    Button(frame, text="Process Image", command=choose_image, width=25, height=2, font=("Helvetica", 12), bg="#4CAF50", fg="white", relief="raised").pack(pady=10)
    Button(frame, text="Process Video", command=choose_video, width=25, height=2, font=("Helvetica", 12), bg="#4CAF50", fg="white", relief="raised").pack(pady=10)

    # 添加校准说明
    info_text = f"Current PPM: {PIXELS_PER_METER}\n"
    Label(frame, text=info_text, font=("Helvetica", 12), bg="#f0f0f0").pack(pady=10)

    root.mainloop()

# === 主程序启动点 ===
if __name__ == "__main__":
    main_ui()
