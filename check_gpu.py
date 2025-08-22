from ultralytics import YOLO

# 加载训练好的模型
model = YOLO("models/yolov8n_run12/weights/best.pt")

# 对图片进行预测
results = model("C:/Users/User/vehicle counting ass/image/car1.jpg", show=True)
