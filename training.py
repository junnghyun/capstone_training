from ultralytics import YOLO

# 모델 초기화
model = YOLO('yolov8n.pt')

# 모델 훈련
model.train(data='/home/nsugis/문서/capstone_11/end/caps.yaml', epochs=300, batch_size=4, imgsz=(3840,2160), early_stop=True, patience=30, device='0,1,2,3')
