import os
from ultralytics import YOLO

# 환경 변수 설정
os.environ['TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC'] = '1200'
os.environ['TORCH_NCCL_ENABLE_MONITORING'] = '0'
os.environ['TORCH_NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_P2P_LEVEL'] = 'LOC'
os.environ['NCCL_DEBUG_SUBSYS'] = 'INIT,GRAPH'

# 모델 초기화
model = YOLO('yolov8n.pt')

# 모델 훈련
model.train(
    data='/home/nsugis/문서/capstone_11/end/caps.yaml',
    epochs=300,
    batch=4,
    imgsz=(3840, 2160),
    device='0,1,2,3'
)
