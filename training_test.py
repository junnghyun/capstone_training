import os
from ultralytics import YOLO
import torch
import torch.distributed as dist

# PyTorch DDP 초기화
dist.init_process_group(backend='nccl', init_method='env://')

# 모델 초기화
model = YOLO('yolov8n.pt')

# GPU 장치 선택
gpu_ids = [0, 1, 2, 3]  # 사용할 GPU ID 목록
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))

# 데이터 로더 최적화
# 데이터 로더의 num_workers 값을 조정하여 성능 최적화
data_loader_params = {
    'batch_size': 4,
    'num_workers': 4,  # 시스템에 맞게 조정
    'pin_memory': True,
}

# 모델 훈련
model.train(
    data='/home/nsugis/문서/capstone_11/end/caps.yaml',
    epochs=300,
    batch=4,
    imgsz=(3840, 2160),
    device=','.join(map(str, gpu_ids)),  # 사용할 GPU 장치 지정
    dataloader_params=data_loader_params
)

# DDP 종료
dist.destroy_process_group()