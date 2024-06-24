import os
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from ultralytics import YOLO

# NCCL 하트비트 타임아웃을 늘리기 위해 환경 변수 설정
os.environ['TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC'] = '1200'

# 분산 학습 초기화
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12355'
torch.distributed.init_process_group(backend='nccl')
local_rank = torch.distributed.get_rank()
torch.cuda.set_device(local_rank)

# 모델 로드
model = YOLO('yolov8n.pt').to(local_rank)
model = DDP(model, device_ids=[local_rank])

# 학습 데이터 경로
data_path = '/home/nsugis/문서/capstone_11/end/caps.yaml'

# 학습 설정
train_kwargs = {
	'data': data_path,
	'epochs': 200,
	'batch_size': 4,
	'early_stop': True,
	'patience': 30,
	'imgsz': (3840, 2160),
	'device': f'cuda:{local_rank}',
	'exist_ok': True		# 이전 학습 세션 덮어쓰기 허용
}

# 학습 실행
model.train(**train_kwargs)
