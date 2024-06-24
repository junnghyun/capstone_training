import os
import ruamel.yaml

# 예시 데이터 (실제 데이터로 교체 필요)
train_data = ['/home/nsugis/문서/capstone_11/end/train/images/*']
train_labels = ['/home/nsugis/문서/capstone_11/end/train/labels/*']

val_data = ['/home/nsugis/문서/capstone_11/end/val/images/*']
val_labels = ['/home/nsugis/문서/capstone_11/end/val/labels/*']

test_data = ['/home/nsugis/문서/capstone_11/end/test/images/*']
test_labels = ['/home/nsugis/문서/capstone_11/end/test/labels/*']

# YAML 내용 구성
data = {
    'train': {
        'data': train_data,
        'labels': train_labels
    },
    'val': {
        'data': val_data,
        'labels': val_labels
    },
    'test': {
        'data': test_data,
        'labels': test_labels
    }
}

# 저장할 디렉토리 경로 지정
save_dir = '/home/nsugis/문서/capstone_11/end/'

# YAML 파일 작성
yaml_file = os.path.join(save_dir, 'dataset_info.yaml')
with open(yaml_file, 'w', encoding='utf-8') as outfile:
    ruamel.yaml.dump(data, outfile, default_flow_style=False, allow_unicode=True)

print(f"YAML 파일 '{yaml_file}'이(가) 성공적으로 생성되었습니다.")
