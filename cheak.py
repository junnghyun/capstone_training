from ultralytics import YOLO

trained_model_path = '/runs/detect/train4/weights/best.pt'
model = YOLO(trained_model_path)

image_path = '/test/test_images/20220913_1123_20m_90-_7073_SUNNY_000012.PNG'

results = model.predict(source=image_path, save=True)
