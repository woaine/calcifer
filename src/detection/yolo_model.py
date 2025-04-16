import torch

from ultralytics import YOLO

class YOLO11Face:
    def __init__(self, name: str, model_file: str, conf_thres: float=0.5, iou_thres: float=0.75, image_size: int=320):
        self.name = name
        
        self.model_path = f"../models/yolo/{model_file}"
        self.model = self._load_model(self.model_path)
        
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.image_size = image_size
    
    def _load_model(self, model_path: str):
        try:
            model = YOLO(model_path)
            model.to(torch.device("cuda"))
            
            return model
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")
        
    def track(self, frame):
        return self.model.track(
            frame, 
            conf=self.conf_threshold, 
            iou=self.iou_threshold, 
            imgsz=self.image_size,
            device="0",
            verbose=False,
            persist=True
        )