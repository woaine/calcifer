import torch
import os
import cv2

from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from utilities import Logger, singleton

class YOLOBaseClass:
    def __init__(self, name, type, model_file: str, conf_thres: float, iou_thres: float):
        self.name = name
        self.type = type

        model_path = os.path.join(os.path.dirname(__file__), '../../models/yolo')
        self.model = self._load_model(f"{model_path}/{model_file}")
        
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
    
    def _load_model(self, model_path: str):
        try:
            model = YOLO(model_path)
            model.to(torch.device("cuda"))
            Logger().log(f"Loaded {self.name} model successfully.")
            return model
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")
        
    def track(self, frame):
        detections = self.model.track(
            frame, 
            conf=self.conf_threshold, 
            iou=self.iou_threshold, 
            imgsz=frame.shape[1],
            device="0",
            verbose=False,
            persist=True
        )

        return detections

    def plot(self):
        pass

@singleton
class YOLO11Face(YOLOBaseClass):
    def __init__(self, name="Y11Face", type="Face", model_file="y11face.pt", conf_thres = 0.1, iou_thres = 0.75):
        super().__init__(name, type, model_file, conf_thres, iou_thres)

    def plot(self, frame, boxes: list, roi: list, tracked: dict, predicting: bool):
        annotator = Annotator(frame, line_width=1, font_size=0.5)
        
        if boxes.id != None:
            for b in boxes:
                id = int(b.id.item())
                label = f"ID:{id}"
                box = b.xyxy.squeeze()

                if tracked and id in tracked.keys():
                    face_info = tracked[id]
                    label = label + f", Tg: {face_info['Tg']:.2f}, Ta: {face_info['Ta']:.2f}"

                    if 'Tp' in face_info.keys() and predicting:
                        label = label + f", Tp: {face_info['Tp']:.2f}"
                
                annotator.box_label(box, label, color=(0, 0, 255))
            
            frame = annotator.result()

            for keypoint in roi:
                cv2.circle(frame, keypoint, radius=1, color=(0, 0, 255))

        return frame

@singleton
class YOLO11Person(YOLOBaseClass):
    def __init__(self, name="Y11Person", type="Person", model_file="y11person.pt", conf_thres = 0.1, iou_thres = 0.75):
        super().__init__(name, type, model_file, conf_thres, iou_thres)

    def plot(self, frame, detections):
        annotator = Annotator(frame, line_width=1, font_size=0.5)
        kpts = detections[0].keypoints.data[0]

        annotator.kpts(kpts, frame.shape[:2])
        frame = annotator.result()
                    
        return frame