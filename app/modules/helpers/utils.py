import numpy as np
import cv2

from filterpy.kalman import KalmanFilter
from src.detection import YOLO11Face

class FaceTracker:
    def __init__(self, prediction_interval=15):
        self.tracked_faces = {} 
        self.prediction_interval = prediction_interval
    
    def _create_kalman_filter(self, initial_temperature):
        kf = KalmanFilter(dim_x=2, dim_z=1)
        
        kf.F = np.array([[1., 1.],
                         [0., 1.]])
        kf.H = np.array([[1., 0.]])
        kf.R *= 10
        kf.Q = np.eye(2) * 0.1
        
        kf.x = np.array([initial_temperature, 0.])
        kf.P *= 1000.
        
        return kf
    
    def update(self, boxes, Tg, Ta):
        current_ids = set()
        if boxes.id != None:
            for b, t in zip(boxes, Tg):
                track_id = int(b.id.item())
                current_ids.add(track_id)
                
                if track_id not in self.tracked_faces:
                    self.tracked_faces[track_id] = {
                        'kalman_filter': self._create_kalman_filter(t),
                        'detection_count': 1,
                        'Tg': t,
                        'Ta': Ta,
                    }
                else:
                    face_info = self.tracked_faces[track_id]
                    
                    kf = face_info['kalman_filter']
                    kf.predict()
                    kf.update(t)
                    
                    face_info['detection_count'] += 1
                    face_info['Tg'] = kf.x[0]
                    face_info['Ta'] = Ta
                    
                    if face_info['detection_count'] % self.prediction_interval == 0:
                        face_info['detection_count'] = 0
            
        ids_to_remove = set(self.tracked_faces.keys()) - current_ids
        for track_id in ids_to_remove:
            del self.tracked_faces[track_id]
        
        return self.tracked_faces
    
    def get_stable(self):
        stable_readings = {}
        for track_id, face_info in self.tracked_faces.items():
            if face_info['detection_count'] == 0:
                stable_readings[track_id] = {
                    'Tg': face_info['Tg'],
                    'Ta': face_info['Ta']
                }
        
        return stable_readings
    
def process_detections(detections: list):
    roi = []
    boxes = detections[0].boxes
    keypoints = detections[0].keypoints.xy.int().cpu().numpy()
    
    if keypoints.size:
        for keypoint in keypoints:
            x1, y1 = keypoint[0]
            x2, y2 = keypoint[1]
            midpoint_x = (x1 + x2) // 2
            midpoint_y = (y1 + y2) // 2

            roi.append((midpoint_x, midpoint_y))

    return boxes, roi

def get_temperature(frame: np.ndarray, roi: list):
    temperatures = []
    
    for r in roi:
        x, y = r
        x = int(x/4)
        y = int(y/4) 

        temperatures.append(frame[y][x]/100 - 273.15)
    
    return temperatures