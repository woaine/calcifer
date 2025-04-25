import customtkinter as ctk
import numpy as np
import threading
import cv2
import os
import atexit

from datetime import datetime
from PIL import Image, ImageTk, ImageDraw

from modules import State, Sensor, ThermalCamera, Model, Logger, Database
from src.superresolution import SuperResolution
from src.detection import YOLO11Face

from modules import get_temperature, process_detections, FaceTracker

class Display(ctk.CTkFrame):
    def __init__(self, master=None, **kwargs):
        super().__init__(master, **kwargs)

        self.grid(row=1, column=0, sticky="nsew", padx=25, pady=25)

        self._canvas = ctk.CTkCanvas(self, bg="gray16")
        self._canvas.pack(fill="both", expand=True)

        self._out = None
        self._streaming = False

        self._start_time_recording = None
        self._temp_file_path = None
        self._assets_path = "../calcifer/app/resources/recordings"
        os.makedirs(self._assets_path, exist_ok=True)

        threading.Thread(target=self._start_stream, daemon=True).start()
        self._trace_variables()
        atexit.register(self._cleanup)

    def _trace_variables(self):
        State().recording.add_listener(self._update_recording_state)

    def _start_stream(self):
        tracker = FaceTracker()
        
        self._streaming = True
        while self._streaming:
            frame, raw_frame = self._capture_frame()
            if frame is not None:
                frame = self._process_frame(frame)    
                boxes, roi = self._detect_face(frame)
                tracked = self._get_data(raw_frame, boxes, roi, tracker)
                stable_readings = self._predict(tracker)

                if stable_readings:
                    for face_id, face_info in stable_readings.items():
                        tracked[face_id]['Tp'] = face_info['Tp']

                frame = self._plot(frame, boxes, roi, tracked)
                
                if State().recording.value and self._out:
                    self._out.write(frame)

                frame = self._resize_and_convert_frame(frame)

                if State().recording.value and self._out:
                    self._draw_recording_indicator(frame)

                self._display(frame)

    def _capture_frame(self):
        frame = ThermalCamera().get_data()
        return frame, np.copy(frame)

    def _process_frame(self, frame):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        frame = clahe.apply(normalized)
        frame = SuperResolution().resolve(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return frame

    def _detect_face(self, frame):
        detections = YOLO11Face().track(frame)

        return process_detections(detections)

    def _get_data(self, raw_frame, boxes, roi, tracker):
        Tg = get_temperature(raw_frame, roi)
        Ta = Sensor().get_data()

        tracked = tracker.update(boxes, Tg, Ta)

        return tracked

    def _predict(self, tracker):
        stable_readings = None
        if State().predicting:
            stable_readings = tracker.get_stable()
            
            if stable_readings:
                Tg = [face_info['Tg'] for face_info in stable_readings.values()]
                Ta = [face_info['Ta'] for face_info in stable_readings.values()]
                predictions = Model().model.predict(np.column_stack((Tg, Ta)))

                for (track_id, face_info), prediction in zip(stable_readings.items(), predictions):
                    face_info['Tp'] = prediction[0]

                    if State().saving:
                        Database().insert_data(Model().model.name, face_info['Tg'], face_info['Ta'], face_info['Tp'])
                        
        return stable_readings

    def _plot(self, frame, boxes, roi, tracked):
        frame = cv2.applyColorMap(frame, cv2.COLORMAP_INFERNO)
        frame = YOLO11Face().plot(frame, boxes, roi, tracked, State().predicting)

        return frame

    def _resize_and_convert_frame(self, frame):
        frame = self._resize_image_to_fit_canvas(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return Image.fromarray(frame)

    def _display(self, frame):
        tk_image = ImageTk.PhotoImage(frame)
        self._canvas.create_image(0, 0, image=tk_image, anchor='nw')
        self._canvas.image = tk_image

    def _update_recording_state(self, *args):
        if State().recording.value:
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self):
        self._start_time_recording = datetime.now().strftime('%Y-%m-%d, %H-%M-%S')
        self._temp_file_path = self._generate_file()
        frame_size = (640, 480)
        self._out = cv2.VideoWriter(self._temp_file_path, cv2.VideoWriter_fourcc(*'mp4v'), 12.0, frame_size)
        Logger().log(f"Recording {self._temp_file_path}.")

    def _generate_file(self):
        temp_file_name = f"app_{self._start_time_recording}_unsaved.mp4"

        return f"{self._assets_path}/{temp_file_name}"

    def _get_frame_size(self):
        frame_width = int(ThermalCamera().cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(ThermalCamera().cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return frame_width, frame_height

    def _draw_recording_indicator(self, frame):
        draw = ImageDraw.Draw(frame)
        radius, x, y = 15, frame.width - 70, 50
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill="red")
        draw.text((x + radius + 5, y - radius), "REC", fill="red")

    def _stop_recording(self):
        if self._out:
            self._out.release()
            self._out = None
            self._finalize_recording()

    def _finalize_recording(self):
        end_time_recording = datetime.now().strftime('%Y-%m-%d, %H-%M-%S')
        final_file_name = f"app_{self._start_time_recording} - {end_time_recording}.mp4"
        final_file_path = f"{self._assets_path}/{final_file_name}"
        os.rename(self._temp_file_path, final_file_path)
        Logger().log(f"Recording saved as {final_file_name}.")

    def _cleanup(self):
        Logger().log("Closing program...")
        self._streaming = False

        self._stop_recording()
        
        Sensor().disconnect()
        ThermalCamera().close()
        Database().disconnect()
    
    def _resize_image_to_fit_canvas(self, image):
        canvas_width, canvas_height = self._canvas.winfo_width(), self._canvas.winfo_height()
        image_height, image_width = image.shape[:2]

        image_aspect = image_width / image_height
        canvas_aspect = canvas_width / canvas_height

        if image_aspect > canvas_aspect:
            new_height = canvas_height
            new_width = int(canvas_height * image_aspect)
        else:
            new_width = canvas_width
            new_height = int(canvas_width / image_aspect)

        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

        x_offset = (new_width - canvas_width) // 2
        y_offset = (new_height - canvas_height) // 2

        return resized_image[y_offset:y_offset + canvas_height, x_offset:x_offset + canvas_width]