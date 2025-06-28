 Vehicle-speed-and-violation-detection-using-Machine-learning-strategies-and-LPR-
"Vehicle Speed &amp; Violation Detection using ML &amp; LPR. Detects speeding vehicles, recognizes license plates &amp; identifies traffic violations. Utilizes ML (TensorFlow, PyTorch), OpenCV &amp; LPR to improve road safety &amp; automate traffic monitoring."
Vehicle-speed-and-violation-detection-using-Machine-learning-strategies-and-LPR/
│
├── main.py
├── speed_estimation.py
├── interpolate.py
├── visualize.py
├── util.py
#
1)Main.py


from ultralytics import YOLO
import cv2
import numpy as np
import util
from sort.sort import *
from util import get_car, read_license_plate, write_csv

results = {}
mot_tracker = Sort()
coco_model = YOLO('yolov8n.pt')
license_plate_detector = YOLO('license_plate_detector.pt')
cap = cv2.VideoCapture("Traffic Control CCTV.mp4")
vehicles = [2, 3, 5, 7]
frame_nmr = -1

while True:
    frame_nmr += 1
    ret, frame = cap.read()
    if not ret:
        break
    results[frame_nmr] = {}
    detections = coco_model(frame)[0]
    detections_ = [
        [x1, y1, x2, y2, score]
        for x1, y1, x2, y2, score, class_id in detections.boxes.data.tolist()
        if int(class_id) in vehicles
    ]
    if len(detections_) > 0:
        track_ids = mot_tracker.update(np.asarray(detections_))
    else:
        track_ids = np.empty((0, 5))
    license_plates = license_plate_detector(frame)[0]
    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
        if car_id != -1:
            license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
            license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
            _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
            license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh)
            if license_plate_text is not None:
                results[frame_nmr][car_id] = {
                    'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                    'license_plate': {
                        'bbox': [x1, y1, x2, y2],
                        'text': license_plate_text,
                        'bbox_score': score,
                        'text_score': license_plate_text_score,
                    },
                }
write_csv(results, './test.csv')

2)Speed_estimation.py
import cv2
from ultralytics import solutions

input_video_path = "car_video_1.mp4"
output_video_path = "speed_estimation.avi"
resize_width, resize_height = 640, 360

cap = cv2.VideoCapture(input_video_path)
original_fps = int(cap.get(cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter(
    output_video_path, cv2.VideoWriter_fourcc(*"mp4v"), original_fps, (resize_width, resize_height)
)

speed_obj = solutions.SpeedEstimator(
    region=[(0, resize_height // 2), (resize_width, resize_height // 2)],
    model="yolo11n.pt",
    classes=[2, 3, 5, 7],
    show=True,
)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    resized_frame = cv2.resize(frame, (resize_width, resize_height))
    processed_frame = speed_obj.estimate_speed(resized_frame)
    video_writer.write(processed_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
video_writer.release()
cv2.destroyAllWindows()
