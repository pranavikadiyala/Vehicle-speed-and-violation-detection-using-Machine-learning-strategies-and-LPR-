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


3)interpolate.py

import csv
import numpy as np
from scipy.interpolate import interp1d

def interpolate_bounding_boxes(data):
    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
    car_ids = np.array([int(float(row['car_id'])) for row in data])
    car_bboxes = np.array([list(map(float, row['car_bbox'][1:-1].split())) for row in data])
    license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'][1:-1].split())) for row in data])

    interpolated_data = []
    unique_car_ids = np.unique(car_ids)
    for car_id in unique_car_ids:
        frame_numbers_ = [p['frame_nmr'] for p in data if int(float(p['car_id'])) == int(float(car_id))]
        car_mask = car_ids == car_id
        car_frame_numbers = frame_numbers[car_mask]
        car_bboxes_interpolated = []
        license_plate_bboxes_interpolated = []
        first_frame_number = car_frame_numbers[0]
        for i in range(len(car_bboxes[car_mask])):
            frame_number = car_frame_numbers[i]
            car_bbox = car_bboxes[car_mask][i]
            license_plate_bbox = license_plate_bboxes[car_mask][i]
            if i > 0:
                prev_frame_number = car_frame_numbers[i-1]
                prev_car_bbox = car_bboxes_interpolated[-1]
                prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]
                if frame_number - prev_frame_number > 1:
                    frames_gap = frame_number - prev_frame_number
                    x = np.array([prev_frame_number, frame_number])
                    x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
                    interp_func = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind='linear')
                    interpolated_car_bboxes = interp_func(x_new)
                    interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0, kind='linear')
                    interpolated_license_plate_bboxes = interp_func(x_new)
                    car_bboxes_interpolated.extend(interpolated_car_bboxes[1:])
                    license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])
            car_bboxes_interpolated.append(car_bbox)
            license_plate_bboxes_interpolated.append(license_plate_bbox)
        for i in range(len(car_bboxes_interpolated)):
            frame_number = first_frame_number + i
            row = {
                'frame_nmr': str(frame_number),
                'car_id': str(car_id),
                'car_bbox': ' '.join(map(str, car_bboxes_interpolated[i])),
                'license_plate_bbox': ' '.join(map(str, license_plate_bboxes_interpolated[i])),
            }
            if str(frame_number) not in frame_numbers_:
                row['license_plate_bbox_score'] = '0'
                row['license_number'] = '0'
                row['license_number_score'] = '0'
            else:
                original_row = [p for p in data if int(p['frame_nmr']) == frame_number and int(float(p['car_id'])) == int(float(car_id))][0]
                row['license_plate_bbox_score'] = original_row.get('license_plate_bbox_score', '0')
                row['license_number'] = original_row.get('license_number', '0')
                row['license_number_score'] = original_row.get('license_number_score', '0')
            interpolated_data.append(row)
    return interpolated_data

with open('test.csv', 'r') as file:
    reader = csv.DictReader(file)
    data = list(reader)

interpolated_data = interpolate_bounding_boxes(data)
header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
with open('test_interpolated.csv', 'w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=header)
    writer.writeheader()
    writer.writerows(interpolated_data)

4) visualise.py


import ast
import cv2
import numpy as np
import pandas as pd

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, line_length_x=200, line_length_y=200):
    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)
    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)
    return img

results = pd.read_csv('./test_interpolated.csv')
cap = cv2.VideoCapture('Traffic Control CCTV.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('./out.mp4', fourcc, fps, (width, height))
license_plate = {}
for car_id in np.unique(results['car_id']):
    max_ = np.amax(results[results['car_id'] == car_id]['license_number_score'])
    license_plate[car_id] = {
        'license_crop': None,
        'license_plate_number': results[(results['car_id'] == car_id) & (results['license_number_score'] == max_)]['license_number'].iloc[0]
    }
    cap.set(cv2.CAP_PROP_POS_FRAMES, results[(results['car_id'] == car_id) & (results['license_number_score'] == max_)]['frame_nmr'].iloc[0])
    ret, frame = cap.read()
    x1, y1, x2, y2 = ast.literal_eval(results[(results['car_id'] == car_id) & (results['license_number_score'] == max_)]['license_plate_bbox'].iloc[0].replace('[ ', '[').replace(' ', ','))
    license_crop = frame[int(y1):int(y2), int(x1):int(x2), :]
    license_crop = cv2.resize(license_crop, (int((x2 - x1) * 400 / (y2 - y1)), 400))
    license_plate[car_id]['license_crop'] = license_crop

frame_nmr = -1
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
while True:
    ret, frame = cap.read()
    frame_nmr += 1
    if not ret:
        break
    df_ = results[results['frame_nmr'] == frame_nmr]
    for _, row in df_.iterrows():
        car_x1, car_y1, car_x2, car_y2 = ast.literal_eval(row['car_bbox'].replace('[ ', '[').replace(' ', ','))
        draw_border(frame, (int(car_x1), int(car_y1)), (int(car_x2), int(car_y2)), (0, 255, 0), 25)
        x1, y1, x2, y2 = ast.literal_eval(row['license_plate_bbox'].replace('[ ', '[').replace(' ', ','))
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 12)
        license_crop = license_plate[row['car_id']]['license_crop']
        H, W, _ = license_crop.shape
        try:
            frame[int(car_y1)-H-100:int(car_y1)-100, int((car_x2+car_x1-W)/2):int((car_x2+car_x1+W)/2), :] = license_crop
            frame[int(car_y1)-H-400:int(car_y1)-H-100, int((car_x2+car_x1-W)/2):int((car_x2+car_x1+W)/2), :] = (255, 255, 255)
            (text_width, text_height), _ = cv2.getTextSize(license_plate[row['car_id']]['license_plate_number'], cv2.FONT_HERSHEY_SIMPLEX, 4.3, 17)
            cv2.putText(frame, license_plate[row['car_id']]['license_plate_number'], (int((car_x2+car_x1-text_width)/2), int(car_y1-H-250+(text_height/2))), cv2.FONT_HERSHEY_SIMPLEX, 4.3, (0, 0, 0), 17)
        except:
            pass
    out.write(frame)
out.release()
cap.release()
