from typing import List, Tuple
from utils import metric_total, metric_IOU, visualize_images, get_rgb_from_id, create_video
import os
from ultralytics import YOLO
import cv2
from yolo import yolo_detection
from Association import association
from filterpy.kalman import KalmanFilter
import time
import numpy as np
import copy


def KalmanFilter4D(R_std: int = 10, Q_std: float = 0.01):

    # Create a Kalman filter with 8 state variables and 4 measurement variables
    kf = KalmanFilter(dim_x=8, dim_z=4)

    # State transition matrix F
    kf.F = np.array([[1, 1, 0, 0, 0, 0, 0, 0],
                     [0, 1, 0, 0, 0, 0, 0, 0],
                     [0, 0, 1, 1, 0, 0, 0, 0],
                     [0, 0, 0, 1, 0, 0, 0, 0],
                     [0, 0, 0, 0, 1, 1, 0, 0],
                     [0, 0, 0, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 1, 1],
                     [0, 0, 0, 0, 0, 0, 0, 1]])

    # Initialize covariance matrix P
    kf.P *= 1000

    # Measurement noise covariance matrix R
    kf.R[2:, 2:] *= R_std

    # Process noise covariance matrix Q
    kf.Q[-1, -1] *= Q_std
    kf.Q[4:, 4:] *= Q_std

    return kf

def state_transition_matrix(dt: float):
    # Define the state transition matrix F based on the time step (dt)
    return np.array([
        [1, dt, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, dt, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, dt, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, dt],
        [0, 0, 0, 0, 0, 0, 0, 1]
    ])



class ObstacleSORT:
    def __init__(self, id: int, bbox: List[int], time: float, age: int = 1, unmatch_age: int = 0) -> None:
        self.id = id
        self.bbox = bbox
        self.age = age
        self.unmatch_age = unmatch_age
        self.time = time
        self.kf = KalmanFilter4D()  # Initialize a 4D Kalman filter for tracking
        # Initialize the state vector x with bbox coordinates and zero velocities
        self.kf.x = np.array([bbox[0], 0, bbox[1], 0, bbox[2], 0, bbox[3], 0])
        # Measurement matrix H for extracting x, y, width, and height from the state vector
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0]])



def tracking_SORT(image: List[int]) -> Tuple[List[int], List[List[int]]]:

    global old_obstacles
    global id

    # 0. Copy image
    image_copy = copy.deepcopy(image)

    # 1. Run YOLO Object Detection to get new detections
    _, new_detections_bbox = yolo_detection(image_copy, model, label_class={'car', 'truck'})

    # 2. Get the current time in seconds
    current_time = time.time()

    # 2. Create empty list for new obstacles
    new_obstacles = []

    # 3. Get bounding box of old obstacles
    old_obstacles_bbox = [obstacle.bbox for obstacle in old_obstacles]

    # 4. Associate new detections bbox (detections) and old obstacles bbox (tracks)
    match_indices, matches, unmatched_detections, unmatched_trackers = association(tracks=old_obstacles_bbox,
                                                                                   detections=new_detections_bbox,
                                                                                   metric_function=metric_total)
    # Print the results
    # print("Size: ", len(match_indices), "| Match Indices:", match_indices)
    print("Size: ", len(matches), "| Matches:", matches)
    print("Size: ", len(unmatched_detections), "| Unmatched Detections:", unmatched_detections)
    print("Size: ", len(unmatched_trackers), "| Unmatched Trackers:", unmatched_trackers)

    # 5. Matches: Creating new obstacles based on match indices
    for index in match_indices:
        # Get ID of old obstacles
        id = old_obstacles[index[0]].id
        # Get bounding box of new detections
        detection_bbox = new_detections_bbox[index[1]]
        # Get age of old obstacles and increment by 1
        age = old_obstacles[index[0]].age + 1
        # Create an obstacle based on id of old obstacle and bounding box of new detection
        obstacle = ObstacleSORT(id=id, bbox=detection_bbox, age=age, time=current_time)
        # PREDICTION
        F = state_transition_matrix(current_time - obstacle.time)
        obstacle.kf.F = F
        obstacle.kf.predict()
        obstacle.time = current_time
        obstacle.bbox = [int(obstacle.kf.x[0]), int(obstacle.kf.x[2]), int(obstacle.kf.x[4]), int(obstacle.kf.x[6])]
        # UPDATE
        measurement = new_detections_bbox[index[1]]
        obstacle.kf.update(np.array(measurement))
        # Append obstacle to new obstacles list
        new_obstacles.append(obstacle)

    # 6. New (Unmatched) Detections: Give the new detections an id and register their bounding box coordinates
    for unmatched_detection_bbox in unmatched_detections:
        # Create new obstacle with the unmatched detections bounding box
        obstacle = ObstacleSORT(id=id, bbox=unmatched_detection_bbox, time=current_time)
        # Append obstacle to new obstacles list
        new_obstacles.append(obstacle)
        # Update id
        id += 1

    # 7. Unmatched tracking: Update unmatch age of tracks in unmatch trackers
    for tracks in unmatched_trackers:
        # Get index of bounding box tracks in old obstacles that match with unmatched trackers
        index = old_obstacles_bbox.index(tracks)
        # If we have a match
        if index is not None:
            # Based on index get the obstacle from old obstacles list
            obstacle = old_obstacles[index]
            # PREDICTION
            F = state_transition_matrix(current_time - obstacle.time)
            obstacle.kf.F = F
            obstacle.kf.predict()
            obstacle.time = current_time
            obstacle.bbox = [int(obstacle.kf.x[0]), int(obstacle.kf.x[2]), int(obstacle.kf.x[4]), int(obstacle.kf.x[6])]
            # Increment unmatch age of obstacle
            obstacle.unmatch_age += 1
            # Append obstacle to new obstacles list
            new_obstacles.append(obstacle)

    # 8. Draw bounding boxes of new obstacles with their corresponding id
    for _, obstacle in enumerate(new_obstacles):
        # Remove false negative: Filter out obstacles that have not been detected for a long time ("MAX_UNMATCHED_AGE")
        if obstacle.unmatch_age > MAX_UNMATCHED_AGE:
            new_obstacles.remove(obstacle)

        # Remove false positive: Display detections only when appeared "MIN_HIT_STREAK" times
        if obstacle.age >= MIN_HIT_STREAK:
            x1, y1, x2, y2 = obstacle.bbox
            color = get_rgb_from_id(obstacle.id*20)
            overlay = image_copy.copy()
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, thickness=cv2.FILLED)
            cv2.addWeighted(overlay, 0.5, image_copy, 1 - 0.5, 0, image_copy)
            text = f"ID: {obstacle.id}"
            cv2.putText(image_copy, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, color, 2)

    # 9. The new obstacles become old obstacles
    old_obstacles = new_obstacles

    return image_copy, old_obstacles





if __name__ == '__main__':
    # Go back to the parent directory
    parent_directory = os.path.dirname(os.getcwd())
    # print(parent_directory)

    # Instantiate model
    weights_path = os.path.join(parent_directory, 'Weights', 'yolov8n.pt')
    model = YOLO(weights_path)
    names = model.names
    model.conf = 0.6
    model.iou = 0.5
    # print(names)

    # Set input directory
    image_folder = os.path.join(parent_directory, 'Data', 'Data_3')

    # Get a list of images
    from natsort import natsorted
    image_files = natsorted([os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith((".jpg", ".jpeg", ".png"))])

    start_index = 10
    end_index = 12 # start_index + len(image_files) - 1

    # Load and visualize the selected images
    images = []
    for img_path in image_files[start_index:end_index]:
        print(img_path)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        images.append(img)

    # Parameters
    old_obstacles = []
    id = 0
    MIN_HIT_STREAK = 1  # Number of matches required before considering an obstacle
    MAX_UNMATCHED_AGE = 2  # Number of unmatched required before considering a lost track

    ### ----- Run tracking on list of images ----- ###
    result_images = []
    for image in images:
        result_image, old_obstacles = tracking_SORT(image)
        result_images.append(result_image)

    # Visualize result
    visualize_images(result_images)

    # Create video
    # create_video(result_images, output_filename='output_video_4.mp4', fps=10, codec='XVID')


