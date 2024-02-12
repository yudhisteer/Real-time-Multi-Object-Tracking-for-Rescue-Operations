# Real-time Multi-Object Tracking for Rescue Operations

## Problem Statement
On January 16th, 2024, a tropical storm, **Belal**, hit close to the southern coast of **Mauritius** and was now heading eastwards out into the Indian Ocean. At least **one person has died** from the powerful winds and rains of the cyclone. On the nearby French island of **Réunion**, thousands of residents remain without **power** and **water outages** following the cyclone's impact the day before. Seeing the calamity that the cyclone has done to my country, Mauritius, I wanted to investigate how I could use my computer vision knowledge to help the **authorities**. 

After watching distressing footage online showing people **trapped** atop their cars amidst floods, as described vividly by Hindustan Times as "**Cars swept away like toys**," I recognized the urgent need for authorities to **track** and **assess** the number of individuals in need of **rescue**. In the chaotic scenario of people stuck in cars drifting in floodwaters, it's crucial to accurately track each person's **location** for successful rescue operations.

<p align="center">
  <img src="https://github.com/yudhisteer/Real-time-Multi-Object-Tracking-for-Rescue-Operations/assets/59663734/e3e921a6-ea73-4168-880f-0bba82a7bf4b" width="60%" />
</p>
<div align="center">
    <p>Image Source: <a href="https://www.hindustantimes.com/world-news/mauritius-floods-mauritius-cyclone-cyclone-belal-mauritius-rainfall-mauritius-floods-videos-101705322329348.html">Cars swept away like toys as cyclone Belal causes havoc in Mauritius</a></p>
</div>


## Abstract
This project involves finding a solution for emergency responders to track people trapped in times of disasters like cyclones, flash floods, or earthquakes. The proposed system will utilize surveillance cameras, drones, or other relevant sources to continuously monitor affected areas and track the movements of individuals in real-time. I show how to design tracking algorithms like **SORT** from scratch which uses the **Hungarian** algorithm for the **association** of tracks and the **Kalman Filter** to model the motion of objects of interest. However, due to poor results during occlusion and a high number of identity switches, I then demonstrate how to implement the **Deep SORT** algorithm which not only models the **motion** but also the **appearance** of objects using a **Siamese** network. The goal of this project is to help disaster response teams by improving their awareness, and coordination, and ultimately, saving lives during emergencies.


<div style="text-align: center;">
  <video src="https://github.com/yudhisteer/Real-time-Multi-Object-Tracking-for-Rescue-Operations/assets/59663734/d480ba65-4825-46c3-a200-f7e3d9b00b9c" controls="controls" style="max-width: 730px;">
  </video>
</div>



## Table of Contents
1. [Object Detection](#od)
2. [Object Tracking](#ot)
3. [The Hungarian Algorithm](#ha)
4. [SORT: Simple Online Realtime Tracking](#s)
5. [Deep SORT: Simple Online and Realtime Tracking with a Deep Association Metric](#ds)

------------
<a name="od"></a>
## 1. Object Detection
Before we talk about object tracking let's talk about the two main categories of object detection models: **one** and **two-stage detectors**.

### 1.1 Two-Stage Detectors

- It consists of two separate modules: **Region Proposal Network (RPN)** and a **Classification Network**.
- RPN suggests a collection of **regions of interest** that are likely to contain an object.
- The classification network then **classifies** the proposals and **refines** their bounding boxes.
- Two-stage detectors are **accurate**, especially for detecting small objects but **slower** than one-stage detectors.
- Examples: Models in the R-CNN family, SPP-Net, FPN...

### 1.2 One-Stage Detectors

- They perform object detection in a **single pass** through the network, without explicitly generating region proposals.
- The image is divided into a grid, and each grid cell predicts **bounding boxes** and **class probabilities** directly.
- One-stage detectors are **simpler** and **faster** because they do not need a separate region proposal stage.
- Used in real-time applications but may sacrifice some accuracy, especially for small objects.
- Examples: YOLO (You Only Look Once), SSD (Single Shot MultiBox Detector).

Now that we know what is object detection, why do we need tracking? We need to model the **temporal** aspect of objects. That is, when detection fails due to occlusion or other factors, we want to at least know where the object might be. Then, we do not only want to detect the object but also be able to **predict** the trajectory in the future. 

Once we have a set of detections per frame that we have found independently for each frame, the actual task is to find the detections that **match** from one frame to another and to form a **trajectory**.

------------
<a name="ot"></a>
## 2. Object Tracking
In my other project - [**Optical Flow Obstacle Avoidance for UAV**](https://github.com/yudhisteer/Optical-Flow-Obstacle-Avoidance-for-UAV) - we saw how we can analyze the **apparent motion** of objects in a sequence of images using ```Optical Flow```. We tested with **sparse**, **dense**, and **deep optical flow** to eventually see how every point in the scene is moving from frame to frame in a video sequence. In object tracking, we want to track **objects** or **regions** from frame to frame. 

### 2.1 Change Detection
Change detection refers to a way to detect meaningful changes in a sequence of frames. Suppose we have a static camera filming the environment as shown in the video below, the meaningful changes will be the moving objects - me running. In other words, we will be doing a real-time **classification** (```foreground-background classification```) of each pixel as belonging to the **foreground** (```meaningful change```) or belonging to a **background** (```static```).


To build this classification, we will need to get rid of unmeaningful changes or uninterested changes which are:

- Background fluctuations such as leaves moving due to wind.
- Image noise due to low light intensity.
- Natural turbulence such as snow or rain.
- Shadows of people moving.
- Camera shake such as in the video below.

<div style="text-align: center;">
  <video src="https://github.com/yudhisteer/Real-time-Ego-Tracking-A-Tactical-Solution-for-Rescue-Operations/assets/59663734/38a4e14a-6e03-4ead-adee-7a9a13bb01a5" controls="controls" style="max-width: 730px;">
  </video>
</div>

There are a few ways to detect meaningful changes:
1. We calculate the difference between the current frame and the previous one. Wherever this difference is significantly higher than a set threshold, we consider it a change. However, we may experience a lot of noise and register uninterested changes such as background fluctuations.

2. A more robust method is to compute a background image, i.e., the **average** of the first ```K``` frames of the scene. We then compare our pixel value (subsequent frames) with the average background image. However, we are storing only one value for pixel as the model and we comparing all the future values with that value and may not handle the movement of leaves or change in lighting.

3. A better method than the average of the first ```K``` frames would be to instead calculate the **median** of the first K frames. It is more stable than the average, but a value in any given frame can vary substantially from the most popular value.

4. An even more robust model is to build an **adaptive** one such that the median is computed using the last few frames and not the first few frames. That is, we are going to recompute the median of the background model and it is going to adapt to changes. We can experience substantial improvement, especially for background fluctuations.


### 2.2 Template Matching
Now that we can spot important changes in videos, we choose one of these changes to represent an object or area we want to focus on. Our goal is to follow and track this object throughout the entire video sequence. We can do so by using a region of interest (ROI) as a template and applying template matching from frame to frame.

#### 2.2.1 Appearance-based Tracking
One approach involves using the entire image region to create **templates** and then applying **template matching**. This process involves creating a template from the initial image, using it to locate the object in the next frame, and then updating the template for further template matching in subsequent frames.

<p align="center">
  <img src="https://github.com/yudhisteer/Real-time-Ego-Tracking-A-Tactical-Solution-for-Rescue-Operations/assets/59663734/a4294f57-94ea-4d66-8863-3dcf4830b165" width="70%" />
</p>

In the example above, we take the grey car in frame ```t-1``` in the red window as a template.  We then apply that template within a search window (green) in the next frame, ```t```. Wherever we find a good match (blue), we declare it as the new position of the object. The condition is that the **change in appearance** of the object between time ```t-1``` and ```t``` is **very small**. However, this method does not handle well large changes in **scale**, **viewpoint**, or **occlusion**.


#### 2.2.2 Histogram-based Tracking

In histogram-based tracking, rather than using the entire image region, we compute a **histogram** - 1-dimensional (grayscale image) or high-dimensional histogram (RGB image). This histogram serves as a **template**, and the tracking process involves **matching** these histograms between images to effectively track the object.

<p align="center">
  <img src="https://github.com/yudhisteer/Real-time-Ego-Tracking-A-Tactical-Solution-for-Rescue-Operations/assets/59663734/d40b0246-bbe9-4191-adbf-16554e7adf93" width="90%" />
</p>

We want to track an object within a region of interest (ROI). However, the reliability of points in the ROI decreases towards the **edges** due to potential **background interference**. To address this, a **weighted histogram**, like the ```Epanechnikov kernel```, is used. This weights pixel contributions differently based on their distance from the center of the window. The weighted histograms are then employed for **matching** between frames, similar to **template matching**. This method, relying on histogram matching, proves more **resilient** to changes in **object pose**, **scale**, **illumination**, and **occlusion** compared to appearance-based template matching.

### 2.3 Tracking-by-Detection

#### 2.3.1 Matching Features using SIFT

1. Given a frame at time ```t-1```, we can either use an object detection algorithm to detect an object and draw a bounding box around it, or we can manually draw a bounding box around an object of interest. 
2. We then compute **SIFT** of similar features for the frame. Note that SIFT has the location and also the descriptor of the features.
3. We classify the features within our bounding box as an **object** and declare them to belong to set ```O```.
4. We then classify the remaining features (outside the bounding box) as **background** and declare them to belong to set ```B```.
5. For the next frame ```t```, we again calculate SIFT **features** and **descriptors**.
6. For each feature and descriptor in frame ```t```, we calculate the distance, ```d_O```, between the current feature and the best matching feature in the object model, ```O```.
7. For each feature and descriptor in frame ```t```, we calculate the distance, ```d_B```, between the current feature and the best matching feature in the background model, ```B```.
8. If ```d_O``` is much smaller than ```d_B``` then we give a confidence value of ```+1``` that the feature belongs to the object else we give a confidence value of ```-1``` that it does not belong to the object.
9. We then take the bounding box in the previous frame ```t-1``` and place it in the current frame ```t```.
10. We will **distort** this window that has changed its position and shape to grab as many object features as possible.
11. We want to find the window for which we have the **largest** number of **object features** inside and a small number of background features such that it becomes the **new position** of the object.
12. Recall that the object may have changed in appearance slightly so we're going to then take the features inside to update the object model, ```O```, and the features outside to update the background model, ```B```.
134. We repeat the process for all the remaining frames and track the object of interest.


#### 2.3.2 Similarity Learning using Siamese Network
SIFT detects and describes local features in images. However, when we have more than one object we need to track, we will need a more robust solution. In my other project, [**Face Recognition with Masks**](https://github.com/yudhisteer/Face-Recognition-with-Masks), I talked about how we could use a Siamese Network to do face recognition. 

- The Siamese Network has **two identical networks** that share the same **architecture** and **weights**.
- Each network takes an input and processes it **independently** to produce an **embedding** of the input data in a **latent space**.
- To **train** the model, we use pairs of images containing the **same** person which would be labeled as **similar**, while pairs of images containing **different** people would be labeled as **dissimilar**.
- We need to learn embeddings such that the distance (Euclidean distance, contrastive loss, or triplet loss) between embeddings of **similar** pairs is **minimized**, while the distance between embeddings of **dissimilar** pairs is **maximized**.
- During **inference**, the Siamese network computes the **embeddings** for each input using the shared networks.
- The **similarity** between the inputs is then determined based on the **distance** between their embeddings in the embedding space.

In the image below we see how when applying a Siamese network, it tells us which car in frame ```t-1``` is matching to which one in frame ```t```. In that way, we will be able to model the appearance of each object and track its trajectory in subsequent frames.

<p align="center">
  <img src="https://github.com/yudhisteer/Real-time-Ego-Tracking-A-Tactical-Solution-for-Rescue-Operations/assets/59663734/cd85c8a9-1bf8-4a8f-a695-a1495eecfe63" width="90%" />
</p>




### 2.4 Cost Function
Later in the Hungarian algorithm section, we will see we will have an algorithm we need to optimize. To do that, we will need a **cost function**. Below we discuss some of them.

#### 2.4.1 Intersection over Union (IoU)
Suppose we have an object we detected in frame ```t-1``` with its corresponding bounding box. For the next frame ```t```, we will again detect the same object with a new bounding box location. Now to make it more complex, suppose we have 2 or more bounding boxes overlapping on the old bounding box in frame ```t-1```. How do we assign a bounding box to the object?

One way to do that is to hypothesize that the bounding box in the subsequent frame will be close to the bounding box in the previous frame. If so, we can select the pair of bounding boxes with the most **overlap**, i.e., ```IoU >= 0.5```. Now since our building boxes may change scale depending on the position of the object, we want to normalize the overlap value hence the formula below:

<p align="center">
  <img src="https://github.com/yudhisteer/Real-time-Multi-Object-Tracking-for-Rescue-Operations/assets/59663734/d88c15a4-3b38-4f30-b4ab-4249531a0fde" />
</p>

The IoU metric is a **similarity measure** that has a range between ```0``` and ```1``` with ```0``` meaning no two bounding boxes intersect and ```1``` meaning a perfect match between detections and predictions. In our scenario, we will try to map tracks (detections at time ```t-1```) and detections at time ```t``` to find an optimal solution for the Hungarian algorithm.

#### 2.4.2 Sanchez-Matilla
More advanced tracking models use various similarity measures categorized into **distance**, **shape**, and **appearance** metrics. These metrics are combined with different **weights** to determine the overall similarity between bounding boxes. **Distance** and **shape** metrics focus on the **position** and **geometry** of the boxes, while **appearance** metrics analyze **pixel** data within each box.

Sanchez-Matilla et al. from the paper [**Online multi-target tracking with strong and weak detections**](http://eecs.qmul.ac.uk/~andrea/papers/2016_ECCVW_MOT_OnlineMTTwithStrongAndWeakDetections_Sanchez-Matilla_Poiesi_Cavallaro.pdf) proposed an affinity measure that promote bounding boxes that are similar both in **shape** and in **position**, rather than in only one aspect. In the formula below ```(H, W)``` are the height and width of the image whereas ```(X_A, Y_A) (X_B, Y_B)``` are the center coordinates of the two bounding boxes.

<p align="center">
  <img src="https://github.com/yudhisteer/Real-time-Multi-Object-Tracking-for-Rescue-Operations/assets/59663734/5e355475-9173-42ab-aa7a-d77aa683dc6c" />
</p>


#### 2.4.3 Yu
Fengwei Yu et al. from the paper [**POI: Multiple Object Tracking with High Performance Detection and Appearance Feature**](https://arxiv.org/abs/1610.06136) proposed another affinity measure that computes the difference in position and shape **exponentially**, by also using **appearance** features. The features are extracted by a CNN with a 128-dimensional output and their **cosine similarity** is computed. Notice that ```w1``` and ```w2``` are **weights** with values of ```0.5``` and ```1.5``` respectively. Similar to the IOU is has a range value between ```0``` and ```1```, denoting zero and complete similarity.

<p align="center">
  <img src="https://github.com/yudhisteer/Real-time-Multi-Object-Tracking-for-Rescue-Operations/assets/59663734/35b8a583-063c-40c6-ac55-d93cc4d0c44c" />
</p>


### 2.4 Metric
The choice of detection accuracy and precision significantly impacts the performance of tracking-by-detection models. Therefore, direct comparisons between trackers using different detectors are less meaningful. Even with similar detection accuracy and precision, tracking model quality can vary widely, necessitating additional metrics.

#### 2.5.1 MOTA
Multiple object tracking accuracy (MOTA) combines false positives, missed targets, and identity switches. In the formula below **FN** is the number of **false negatives**, **FP** is the number of **false positives**, **IDS** is the number of **identity switches** and **GT** is the number of objects present.

<p align="center">
  <img src="https://github.com/yudhisteer/Real-time-Multi-Object-Tracking-for-Rescue-Operations/assets/59663734/ad07dbea-23fe-44d1-b5b2-f34d1b3bfd8e" />
</p>

#### 2.5.2 MOTP
Multiple object tracking precision (MOTP) measures the **misalignment** of the **predicted bounding boxes** and the **ground truth**. ```d_i``` is the **distance** between the ground truth and the prediction of object ```i```, and ```c_t``` is the number of matches found in frame ```t```.

<p align="center">
  <img src="https://github.com/yudhisteer/Real-time-Multi-Object-Tracking-for-Rescue-Operations/assets/59663734/f6ebd4d8-83fd-4ef6-b5c3-2dcfdfea0922" />
</p>

#### 2.5.3 IDF1
IDF1 score is the **ratio** of correctly identified detections over the average of ground truth and predicted detections.  **IDTP** is Identity True Positives, **IDFP** is Identity False Positives, and **IDFN** is Identity False Negatives.

<p align="center">
  <img src="https://github.com/yudhisteer/Real-time-Multi-Object-Tracking-for-Rescue-Operations/assets/59663734/3caa0642-3e2d-46e7-beb1-88be576a858e" />
</p>


---------------
<a name="ha"></a>
## 3. The Hungarian Algorithm
The Hungarian algorithm(Kuhn-Munkres algorithm) is a **combinatorial optimization algorithm** used for solving ```assignment problems```. In the context of object tracking, it is employed to find the optimal association between multiple **tracks** and **detections**, optimizing the **cost** of assignments based on metrics such as **Intersection over Union (IoU)**. But why do we need the Hungarian algorithm? Why don't we choose the highest IOU? Here's why:

- It can deal with cases where not all tracks are associated with detections or vice versa.
- It can handle scenarios with multiple tracks and detections, ensuring coherent and consistent assignments. Suppose for one object we have three IOUs: ```0.29```, ```0.30```, and ```0.31```. If we had to choose the highest IOU we would choose ```0.31``` but this also means that we selected this IOU over the lowest one (```0.29```) with only a difference of ```0.02```. Selecting the highest IOU would be a naive approach.
- It considers all possible associations simultaneously, optimizing the overall assignment of tracks to detections.

Now let's take an example of three bounding boxes as shown below. The **black** ones are the **tracks** at time ```t-1``` and the **red** ones are the **detections** at time ```t```. From the image, we can already see which track will associate with which detection. Note that this is a simple scenario where we have no two or more detections for one track.

<p align="center">
  <img src="https://github.com/yudhisteer/Multi-Object-Tracking-with-Deep-SORT/assets/59663734/00b43dbd-7929-4ec2-9023-09b6a4e47e45" width="70%" />
</p>

The next step will be to calculate the IOU for each combination of detection and track and put them in a matrix as shown below. Again, we can already see a pattern of association emerging for the detection and track from the value of IOU only.

<p align="center">
  <img src="https://github.com/yudhisteer/Multi-Object-Tracking-with-Deep-SORT/assets/59663734/95b5f22f-13bc-48b1-9fe7-ac7db1090f85" width="50%" />
</p>

Below is the step-by-step process of the Hungarian algorithm. We won't need to code it from scratch but use a function from ```scipy```.

<p align="center">
  <img src="https://github.com/yudhisteer/Multi-Object-Tracking-with-Deep-SORT/assets/59663734/38d83258-89c1-424a-ad84-8ec151d62090" width="50%" />
</p>

For our scenario since our metric is IOU, meaning the highest IOU equal to the highest overlap between two bounding boxes, it is a **maximization** problem. Hence, we introduce a **minus** sign in the IOU matrix when putting it as a parameter in the ```linear_sum_assignment``` function.

```python
row_ind, col_ind = linear_sum_assignment(-iou_matrix)
```
We then select an IOU **threshold** ```(0.4```), to filter matches and unmatched items for determining associations between detections and trackings. This threshold allows us to control the level of overlap required for a match. From the results, we may have three possible scenarios:

- **Matches**: Associations between detected objects at time ```t``` and existing tracks from time ```t-1```, indicating the continuity of tracking from one frame to the next.

- **Unmatched Detections**: Detected objects at time ```t``` that do not have corresponding matches with existing tracks from time ```t-1```. These represent newly detected objects or objects for which tracking continuity couldn't be established.

- **Unmatched Trackings**: Existing tracks from time ```t-1``` that do not find corresponding matches with detected objects at time ```t```. This may occur when a tracked object is not detected in the current frame or is incorrectly associated with other objects.

```python
matches = [(old[i], new[j]) for i, j in zip(row_ind, col_ind) if iou_matrix[i, j] >= 0.4]
unmatched_detections = [(new[j]) for i, j in zip(row_ind, col_ind) if iou_matrix[i, j] < 0.4]
unmatched_trackings = [(old[i]) for i, j in zip(row_ind, col_ind) if iou_matrix[i, j] < 0.4]
```
The output:

```python
Matches: [([100, 80, 150, 180], [110, 120, 150, 180]), ([250, 160, 300, 220], [250, 180, 300, 240])]
Unmatched Detections: [[350, 160, 400, 220]]
Unmatched Trackings: [[400, 80, 450, 140]]
```

------------
<a name="s"></a>
## 4. SORT: Simple Online Realtime Tracking
The authors of the SORT paper offer a lean approach for MOT for online and real-time applications. They argue that in the tracking-by-detection method, detection holds a key factor whereby the latter can increase the tracking accuracy by ```18.9%```. SORT focuses on a frame-to-frame prediction using the **Kalman Filter** and association using the **Hungarian algorithm**. Their method achieves speed and accuracy comparable to, at that time, SOTA online trackers. Below is my implementation of the SORT algorithm. Though it is not the same as the official SORT GitHub repo, my approach offers a simpler method with not-so-bad accuracy. Most of the explanations below have been extracted from the [SORT](https://arxiv.org/abs/1602.00763) paper itself and rewritten by me.

### 4.1 Detection
The authors of the SORT paper use a Faster Region CNN - FrRCNN as their object detector. In my implementation, I will use the [YOLOv8m](https://github.com/ultralytics/ultralytics) model. I have created a **yolo_detection** function which takes in as parameters an image, the YOLO model, and the label classes we want to detect.

```python
    # 1. Run YOLO Object Detection to get new detections
    _, new_detections_bbox = yolo_detection(image_copy, model, label_class={'car', 'truck', 'person'})
```

### 4.2 Estimation Model
In the SORT algorithm, a ```first-order four-dimensional (4D) Kalman filter``` is employed for object tracking. Each **tracked object** is represented by a ```4D state vector```, incorporating **position** coordinates and **velocities**. The Kalman filter is initialized upon detection, setting its initial state based on the bounding box. In each frame, the filter **predicts** the object's **next state**, updating the bounding box accordingly. When a detection aligns with a track, the Kalman filter is **updated** using the observed information. In my project [UAV Drone: Object Tracking using Kalman Filter](https://github.com/yudhisteer/UAV-Drone-Object-Tracking-using-Kalman-Filter), I explain more about the Kalman Filter in detail.

```python
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
```

### 4.3 Data Association
When associating detections with existing targets, the algorithm estimates each target's bounding box by predicting its position in the current frame. The assignment cost matrix is computed using the IOU distance, measuring the overlap between detections and predicted bounding boxes. The **Hungarian** algorithm optimally solves the **assignment problem**, with a minimum IOU threshold rejecting assignments with insufficient overlap. Below I wrote an **association** function that computes the Hungarian and returns the indices of match detections, bounding boxes of unmatched detections, and bounding boxes of unmatched trackers as explained in the Hungarian section.

```python
    # 4. Associate new detections bbox (detections) and old obstacles bbox (tracks)
    match_indices, matches, unmatched_detections, unmatched_trackers = association(tracks=old_obstacles_bbox,
                                                                                   detections=new_detections_bbox,
                                                                                  metric_function=metric_total)
```

In this code snippet, for each pair of matched indices representing existing targets and new detections, the algorithm retrieves the ID, bounding box, and age of the old obstacle. It increments the age and creates a new obstacle instance with the corresponding information, including the current time. The Kalman filter predicts the next state. Subsequently, the Kalman filter of the obstacle is then updated with the measurement (bounding box) from the new detection. The obstacle's time is updated, and its bounding box is adjusted according to the Kalman filter's predicted values. Finally, the newly updated obstacle is appended to the list of new obstacles.

```python
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
```


### 4.4 Creation and Deletion of Track Identities

In the code below, for each unmatched detection, a new obstacle is created with a unique ID, using the bounding box coordinates of the unmatched detection and the current time. This ensures that each new detection, not associated with any existing target, is assigned a distinct identifier. The newly created obstacle is then added to the list of new obstacles, and the ID counter is incremented to maintain uniqueness for the next unmatched detection.

```python
    # 6. New (Unmatched) Detections: Give the new detections an id and register their bounding box coordinates
    for unmatched_detection_bbox in unmatched_detections:
        # Create new obstacle with the unmatched detections bounding box
        obstacle = ObstacleSORT(id=id, bbox=unmatched_detection_bbox, time=current_time)
        # Append obstacle to new obstacles list
        new_obstacles.append(obstacle)
        # Update id
        id += 1
```

Here, the unmatched trackers, which represent existing targets that were not successfully matched with any detection in the current frame, are processed. For each unmatched tracker, the corresponding obstacle is retrieved from the list of old obstacles based on the bounding box. The Kalman filter associated with the obstacle is then updated by predicting its state using the state transition matrix and the time difference since the last update. The unmatch age of the obstacle is incremented, indicating how many frames it has remained unmatched. The obstacle's bounding box is also updated, and it is added to the list of new obstacles to continue tracking.

```python
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
```

Tracks are terminated after not being detected for a duration defined by "MAX_UNMATCHED_AGE". It avoids issues where predictions continue for a long time without being corrected by the detector. The author argues that "MAX_UNMATCHED_AGE" is set to ```1``` in experiments for two reasons: the constant velocity model is a poor model of the true dynamics, and secondly, the focus is on frame-to-frame tracking rather than **re-identification**. 

```python
    # Draw bounding boxes of new obstacles with their corresponding id
    for _, obstacle in enumerate(new_obstacles):
        # Remove false negative: Filter out obstacles that have not been detected for a long time ("MAX_UNMATCHED_AGE")
        if obstacle.unmatch_age > MAX_UNMATCHED_AGE:
            new_obstacles.remove(obstacle)

        # Remove false positive: Display detections only when appeared "MIN_HIT_STREAK" times
        if obstacle.age >= MIN_HIT_STREAK:
            x1, y1, x2, y2 = obstacle.bbox
            color = get_rgb_from_id(obstacle.id*20)
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, thickness=cv2.FILLED)
```


--------------
<a name="ds"></a>
## 4. Deep SORT: Simple Online and Realtime Tracking with a Deep Association Metric
With SORT, we were able to model our target's **motion**. That is, predict where the target is and will be. But now, we also want to model our target's **appearance**. That is, what our target looks like. Hence, the author of the DeepSORT paper combines **appearance information** to track objects over a **longer period** even when they are **occluded**, and to **reduce** the number of **identity switches**.  

### 4.1 Track Handling and State Estimation
The authors use a **constant velocity** motion and **linear observation model** such that they take the bounding box coordinates of the object as input parameters. For each track, we track the number of frames since the last successful measurement association. Tracks exceeding a maximum age are removed. New tracks are initiated for unassociated detections. If no successful association is made within the first three frames, these tracks are deleted.

### 4.2 Assignment Problem
Similar to SORT, they use the Hungarian algorithm for the association between the predicted Kalman states and the measurements. However, this time they integrate motion and appearance into their cost function metric. They use the **Mahalanobis distance** to inform about potential object locations based on motion which helps short-term predictions. But also the **cosine distance** which incorporates appearance information for identity recovery during long-term occlusions when motion information is less reliable. This combination allows for **robust association**. In my implementation, I used a weighted combination of **IoU**, **Sanchez-Matilla**, **Yu**, and **cosine similarity** such that we have an association if it is within a certain threshold.

```python
def metric_total_feature(bbox1: Tuple[int, int, int, int],
                         bbox2: Tuple[int, int, int, int],
                         old_features: np.ndarray,
                         new_features: np.ndarray,
                         iou_threshold: float = 0.6,
                         sm_threshold: float = 1e-10,
                         yu_threshold: float = 0.5,
                         feature_threshold: float = 0.98):

    iou_cost = metric_IOU(bbox1, bbox2)
    sm_cost = metric_sanchez_matilla(bbox1, bbox2)
    feature_cost = cosine_similarity(old_features, new_features)[0][0]
    yu_cost = metric_yu(bbox1, bbox2, old_features, new_features)

    # Condition
    if iou_cost > iou_threshold and feature_cost > feature_threshold and sm_cost > sm_threshold and yu_cost > yu_threshold:
        return iou_cost
    # Condition not met
    return 0
```

### 4.3 Matching Cascade
The matching cascade in Deep SORT is a two-stage process for associating detections with existing tracks. In the first stage, simple distance metrics quickly identify potential matches between detections and tracks. The second stage refines these matches using the Hungarian algorithm using the IoU metric. This cascade ensures **accurate** and **robust tracking** by maintaining track identities and handling occlusions effectively. For my implementation, I used the Hungarian algorithm only.

### 4.4 Deep Appearance Descriptor
As mentioned before, we want to model the **appearance** of objects. We first **detect** the objects using YOLOv8 and then **cropped** the objects at their bounding boxes. We use a **Siamese** network to process the **features** of the cropped objects with the shape: ```[#Channels, 128, 128]```. For each object's features in frame ```t-1``` we use a **cosine similarity** to associate to objects' features in frame ```t```.

```python
    # Extract Features
    siamese_model = torch.load("model640.pt", map_location=torch.device('cpu'))
    siamese_model = siamese_model.eval()

    features_1 = extract_features(model=siamese_model, crop_tensor=crop_tensor_1)
    print("Feature 1 shape: ", features_1.shape)
    features_2 = extract_features(model=siamese_model, crop_tensor=crop_tensor_2)
    print("Feature 2 shape: ", features_2.shape)
    cosine_result = cosine_similarity(features_1, features_2)
    print("Cosine result shape: ", cosine_result.shape)
```

```python
    # Association
    match_indices, matches, unmatched_detections, unmatched_trackers = association(tracks=bboxs[0],
                                                                                   detections=bboxs[1],
                                                                                   metric_function=metric_total_feature,
                                                                                   old_features=features_1,
                                                                                   new_features=features_2)
```

In summary, DeepSORT uses an association metric that combines both **motion** and **appearance** descriptors. DeepSORT can be defined as the tracking algorithm that tracks objects not only based on motion but also on their appearance. 

-------
## Conclusion
Below is the result of Deep SORT on footage during the cyclone in Mauritius. We can easily **count** the number of people in the danger areas and **track** their movements when they are being carried away by powerful tides. Although this project shows the implementation of SORT and Deep SORT from scratch, we have more cutting-edge obstacle-tracking algorithms such as **CenterTrack**, **ByteTrack**, **FairMOT**, **Strong SORT**, and so on. 

<div style="text-align: center;">
  <video src="https://github.com/yudhisteer/Real-time-Multi-Object-Tracking-for-Rescue-Operations/assets/59663734/d480ba65-4825-46c3-a200-f7e3d9b00b9c" controls="controls" style="max-width: 730px;">
  </video>
</div>

----------------

## References
[1] Arshren. (n.d.). Hungarian Algorithm. [Medium Article]. [https://arshren.medium.com/hungarian-algorithm-6cde8c4065a3](https://arshren.medium.com/hungarian-algorithm-6cde8c4065a3)

[2] Think Autonomous. (n.d.). Understanding the Hungarian Algorithm. [Blog Post]. [https://www.thinkautonomous.ai/blog/hungarian-algorithm/](https://www.thinkautonomous.ai/blog/hungarian-algorithm/)

[3] Augmented Startups. (n.d.). DeepSORT: Deep Learning Applied to Object Tracking. [Medium Article]. [https://medium.com/augmented-startups/deepsort-deep-learning-applied-to-object-tracking-924f59f99104](https://medium.com/augmented-startups/deepsort-deep-learning-applied-to-object-tracking-924f59f99104)

[4] YouTube. (n.d.). Visual Object Tracking: DeepSORT Explained. [Video]. [https://www.youtube.com/watch?v=QtAYgtBnhws&ab_channel=DynamicVisionandLearningGroup](https://www.youtube.com/watch?v=QtAYgtBnhws&ab_channel=DynamicVisionandLearningGroup)

[5] YouTube. (n.d.). Multiple Object Tracking in Videos. [Video]. [https://www.youtube.com/watch?app=desktop&v=ezSx8OyBZVc&ab_channel=ShokoufehMirzaei](https://www.youtube.com/watch?app=desktop&v=ezSx8OyBZVc&ab_channel=ShokoufehMirzaei)

[6] Brilliant. (n.d.). Hungarian Matching. [Webpage]. [https://brilliant.org/wiki/hungarian-matching/](https://brilliant.org/wiki/hungarian-matching/)

[7] YouTube. (n.d.). Understanding Hungarian Algorithm - AI Coffee Break with Letitia. [Video]. [https://www.youtube.com/watch?v=BLRSIwal7Go&list=PL2zRqk16wsdoYzrWStffqBAoUY8XdvatV&index=12&ab_channel=FirstPrinciplesofComputerVision](https://www.youtube.com/watch?v=BLRSIwal7Go&list=PL2zRqk16wsdoYzrWStffqBAoUY8XdvatV&index=12&ab_channel=FirstPrinciplesofComputerVision)

[8] Learn OpenCV. (n.d.). Object Tracking and Re-identification with FairMOT. [Tutorial]. [https://learnopencv.com/object-tracking-and-reidentification-with-fairmot/](https://learnopencv.com/object-tracking-and-reidentification-with-fairmot/)

[9] Stiefel, S., & Sanchez, A. (2006). An Efficient Implementation of the Hungarian Algorithm for the Assignment Problem. [PDF]. [https://cvhci.anthropomatik.kit.edu/~stiefel/papers/ECCV2006WorkshopCameraReady.pdf](https://cvhci.anthropomatik.kit.edu/~stiefel/papers/ECCV2006WorkshopCameraReady.pdf)

[10] Learn OpenCV. (n.d.). Understanding Multiple Object Tracking using DeepSORT. [Tutorial]. [https://learnopencv.com/understanding-multiple-object-tracking-using-deepsort/](https://learnopencv.com/understanding-multiple-object-tracking-using-deepsort/)

[11] Sanchez-Matilla, R., Poiesi, F., & Cavallaro, A. (2016). Online Multi-Target Tracking Using Recurrent Neural Networks. [PDF]. [https://eecs.qmul.ac.uk/~andrea/papers/2016_ECCVW_MOT_OnlineMTTwithStrongAndWeakDetections_Sanchez-Matilla_Poiesi_Cavallaro.pdf](https://eecs.qmul.ac.uk/~andrea/papers/2016_ECCVW_MOT_OnlineMTTwithStrongAndWeakDetections_Sanchez-Matilla_Poiesi_Cavallaro.pdf)

[12] arXiv. (n.d.). POI: Multiple Object Tracking with High-Performance Detection and Appearance Feature [Research Paper]. [https://arxiv.org/abs/1610.06136](https://arxiv.org/abs/1610.06136)

[13] arXiv. (n.d.). Simple Online and Realtime Tracking with a Deep Association Metric. [Research Paper]. [https://arxiv.org/abs/1610.06136](https://arxiv.org/abs/1610.06136)

[14] arXiv. (n.d.). Simple Online and real-time tracking. [Research Paper]. [https://arxiv.org/abs/1602.00763](https://arxiv.org/abs/1602.00763)


