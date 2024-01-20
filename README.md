# Real-time Ego Tracking: A Tactical Solution for Rescue Operations

## Problem Statement

## Abstract

## Table of Contents
1. [Object Detection](#od)
2. [Object Tracking](#ot)
3. [The Hungarian Algorithm](#ha)
4. [SORT: Simple Online Realtime Tracking](#s)
5. [Deep SORT: Simple Online and Realtime Tracking with a Deep Association Metric](#ds)

------------
<a name="od"></a>
## 1. Object Detection

------------
<a name="ot"></a>
## 2. Object Tracking
In my other project - [Optical Flow Obstacle Avoidance for UAV](https://github.com/yudhisteer/Optical-Flow-Obstacle-Avoidance-for-UAV) - we saw how we can analyze the apparent motion of objects in a sequence of images. We tested with sparse, dense and deep optical flow to eventually see how every point in the scene is moving from frame to frame in a video sequence. In object tracking, we want to track objects or regions from frame to frame. 

### 2.1 Change Detection



https://github.com/yudhisteer/Real-time-Ego-Tracking-A-Tactical-Solution-for-Rescue-Operations/assets/59663734/38a4e14a-6e03-4ead-adee-7a9a13bb01a5



### 2.2 Gaussian Mixture Model

### 2.3 Template Matching

### 2.4 Tracking-by-Detection

### 2.5 Metric

#### 2.5.1 Intersection over Union (IoU)


#### 2.5.2 Sanchez-Matilla

#### 2.5.3 Yu

#### 2.1.4 MOTA

#### 2.1.5 IDF1

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




----------------

## References
1. https://arshren.medium.com/hungarian-algorithm-6cde8c4065a3
2. https://www.thinkautonomous.ai/blog/hungarian-algorithm/
3. https://medium.com/augmented-startups/deepsort-deep-learning-applied-to-object-tracking-924f59f99104
4. https://www.youtube.com/watch?v=QtAYgtBnhws&ab_channel=DynamicVisionandLearningGroup
5. https://www.youtube.com/watch?app=desktop&v=ezSx8OyBZVc&ab_channel=ShokoufehMirzaei
6. https://brilliant.org/wiki/hungarian-matching/
7. https://www.youtube.com/watch?v=BLRSIwal7Go&list=PL2zRqk16wsdoYzrWStffqBAoUY8XdvatV&index=12&ab_channel=FirstPrinciplesofComputerVision
