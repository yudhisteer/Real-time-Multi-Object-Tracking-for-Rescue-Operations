# Multi-Object Tracking with DeepSORT

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


### 2.1 Metric: IOU

```python
def calculate_IOU(bbox1: List[int], bbox2: List[int]):

    # Get max and min coordinates 
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    # Calculate the distance between coordinates
    width = x2 - x1
    height = y2 - y1

    # Condition to see if there is an overlap
    if width <= 0 or height <= 0:
        # No overlap
        iou = 0
        return iou

    # Calculate overlap area
    overlap = width * height

    # Calculate union area
    area_bbox1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area_bbox2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = area_bbox1 + area_bbox2 - overlap
    
    # Calculate IOU
    iou = round(overlap/union_area,5)
    
    return iou
```

---------------
<a name="ha"></a>
## 3. The Hungarian Algorithm

<p align="center">
  <img src="https://github.com/yudhisteer/Multi-Object-Tracking-with-Deep-SORT/assets/59663734/00b43dbd-7929-4ec2-9023-09b6a4e47e45" width="70%" />
</p>




<p align="center">
  <img src="https://github.com/yudhisteer/Multi-Object-Tracking-with-Deep-SORT/assets/59663734/38d83258-89c1-424a-ad84-8ec151d62090" width="50%" />
</p>

```python
row_ind, col_ind = linear_sum_assignment(-iou_matrix)
```

```python
matches = [(old[i], new[j]) for i, j in zip(row_ind, col_ind) if iou_matrix[i, j] >= 0.4]
unmatched_detections = [(new[j]) for i, j in zip(row_ind, col_ind) if iou_matrix[i, j] < 0.4]
unmatched_trackings = [(old[i]) for i, j in zip(row_ind, col_ind) if iou_matrix[i, j] < 0.4]
```

```python
Matches: [([100, 80, 150, 180], [110, 120, 150, 180]), ([250, 160, 300, 220], [250, 180, 300, 240])]
Unmatched Detections: [[350, 160, 400, 220]]
Unmatched Trackings: [[400, 80, 450, 140]]
```


<p align="center">
  <img src="https://github.com/yudhisteer/Multi-Object-Tracking-with-Deep-SORT/assets/59663734/95b5f22f-13bc-48b1-9fe7-ac7db1090f85" width="50%" />
</p>


----------------

## References
1. https://arshren.medium.com/hungarian-algorithm-6cde8c4065a3
2. https://www.thinkautonomous.ai/blog/hungarian-algorithm/
3. https://medium.com/augmented-startups/deepsort-deep-learning-applied-to-object-tracking-924f59f99104
4. https://www.youtube.com/watch?v=QtAYgtBnhws&ab_channel=DynamicVisionandLearningGroup
5. https://www.youtube.com/watch?app=desktop&v=ezSx8OyBZVc&ab_channel=ShokoufehMirzaei
6. https://brilliant.org/wiki/hungarian-matching/
7. https://www.youtube.com/watch?v=BLRSIwal7Go&list=PL2zRqk16wsdoYzrWStffqBAoUY8XdvatV&index=12&ab_channel=FirstPrinciplesofComputerVision
