import numpy as np
from scipy.optimize import linear_sum_assignment
from utils import metric_IOU
from typing import List, Tuple, Callable, Any


# Calculate Hungarian
def hungarian(tracks: List[List[int]], detections: List[List[int]],
              metric_function: Callable[[List[int], List[int]], float]) -> tuple[
              list[tuple[Any, Any]], list[tuple[list[int] | list[list[int]],
              list[int] | list[list[int]]]],
              list[list[int] | list[list[int]] | Any],
              list[list[int] | list[list[int]] | Any]]:

    # Generate empty iou matrix
    iou_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)

    for i, old_box in enumerate(tracks):
        for j, new_box in enumerate(detections):
            iou_matrix[i][j] = metric_function(old_box, new_box)
    # print("IOU Matrix: ", "\n", iou_matrix)

    # Call the Hungarian Algorithm
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    hungarian_matrix = np.column_stack((row_ind, col_ind))

    # Initialize lists for matches, unmatched detections, and unmatched trackers
    matches, unmatched_detections, unmatched_trackers = [], [], []

    # Indices of matches from frame 1 and frame 2
    # [[frame 1, frame 2]
    #  [frame 1, frame 2]]
    match_indices = np.array([(h[0], h[1]) for h in hungarian_matrix if iou_matrix[h[0], h[1]] >= 0.5])

    matches = [(tracks[h[0]], detections[h[1]]) for h in hungarian_matrix if iou_matrix[h[0], h[1]] >= 0.5]
    unmatched_trackers.extend([tracks[h[0]] for h in hungarian_matrix if iou_matrix[h[0], h[1]] < 0.5])
    unmatched_detections.extend([detections[h[1]] for h in hungarian_matrix if iou_matrix[h[0], h[1]] < 0.5])

    # Check if matches_indices is empty
    if len(match_indices) == 0:
        matches_indices = np.empty((0, 2), dtype=int)

    # Add unmatched trackers
    unmatched_trackers.extend([trk for t, trk in enumerate(tracks) if t not in hungarian_matrix[:, 0]])

    # Add unmatched detections
    unmatched_detections.extend([det for d, det in enumerate(detections) if d not in hungarian_matrix[:, 1]])

    return match_indices, matches, unmatched_detections, unmatched_trackers












if __name__ == "__main__":
    # Detections at time 0
    Trk_1 = [768, 272, 823, 332]
    Trk_2 = [1001, 136, 1023, 158]
    Trk_3 = [1479, 607, 1647, 746]

    # Detections at time 1
    Det_1 = [769, 272, 823, 332]
    Det_2 = [866, 191, 910, 234]
    Det_3 = [1477, 607, 1647, 746]


    tracks = [Trk_1, Trk_2, Trk_3]
    detections = [Det_1, Det_2, Det_3]

    match_indices, matches, unmatched_detections, unmatched_trackers = hungarian(tracks=tracks,
                                                                   detections=detections,
                                                                   metric_function=metric_IOU)

    # Print the results
    print("Size: ", len(match_indices), "| Match Indices:", match_indices)
    print("Size: ", len(matches), "| Matches:", matches)
    print("Size: ", len(unmatched_detections), "| Unmatched Detections:", unmatched_detections)
    print("Size: ", len(unmatched_trackers), "| Unmatched Trackers:", unmatched_trackers)