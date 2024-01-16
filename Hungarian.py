import numpy as np
from scipy.optimize import linear_sum_assignment
from utils import calculate_IOU

# Detections at time 0
Trk_1 = [100, 80, 150, 180]
Trk_2 = [250, 160, 300, 220]
Trk_3 = [400, 80, 450, 140]

# Detections at time 1
Det_1 = [110, 120, 150, 180]
Det_2 = [250, 180, 300, 240]
Det_3 = [350, 160, 400, 220]




if __name__ == "__main__":
    old = [Trk_1, Trk_2, Trk_3]
    new = [Det_1, Det_2, Det_3]
    print(old)
    print(new)

    iou_matrix = np.zeros((len(old), len(new)), dtype=np.float32)

    for i, old_box in enumerate(old):
        for j, new_box in enumerate(new):
            iou_matrix[i][j] = calculate_IOU(old_box, new_box)

    print(iou_matrix)

    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    print(row_ind, col_ind)

    cost = iou_matrix[row_ind, col_ind].sum()
    print("Cost: ", cost)

    # Interpret the results
    matches = [(old[i], new[j]) for i, j in zip(row_ind, col_ind) if iou_matrix[i, j] >= 0.4]
    unmatched_detections = [(new[j]) for i, j in zip(row_ind, col_ind) if iou_matrix[i, j] < 0.4]
    unmatched_trackings = [(old[i]) for i, j in zip(row_ind, col_ind) if iou_matrix[i, j] < 0.4]

    # Print the results
    print("Matches:", matches)
    print("Unmatched Detections:", unmatched_detections)
    print("Unmatched Trackings:", unmatched_trackings)
