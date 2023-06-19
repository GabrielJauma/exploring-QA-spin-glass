import numpy as np
from numba import njit, prange


# %%
@njit(fastmath=True)
def dist_from_node(D):
    size = D.shape[0]
    l_max = np.max(D)
    ls = np.arange(1, l_max+1)
    nodes_at_l = np.zeros(len(ls), dtype='int64')

    for k in range(len(ls)):
        for i in range(size):
            nodes_at_l[k] += np.sum(D[i, :] <= ls[k]) -1
        nodes_at_l[k] /= size
    return ls, nodes_at_l


def box_counting(D):
    size = D.shape[0]
    l_max = np.max(D)
    ls = np.arange(2, l_max)
    n_boxes_vs_l = np.zeros(len(ls), dtype='int64')
    boxes_vs_l = [[]] * len(ls)

    for k in range(len(ls)):
        uncovered = np.ones(size, dtype='int64')
        boxes = []
        while np.any(uncovered == 1):
            n_boxes_vs_l[k] += 1
            box = []
            box.append(np.random.choice(np.where(uncovered == 1)[0]))
            # box.append(np.where(uncovered == 1)[0][0])
            uncovered[box[0]] = 0
            for i in range(size):
                box_distances = np.empty(len(box), dtype='int64')
                for j in range(len(box)):
                    if D[i, box[j]] == 0:
                        box_distances[j] = 30000  # Hardcoded value to handle non connected nodes
                    else:
                        box_distances[j] = D[i, box[j]]
                if uncovered[i] == 1 and np.all(box_distances < ls[k]):
                    uncovered[i] = 0
                    box.append(i)
            boxes.append(box)
        boxes_vs_l[k] = boxes

    # ls = np.arange(1, l_max + 1)
    # n_boxes_vs_l = np.append(n_boxes_vs_l, 1)
    # n_boxes_vs_l = np.append(size, n_boxes_vs_l)
    return ls, n_boxes_vs_l, boxes_vs_l


@njit(fastmath=True)
def box_counting_numba(D):
    size = D.shape[0]
    l_max = np.max(D)
    ls = np.arange(2, l_max+1)
    n_boxes_vs_l = np.zeros(len(ls), dtype='int64')

    for k in range(len(ls)):
        uncovered = np.ones(size, dtype='int64')
        while np.any(uncovered == 1):
            n_boxes_vs_l[k] += 1
            box = []
            box.append(np.random.choice(np.where(uncovered == 1)[0]))
            # box.append(np.where(uncovered == 1)[0][0])
            uncovered[box[0]] = 0
            for i in range(size):
                box_distances = np.empty(len(box), dtype='int64')
                for j in range(len(box)):
                    if D[i, box[j]] == 0:
                        box_distances[j] = 30000  # Hardcoded value to handle non connected nodes
                    else:
                        box_distances[j] = D[i, box[j]]
                if uncovered[i] == 1 and np.all(box_distances < ls[k]):
                    uncovered[i] = 0
                    box.append(i)
    return ls, n_boxes_vs_l


# This is broken FUCK
@njit(fastmath=True)
def compact_box_counting_numba(D):
    size = D.shape[0]
    l_max = np.max(D)

    ls = np.arange(2, l_max+1)
    n_boxes_vs_l = np.zeros(len(ls), dtype='int64')

    for k in range(len(ls)):
        uncovered = np.ones(size, dtype='uint8')
        while np.any(uncovered == 1):
            # box = []
            n_boxes_vs_l[k] += 1
            candidates = uncovered.copy()
            while np.any(candidates == 1):
                candidates_indices = np.where(candidates == 1)[0]
                seed_node = np.random.choice(candidates_indices)
                # seed_node = candidates_indices[0]
                # box.append(seed_node)
                uncovered[seed_node] = 0
                candidates[seed_node] = 0
                for i in candidates_indices:
                    if D[seed_node, i] >= ls[k] or D[seed_node, i] == 0:  # Non connected nodes are at 0 distance
                        candidates[i] = 0
            # Check box
            # box_distances = np.zeros((len(box), len(box)), dtype='int64')
            # for i in range(len(box)):
            #     for j in range(len(box)):
            #         box_distances[i, j] = D[box[i], box[j]]
            #     if np.any(box_distances >= ls[k]):
            #         return 0

    # ls = np.arange(1, l_max + 1)
    # n_boxes_vs_l = np.append(n_boxes_vs_l, 1)
    return ls, n_boxes_vs_l


# def maximum_excluded_mass_burning_numba(D):



def dimension(D):
    ls, n_boxes_vs_l = box_counting_numba(D)
    # ls, n_boxes_vs_l = compact_box_counting_numba(D)
    # ls, n_boxes_vs_l = dimension_numba_fast(D)
    # return np.polyfit(-np.log(ls[1:]), np.log(n_boxes_vs_l[1:]), 1)[0]
    return np.polyfit(-np.log(ls), np.log(n_boxes_vs_l), 1)[0]
