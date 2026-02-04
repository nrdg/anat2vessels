import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from tqdm import tqdm


def num_neighbors(skeleton):
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    return ndi.convolve(skeleton, kernel, mode="constant", cval=0)


def bifurcation_endpoint_arrays(skeleton):
    neighbor_count = num_neighbors(skeleton)
    bifurcations = neighbor_count > 2
    endpoints = neighbor_count == 1
    bifurcations = bifurcations * skeleton
    endpoints = endpoints * skeleton

    return bifurcations, endpoints


def labeled_branches(skeleton):
    neighbor_count = num_neighbors(skeleton)
    skeleton = skeleton.astype(np.uint8, copy=True)
    bifurcation = neighbor_count > 2
    skeleton[bifurcation] = 0

    struct = np.ones((3, 3, 3), dtype=np.uint8)
    ll, num_branches = ndi.label(skeleton, structure=struct)

    branch_idx = []
    for ii in range(1, num_branches + 1):
        if np.sum(ll == ii) < 2:
            ll[ll == ii] = 0
        else:
            branch_idx.append(ii)
    return ll, branch_idx


def extract_radius(segmentation, centerlines, voxel_spacing):
    image = segmentation
    skeleton = centerlines
    transf = ndi.distance_transform_edt(
        image, return_indices=False, sampling=voxel_spacing
    )
    radius_matrix = transf * skeleton
    radius_matrix = radius_matrix[skeleton > 0]
    return radius_matrix


def skeleton(segmentation):
    return skeletonize(segmentation, method="lee").astype(np.uint8, copy=False)


def tortuosities_and_lengths(labeled_branches, branch_labels, vox_spacing):
    branches = []
    for i in tqdm(branch_labels):
        branch = (labeled_branches == i).astype(np.int8)
        points = _get_points_in_order(branch, vox_spacing)

        if len(points) == 0:
            continue
        if len(points) == 1:
            branches.append({"tortuosity": 1, "full_path": 1, "straight_path": 1})
        else:
            full_path = _calc_full_path_from_points(points)

            straight_path = _calc_shortest_path_from_points(points)

            if straight_path > full_path:
                print(
                    f"WARNING: Branch {i} full path: {full_path} < straight path: {straight_path}"
                )

            tortuosity = full_path / straight_path
            branches.append(
                {
                    "tortuosity": tortuosity,
                    "full_path": full_path,
                    "straight_path": straight_path,
                }
            )
    return branches


def _get_branch_array_by_label(labeled_branches, branch_label):
    return (labeled_branches == branch_label).astype(np.int8)


def _get_points_in_order(branch, vox_spacing):
    path_cords = np.argwhere(branch == 1)
    endpoints = np.argwhere(bifurcation_endpoint_arrays(branch)[1] == 1)

    if len(endpoints) == 1:
        return endpoints
    if len(endpoints) < 1:
        return []
    remaining_cords = set(map(tuple, path_cords))
    current_pos = tuple(endpoints[0])
    remaining_cords.remove(current_pos)

    array = [np.array(current_pos) * vox_spacing]

    while remaining_cords:
        neighbors = [
            (current_pos[0] + dx, current_pos[1] + dy, current_pos[2] + dz)
            for dx in [-1, 0, 1]
            for dy in [-1, 0, 1]
            for dz in [-1, 0, 1]
            if (dx, dy, dz) != (0, 0, 0)
        ]

        next_pos = next((n for n in neighbors if tuple(n) in remaining_cords), None)

        if next_pos is None:
            raise ValueError("Invalid path, disconnected branch")

        array.append(np.array(next_pos) * vox_spacing)

        current_pos = next_pos
        remaining_cords.remove(next_pos)

    return np.array(array)


def _calc_full_path_from_points(points):
    if len(points) < 2:
        return 1
    length = 0
    for i in range(0, len(points) - 1):
        diff = np.array(points[i + 1]) - np.array(points[i])
        length += np.linalg.norm(diff)

    return length


def _calc_shortest_path_from_points(points):
    if len(points) < 2:
        return 1
    diff = np.array(points[0]) - np.array(points[-1])
    length = np.linalg.norm(diff)
    return length
