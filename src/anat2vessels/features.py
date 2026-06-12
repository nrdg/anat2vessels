import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from tqdm import tqdm


def _load_seg_spacing(nifti_path):
    """Load a binary segmentation and its voxel spacing from a NIfTI file.

    Parameters
    ----------
    nifti_path : str
        Path to a NIfTI file containing a binary vessel segmentation.

    Returns
    -------
    segmentation : np.ndarray
        Boolean array of the segmentation.
    voxel_spacing : tuple of float
        Voxel dimensions (mm) along each axis.
    """
    img = nib.load(nifti_path)
    voxel_spacing = img.header.get_zooms()[:3]
    segmentation = img.get_fdata().astype(bool)
    return segmentation, voxel_spacing


def _get_bifurcation_endpoint_arrays(skeleton):
    """Identify bifurcation and endpoint voxels in a skeleton.

    Bifurcations have >2 neighbors; endpoints have exactly 1 neighbor.

    Parameters
    ----------
    skeleton : np.ndarray
        Binary skeleton array.

    Returns
    -------
    bifurcations : np.ndarray
        Binary array marking bifurcation voxels.
    endpoints : np.ndarray
        Binary array marking endpoint voxels.
    """
    num_neighbors = _get_num_neighbors(skeleton)

    bifurcations = num_neighbors > 2
    endpoints = num_neighbors == 1

    bifurcations = bifurcations * skeleton
    endpoints = endpoints * skeleton

    return bifurcations, endpoints


def _get_labeled_branches(skeleton):
    """Label connected branch segments after removing bifurcation voxels.

    Bifurcation voxels are zeroed so each connected component is a single
    unbranched vessel segment. Segments with fewer than 2 voxels are discarded.

    Parameters
    ----------
    skeleton : np.ndarray
        Binary skeleton array.

    Returns
    -------
    labeled_branches : np.ndarray
        Integer array where each branch has a unique label.
    branch_names : list of int
        Valid branch labels (segments with >=2 voxels).
    """
    neighbor_count = _get_num_neighbors(skeleton)
    skeleton = skeleton.astype(np.uint8, copy=True)

    bifurcation = neighbor_count > 2
    skeleton[bifurcation] = 0

    struct = np.ones((3, 3, 3), dtype=np.uint8)
    labeled_branches, num_branches = ndi.label(skeleton, structure=struct)

    branch_names = []
    for i in range(1, num_branches + 1):
        if np.sum(labeled_branches == i) < 2:
            labeled_branches[labeled_branches == i] = 0
        else:
            branch_names.append(i)
    return labeled_branches, branch_names


def _extract_radius(segmentation, centerlines, voxel_spacing):
    """Extract vessel radii at centerline voxels using a distance transform.

    Parameters
    ----------
    segmentation : np.ndarray
        Binary vessel segmentation.
    centerlines : np.ndarray
        Binary skeleton (centerlines) of the segmentation.
    voxel_spacing : tuple of float
        Voxel dimensions (mm).

    Returns
    -------
    radius_matrix : np.ndarray
        Radius values at centerline voxels (zeros elsewhere).
    """
    image = segmentation
    skeleton = centerlines
    transf = ndi.distance_transform_edt(
        image, return_indices=False, sampling=voxel_spacing
    )
    radius_matrix = transf * skeleton
    radius_matrix = radius_matrix[skeleton > 0]
    return radius_matrix


def _extract_skeleton(segmentation):
    """Skeletonize a binary segmentation using the Lee method.

    Parameters
    ----------
    segmentation : np.ndarray
        Binary segmentation array.

    Returns
    -------
    skeleton : np.ndarray
        Binary skeleton (uint8).
    """
    image = segmentation
    skeleton = skeletonize(image, method="lee")
    return skeleton.astype(np.uint8, copy=False)


def _get_num_neighbors(skeleton):
    """Count the number of 26-connected neighbors for each foreground voxel.

    Parameters
    ----------
    skeleton : np.ndarray
        Binary skeleton array.

    Returns
    -------
    num_neighbors : np.ndarray
        Integer array of neighbor counts (same shape as input).
    """
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    num_neighbors = ndi.convolve(skeleton, kernel, mode="constant", cval=0)
    return num_neighbors


def _calc_tortuosities_also_lengths(labeled_branches, branch_labels, vox_spacing):
    """Compute tortuosity, full path, and straight-line length per branch.

    Parameters
    ----------
    labeled_branches : np.ndarray
        Integer array of labeled branches.
    branch_labels : list of int
        Branch labels to process.
    vox_spacing : tuple of float
        Voxel dimensions (mm).

    Returns
    -------
    branches : list of dict
        Each dict has keys ``tortuosity``, ``full_path``, ``straight_path``.
    """
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
                    f"WARNING: Branch {i} full path: {full_path}"
                    f" < straight path: {straight_path}"
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
    """Extract a single branch as a binary array.

    Parameters
    ----------
    labeled_branches : np.ndarray
        Integer array of labeled branches.
    branch_label : int
        Label of the branch to extract.

    Returns
    -------
    branch_array : np.ndarray
        Binary array (int8) with ones at the target branch.
    """
    return (labeled_branches == branch_label).astype(np.int8)


def _get_points_in_order(branch, vox_spacing):
    """Walk along a branch from one endpoint to the other, returning ordered points.

    Parameters
    ----------
    branch : np.ndarray
        Binary array of a single branch.
    vox_spacing : tuple of float
        Voxel dimensions (mm).

    Returns
    -------
    points : np.ndarray
        (N, 3) array of ordered point coordinates in physical space (mm).

    Raises
    ------
    ValueError
        If the branch skeleton is disconnected.
    """
    path_cords = np.argwhere(branch == 1)
    endpoints = np.argwhere(_get_bifurcation_endpoint_arrays(branch)[1] == 1)

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
    """Compute the cumulative path length along a sequence of points.

    Parameters
    ----------
    points : np.ndarray
        (N, 3) array of ordered points.

    Returns
    -------
    length : float
        Cumulative Euclidean distance along the path. Returns 1 for
        single-point or empty inputs to avoid division by zero.
    """
    if len(points) < 2:
        return 1
    length = 0
    for i in range(0, len(points) - 1):
        diff = np.array(points[i + 1]) - np.array(points[i])
        length += np.linalg.norm(diff)

    return length


def _calc_shortest_path_from_points(points):
    """Compute the straight-line distance between the first and last point.

    Parameters
    ----------
    points : np.ndarray
        (N, 3) array of ordered points.

    Returns
    -------
    length : float
        Euclidean distance between endpoints. Returns 1 for single-point
        or empty inputs to avoid division by zero.
    """
    if len(points) < 2:
        return 1
    diff = np.array(points[0]) - np.array(points[-1])
    length = np.linalg.norm(diff)
    return length


def compute_features(segmentation, voxel_spacing):
    """Run the full feature extraction pipeline on a binary segmentation array.

    Computes skeleton, bifurcations, endpoints, radii, branch statistics,
    tortuosity, and total volume.

    Parameters
    ----------
    segmentation : np.ndarray
        Binary vessel segmentation array.
    voxel_spacing : tuple of float
        Voxel dimensions (mm).

    Returns
    -------
    features : dict
        Keys include ``branch_list``, ``bifurcations``, ``endpoints``,
        ``radius_list``, ``total_volume``, ``num_branches``.
    """
    skeleton = _extract_skeleton(segmentation).astype(np.uint8)
    bifurcations, endpoints = _get_bifurcation_endpoint_arrays(skeleton)
    radius_matrix = _extract_radius(segmentation, skeleton, voxel_spacing)
    radius_list = radius_matrix[np.nonzero(radius_matrix)].tolist()
    labeled_branches, branch_labels = _get_labeled_branches(skeleton)
    branches = _calc_tortuosities_also_lengths(
        labeled_branches, branch_labels, voxel_spacing
    )
    return {
        "branch_list": branches,
        "bifurcations": bifurcations,
        "endpoints": endpoints,
        "radius_list": radius_list,
        "total_volume": float(segmentation.sum() * np.prod(voxel_spacing)),
        "num_branches": len(branch_labels),
    }


def extract_features(nifti_path):
    """Load a NIfTI segmentation and compute all vessel features.

    Convenience wrapper around :func:`compute_features`.

    Parameters
    ----------
    nifti_path : str
        Path to a NIfTI file containing a binary vessel segmentation.

    Returns
    -------
    features : dict
        See :func:`compute_features` for details.
    """
    segmentation, voxel_spacing = _load_seg_spacing(nifti_path)
    return compute_features(segmentation, voxel_spacing)
