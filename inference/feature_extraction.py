import numpy as np
import nibabel as nib
from nibabel.processing import resample_to_output
from scipy import ndimage as ndi
from skimage.morphology import skeletonize
from tqdm import tqdm

def _get_skel_seg_spacing(nifit_path):
    nifit_img = nib.load(nifit_path)
    voxel_spacing = nifit_img.header.get_zooms()[:3]

    segmentation = nifit_img.get_fdata().astype(bool)
    skeleton = _extract_skeleton(segmentation).astype(np.uint8)

    return skeleton, segmentation, voxel_spacing

def _get_bifurcation_endpoint_arrays(skeleton):
    num_neighbors = _get_num_neighbors(skeleton)

    bifurcations = num_neighbors > 2
    endpoints = num_neighbors == 1

    bifurcations = bifurcations * skeleton
    endpoints = endpoints * skeleton

    return bifurcations, endpoints

def _get_labeled_branches(skeleton):
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
    image = segmentation
    skeleton = centerlines
    transf = ndi.distance_transform_edt(image, return_indices=False, sampling=voxel_spacing)
    radius_matrix = transf*skeleton
    radius_matrix = radius_matrix[skeleton > 0]
    return radius_matrix

def _extract_skeleton(segmentation):
    image = segmentation
    skeleton = skeletonize(image, method='lee')
    return skeleton.astype(np.uint8, copy=False)

def _get_num_neighbors(skeleton):
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    num_neighbors = ndi.convolve(skeleton, kernel, mode='constant', cval=0)
    return num_neighbors

def _calc_tortuosities_also_lengths(labeled_branches, branch_labels, vox_spacing):
    branches = []
    for i in tqdm(branch_labels):
        branch = (labeled_branches == i).astype(np.int8)
        points = _get_points_in_order(branch, vox_spacing)

        if len(points) == 0:
            continue
        if len(points) == 1:
            branches.append({'tortuosity': 1, 'full_path': 1, 'straight_path': 1})
        else:
            full_path = _calc_full_path_from_points(points)

            straight_path = _calc_shortest_path_from_points(points)

            if straight_path > full_path:
                print(f'WARNING: Branch {i} full path: {full_path} < straight path: {straight_path}')

            tortuosity = full_path / straight_path
            branches.append({'tortuosity': tortuosity, 'full_path': full_path, 'straight_path': straight_path})
    return branches

def _get_branch_array_by_label(labeled_branches, branch_label):
    return (labeled_branches == branch_label).astype(np.int8)

def _get_points_in_order(branch, vox_spacing):
    path_cords = np.argwhere(branch == 1)
    endpoints = np.argwhere(_get_bifurcation_endpoint_arrays(branch)[1] == 1)

    if len(endpoints) == 1:
        return endpoints
    if len(endpoints) < 1:
        return []
    remaining_cords = set(map(tuple, path_cords))
    current_pos = tuple(endpoints[0])
    remaining_cords.remove(current_pos)

    array = [np.array(current_pos)*vox_spacing]

    while remaining_cords:
        neighbors = [
            (current_pos[0] + dx, current_pos[1] + dy, current_pos[2] + dz)
            for dx in [-1, 0, 1] for dy in [-1, 0, 1] for dz in [-1, 0, 1]
            if (dx, dy, dz) != (0, 0, 0)
        ]

        next_pos = next(
            (n for n in neighbors if tuple(n) in remaining_cords), None
        )

        if next_pos is None:
            raise ValueError("Invalid path, disconnected branch")

        array.append(np.array(next_pos)*vox_spacing)

        current_pos = next_pos
        remaining_cords.remove(next_pos)

    return np.array(array)

def _calc_full_path_from_points(points):
    if len(points) < 2:
        return 1
    length = 0
    for i in range(0, len(points)-1):
        diff = np.array(points[i+1]) - np.array(points[i])
        length += np.linalg.norm(diff)

    return length

def _calc_shortest_path_from_points(points):
    if len(points) < 2:
        return 1
    diff = np.array(points[0]) - np.array(points[-1])
    length = np.linalg.norm(diff)
    return length


def extract_features(nifti_path):
    out = {}

    skeleton, segmentation, voxel_spacing = _get_skel_seg_spacing(nifti_path)

    bifurcations, endpoints = _get_bifurcation_endpoint_arrays(skeleton)

    # out['bifurcations'], out['endpoints'] = float(bifurcations.sum()), endpoints.sum()
    #
    # out['total_volume'] = float(segmentation.sum() * np.prod(voxel_spacing))
    ##
    radius_matrix = _extract_radius(segmentation, skeleton, voxel_spacing)
    radius_list = radius_matrix[np.nonzero(radius_matrix)].tolist()
    # out['radius_list'] = radius.tolist()
    # out['mean_radius'] = float(radius.sum() / skeleton.sum())
    # out['max_radius'] = float(radius.max())
    # out['min_radius'] = float(radius.min())

    labeled_branches, branch_labels = _get_labeled_branches(skeleton)

    # out['num_branches'] = len(branch_labels)

    branches = _calc_tortuosities_also_lengths(labeled_branches, branch_labels, voxel_spacing)

    # tortiousities = [branch['tortuosity'] for branch in branches]
    # out['mean_tortuosity'] = np.mean(tortiousities)
    # out['max_tortuosity'] = np.max(tortiousities)
    # out['min_tortuosity'] = np.min(tortiousities)
    # out['tortuosity_list'] = tortiousities

    branch_list = [{'full_path': float(branch['full_path']),
                    'straight_path': float(branch['straight_path']),
                    'tortuosity': float(branch['tortuosity'])} for branch in branches]
    # out['branch_list'] = branch_list
    # out['branch_lengths_list'] = [float(branch['full_path']) for branch in branches]
    # out['total_branch_length'] = np.sum(out['branch_lengths_list'])
    # out['mean_branch_length'] = np.mean(out['branch_lengths_list'])
    # out['max_branch_length'] = np.max(out['branch_lengths_list'])

    '''
    Stuff we actually do:
    radius_list
    branches
    bifurcations
    endpoints
    '''
    out = {
        'branch_list': branches,
        'bifurcations': bifurcations,
        'endpoints': endpoints,
        'radius_list': radius_list,
        'total_volume': float(segmentation.sum() * np.prod(voxel_spacing)),
        'num_branches': len(branch_labels)
        }
    return out