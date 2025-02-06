from typing import List, Tuple

import numpy as np


def calculate_patch_starts(dimension_size: int, patch_size: int, overlap: int) -> List[int]:
    """
    Calculate the starting positions of patches along a single dimension
    with specified overlap to cover the entire dimension.

    Parameters:
    -----------
    dimension_size : int
        Size of the dimension
    patch_size : int
        Size of the patch in this dimension
    overlap : int
        Overlap size (must be an integer less than patch size)

    Returns:
    --------
    List[int]
        List of starting positions for patches
    """
    if dimension_size <= patch_size:
        return [0]

    stride = patch_size - overlap
    positions = list(range(0, dimension_size - patch_size + 1, stride))

    # Ensure the last patch covers the end
    if positions[-1] + patch_size < dimension_size:
        positions.append(dimension_size - patch_size)

    return positions

def make_weight_3d(d, h, w, front=0, top=0, left=0, back=0, bottom=0, right=0):
    """
    Create a 3D weight map with linear gradients for overlapping regions.
    
    Parameters:
    -----------
    d : int
        Depth of the weight map
    h : int
        Height of the weight map
    w : int
        Width of the weight map
    top : int
        Number of rows at the top to create a gradient
    left : int
        Number of columns on the left to create a gradient
    bottom : int
        Number of rows at the bottom to create a gradient
    right : int
        Number of columns on the right to create a gradient
    front : int
        Number of slices at the front to create a gradient
    back : int
        Number of slices at the back to create a gradient

    Returns:
    --------
    np.ndarray
        A 3D weight map
    """
    weight = np.full((d, h, w), fill_value=1)

    return weight


def extract_3d_patches_minimal_overlap(
    arrays: List[np.ndarray], 
    patch_size_z: int,
    patch_size_x: int,
    patch_size_y: int,
    overlap_z: int,
    overlap_x: int,
    overlap_y: int
) -> Tuple[List[np.ndarray], List[Tuple[int, int, int]]]:
    """
    Extract 3D patches from input arrays with specified overlap.

    Parameters:
    -----------
    arrays : List[np.ndarray]
        List of 3D arrays to extract patches from
    patch_size_x, patch_size_y, patch_size_z : int
        Size of the patches in each dimension
    overlap : int
        Overlap size (must be an integer less than patch size)

    Returns:
    --------
    Tuple[List[np.ndarray], List[Tuple[int, int, int]]]
        Extracted patches and their coordinates
    """
    if not arrays or not isinstance(arrays, list):
        raise ValueError("Input must be a non-empty list of arrays")

    # Verify all arrays have the same shape
    shape = arrays[0].shape
    if not all(arr.shape == shape for arr in arrays):
        raise ValueError("All input arrays must have the same shape")

    l, m, n = shape
    patches = []
    coordinates = []

    # Calculate starting positions for each dimension
    z_starts = calculate_patch_starts(l, patch_size_z, overlap_z)
    x_starts = calculate_patch_starts(m, patch_size_x, overlap_x)
    y_starts = calculate_patch_starts(n, patch_size_y, overlap_y)
    
    # Extract patches from each array
    for arr in arrays:
        for z in z_starts:
            for x in x_starts:
                for y in y_starts:
                    patch = arr[z : z + patch_size_z, x : x + patch_size_x, y : y + patch_size_y]
                    patches.append(patch)
                    coordinates.append((z, x, y))

    return patches, coordinates

def reconstruct_array(
    patches: List[np.ndarray],
    coordinates: List[Tuple[int, int, int]],
    original_shape: Tuple[int, int, int, int],
    patch_size: Tuple[int, int, int]
) -> np.ndarray:
    """
    Reconstruct array from patches with weighted averaging for overlapping regions.

    Parameters:
    -----------
    patches : List[np.ndarray]
        List of patches to reconstruct from
    coordinates : List[Tuple[int, int, int]]
        Starting coordinates for each patch
    original_shape : Tuple[int, int, int, int]
        Shape of the original array
    patch_size : Tuple[int, int, int]
        Size of the patches in each dimension (z, x, y)

    Returns:
    --------
    np.ndarray
        Reconstructed array
    """
    reconstructed = np.zeros(original_shape, dtype=np.float32)
    weight_map = np.zeros(original_shape, dtype=np.float32)

    patch_size_z, patch_size_x, patch_size_y = patch_size

    for patch, (z, x, y) in zip(patches, coordinates):
        # Generate the 3D weight map for the current patch (local size, based on the patch position)
        patch_weight = make_weight_3d(
            patch_size_z, patch_size_x, patch_size_y,
            front=y, top=z, left=x,
            back=y + patch_size_y,
            bottom=z + patch_size_z,
            right=x + patch_size_x
        )

        reconstructed[:, z : z + patch_size_z, x : x + patch_size_x, y : y + patch_size_y] += patch * patch_weight
        weight_map[:, z : z + patch_size_z, x : x + patch_size_x, y : y + patch_size_y] += patch_weight

    reconstructed /= weight_map

    return reconstructed
