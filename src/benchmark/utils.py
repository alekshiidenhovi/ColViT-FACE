import os
from collections import defaultdict


def estimate_flops(n_vectors: int, vector_dim: int, n_operations: int = 1) -> int:
    """Estimate FLOPs for vector operations.
    
    Args:
        n_vectors: Number of vectors processed
        vector_dim: Dimension of each vector
        n_operations: Number of operations per element (default: 1)
        
    Returns:
        Estimated number of FLOPs
    """
    return n_vectors * vector_dim * n_operations


def partition_lfw_images(dir_path: str, max_images_per_identity: int):
    """Partition LFW images into identities with a single image and those with multiple images. Return also the corresponding image paths.
    """
    
    all_identities = [
        d for d in os.listdir(dir_path)
        if os.path.isdir(os.path.join(dir_path, d))
    ]
    all_image_paths = []
    all_image_path_to_identity = {}
    test_identities = []
    test_image_paths = []
    test_image_path_to_identity = {}
    
    for identity in all_identities:
        identity_path = os.path.join(dir_path, identity)
        image_file_names = [img for img in os.listdir(identity_path) 
                  if img.lower().endswith((".jpg", ".jpeg", ".png"))]
        image_file_names = image_file_names[:max_images_per_identity]
        if len(image_file_names) > 1:
            test_identities.append(identity)
            test_image_paths.extend([os.path.join(identity_path, img_file_name) for img_file_name in image_file_names])
            for img_file_name in image_file_names:
                image_path = os.path.join(identity_path, img_file_name)
                test_image_path_to_identity[image_path] = identity
        all_identities.append(identity)
        for img_file_name in image_file_names:
            image_path = os.path.join(identity_path, img_file_name)
            all_image_paths.append(image_path)
            all_image_path_to_identity[image_path] = identity
    
    return all_identities, all_image_paths, all_image_path_to_identity, test_identities, test_image_paths, test_image_path_to_identity