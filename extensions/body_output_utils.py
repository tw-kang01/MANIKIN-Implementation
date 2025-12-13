"""
Body Output Type Handling
Python 버전별 body_model 반환 타입 처리
"""

import torch
import numpy as np


def get_joint_positions(body_output):
    """
    Extract joint positions from body_output
    Handles dict/object differences across Python versions
    
    Args:
        body_output: output from BodyModel()
    
    Returns:
        np.ndarray: (T, 22, 3) joint positions
    """
    # Try different access patterns
    if hasattr(body_output, 'Jtr'):
        joints = body_output.Jtr
    elif isinstance(body_output, dict) and 'Jtr' in body_output:
        joints = body_output['Jtr']
    elif hasattr(body_output, 'joints'):
        joints = body_output.joints
    else:
        raise AttributeError(
            f"Cannot extract joints from {type(body_output)}. "
            f"Available attributes: {dir(body_output)}"
        )
    
    # Convert to numpy
    if isinstance(joints, torch.Tensor):
        joints = joints.cpu().numpy()
    
    return joints[:, :22, :]


def get_vertices(body_output):
    """
    Extract vertices for mesh rendering
    
    Args:
        body_output: output from BodyModel()
    
    Returns:
        np.ndarray: (T, 6890, 3) vertices
    """
    if hasattr(body_output, 'v'):
        verts = body_output.v
    elif isinstance(body_output, dict) and 'v' in body_output:
        verts = body_output['v']
    elif hasattr(body_output, 'vertices'):
        verts = body_output.vertices
    else:
        raise AttributeError(
            f"Cannot extract vertices from {type(body_output)}. "
            f"Available attributes: {dir(body_output)}"
        )
    
    if isinstance(verts, torch.Tensor):
        verts = verts.cpu().numpy()
    
    return verts


def get_faces(body_model):
    """
    Extract face indices from body model
    
    Args:
        body_model: BodyModel instance
    
    Returns:
        np.ndarray: (N, 3) face indices
    """
    if hasattr(body_model, 'f'):
        faces = body_model.f
    elif hasattr(body_model, 'faces'):
        faces = body_model.faces
    else:
        raise AttributeError(
            f"Cannot extract faces from body model. "
            f"Available attributes: {dir(body_model)}"
        )
    
    if isinstance(faces, torch.Tensor):
        faces = faces.cpu().numpy()
    
    return faces


def safe_to_numpy(tensor_or_array):
    """
    Safely convert torch tensor or numpy array to numpy array
    
    Args:
        tensor_or_array: torch.Tensor or np.ndarray
    
    Returns:
        np.ndarray
    """
    if isinstance(tensor_or_array, torch.Tensor):
        return tensor_or_array.cpu().numpy()
    elif isinstance(tensor_or_array, np.ndarray):
        return tensor_or_array
    else:
        return np.array(tensor_or_array)


class BodyOutputWrapper:
    """
    Makes dict-based body_output compatible with attribute access
    
    This wrapper allows code expecting body_output.v or body_output.Jtr
    to work with dict-based body_output from Python 3.13+
    
    Usage:
        body_output = bm(**body_parms)
        wrapped = BodyOutputWrapper(body_output)
        vertices = wrapped.v  # Works regardless of dict/object type
    """
    def __init__(self, body_output):
        if isinstance(body_output, dict):
            # Dict-based: copy all items as attributes
            for key, value in body_output.items():
                setattr(self, key, value)
        else:
            # Object-based: copy common attributes
            for key in ['v', 'Jtr', 'f', 'joints', 'vertices']:
                if hasattr(body_output, key):
                    setattr(self, key, getattr(body_output, key))

