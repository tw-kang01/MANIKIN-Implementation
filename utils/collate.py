"""
Custom collate function for MANIKIN DataLoader

Handles batching of:
- bone_lengths: dict with scalar values per sample → dict with (B,) tensors
- filename: list of strings
- All tensor fields: stacked into batch dimension
"""

import torch
from torch.utils.data._utils.collate import default_collate


def manikin_collate_fn(batch):
    """
    Custom collate function for MANIKIN dataset

    Handles:
    - bone_lengths: dict per sample → dict with (B,) tensors
    - filename: list of strings (not batched)
    - Standard tensors: stacked along batch dimension

    Args:
        batch: list of dicts from MANIKINDataset.__getitem__()

    Returns:
        dict: batched data ready for model.feed_data()
    """
    if len(batch) == 0:
        return {}

    result = {}
    elem = batch[0]

    for key in elem:
        if key == 'bone_lengths':
            # Stack bone_lengths dict values into (B,) tensors
            # Input: each sample has {'left_humerus': 0.3, 'left_radius': 0.25, ...}
            # Output: {'left_humerus': tensor([0.3, 0.3, ...]), ...}
            bone_keys = batch[0]['bone_lengths'].keys()
            result['bone_lengths'] = {
                bk: torch.tensor([b['bone_lengths'][bk] for b in batch], dtype=torch.float32)
                for bk in bone_keys
            }

        elif key == 'filename':
            # Keep filenames as list of strings
            result['filename'] = [b['filename'] for b in batch]

        elif isinstance(elem[key], torch.Tensor):
            # Stack tensors along batch dimension
            result[key] = torch.stack([b[key] for b in batch], dim=0)

        elif isinstance(elem[key], (int, float)):
            # Convert scalars to tensor
            result[key] = torch.tensor([b[key] for b in batch])

        elif elem[key] is None:
            # Keep None values
            result[key] = None

        else:
            # Default collation for other types
            try:
                result[key] = default_collate([b[key] for b in batch])
            except Exception:
                # If default collate fails, keep as list
                result[key] = [b[key] for b in batch]

    return result


def manikin_test_collate_fn(batch):
    """
    Collate function for test mode (batch_size=1)

    Simpler handling since we process one sample at a time.
    Still converts bone_lengths to tensor format for consistency.

    Args:
        batch: list with single dict from MANIKINDataset.__getitem__()

    Returns:
        dict: data ready for model.feed_data()
    """
    if len(batch) == 0:
        return {}

    # For test mode with batch_size=1, we still need consistent format
    return manikin_collate_fn(batch)
