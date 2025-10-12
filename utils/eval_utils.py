"""Utilities for evaluation.
"""
import os
import glob
import numpy as np
import cv2
import torchvision.utils as vutils
import torch


def read_image(path):
    """Read image from the given path.

    Args:
        path (str): The path of the image.

    Returns:
        array: RGB image.
    """
    # RGB
    img = cv2.imread(path)[:, :, ::-1]
    return img


def read_seq_images(path):
    """Read a sequence of images.

    Args:
        path (str): The path of the image sequence.

    Returns:
        array: (N, H, W, C) RGB images.
    """
    imgs_path = sorted(glob.glob(os.path.join(path, '*')))
    imgs = [read_image(img_path) for img_path in imgs_path]
    imgs = np.stack(imgs, axis=0)
    return imgs


def index_generation(num_output_frames, num_GT):
    """Generate index list for evaluation. 
    Each list contains num_output_frames indices.

    Args:
        num_output_frames (int): Number of output frames.
        num_GT (int): Number of ground truth.

    Returns:
        list[list[int]]: A list of indices list for testing.
    """

    indices_list = []
    right = num_output_frames
    while (right <= num_GT):
        indices = list(range(right - num_output_frames, right))
        indices_list.append(indices)
        right += num_output_frames - 1

    # Check if the last frame is included
    if right - num_output_frames < num_GT - 1:
        indices = list(range(num_GT - num_output_frames, num_GT))
        indices_list.append(indices)
    return indices_list


def write_images(image_outputs, display_image_num, file_name):
    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs]  # expand gray-scale images to 3 channels
    image_tensor = torch.cat([images[:display_image_num] for images in image_outputs], 0)
    image_grid = vutils.make_grid(image_tensor.data, nrow=display_image_num, padding=0, normalize=False)
    vutils.save_image(image_grid, file_name, nrow=1)
