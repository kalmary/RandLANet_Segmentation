import numpy as np
import torch

def rotate_points(points: torch.Tensor, device = torch.device('cpu')) -> torch.Tensor:
    """
    Rotates 3D points around the Z-axis (up-axis) by a given angle using PyTorch tensors.

    Args:
        points (torch.Tensor): A (N, 3) PyTorch tensor of points, where columns are (x, y, z).
                               Expected dtype is torch.float32 or torch.float64.
        angle_degrees (float): Rotation angle in degrees.

    Returns:
        torch.Tensor: The rotated points as a (N, 3) PyTorch tensor.
                      The dtype will match the input 'points' tensor.
    """
    angle_degrees = torch.rand(1) * 360.
    angle_degrees = angle_degrees.item()

    # Convert angle from degrees to radians using PyTorch's function
    angle_radians = torch.deg2rad(torch.tensor(angle_degrees, dtype=points.dtype))

    # Calculate cosine and sine of the angle
    cos_theta = torch.cos(angle_radians)
    sin_theta = torch.sin(angle_radians)

    # Define the 3x3 rotation matrix for rotation around the Z-axis
    # Ensure the dtype matches the input points tensor for consistency
    rotation_matrix = torch.tensor([
        [cos_theta, -sin_theta, 0.0],
        [sin_theta, cos_theta,  0.0],
        [0.0,       0.0,        1.0]
    ], dtype=points.dtype).to(device)

    # Apply rotation using tensor matrix multiplication
    # The @ operator performs matrix multiplication for PyTorch tensors
    # We transpose the rotation_matrix because 'points' is (N, 3) and we want to multiply
    # each point (row vector) by the rotation matrix.
    # Alternatively, you could do torch.matmul(points, rotation_matrix.T)
    # or if points were (3, N), then torch.matmul(rotation_matrix, points)
    points = points @ rotation_matrix.T
    return points

def tilt_points(
    points: torch.Tensor,
    max_x_tilt_degrees: float = 10.0,
    max_y_tilt_degrees: float = 10.0,
    device = torch.device('cpu')
) -> torch.Tensor:
    """
    Rotates 3D points randomly around the X and Y axes within specified tilt limits.

    Args:
        points (torch.Tensor): A (N, 3) PyTorch tensor of points, where columns are (x, y, z).
                               Expected dtype is torch.float32 or torch.float64.
        max_x_tilt_degrees (float): Maximum absolute rotation angle in degrees for the X-axis.
                                    The actual angle will be sampled uniformly from
                                    [-max_x_tilt_degrees, +max_x_tilt_degrees].
        max_y_tilt_degrees (float): Maximum absolute rotation angle in degrees for the Y-axis.
                                    The actual angle will be sampled uniformly from
                                    [-max_y_tilt_degrees, +max_y_tilt_degrees].

    Returns:
        torch.Tensor: The randomly tilted points as a (N, 3) PyTorch tensor.
                      The dtype will match the input 'points' tensor.
    """
    dtype = points.dtype

    # Generate random angles for X and Y axes
    # torch.rand(1) generates a number between 0 and 1.
    # (rand * 2 * max_deg) - max_deg maps it to [-max_deg, +max_deg]
    rand_x_angle_degrees = (torch.rand(1, dtype=dtype) * 2 * max_x_tilt_degrees - max_x_tilt_degrees).item()
    rand_y_angle_degrees = (torch.rand(1, dtype=dtype) * 2 * max_y_tilt_degrees - max_y_tilt_degrees).item()

    # Convert angles from degrees to radians
    angle_x_radians = torch.deg2rad(torch.tensor(rand_x_angle_degrees, dtype=dtype))
    angle_y_radians = torch.deg2rad(torch.tensor(rand_y_angle_degrees, dtype=dtype))

    cos_x = torch.cos(angle_x_radians)
    sin_x = torch.sin(angle_x_radians)
    cos_y = torch.cos(angle_y_radians)
    sin_y = torch.sin(angle_y_radians)

    # Define the 3x3 rotation matrix for X-axis (Roll)
    rotation_matrix_x = torch.tensor([
        [1.0, 0.0,   0.0],
        [0.0, cos_x, -sin_x],
        [0.0, sin_x, cos_x]
    ], dtype=dtype).to(device)

    # Define the 3x3 rotation matrix for Y-axis (Pitch)
    rotation_matrix_y = torch.tensor([
        [cos_y,  0.0, sin_y],
        [0.0,    1.0, 0.0],
        [-sin_y, 0.0, cos_y]
    ], dtype=dtype).to(device)

    # Apply rotations sequentially
    # Order matters: typically apply X, then Y, then Z or vice-versa depending on convention.
    # Here, we'll apply X-rotation first, then Y-rotation.
    points = points @ rotation_matrix_x.T
    points = points @ rotation_matrix_y.T

    return points

def transform_points(
    points: torch.Tensor,
    min_scale: float = 0.9,
    max_scale: float = 1.1,
    device = torch.device('cpu')
) -> torch.Tensor:
    
    """
    Applies a random scaling transformation to 3D points.

    Args:
        points (torch.Tensor): A (N, 3) PyTorch tensor of points, where columns are (x, y, z).
                               Expected dtype is torch.float32 or torch.float64.
        min_scale (float): Minimum scaling factor to apply.
        max_scale (float): Maximum scaling factor to apply.

    Returns:
        torch.Tensor: The scaled points as a (N, 3) PyTorch tensor.
                      The dtype will match the input 'points' tensor.
    
    """

    dtype = points.dtype

    scale_x = (torch.rand(1, dtype=dtype) * (max_scale - min_scale) + min_scale).item()
    scale_y = (torch.rand(1, dtype=dtype) * (max_scale - min_scale) + min_scale).item()
    scale_z = (torch.rand(1, dtype=dtype) * (max_scale - min_scale) + min_scale).item()

    rotation_matrix = torch.tensor([
        [scale_x, 0.,   0.],
        [0., scale_y, 0.],
        [0., 0., scale_z]
    ], dtype=dtype).to(device)
    points = points @ rotation_matrix.T
    return points
