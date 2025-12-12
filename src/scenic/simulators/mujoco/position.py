
import math

def position_to_mujoco(pos):
    """Convert a position from Scenic coordinates to MuJoCo coordinates.
    Args:
        pos: A tuple or list of three floats representing the position in Scenic coordinates (x, y, z).

    Returns:
        A tuple of three floats representing the position in MuJoCo coordinates (x, y, z).
    """
    return pos