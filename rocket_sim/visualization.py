"""
Shared plotting/visualization helpers used by both the standalone
animation script (examples/animate_ascent.py) and the Streamlit app
(app.py), so the rocket marker only needs to be defined once.
"""

from matplotlib.path import Path
from matplotlib.markers import MarkerStyle
from matplotlib.transforms import Affine2D


def rocket_marker(angle_deg):
    """
    Build a small rocket-silhouette marker (nose cone + body + two fins)
    that points "up" by default and is rotated to match a given flight
    path angle.

    Parameters
    ----------
    angle_deg : float
        Flight path angle, degrees (90 = straight up, matching the
        marker's default orientation). The marker is rotated by
        (angle_deg - 90) degrees to reflect the vehicle's actual pitch.
    """
    verts = [
        (0.00, 1.00),
        (0.25, 0.35),
        (0.22, -0.55),
        (0.45, -1.00),
        (0.10, -0.65),
        (0.00, -0.80),
        (-0.10, -0.65),
        (-0.45, -1.00),
        (-0.22, -0.55),
        (-0.25, 0.35),
        (0.00, 1.00),
    ]
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
    path = Path(verts, codes)
    transform = Affine2D().rotate_deg(angle_deg - 90)
    return MarkerStyle(path, transform=transform)
