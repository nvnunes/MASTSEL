"""Every PSF is shown when positions do not form a complete grid."""
from types import SimpleNamespace
import matplotlib.pyplot as plt
import numpy as np
from mastsel.mavisUtilities import tiledDisplay


def test_irregular_109_direction_field_preserves_every_image(monkeypatch):
    monkeypatch.setattr(plt, 'show', lambda: None)
    results = [SimpleNamespace(sampling=np.full((3, 3), k + 1.)) for k in range(109)]
    coords = np.column_stack((np.arange(109), np.sin(np.arange(109))))
    tiledDisplay(results, coords)
    fig = plt.gcf()
    try:
        images = [axis.images[0] for axis in fig.axes if axis.get_visible()]
        assert len(images) == 109
        assert all(img.origin == 'lower' for img in images)
        assert sorted(float(img.get_array()[0, 0]) for img in images) == sorted(np.log(np.arange(1, 110)))
    finally:
        plt.close(fig)
