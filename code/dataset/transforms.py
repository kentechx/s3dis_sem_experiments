import numpy as np
from scipy.spatial.transform import Rotation
from typing import TypedDict


def exists(val):
    return val is not None


def default(*vals):
    for val in vals:
        if exists(val):
            return val


class Inputs(TypedDict):
    xyz: np.ndarray
    rgb: np.ndarray
    label: np.ndarray


class Transform:
    def __call__(self, inputs: Inputs):
        raise NotImplementedError


class TransformCompose(Transform):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, inputs: Inputs):
        for transform in self.transforms:
            inputs = transform(inputs)
        return inputs


# --------------------
# point clouds
# --------------------
class Jitter(Transform):
    def __init__(self, sigma=0.001, clip=0.005, p=0.5):
        self.sigma = sigma
        self.clip = clip
        self.p = p

    def __call__(self, inputs: Inputs):
        if np.random.rand() < self.p:
            xyz = inputs['xyz']
            inputs['xyz'] = xyz + np.clip(self.sigma * np.random.randn(*xyz.shape), -self.clip, self.clip)
        return inputs


class Rotate(Transform):
    def __init__(self, angle=(0, 0, 15.), p=0.5):
        self.angle = angle
        self.p = p

    def __call__(self, inputs: Inputs):
        # xyz: (n, 3)
        if np.random.rand() < self.p:
            angles = np.random.uniform(-1., 1., 3) * np.array(self.angle)
            R = Rotation.from_euler('xyz', angles, degrees=True).as_matrix()
            inputs['xyz'] = inputs['xyz'] @ R
        return inputs


class AnisotropicScale(Transform):
    def __init__(self, scale=(0.8, 1.2), p=0.5):
        self.scale = scale
        self.p = p

    def __call__(self, inputs: Inputs):
        # xyz: (n, 3)
        if np.random.rand() < self.p:
            xyz = inputs['xyz']
            scale = np.random.uniform(self.scale[0], self.scale[1], size=xyz.shape[-1])
            inputs['xyz'] = xyz * scale
        return inputs


class XYZAlign(Transform):
    def __init__(self, gravity_dim=2):
        self.gravity_dim = gravity_dim

    def __call__(self, inputs: Inputs):
        # xyz: (n, 3)
        xyz = inputs['xyz']
        xyz -= xyz.mean(axis=0, keepdims=True)
        xyz[:, self.gravity_dim] -= np.min(xyz[:, self.gravity_dim])
        inputs['xyz'] = xyz
        return inputs


# --------------------
# colors
# --------------------
class ColorAutoContrast(Transform):
    def __init__(self, p=0.2, blend_factor=None):
        self.p = p
        self.blend_factor = blend_factor

    def __call__(self, inputs: Inputs):
        if np.random.rand() < self.p:
            rgb = inputs['rgb']
            lo = np.min(rgb, 0, keepdims=True)
            hi = np.max(rgb, 0, keepdims=True)
            scale = 255 / np.maximum(hi - lo, 1.)
            contrast_feat = (rgb - lo) * scale
            blend_factor = default(self.blend_factor, np.random.rand())
            rgb = (1 - blend_factor) * rgb + blend_factor * contrast_feat
            inputs['rgb'] = rgb
        return inputs


class ColorAllDrop(Transform):
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, inputs: Inputs):
        if np.random.rand() < self.p:
            inputs['rgb'][:] = 0
        return inputs


class ColorDrop(Transform):
    def __init__(self, p=0.2):
        self.p = p

    def __call__(self, inputs: Inputs):
        rgb = inputs['rgb']
        mask = np.random.rand(rgb.shape[0]) < self.p
        rgb[mask] = 0
        inputs['rgb'] = rgb
        return inputs


class ColorNormalize(Transform):
    def __init__(self,
                 color_mean=[0.5136457, 0.49523646, 0.44921124],
                 color_std=[0.18308958, 0.18415008, 0.19252081]):
        self.color_mean = color_mean
        self.color_std = color_std

    def __call__(self, inputs: Inputs):
        rgb = inputs['rgb']
        if rgb.max() > 1:
            rgb /= 255.

        rgb = (rgb - self.color_mean) / self.color_std
        inputs['rgb'] = rgb
        return inputs


if __name__ == '__main__':
    inputs = Inputs(
        xyz=np.random.rand(100, 3),
        rgb=np.random.rand(100, 3) * 255,
        label=np.random.randint(0, 10, (100, 1))
    )
    transform = TransformCompose([
        Jitter(p=1),
        Rotate(p=1),
        AnisotropicScale(p=1),
        XYZAlign(),
        ColorAutoContrast(p=1),
        ColorDrop(p=0.5),
        ColorNormalize()
    ])
    inputs = transform(inputs)
