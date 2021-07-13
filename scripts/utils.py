import cv2
#from efficientnet_pytorch import EfficientNet
from fastai.vision.all import *
from functools import wraps
from albumentations.core.transforms_interface import BasicTransform,ImageOnlyTransform
from albumentations import functional as FF
import albumentations as A


def _parent_idxs(items, name):
    def _inner(items, name): return mask2idxs(Path(o).parent.name == name for o in items)
    return [i for n in L(name) for i in _inner(items,n)]


def ParentSplitter(train_name='train', valid_name='valid'):
    "Split `items` from the grand parent folder names (`train_name` and `valid_name`)."
    def _inner(o):
        return _parent_idxs(o, train_name),_parent_idxs(o, valid_name)
    return _inner



class AlbumentationsTransform(DisplayedTransform):
    split_idx,order=0,2
    def __init__(self, train_aug): store_attr()
    
    def encodes(self, img: PILImage):
        aug_img = self.train_aug(image=np.array(img))['image']
        return PILImage.create(aug_img)


def preserve_shape(func):
    """
    Preserve shape of the image
    """

    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        shape = img.shape
        result = func(img, *args, **kwargs)
        result = result.reshape(shape)
        return result

    return wrapped_function
@preserve_shape
def downscale(img, scale, interpolation=cv2.INTER_CUBIC):
    h, w = img.shape[:2]

    need_cast = interpolation != cv2.INTER_CUBIC and img.dtype == np.uint8
    if need_cast:
        img = to_float(img)
    downscaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    upscaled = cv2.resize(downscaled, (w, h), interpolation=interpolation)
    if need_cast:
        upscaled = from_float(np.clip(upscaled, 0, 1), dtype=np.dtype("uint8"))
    return upscaled

class Downscale(ImageOnlyTransform):
    """Decreases image quality by downscaling and upscaling back.
    Args:
        scale_min (float): lower bound on the image scale. Should be < 1.
        scale_max (float):  lower bound on the image scale. Should be .
        interpolation: cv2 interpolation method. cv2.INTER_NEAREST by default
    Targets:
        image
    Image types:
        uint8, float32
    """

    def __init__(
        self,
        scale_min=0.25,
        scale_max=0.25,
        interpolation=cv2.INTER_CUBIC,
        always_apply=False,
        p=0.5,
    ):
        super(Downscale, self).__init__(always_apply, p)
        if scale_min > scale_max:
            raise ValueError("Expected scale_min be less or equal scale_max, got {} {}".format(scale_min, scale_max))
        if scale_max >= 1:
            raise ValueError("Expected scale_max to be less than 1, got {}".format(scale_max))
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.interpolation = interpolation

    def apply(self, image, scale, interpolation, **params):
        return FF.downscale(image, scale=scale, interpolation=interpolation)

    def get_params(self):
        return {
            "scale": np.random.uniform(self.scale_min, self.scale_max),
            "interpolation": self.interpolation,
        }

    def get_transform_init_args_names(self):
        return "scale_min", "scale_max", "interpolation"