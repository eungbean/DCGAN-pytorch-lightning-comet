import albumentations as A
import functools
import numpy as np
import random
from torchvision import transforms

'''
Image Augmentation with Albumentation.
https://github.com/albumentations-team/albumentations_examples/blob/master/notebooks/example.ipynb
'''

def get_augmentation(_C, is_train):
    """
    """
    if is_train:
        augmentation = [
            # random flip
            A.HorizontalFlip(
                p=_C.TRANSFORM.TRAIN_HORIZONTAL_FLIP_PROB
            ),
            A.VerticalFlip(
                p=_C.TRANSFORM.TRAIN_VERTICAL_FLIP_PROB
            ),
            # random rotate
            A.ShiftScaleRotate(
                scale_limit=0.0,
                rotate_limit=_C.TRANSFORM.TRAIN_RANDOM_ROTATE_DEG,
                shift_limit=0.0,
                p=_C.TRANSFORM.TRAIN_RANDOM_ROTATE_PROB,
                border_mode=0),
            # random crop
            A.RandomCrop(
                width=_C.TRANSFORM.TRAIN_RANDOM_CROP_SIZE[0],
                height=_C.TRANSFORM.TRAIN_RANDOM_CROP_SIZE[1],
                p=_C.TRANSFORM.TRAIN_RANDOM_CROP_PROB,
            ),
            # speckle noise
            A.Lambda(
                image=functools.partial(
                    _random_speckle_noise,
                    speckle_std=_C.TRANSFORM.TRAIN_SPECKLE_NOISE_STD,
                    p=_C.TRANSFORM.TRAIN_SPECKLE_NOISE_PROB
                )
            ),
            # blur
            A.OneOf([
                A.MotionBlur(p=_C.TRANSFORM.TRAIN_BLUR_MOTION_PROB),
                A.MedianBlur(blur_limit=_C.TRANSFORM.TRAIN_BLUR_MEDIAN_LIMIT, p=_C.TRANSFORM.TRAIN_BLUR_MEDIAN_PROB),
                A.Blur(blur_limit=_C.TRANSFORM.TRAIN_BLUR_LIMIT, p=_C.TRANSFORM.TRAIN_BLUR_PROB),
                ], p=_C.TRANSFORM.TRAIN_BLUR_ONEOF),

            # random brightness
            A.Lambda(
                image=functools.partial(
                    _random_brightness,
                    brightness_std=_C.TRANSFORM.TRAIN_RANDOM_BRIGHTNESS_STD,
                    p=_C.TRANSFORM.TRAIN_RANDOM_BRIGHTNESS_PROB
                ),
            # transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            # ])
            ),
        ]
    else:
        augmentation = [
            A.PadIfNeeded(
                min_width=_C.TRANSFORM.TEST_SIZE[0],
                min_height=_C.TRANSFORM.TEST_SIZE[1],
                always_apply=True,
                border_mode=0
            )
        ]
    return A.Compose(augmentation)


def _random_speckle_noise(image, speckle_std, p=1.0, **kwargs):
    """
    """
    if speckle_std <= 0:
        return image

    if random.random() >= p:
        return image

    im_shape = image.shape
    gauss = np.random.normal(0, speckle_std, im_shape)
    gauss = gauss.reshape(im_shape)
    speckle_noise = gauss * image
    noised = image + speckle_noise

    return noised


def _random_brightness(image, brightness_std, p=1.0, **kwargs):
    """
    """
    if brightness_std <= 0:
        return image

    if random.random() >= p:
        return image

    gauss = np.random.normal(0, brightness_std)
    brightness_noise = gauss * image
    noised = image + brightness_noise

    return noised
