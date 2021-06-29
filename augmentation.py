import numpy as np
import albumentations as albu
# from tensorflow.python.util.tf_export import API_ATTRS


def get_augmentation(TRANSFER=False):
    """ Make albumentation's augmentation sequences.

        Args:
            TRANSFER (bool, optional): If you train model to work with other data, set True. Defaults to False.

        Returns:
            albu.Compose(transform): Augmentation sequences.
    """

    if TRANSFER:
        transform = [
            
            albu.OneOf([ # Distortion
                albu.OpticalDistortion(p=1),
                albu.GridDistortion(p=1),
                # albu.ElasticTransform(alpha=100,p=1)
            ],p=1),
            
            albu.OneOf([ # Color
                albu.CLAHE(p=1),
                albu.RandomGamma(p=1),
                albu.RandomBrightnessContrast(p=1),
            ],p=1),
            
            albu.OneOf([
                albu.Sequential([
                    albu.HorizontalFlip(p=1),
                    albu.ShiftScaleRotate(shift_limit=0.2,scale_limit=0.1,rotate_limit=[-40,-50],p=1), 
                ]),
                albu.Sequential([
                    albu.ShiftScaleRotate(shift_limit=0.2,scale_limit=0.1,rotate_limit=10,p=1), # Affine transformation（Shift, Scale, Rotation）
                ])
            ],p=1)
        ]
        return albu.Compose(transform)
        
    else:
        transform = [
            albu.OneOf([ # Distortion
                albu.OpticalDistortion(p=1),
                albu.GridDistortion(p=1),
                # albu.ElasticTransform(alpha=100,p=1)
            ],p=1),
            albu.OneOf([ # Color
                albu.CLAHE(),
                albu.RandomGamma(p=1),
                albu.RandomBrightnessContrast(p=1),
            ],p=1),
            
            albu.Rotate(limit=10,p=1),
        ]
        return albu.Compose(transform)


def data_augmentation(X,Y,times=10, TRANSFER=False):
    
    transforms = get_augmentation(TRANSFER)
    
    X_aug = []
    Y_aug = []
    for i in range(len(X)):
        for _ in range(times):
            augmented = transforms(image=X[i],mask=Y[i])
            X_aug.append(augmented['image'])
            Y_aug.append(augmented['mask'])
            
    return np.array(X_aug),np.array(Y_aug)    


##### For self-custom augmentation #####

# from albumentations.core.transforms_interface import ImageOnlyTransform

# class Reflection(ImageOnlyTransform):
#     def __init__(self, m=1, always_apply=False, p=0.5):
#         super(Reflection, self).__init__(always_apply, p)
#         self.m = m
        
#     def apply(self, img, **params):
#         return self.reflection(img, self.m)
    
#     def reflection(self, x, m):
#         """
#         https://python.atelierkobato.com/symmetric/
#         """
#         return np.fliplr(np.flipud(x.T))
    

""" Augmentation Example

    References
    - https://qiita.com/Takayoshi_Makabe/items/79c8a5ba692aa94043f7
    - https://idiotdeveloper.com/data-augmentation-for-semantic-segmentation-deep-learning/
    - https://qiita.com/kurilab/items/b69e1be8d0224ae139ad
    - http://maruo51.com/2020/07/22/albumentations/
    - https://github.com/katsura-jp/tour-of-albumentations

    # Distortion
    - OptivalDistortion
    - GridDistortion
    - ElasticTransform

    # Color
    - Random Gamma
    - Random Brightness
    - Random Contrast
    - Random Brightness Contrast
    - CLAHE

    # Dropout
    - Cutout
    - CoarseDropout

    def strong_aug(p=0.5):
        return Compose([
            RandomRotate90(),
            Flip(),
            Transpose(),
            OneOf([
                IAAAdditiveGaussianNoise(),
                GaussNoise(),
            ], p=0.2),
            OneOf([
                MotionBlur(p=0.2),
                MedianBlur(blur_limit=3, p=0.1),
                Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
            OneOf([
                OpticalDistortion(p=0.3),
                GridDistortion(p=0.1),
                IAAPiecewiseAffine(p=0.3),
            ], p=0.2),
            OneOf([
                CLAHE(clip_limit=2),
                IAASharpen(),
                IAAEmboss(),
                RandomBrightnessContrast(),
            ], p=0.3),
            HueSaturationValue(p=0.3),
        ], p=p)

"""