import os
import cv2
import labelme
import json
import tifffile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageEnhance


def crop_image(X,kind,TB=True):
    """ Cut off the area outside the designated area.

    Args:
        X (numpy array): Images.
        kind (str): "GC" or "pa" classes.
        TB (bool, optional): Cut off the extra vertical area. Defaults to True.

    Returns:
        X (numpy array): Cropped images.
    """
    
    assert X.ndim >= 3, "Must be multiple images array."
    assert (kind=="GC") or (kind=="pa"), f"kind {kind} is invalid."

    top, bottom = 78, 334
    border = 256

    # crop from top to bottom
    if TB==True: 
        X = X[:,top:bottom,:]

    # extract Nematode area (right(256:)? or left(:256)?)
    if kind == "pa":
        X = X[:,:,:border]
    elif kind == "GC":
        X = X[:,:,border:]    
    
    return X


def save_imgs(imgs,names,save_path):
    """ Save all images.

    Args:
        imgs (array): Images to be saved.
        names (array): Names of images.
        save_path (str): Path of Saving folder.

    Note: 
        The order of the contents of A and B should be aligned.
    """
    
    assert len(imgs)==len(names),"Length of images and names must be the same."
    os.makedirs(save_path,exist_ok=True)
    
    for i in range(len(imgs)):    
        save_name = os.path.join(save_path, f"{names[i]}.jpg")
        plt.imsave(save_name, imgs[i])
    return print(f"Completed saving images to {save_path}.")


def get_image_name(img_paths):
    """ Get names of images.

    Args:
        img_paths (array): Paths of images.

    Returns:
        names (array): Names of images without extension.

    Example: 
        before: /data/Users/katafuchi/images/image0.jpg
        after:  image0
    """

    names=[]
    for path in img_paths:
        name = os.path.splitext(os.path.basename(path))[0]
        names.append(name)

    return names


def to_uint8(img):
    """ Convert uint16 to uint8. 
        .jpg image converted from .tif is sometimes uint16 data type.
        There are some ways to convert uint16 to uint8, but this is the best way not to lose image's detail information.

    Args:
        img (image): uint16 image data.

    Returns:
        img (image): uint8 image data.
    """
    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)
    img = img.astype(np.uint8)
    return img


def get_image_data(Imgs_path, flip=True):
    """ Get uint8 jpg images.

    Args:
        Imgs_path (array): Array of paths to images.
        flip (bool, optional): Flip image or not. Defaults to True.

    Returns:
        Imgs (numpy array): Array of images which is grayscale and uint8 data type.
    """
    Imgs=[]
    for path in Imgs_path:
        # TIFF to JPG
        img = tifffile.imread(path) 
        img = to_uint8(img)

        if flip:
            img = img[:,::-1] # flip

        if img.ndim>=3:
           img = img[:,:,0] 
        
        Imgs.append(img)
    return np.array(Imgs)


def get_labelme_mask(PATH):
    """ Get mask from .json file made using labelme.

    Args:
        PATH (str): path to json file.

    Returns:
        mask (numpy array): a mask image.
    """

    with open(PATH, "r",encoding="utf-8") as f:
        dj = json.load(f)

    mask = labelme.utils.shape_to_mask(
            (dj['imageHeight'],dj['imageWidth']), 
            dj['shapes'][0]['points'], 
            shape_type=None, line_width=1, point_size=1
    ) # dj['shapes'][0] is for getting only first label.
    
    mask = mask.astype(np.int) # bool to [0,1] (False:0,True:1)
    return mask


def image_preprocess(img, kind="GC"):
    """ Preprocess image.

    Args:
        img (arr): image array.
        kind (str, optional): "GC" or "pa" classes. Defaults to "GC".

    Returns:
        img: preprocessed image.
    """
    
    assert (kind=="GC") or (kind=="pa"), "wrong kind"
    
    # degree of emphasis
    if kind=="GC": # Left
        factor_color = 1
        factor_brightness = 1
        factor_contrast = 2
        factor_sharpness = 2

    elif kind=="pa": # Right
        factor_color = 1
        factor_brightness = 2
        factor_contrast = 3
        factor_sharpness = 4
        
    img = Image.fromarray(np.uint8(img))
    
    # Chroma
    enhancer = ImageEnhance.Color(img)  
    img = enhancer.enhance(factor_color)
        
    # Brightness
    enhancer = ImageEnhance.Brightness(img)  
    img = enhancer.enhance(factor_brightness)

    # Contrast
    enhancer = ImageEnhance.Contrast(img)  
    img = enhancer.enhance(factor_contrast)
    
    # Sharpness
    enhancer = ImageEnhance.Sharpness(img)  
    img = enhancer.enhance(factor_sharpness)
    
    return np.array(img)
