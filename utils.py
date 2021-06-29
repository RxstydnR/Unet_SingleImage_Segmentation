import cv2
import numpy as np

def prediction_superimpose(X,preds):
    """ Create a superimposed image of the prediction and the original image.

    Args:
        X (array): Images (Only 1-channel images are supported).
        preds (array): Model predictions.

    Returns:
        preds_superimpose (array): Superimposed images.

    """
    # Input images
    X_3ch = np.stack((X,)*3, axis=-1)
    X_3ch = (X_3ch*255).astype('uint8')
        
    # Make 3ch red prediction
    preds = np.squeeze(preds)
    preds_red = np.zeros((*preds.shape,3))
    preds_red[...,2] = preds
    preds_red = (preds_red*255).astype('uint8')
    
    # Superimpose
    alpha, beta = 0.5, 0.5
    preds_superimpose = cv2.addWeighted(X_3ch, alpha, preds_red, beta, 0)
            
    return preds_superimpose

def mask2bin(masks,thres=0.5):
    """ Convert a [0,1] prediction into a binary image.

    Args:
        masks (array): Mask images (model predictions).
        thres (float, optional): Threshold for binarizing predictions. Defaults to 0.5.

    Returns:
        bin_masks: Binary mask images.
    """
    bin_masks = np.where(masks>thres,1,0)
    return bin_masks

def mask2ind(masks):
    """ Convert mask images into index-referenced mask images.

    Args:
        masks (array): Binary mask images.

    Returns:
        anno_arr (array): Index-referenced mask images.

    """
    anno_arr=[]
    for i in range(len(masks)):
        anno_arr.append(
            np.where(masks[i]==1)
        )
    anno_arr = np.array(anno_arr)
    return anno_arr

def ind2mask(anno_arr, img_size=(256,256)):
    """ Convert ndex-referenced mask images into mask images.

    Args:
        anno_arr (array): Index-referenced mask images.
        img_size (tuple): Array for Index-referenced mask images. Defaults to (256,256).

    Returns:
        masks (Array): Binary mask images.
    """
    masks=[]
    for i in range(len(anno_arr)):
        mask = np.zeros(img_size)
        for j in range(len(anno_arr[i,0])):
            mask[anno_arr[i,0][j],anno_arr[i,1][j]]=1
        masks.append(mask)
    return np.array(masks)

def data_shuffle(X,Y): 
    """ Shuffle data while maintaining the correspondence.

    Args:
        X (array): Images here.
        Y (array): Labels here.

    Returns:
        X[p],Y[p]: Shuffled data.
    """
    np.random.seed(0)
    assert len(X)==len(Y), "length of X and Y is not same."
    p = np.random.permutation(len(X))
    return X[p],Y[p]
