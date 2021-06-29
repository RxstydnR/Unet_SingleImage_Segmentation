import os
import cv2
import tifffile
import argparse

from data import to_uint8

if __name__ == "__main__":

    """ Convert first frame .tif image to .jpg image for labelme annotation.

    Example:
        python make_firstframe_jpg.py --PATH [folder containing .tif images]
        python make_firstframe_jpg.py --PATH /data/Users/katafuchi/RA/Nematode/2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-11
    """

    parser = argparse.ArgumentParser(description='Make JPG image of first frame.') 
    parser.add_argument('--PATH', type=str, required=True, help='path to a directory that contains .tif first frame') 
    opt = parser.parse_args() 

    data_name = os.path.join(opt.PATH, os.path.basename(opt.PATH)+"_0000.tif")
    print(data_name)
    assert os.path.isfile(data_name)==True, "Could not find the first frame .tif image."

    x = tifffile.imread(data_name) 
    x = to_uint8(x)
    x = x[:,::-1]

    # 3ch gray to 1ch gray
    if x.ndim>=3:
        x = x[:,:,0] 

    save_dir = "/data/Users/katafuchi/RA/Nematode/labelme_mask" # Saving folder to store images to be annotated with labelme.
    os.makedirs(save_dir, exist_ok=True)

    save_name = os.path.join(save_dir, os.path.splitext(os.path.basename(data_name))[0]+".jpg")
    cv2.imwrite(save_name, x)
    print(f"Saved as {save_name}")