import os
import cv2
import glob
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from data import get_image_data, crop_image, image_preprocess


def main():

    # get images
    X = get_image_data(sorted(glob.glob(f"{opt.DATA_PATH}/*.tif"))) 
    X = crop_image(X, kind="pa", TB=False)
    X = X[1:] # remove first image. (the first image was used as a reference to bbox in siamFC++ algorithm.)

    # apply preprocessing
    if opt.preprocess==True:
        X = np.array([image_preprocess(x, kind="pa") for x in X])

    # get csv file
    csv_file = pd.read_csv(opt.CSV_PATH, header=None)
    
    assert len(X) == len(csv_file), "Length of csv file and images don't match."

    Lvalue_aves=[]
    for i in range(len(csv_file)):

        x = X[i]
        left_x,top_y,right_x,bottom_y = csv_file.iloc[i]

        # get patch img in bbox area
        bbox_x = x[top_y:bottom_y,left_x:right_x]

        # calculate time series quantification of changes in luminance values
        value = np.mean(bbox_x.ravel())
        Lvalue_aves.append(value)
    
    Lvalue_aves = np.array(Lvalue_aves)

    # make save folder
    SAVE_PATH = os.path.join(opt.SAVE_PATH,os.path.basename(opt.DATA_PATH))
    os.makedirs(SAVE_PATH, exist_ok=False)

    # save as a binary file
    # np.save(f"{SAVE_PATH}/quantification-cell-pa", Lvalue_aves) 
    # save as a txt file
    np.savetxt(f"{SAVE_PATH}/quantification-cell-pa.txt", Lvalue_aves)

    plt.figure(figsize=(12,6))
    plt.plot(Lvalue_aves)
    plt.xlabel("Time (image number)",fontsize=13)
    plt.ylabel("Ave of luminance values in the neurites",fontsize=13)
    plt.title("Time series quantification of changes in luminance values",fontsize=14)
    plt.savefig(f"{SAVE_PATH}/quantification-cell-pa.jpg")
    plt.clf()
    plt.close()

    


if __name__ == "__main__":

    """ Quantify the pixel values of cell body in bounding box obtained by siamFC++.

        Ex) python cell_quantify.py \
            --DATA_PATH [Folder storing .tif images] \
            --CSV_PATH  [Csv file storing bbox values.] \
            --SAVE_PATH [Parent saving directory. New saving folder with dataset name will be created in the saving folder.] 

        Ex) python cell_quantify.py \
            --DATA_PATH /data/Users/katafuchi/RA/Nematode/2021_0303_dia10-5-gcy28d-GCaMP6f+paQuasAr3-3 \
            --CSV_PATH  /data/Users/katafuchi/RA/Nematode/cell_bbox/2021_0303_dia10-5-gcy28d-GCaMP6f+paQuasAr3-3_pa.csv \
            --SAVE_PATH /data/Users/katafuchi/RA/Nematode/results_cell_quantify 
    """

    parser = argparse.ArgumentParser(description='Make JPG image of first frame.') 
    parser.add_argument('--DATA_PATH', type=str, required=True, help='path to a directory that contains .tif first frame') 
    parser.add_argument('--CSV_PATH',  type=str, required=True, help='path to a csv file') 
    parser.add_argument('--SAVE_PATH', type=str, required=True, help='path to a save folder') 
    parser.add_argument('--preprocess', action='store_true', required=False, help='whether data is preprocessed or not.') 
    opt = parser.parse_args() 

    assert os.path.isdir(opt.DATA_PATH),f"Data directory {opt.DATA_PATH} does not exist not found."
    assert os.path.isfile(opt.CSV_PATH),f"csv file {opt.CSV_PATH} is not found."

    os.makedirs(opt.SAVE_PATH, exist_ok=True)
    main()
    print("done")

    