import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2" 
os.environ["CUDA_VISIBLE_DEVICES"]=  "4" 

from augmentation import data_augmentation
from model import UNet
from data import image_preprocess, get_labelme_mask, get_image_data, save_imgs, get_image_name
from utils import prediction_superimpose, mask2ind, mask2bin
from img2movie import imgs_to_movie


def main():
    
    # get images
    Img_paths = sorted(glob.glob(f"{opt.DATA_PATH}/*.tif"))
    Imgs      = get_image_data(Img_paths) 
    Img_names = get_image_name(Img_paths)
    save_imgs(imgs=Imgs, names=Img_names, save_path=f"{opt.SAVE_PATH}/DATA_JPG")
    
    # get a mask of first frame
    mask = get_labelme_mask(opt.MASK_NAME)
    
    # extraction of target area
    _, m_w = mask.shape
    upper_h, lower_h = 78, 334
    
    border = int(m_w//2) # Left side of the boundary: pa, Right one is GC.
    mask = mask[upper_h:lower_h, border:] 
    
    for class_ in ["pa", "GC"]:
        for kind in ["original","preprocess"]:
            
            # extract Nematode area (right(256:)? or left(:256)?)
            if class_ == "pa":
                X = Imgs[:,upper_h:lower_h,:border].copy()
            elif class_ == "GC":
                X = Imgs[:,upper_h:lower_h,border:].copy()
            
            # apply preprocessing
            if kind=="preprocess":
                X = np.array([image_preprocess(x, kind=class_) for x in X])

            save_imgs(imgs=X[1:], names=Img_names[1:], save_path=f"{opt.SAVE_PATH}/DATA_JPG_{class_}_{kind}")

            # preparing the training data
            X_train = np.expand_dims(X[0],axis=0) # initial frame as training data
            Y_train = np.expand_dims(mask,axis=0) # annotation of the initial frame
            X_test = X[1:] # the rest of frames

            # augmentation
            X_train_aug,Y_train_aug = data_augmentation(X_train,Y_train,times=opt.aug_times)

            X_train_aug = X_train_aug.astype('float32')/ 255.
            X_test = X_test.astype('float32')/ 255.

            # U-Net
            model = UNet(input_ch=1, output_ch=1, filters=opt.n_filter).get_model() # 64*1
            model.compile(loss="mse", optimizer="adam")
            
            print("training begins.")
            history = model.fit(
                X_train_aug, Y_train_aug, 
                batch_size=opt.batch_size, 
                epochs=opt.epochs,
                validation_split=0.1,
                verbose=0)

            plt.figure(figsize=(12,4))
            plt.subplot(121)
            plt.plot(history.history["loss"])
            plt.plot(history.history["val_loss"])
            plt.subplot(122)
            plt.plot(history.history["loss"])
            plt.plot(history.history["val_loss"])
            plt.ylim(0,0.05)
            plt.savefig(f"{opt.SAVE_PATH}/history-{class_}-{kind}.jpg")
            plt.clf()
            plt.close()

            print("prediction")
            preds = model.predict(X_test)
            preds = np.squeeze(preds)

            # make a result movie
            preds_superimpose = prediction_superimpose(X_test, preds)
            imgs_to_movie(Imgs=preds_superimpose, SAVE_PATH=opt.SAVE_PATH, MOVIE_NAME=f"movie-{class_}-{kind}", fps=20.0)

            # make masks into binary masks
            masks = mask2bin(preds).astype('uint8')
            # save as a image
            save_imgs(imgs=masks,names=Img_names[1:],save_path=f"{opt.SAVE_PATH}/MASKs-{class_}-{kind}")
            # save as a binary file            
            # np.save(f"{opt.SAVE_PATH}/prediction-{class_}-{kind}", mask2ind(masks))
            
            # calculate time series quantification of changes in luminance values
            X_neurites = X_test*masks
            Lvalue_aves = np.sum(X_neurites, axis=(1,2))/np.count_nonzero(X_neurites, axis=(1,2))
            # save as a binary file
            np.save(f"{opt.SAVE_PATH}/quantification-{class_}-{kind}", Lvalue_aves) 
            # save as a txt file
            np.savetxt(f"{opt.SAVE_PATH}/quantification-{class_}-{kind}.txt", Lvalue_aves)

            plt.figure(figsize=(12,6))
            plt.plot(Lvalue_aves)
            plt.xlabel("Time (image number)",fontsize=13)
            plt.ylabel("Ave of luminance values in the neurites",fontsize=13)
            plt.title("Time series quantification of changes in luminance values",fontsize=14)
            plt.savefig(f"{opt.SAVE_PATH}/quantification-{class_}-{kind}.jpg")
            plt.clf()
            plt.close()

            # save images at max value and min value
            # min_idx = np.argmin(Lvalue_aves)
            # max_idx = np.argmax(Lvalue_aves)

            # fig,axs = plt.subplots(1,2, figsize=(6,3))
            # axs[0].imshow(X_test[min_idx], vmin=0, vmax=1)
            # axs[0].axis("off")
            # axs[0].set_title(f"minimum, ind={min_idx}")

            # axs[1].imshow(X_test[max_idx], vmin=0, vmax=1)
            # axs[1].axis("off")
            # axs[1].set_title(f"maximum, ind={max_idx}")

            # plt.tight_layout()
            # plt.savefig(f"{opt.SAVE_PATH}/minMaximages-{class_}-{kind}.jpg")
            # plt.clf()
            # plt.close()
            
if __name__ == "__main__":

    """ Training model for segmentation with a first frame of dataset.

    Example:
        python main.py \
            --DATA_PATH [Dataset folder storing .tif images] \
            --MASK_NAME [.json mask files] \
            --SAVE_PATH [Saving folder]

        python main.py \
            --DATA_PATH /data/Users/katafuchi/RA/Nematode/2021_0303_dia10-5-gcy28d-GCaMP6f+paQuasAr3-3 \
            --MASK_NAME /data/Users/katafuchi/RA/Nematode/labelme_mask/2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-3_0000.json \
            --SAVE_PATH /Results/2021_0303_dia10-5-gcy28d-GCaMP6f+paQuasAr3-3
    """

    parser = argparse.ArgumentParser(description='Nematode Neurites Segmentation with Single Image.') 
    parser.add_argument('--DATA_PATH', type=str, required=True, help='path to a directory that contains .tif dataset') 
    parser.add_argument('--MASK_NAME', type=str, required=True, help='path to mask label .json file.') 
    parser.add_argument('--SAVE_PATH', type=str, required=True, help='path to save directory') 
    
    parser.add_argument('--aug_times', type=int, required=False, default=100, help='how many the initial frame (training data) should be augmented to.') 
    parser.add_argument('--n_filter', type=int, required=False, default=64, help='the number of first filters in Unet.') 
    parser.add_argument('--batch_size', type=int, required=False, default=16, help='training batch size') 
    parser.add_argument('--epochs', type=int, required=False, default=100, help='training epochs') 
    opt = parser.parse_args() 
    
    assert os.path.isdir(opt.DATA_PATH),"Data directory is not found."
    assert os.path.isfile(opt.MASK_NAME),"Annotatinon file is not found."
    os.makedirs(opt.SAVE_PATH, exist_ok=False)

    main()

    """ DATASET NAME LIST EXAMPLES
        "/data/Users/katafuchi/RA/Nematode"

            "2021_0303_dia10-5-gcy28d-GCaMP6f+paQuasAr3-3"
            "2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-1"
            "2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-3"
            "2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-4"
            "2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-5"
            "2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-6"
            "2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-7"
            "2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-11"
    """

    
    
