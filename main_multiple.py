import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2" 
os.environ["CUDA_VISIBLE_DEVICES"]=  "4" 

import glob
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from model import UNet
from img2movie import imgs_to_movie
from augmentation import data_augmentation
from data import image_preprocess, get_labelme_mask, get_image_data, save_imgs, get_image_name
from utils import prediction_superimpose, mask2ind, mask2bin, data_shuffle

def train():
    """ Training model with multiple first frames of datasets.
    """
    
    # get images and masks
    print("Preparation for training data...")
    X=[]
    Y=[]
    for data_path, mask_path in zip(opt.DATA_PATH_LIST, opt.MASK_NAME_LIST):

        # get first frame image
        x = get_image_data(sorted(glob.glob(f"{data_path}/*.tif"))[0:1]) 
        x = np.squeeze(x)
    
        # get a mask of first frame
        mask = get_labelme_mask(mask_path)
        
        # extraction of target area
        _, m_w = mask.shape
        upper_h, lower_h = 78, 334
        
        border = int(m_w//2) # Left side of the boundary: pa, Right one is GC.
        mask = mask[upper_h:lower_h, border:] 
    
        # extract Nematode area (right(256:)? or left(:256)?)
        if opt.kind == "pa":
            x = x[upper_h:lower_h,:border]
        elif opt.kind == "GC":
            x = x[upper_h:lower_h,border:]
        
        # apply preprocessing
        if opt.preprocess==True:
            x = image_preprocess(x, kind=opt.kind)

        X.append(x)
        Y.append(mask)

    # preparing the training data
    X_train = np.array(X) # initial frame as training data
    Y_train = np.array(Y) # annotation of the initial frame

    # augmentation
    X_train_aug,Y_train_aug = data_augmentation(X_train,Y_train, times=opt.aug_times)
    X_train_aug,Y_train_aug = data_shuffle(X_train_aug,Y_train_aug) # for validation
    X_train_aug = X_train_aug.astype('float32')/ 255.

    # U-Net
    model = UNet(input_ch=1, output_ch=1, filters=opt.n_filter).get_model() # 64*1
    model.compile(loss="mse", optimizer="adam")
    
    print("Training begins...")
    history = model.fit(
        X_train_aug, Y_train_aug, 
        batch_size=opt.batch_size, epochs=opt.epochs, validation_split=0.1, verbose=2)

    # save trained model
    model.save(f'{opt.SAVE_PATH}/model.h5', include_optimizer=False)

    # save trianing hisotry
    plt.figure(figsize=(8,4))
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.savefig(f"{opt.SAVE_PATH}/history-{opt.kind}.jpg")
    plt.clf()
    plt.close()


def test():
    """ Test segmentation with model trained with multiple first frames of datasets.
    """

    for data_path in tqdm(opt.DATA_PATH_LIST):
        
        # make save folder for each test dataset
        SAVE_PATH = os.path.join(opt.SAVE_PATH,os.path.basename(data_path))
        os.makedirs(SAVE_PATH,exist_ok=True)

        # get image data        
        Img_paths = sorted(glob.glob(f"{data_path}/*.tif"))
        Imgs      = get_image_data(Img_paths)
        Img_names = get_image_name(Img_paths)
    
        # extraction of target area
        _, m_w = Imgs[0].shape
        upper_h, lower_h = 78, 334
        border = int(m_w//2) # Left side of the boundary: pa, Right one is GC.

        # extract Nematode area (right(256:)? or left(:256)?)
        if opt.kind == "pa":
            X = Imgs[:,upper_h:lower_h,:border]
        elif opt.kind == "GC":
            X = Imgs[:,upper_h:lower_h,border:]
        
        # apply preprocessing
        if opt.preprocess==True:
            X = np.array([image_preprocess(x, kind=opt.kind) for x in X])

        # save Images
        save_imgs(imgs=X,names=Img_names,save_path=f"{SAVE_PATH}/Imgs-{opt.kind}")

        # save model
        model = load_model(opt.model_path, compile=False)

        print("Prediction...")
        X = X.astype('float32')/255.
        preds = model.predict(X)
        preds = np.squeeze(preds)

        # make a result movie
        preds_superimpose = prediction_superimpose(X, preds)
        imgs_to_movie(Imgs=preds_superimpose, SAVE_PATH=SAVE_PATH, MOVIE_NAME=f"movie-{opt.kind}", fps=20.0)

        # make masks into binary masks
        masks = mask2bin(preds).astype('uint8')
        # save as a image
        save_imgs(imgs=masks,names=Img_names,save_path=f"{SAVE_PATH}/MASKs-{opt.kind}")
        # save as a binary file            
        # np.save(f"{SAVE_PATH}/prediction-{opt.kind}", mask2ind(masks))
        
        # calculate time series quantification of changes in luminance values
        X_neurites = X*masks
        Lvalue_aves = np.sum(X_neurites, axis=(1,2))/np.count_nonzero(X_neurites, axis=(1,2))
        # save as a binary file
        # np.save(f"{SAVE_PATH}/quantification-{opt.kind}", Lvalue_aves) 
        # save as a txt file
        np.savetxt(f"{SAVE_PATH}/quantification-{opt.kind}.txt", Lvalue_aves)

        plt.figure(figsize=(12,6))
        plt.plot(Lvalue_aves)
        plt.xlabel("Time (image number)",fontsize=13)
        plt.ylabel("Ave of luminance values in the neurites",fontsize=13)
        plt.title("Time series quantification of changes in luminance values",fontsize=14)
        plt.savefig(f"{SAVE_PATH}/quantification-{opt.kind}.jpg")
        plt.clf()
        plt.close()

            
if __name__ == "__main__":

    """ Training model with multiple first frames of datasets and test segmentation.

    Training Example:
        python main_multiple.py \
            --DATA_PATH_LIST [Training data folders storing .tif images] \
            --MASK_NAME [.json mask files] \
            --SAVE_PATH [Saving folder] \
            --train [If train, set --train] \
            --kind [GC or pa] \
            --preprocess [If apply image preprocessing, set --preprocess] \
            --aug_times [the number of times] \
            --epochs [the number of epochs]
        
    Test Example:
        python main_multiple.py \
            --DATA_PATH_LIST [Test data folders storing .tif images] \
            --SAVE_PATH [Parent saving directory. New saving folder with dataset name will be created in the saving folder.]  \
            --kind [GC or pa] \
            --preprocess [If apply image preprocessing, set --preprocess] \
            --model_path [Path of model created by training.] \

    
    Training Example:
        python main_multiple.py \
            --DATA_PATH_LIST \
                /data/Users/katafuchi/RA/Nematode/2021_0303_dia10-5-gcy28d-GCaMP6f+paQuasAr3-3 \
                /data/Users/katafuchi/RA/Nematode/2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-3 \
                /data/Users/katafuchi/RA/Nematode/2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-4 \
            --MASK_NAME \
                /data/Users/katafuchi/RA/Nematode/labelme_mask/2021_0303_dia10-5-gcy28d-GCaMP6f+paQuasAr3-3_0000.json \
                /data/Users/katafuchi/RA/Nematode/labelme_mask/2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-3_0000.json \
                /data/Users/katafuchi/RA/Nematode/labelme_mask/2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-4_0000.json \
            --SAVE_PATH /data/Users/katafuchi/RA/Nematode/result_multiple \
            --train \
            --kind pa \
            --preprocess \
            --aug_times 100 \
            --epochs 200 

    Test Example:
        python main_multiple.py \
            --DATA_PATH_LIST \
                /data/Users/katafuchi/RA/Nematode/2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-1 \
                /data/Users/katafuchi/RA/Nematode/2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-2 \
                /data/Users/katafuchi/RA/Nematode/2021_0323_dia10-5-gcy28d-GCaMP6f+paQuasAr3-5 \
            --SAVE_PATH /data/Users/katafuchi/RA/Nematode/result_multiple \
            --kind pa \
            --preprocess \
            --model_path /data/Users/katafuchi/RA/Nematode/result_multiple/model.h5\
        
    """

    parser = argparse.ArgumentParser(description='Nematode Neurites Segmentation with First Frame Images of Multiple Dataset.') 

    parser.add_argument('--DATA_PATH_LIST', type=str, nargs='+', required=True, help='multiple paths to directories that contain .tif dataset.')
    parser.add_argument('--MASK_NAME_LIST', type=str, nargs='+', required=False, help='multiple paths to mask label .json files.') 
    parser.add_argument('--SAVE_PATH', type=str, required=True, help='path to save directory') 
    
    parser.add_argument('--train', action='store_true', required=False, help='if not, test begins.') 

    parser.add_argument('--kind', type=str, required=False, default="GC", choices=['GC','pa'], help='which data is used, pa or GC.') 
    parser.add_argument('--preprocess', action='store_true', required=False, help='whether data is preprocessed or not.') 
    parser.add_argument('--model_path', type=str, required=False, help='test: path to trained model.') 

    parser.add_argument('--aug_times', type=int, required=False, default=50, help='how many the each initial frame (training data) should be augmented to.') 
    parser.add_argument('--n_filter', type=int, required=False, default=64, help='the number of first filters in Unet.') 
    parser.add_argument('--batch_size', type=int, required=False, default=32, help='training batch size') 
    parser.add_argument('--epochs', type=int, required=False, default=100, help='training epochs') 
    opt = parser.parse_args() 

    # make save folder
    os.makedirs(opt.SAVE_PATH, exist_ok=True)

    # train or test
    if opt.train:
        # check the number of dataset
        assert len(opt.DATA_PATH_LIST)>0,"Set multiple paths to dataset."
        assert len(opt.MASK_NAME_LIST)>0,"Set multiple paths to mask .json files."
        
        # check the existence of dataset and file
        for data_path, mask_path in zip(opt.DATA_PATH_LIST, opt.MASK_NAME_LIST):
            assert os.path.isdir(data_path),f"Data directory {data_path} does not exist not found."
            assert os.path.isfile(mask_path),f"Annotatinon file {mask_path} is not found."

        print("training")
        train()
    else:
        # check the number of dataset
        assert len(opt.DATA_PATH_LIST)>0,"Set multiple paths to dataset."
        
        # check the existence of dataset and file
        for data_path in opt.DATA_PATH_LIST:
            assert os.path.isdir(data_path),f"Data directory {data_path} does not exist not found."
        
        print("test")
        assert opt.model_path!=None, "Set path to the trained model."
        test()
