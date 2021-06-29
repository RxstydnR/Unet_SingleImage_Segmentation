import os
import glob
import argparse
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from data import get_image_data,crop_image


def imgs_to_movie(Imgs, SAVE_PATH, MOVIE_NAME="movie", fps=20.0):
    """ Make a movie from an image sequence.

    Args:
        Imgs (array): An image sequence. The order needs to be aligned.
        save_path (str): Path to saving folder.
        fps (float, optional): fps of movie. Defaults to 20.0.

    Note:
        Imgs must be uint8 datatype. If necessary, use the code below.
        ''' 
            Imgs = (Imgs*255).astype("uint8") 
        '''
    """
    Imgs = np.array(Imgs)
    assert len(Imgs)>0,"Got empty an image array."
    
    os.makedirs(SAVE_PATH,exist_ok=True)

    height, width = Imgs[0].shape[0], Imgs[0].shape[1]
    video_size = (width, height)

    if Imgs.ndim==3:
        isColor=False
    elif Imgs.shape[-1]==1:
        isColor=False
    else:
        isColor=True 

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_path = SAVE_PATH+f'/{MOVIE_NAME}.mp4'
    video = cv2.VideoWriter(video_path, fourcc, fps, video_size, isColor=isColor)

    for img in Imgs:
        video.write(img)
    video.release()

    return print(f'Saved {video_path}.')


if __name__ == "__main__":
    """ Make a movie from an image sequence in a folder.

        Ex) python img2movie.py \
            --data_path [Folder storing .tif images] \
            --save_path [Saving folder] \
            --kind [GC or pa classes]

        Ex) python cell_quantify.py \
            --data_path /data/Users/katafuchi/RA/Nematode/2021_0303_dia10-5-gcy28d-GCaMP6f+paQuasAr3-3 \
            --save_path /data/Users/katafuchi/RA/Nematode/movies \
            --kind pa
    """

    parser = argparse.ArgumentParser(prog='make_movie.py')
    parser.add_argument('--data_path', type=str, required=True, help='path of image data.')
    parser.add_argument('--save_path', type=str, required=True, help='path of save folder.')
    parser.add_argument('--fps', type=float, default=20, help='frame per second.')
    parser.add_argument('--kind', type=str, default=None, help='Crop GC or pa area. If none, the entire images will be into a movie.')
    opt = parser.parse_args()

    data_path = opt.data_path 
    save_path = opt.save_path
    os.makedirs(save_path, exist_ok=True)
    movie_name = os.path.basename(data_path)

    Imgs = get_image_data(sorted(glob.glob(data_path+"/*.tif")))

    # Crop GC (right) or pa (left) area
    if opt.kind != None:
        Imgs = crop_image(Imgs, kind=opt.kind, TB=False)
        movie_name = movie_name + f"_{opt.kind}"
    
    imgs_to_movie(Imgs, save_path, movie_name, opt.fps)
