# total files will be: 100 files * 2 image types * 36 slices each = 7,200 files pngs.

import glob 
import os 

from scipy.io import loadmat 
from PIL import Image

import numpy as np 

target="/fileserver/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/diffusion/experiments/s20240209_oecorr/guided-diffusion/data/acdc_png/train/"
source="/fileserver/Rad-Warfield-e2/Groups/Imp-Recons/serge/code/diffusion/experiments/s20240209_oecorr/guided-diffusion/data/acdc/data_ED_ES/train/"
os.makedirs(target, exist_ok=True)

from IPython import embed; embed()

files = glob.glob(source+"*mat")
assert files

for counter, mat in enumerate(files):
    # load .mat 
    mato = loadmat(mat)
    for k in ["image_ED", "image_ES"]: 
        im3d=mato[k]
        
        im3d_i = np.moveaxis(im3d, -1,0)
        L = im3d_i.shape[0]
        for sl, im in enumerate(im3d_i):
            
            # cut to shape 128x128 

            # Calculating the starting point for cropping to keep the center
            start_x = (im.shape[1] - 128) // 2
            start_y = (im.shape[0] - 128) // 2

            # Cropping the image to get the central 128x128 part
            cropped_im = im[start_y:start_y+128, start_x:start_x+128]
            
            # Normalizing the cropped image data to the range [0, 255] for PNG format
            normalized_im = 255 * (cropped_im - cropped_im.min()) / (cropped_im.max() - cropped_im.min())
            normalized_im = normalized_im.astype(np.uint8)
            
            
            # Creating an image object and saving to 'cropped_image.png'
            img = Image.fromarray(normalized_im)
            
            
            # save 
            savename = os.path.basename(mat).replace(".mat", "")
            savename = target + savename + "_" + k + "_sl" + str(sl) + ".png"
            img.save(savename)
            
            # verbose 
            print(f"{counter}:{len(files)}-> {k} -> {sl} -> {L}")
            
            
            

