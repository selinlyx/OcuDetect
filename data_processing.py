import os
import cv2
import numpy as np 
import matplotlib.pyplot as plt
import fundus_image_toolbox as fit
from PIL import Image

# fundus = plt.imread('ODIR-5K/Training Images/1006_right.jpg')

# fit.utils.show([fundus])
# fit.utils.print_type([fundus])

# fundus_cropped = fit.circle_crop.crop(fundus, size=224)
# fit.utils.show([fundus_cropped])
# fit.utils.print_type([fundus_cropped])


for filename in os.listdir('ODIR-5K/Training Images'):
    path = os.path.join('ODIR-5K/Training Images', filename)
    fundus = plt.imread(path)

    # crop and rescale fundus image to 224x224
    fundus_cropped = fit.circle_crop.crop(fundus, size=224)

    # contrast enhancement (CLAHE on lightness channel)
    lab = cv2.cvtColor(fundus_cropped, cv2.COLOR_RGB2LAB) # transform (R, G, B) to (Lightness, a, b) space
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    fundus_contrast = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB) # transform back to (R, G, B) space

    save_path = os.path.join('ODIR-5K/data', filename)
    Image.fromarray(fundus_contrast).save(save_path)
    print(f'{filename} processed and saved.')


for filename in os.listdir('ODIR-5K/Testing Images'):
    path = os.path.join('ODIR-5K/Testing Images', filename)
    fundus = plt.imread(path)

    # crop and rescale fundus image to 224x224
    fundus_cropped = fit.circle_crop.crop(fundus, size=224)

    # contrast enhancement (CLAHE on lightness channel)
    lab = cv2.cvtColor(fundus_cropped, cv2.COLOR_RGB2LAB) # transform (R, G, B) to (Lightness, a, b) space
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge((l_eq, a, b))
    fundus_contrast = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB) # transform back to (R, G, B) space

    save_path = os.path.join('ODIR-5K/data', filename)
    Image.fromarray(fundus_contrast).save(save_path)
    print(f'{filename} processed and saved.')