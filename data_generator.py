import cv2
import numpy as np
from glob import glob
import os
generalpath = os.getcwd()
# print(generalpath)
size = 500
dsize = (size, size)
bname, wname, noise = 984,955,867
for pic in glob('*.jpg'):
    os.chdir(generalpath)
    I = cv2.imread(pic)
    F = cv2.resize(I, dsize)
    cv2.imshow('pic', F)
    key = cv2.waitKey()
    if key & 0xFF == ord('b'): 
        path = os.getcwd()
        os.chdir(path + "/blue")
        cv2.imwrite(str(bname)+'.jpg', I)
        bname += 1
    elif key & 0xFF == ord('w'): 
        path = os.getcwd()
        os.chdir(path + "/white")
        cv2.imwrite(str(wname)+'.jpg', I)
        wname += 1
    elif key & 0xFF == ord('n'): 
        path = os.getcwd()
        os.chdir(path + "/noise")
        cv2.imwrite(str(noise)+'.jpg', I)
        noise += 1
    elif key & 0xFF == 27: 
        break
cv2.destroyAllWindows()