import cv2
import numpy as np

## not useful matrix
def getPSNR(I1, I2):
    '''
    :param I1: represents original image matrix
    :param I2: represents degraded image matrix
    :return: psnr score 
    '''
    # mse calculation
    s1 = cv2.absdiff(I1, I2)
    # cannot make a square on 8 bits
    s1 = np.float32(s1)
    s1 = s1 * s1
    sse = s1.sum()
    # return zero if the difference is extremely small
    if sse <= 1e-10:
        return 0
    else:
        shape = I1.shape
        mse = 1.0 * sse / (shape[0] * shape[1] * shape[2])
        # here 255 denotes the maximum possible 
        # value in a 8-bit channel
        psnr = 10.0 * np.log10((255 * 255) / mse)
        return psnr