import numpy as np
from scipy import signal   # For signal.gaussian function
import myImageFilter as Imfilt
import cv2


def myEdgeFilter(img0, sigma):
    
    #smooth the image
    hsize = int(2 * np.ceil(3 * sigma) + 1)
    gaus_kernal = signal.windows.gaussian(hsize, sigma-1).reshape(hsize, 1)
    gaus_filter = np.outer(gaus_kernal, gaus_kernal)
    gaus_filter /= np.sum(gaus_filter)
    
    img_smooth = Imfilt.myImageFilter(img0, gaus_filter)
    # cv2.imshow('smooth image', img_smooth)
    
    #sobel filters
    sobel_horizontal = np.array([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0 , 1]])
    sobel_vertical = np.array([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2 , -1]])
    sobel_horizontal_2 = np.array([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0 , -1]])
    sobel_vertical_2 = np.array([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2 , 1]])
    
    imgy = Imfilt.myImageFilter(img_smooth, sobel_vertical)
    imgx = Imfilt.myImageFilter(img_smooth, sobel_horizontal)
    imgx_2 = Imfilt.myImageFilter(img_smooth, sobel_horizontal_2)
    imgy_2 = Imfilt.myImageFilter(img_smooth, sobel_vertical_2)
    
    # cv2.imshow("filter gaus1:", imgy)
    # cv2.imshow("filter gaus2:", imgx)
    # cv2.imshow("filter gaus3:", imgy_2)
    # cv2.imshow("filter gaus4:", imgx_2)

    
    mag = np.sqrt(imgx**2 + imgy**2 + imgx_2**2 + imgy_2**2)
    # mag = np.sqrt(imgx**2 + imgy**2)

    dir = np.arctan2(imgy, imgx)*(180/np.pi) + np.arctan2(imgy_2, imgx_2)*(180/np.pi)
    #dir = np.arctan2(imgy, imgx)*(180/np.pi)
    dir_grad = (dir+180)%180
    image_h, image_w = mag.shape
    mag_out = np.zeros_like(mag)
    
    #non maximum suppression
    for i in range(1, image_h-1):
        for j in range(1, image_w-1):
            if(0 <= dir_grad[i, j] <= 22.5):
                #compare with top and down, 0
                mag1 = mag[i, j-1]
                mag2 = mag[i, j+1]
     
            elif(22.5 < dir_grad[i, j] <= 67.5):
                #compare with top right and bottom left, 45
                mag1 = mag[i - 1, j + 1]
                mag2 = mag[i + 1, j -1]
            elif(67.5 < dir_grad[i, j] <= 112.5):
                #compare with up and down, 90
                mag1 = mag[i-1, j]
                mag2 = mag[i+1, j]
            elif(112.5 < dir_grad[i, j] < 157.5):
                #compare with top left and bottom right
                mag1 = mag[i+1, j +1]
                mag2 = mag[i-1, j-1]
                
            if(mag[i, j] >= mag1 and mag[i, j] >= mag2):
                mag_out[i, j] = mag[i, j]
    # cv2.imshow('output:', mag_out)
    # kernal = np.ones((3, 3))
    # cv2.dilate(mag, kernal, iterations = 1)
    # # cv2.imshow('dilate: ', mag)
    # cv2.imshow("filter gaus:", mag)
    return mag_out
    # YOUR CODE HERE
