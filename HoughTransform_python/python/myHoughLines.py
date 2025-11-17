import numpy as np
import cv2  # For cv2.dilate function

def myHoughLines(H, nLines):
    # YOUR CODE HERE
    kernal = np.ones((5, 5))
    img = cv2.dilate(H, kernal, iterations = 1)
    # # cv2.imshow('dilate hough:', img)
    
    height, width = H.shape
    img = H.copy()
    for i in range(1, height-1):
        for j in range(1, width-1):
            #check neighbor pixles 3x3
            suppress = 0
            center = img[i][j]
            for m in range(i-1, i+2):
                for n in range(j-1, j+2):
                    if(suppress == 1):
                        break             
                    if (img[m][n] > center):
                        img[i][j] = 0
                        suppress = 1
                        
                if(suppress == 1):
                    break 
                
    img_1 = img.ravel()
    rho = []
    theta = []
    
    for i in range(nLines):
        row, col = np.unravel_index(np.argmax(img_1), img.shape)
        # print(row, col)
        rho.append(row)
        theta.append(col)
        img_1[np.argmax(img_1)] = 0
            
    return rho, theta