import numpy as np

def myHoughTransform(Im, rhoRes, thetaRes):
    # YOUR CODE HERE
    
    rhoScale = []
    thetaScale = []
    
    im_h, im_w = Im.shape
    rhomax = np.sqrt(im_h**2 + im_w**2)
    rhoScale = np.arange(0, rhomax+rhoRes, rhoRes)
    thetaScale = np.arange(0, 2*np.pi, thetaRes)
    im_hough = np.zeros((len(rhoScale), len(thetaScale)))

    edge_index = np.argwhere(Im)
    
    for y,x in edge_index:
        for theta in range(len(thetaScale)):
            rho = int(round(x*np.cos(thetaScale[theta]) + y*np.sin(thetaScale[theta])))
            if(rho in rhoScale):
                rho_index = np.where(rhoScale == rho)[0]
                if (rho_index.size > 0):
                    im_hough[rho_index[0]][theta] += 0.1


    return im_hough, rhoScale, thetaScale
