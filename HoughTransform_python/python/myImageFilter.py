import numpy as np

def myImageFilter(img0, h):
    # YOUR CODE HERE

    height, width = img0.shape
    if h.ndim == 1:
        h = h[:, np.newaxis]
    filter_h, filter_w = h.shape
    if(filter_h == 1):
        pad_h = 1
    else:
        pad_h = filter_h//2
        
    if(filter_w == 1):
        pad_w = 1
    else:
        pad_w = filter_w//2

    img0_pad = np.pad(img0, ((pad_h, pad_h), (pad_w, pad_w)))
    out_img = np.zeros_like(img0)
    for i in range(filter_h):
        for j in range(filter_w):
            out_img += h[i, j]*img0_pad[i:(i+height), j:(j+width)]
            
    output_img = np.clip(out_img, 0, 1)
    return output_img


