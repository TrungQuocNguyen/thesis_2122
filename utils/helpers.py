import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
label_to_color = {
    0: [0,0,0],  # unannotated: black
    1: [139,0,0],  # wall: maroon
    2: [255, 250, 250],  # floor: snow
    3: [70, 130, 180], #cabinet:  steel blue
    4: [240, 255, 240],  # bed: honeydew
    5: [255, 99, 71],  # chair: tomato
    6: [255, 250, 240], # sofa: floral white
    7: [64, 224, 208], # table: turquoise
    8: [119, 136, 153],  # door: light slate gray
    9: [240, 128, 128], # window: light coral
    10: [255, 245, 238], # bookshelf: sea shell
    11: [255, 160, 122],  # picture: light salmon
    12: [250, 240, 230], # counter: linen  
    13: [218, 165, 32], # blinds: golden rod 
    14: [255, 218, 185], # desk: peach stuff 
    15: [240, 230, 140], # shelves: khaki 
    16: [188, 143, 143], # curtain: rosy brown  
    17: [ 154, 205, 50], # dresser: yellow green 
    18: [244, 164, 96], # pillow: sandy brown 
    19: [124, 252, 0], # mirror: lawn green
    20: [160, 82, 45], # floor mat: sienna 
    21: [0,100,0],  # clothes: dark green
    22: [250, 250, 210], # ceiling: light golden rod yellow 
    23: [0, 255, 0], # books: lime 
    24: [245, 222, 179], # refridgerator: wheat
    25: [152, 251, 152], # television: pale green 
    26: [245, 245, 220], # paper: beige 
    27: [0, 255, 127],  # towel: spring green
    28: [255, 182, 193], # shower curtain: light pink 
    29: [60, 179, 113], # box: medium sea green 
    30: [219, 112, 147], # whiteboard: pale violet red 
    31: [0, 128, 128], # person: teal
    32: [255, 0, 255], # nightstand: magenta
    33: [0, 255, 255], # toilet: cyan
    34: [216, 191, 216], # sink: thistle
    35: [255, 165, 0],  # lamp: orange
    36: [153, 50, 204], # bathtub: dark orchid
    37: [127, 255, 212],  #bag: aqua marine
    38: [147, 112, 219], # otherstructure: medium purple
    39: [178, 34, 34], # other furniture: firebrick 
    40: [72, 61, 139], # otherprop: dark slate blue

}
def mask2pixel(mask): 
    h, w = mask.shape
    img_rgb = np.zeros((h,w,3), dtype = np.uint8)
    for label, rgb in label_to_color.items(): 
        img_rgb[mask == label, :] = rgb
    return img_rgb
def plot_preds(imgs, targets, preds): 
    #imgs: [N, 3, img_size, img_size]
    #targets: [N, img_size, img_size]
    #preds: [N, img_size, img_size]
    N = imgs.size(0)
    num_img_show = N if N <= 4 else 4
    imgs = imgs.permute(0,2,3,1).numpy()
    targets = targets.numpy()
    preds = preds.numpy()
    fig = plt.figure(figsize=(14,8))
    for idx in range(num_img_show): 
        fig.add_subplot(3,num_img_show,idx +1)
        plt.imshow(imgs[idx])
        fig.add_subplot(3,num_img_show,idx +1 + num_img_show)
        plt.imshow(mask2pixel(preds[idx]))
        fig.add_subplot(3,num_img_show,idx +1 + num_img_show*2)
        plt.imshow(mask2pixel(targets[idx]))
    return fig
def initialize_weights(*models):
    for model in models:
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1.)
                m.bias.data.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()