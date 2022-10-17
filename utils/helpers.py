import math
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
CLASS_IDS_TO_COLOR = {
    0:[0, 0, 0], # first index=0
    1:[174, 199, 232],  # 1.wall
    2:[152, 223, 138],  # 2.floor
    3:[31, 119, 180],   # 3.cabinet
    4:[255, 187, 120],  # 4.bed
    5:[188, 189, 34],   # 5.chair
    6:[140, 86, 75],    # 6.sofa
    7:[255, 152, 150],  # 7.table
    8:[214, 39, 40],    # 8.door
    9:[197, 176, 213],  # 9.window
    10:[148, 103, 189],  # 10.bookshelf
    11:[196, 156, 148],  # 11.picture
    12:[23, 190, 207],   # 12.counter
    13:[178, 76, 76],  
    14:[247, 182, 210],  # 14.desk
    15:[66, 188, 102], 
    16:[219, 219, 141],  # 16.curtain
    17:[140, 57, 197], 
    18:[202, 185, 52], 
    19:[51, 176, 203], 
    20:[200, 54, 131], 
    21:[92, 193, 61],  
    22:[78, 71, 183],  
    23:[172, 114, 82], 
    24:[255, 127, 14],   # 24.refrigerator
    25:[91, 163, 138], 
    26:[153, 98, 156], 
    27:[140, 153, 101],
    28:[158, 218, 229],  # 28.shower curtain
    29:[100, 125, 154],
    30:[178, 127, 135],
    31:[120, 185, 128],
    32:[146, 111, 194],
    33:[44, 160, 44],    # 33.toilet
    34:[112, 128, 144],  # 34.sink
    35:[96, 207, 209], 
    36:[227, 119, 194],  # 36.bathtub
    37:[213, 92, 176], 
    38:[94, 106, 211], 
    39:[82, 84, 163],    # 39.otherfurn
    40:[100, 85, 144]    # last index=40

}
CLASS_LABELS = ['unannotated', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floor mat', 
'clothes', 'ceiling', 'books', 'refridgerator', 'television', 'paper', 'towel', 'shower curtain', 'box', 'whiteboard', 'person ', 'nightstand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag', 'otherstructure', 'other furniture', 'otherprop']
def mask2pixel(mask): 
    h, w = mask.shape
    img_rgb = np.zeros((h,w,3), dtype = np.uint8)
    for class_id, rgb in CLASS_IDS_TO_COLOR.items(): 
        img_rgb[mask == class_id, :] = rgb
    return img_rgb
def plot_preds(imgs, targets, preds): 
    #imgs: [N, 3, H, W] [N,3,256, 328]
    #targets: [N, H, W]
    #preds: [N, H, W]
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
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            #elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                #m.weight.fill_(1.)
                #m.bias.fill_(1e-4)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, 0.0001)
                m.bias.data.zero_()
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
        m.weight.data.fill_(1.)
        m.bias.data.fill_(1e-4)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.0001)
        m.bias.data.zero_()
def print_params(model): 
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total params: %d' %(total_params))
    print('Trainable params: %d' %(trainable_params))

def make_intrinsic(fx, fy, mx, my):
    intrinsic = torch.eye(4)
    intrinsic[0][0] = fx
    intrinsic[1][1] = fy
    intrinsic[0][2] = mx
    intrinsic[1][2] = my
    return intrinsic


# create camera intrinsics
def adjust_intrinsic(intrinsic, intrinsic_image_dim, image_dim):
    if intrinsic_image_dim == image_dim:
        return intrinsic
    resize_width = int(math.floor(image_dim[1] * float(intrinsic_image_dim[0]) / float(intrinsic_image_dim[1])))
    intrinsic[0,0] *= float(resize_width)/float(intrinsic_image_dim[0])
    intrinsic[1,1] *= float(image_dim[1])/float(intrinsic_image_dim[1])
    # account for cropping here
    intrinsic[0,2] *= float(image_dim[0]-1)/float(intrinsic_image_dim[0]-1)
    intrinsic[1,2] *= float(image_dim[1]-1)/float(intrinsic_image_dim[1]-1)
    return intrinsic