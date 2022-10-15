import os
from pathlib import Path
import argparse
import numpy as np
import trimesh
import open3d as o3d
from metric import metric
from utils.helpers import CLASS_LABELS
class ConfusionMatrix_numpy(metric.Metric):
    """Constructs a confusion matrix for a multi-class classification problems.
    Does not support multi-label, multi-class problems.
    Keyword arguments:
    - num_classes (int): number of classes in the classification problem.
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.
    Modified from: https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py
    """

    def __init__(self, num_classes, normalized=False):
        super().__init__()

        self.conf = np.ndarray((num_classes, num_classes), dtype=np.int64)
        self.normalized = normalized
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.conf.fill(0)

    def add(self, predicted, target):
        """Computes the confusion matrix
        The shape of the confusion matrix is K x K, where K is the number
        of classes.
        Keyword arguments:
        - predicted (Tensor or numpy.ndarray): Can be an N x K tensor/array of
        predicted scores obtained from the model for N examples and K classes,
        or an N-tensor/array of integer values between 0 and K-1.
        - target (Tensor or numpy.ndarray): Can be an N x K tensor/array of
        ground-truth classes for N examples and K classes, or an N-tensor/array
        of integer values between 0 and K-1.
        """
        # If target and/or predicted are tensors, convert them to numpy arrays

        assert predicted.shape[0] == target.shape[0], \
            'number of targets and predicted outputs do not match'

        if np.ndim(predicted) != 1:
            assert predicted.shape[1] == self.num_classes, \
                'number of predictions does not match size of confusion matrix'
            predicted = np.argmax(predicted, 1)
        else:
            assert (predicted.max() < self.num_classes) and (predicted.min() >= 0), \
                'predicted values are not between 0 and k-1'

        if np.ndim(target) != 1:
            assert target.shape[1] == self.num_classes, \
                'Onehot target does not match size of confusion matrix'
            assert (target >= 0).all() and (target <= 1).all(), \
                'in one-hot encoding, target values should be 0 or 1'
            assert (target.sum(1) == 1).all(), \
                'multi-label setting is not supported'
            target = np.argmax(target, 1)
        else:
            assert (target.max() < self.num_classes) and (target.min() >= 0), \
                'target values are not between 0 and k-1'

        # hack for bincounting 2 arrays together
        x = predicted + self.num_classes * target
        bincount_2d = np.bincount(
            x.astype(np.int64), minlength=self.num_classes**2)
        assert bincount_2d.size == self.num_classes**2
        conf = bincount_2d.reshape((self.num_classes, self.num_classes))

        self.conf += conf

    def value(self):
        """
        Returns:
            Confustion matrix of K rows and K columns, where rows corresponds
            to ground-truth targets and columns corresponds to predicted
            targets.
        """
        if self.normalized:
            conf = self.conf.astype(np.float32)
            return conf / conf.sum(1).clip(min=1e-12)[:, None]
        else:
            return self.conf

class IoU_numpy(metric.Metric):
    """Computes the intersection over union (IoU) per class and corresponding
    mean (mIoU).
    Intersection over union (IoU) is a common evaluation metric for semantic
    segmentation. The predictions are first accumulated in a confusion matrix
    and the IoU is computed from it as follows:
        IoU = true_positive / (true_positive + false_positive + false_negative).
    Keyword arguments:
    - num_classes (int): number of classes in the classification problem
    - normalized (boolean, optional): Determines whether or not the confusion
    matrix is normalized or not. Default: False.
    - ignore_index (int or iterable, optional): Index of the classes to ignore
    when computing the IoU. Can be an int, or any iterable of ints.
    """

    def __init__(self, num_classes, normalized=False, ignore_index=None):
        super().__init__()
        self.conf_metric = ConfusionMatrix_numpy(num_classes, normalized)

        if ignore_index is None:
            self.ignore_index = None
        elif isinstance(ignore_index, int):
            self.ignore_index = (ignore_index,)
        else:
            try:
                self.ignore_index = tuple(ignore_index)
            except TypeError:
                raise ValueError("'ignore_index' must be an int or iterable")

    def reset(self):
        self.conf_metric.reset()

    def add(self, predicted, target):
        """Adds the predicted and target pair to the IoU metric.
        Keyword arguments:
        - predicted (Tensor): (N, H, W) tensor of integer values between 0 and K-1.
        - target (Tensor): (N, H, W) tensor of integer values between 0 and K-1.
        """

        self.conf_metric.add(predicted.flatten(), target.flatten())

    def value(self):
        """Computes the IoU and mean IoU.
        The mean computation ignores NaN elements of the IoU array.
        Returns:
            Tuple: (IoU, mIoU). The first output is the per class IoU,
            for K classes it's numpy.ndarray with K elements. The second output,
            is the mean IoU.
        """
        conf_matrix = self.conf_metric.value()
        if self.ignore_index is not None:
            conf_matrix[:, self.ignore_index] = 0
            conf_matrix[self.ignore_index, :] = 0
        true_positive = np.diag(conf_matrix)
        false_positive = np.sum(conf_matrix, 0) - true_positive # np.sum(conf_matrix, 0) == pred_count
        false_negative = np.sum(conf_matrix, 1) - true_positive # np.sum(conf_matrix, 0) == actual_count (See https://medium.com/@cyborg.team.nitr/miou-calculation-4875f918f4cb )

        # Just in case we get a division by 0, ignore/hide the error
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative) # I = true_positive = diag(conf), U = true_pos + false_pos + false_neg = actual_count + pred_count - I

        return iou, np.nanmean(iou)
def project_to_mesh(from_mesh, to_mesh, attribute, dist_thresh=None):
    """ Transfers attributs from from_mesh to to_mesh using nearest neighbors

    Each vertex in to_mesh gets assigned the attribute of the nearest
    vertex in from mesh. Used for semantic evaluation.

    Args:
        from_mesh: Trimesh with known attributes
        to_mesh: Trimesh to be labeled
        attribute: Which attribute to transfer
        dist_thresh: Do not transfer attributes beyond this distance
            (None transfers regardless of distacne between from and to vertices)

    Returns:
        Trimesh containing transfered attribute
    """

    if len(from_mesh.vertices) == 0:
        to_mesh.vertex_attributes[attribute] = np.zeros((0), dtype=np.uint8)
        to_mesh.visual.vertex_colors = np.zeros((0), dtype=np.uint8)
        return to_mesh

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(from_mesh.vertices)
    kdtree = o3d.geometry.KDTreeFlann(pcd)

    pred_ids = from_mesh.vertex_attributes[attribute]
    pred_colors = from_mesh.visual.vertex_colors

    matched_ids = np.zeros((to_mesh.vertices.shape[0]), dtype=np.uint8)
    matched_colors = np.zeros((to_mesh.vertices.shape[0], 4), dtype=np.uint8)

    for i, vert in enumerate(to_mesh.vertices):
        _, inds, dist = kdtree.search_knn_vector_3d(vert, 1)
        if dist_thresh is None or dist[0]<dist_thresh:
            matched_ids[i] = pred_ids[inds[0]]
            matched_colors[i] = pred_colors[inds[0]]

    mesh = to_mesh.copy()
    mesh.vertex_attributes[attribute] = matched_ids
    mesh.visual.vertex_colors = matched_colors
    return mesh

def main(args): 
    gt_path = args.gt_path
    pred_path = args.pred_path
    with open(args.scene_list) as f:
        lines = f.readlines()
    scene_list = [line.strip() for line in lines]
    num_classes = 41
    ignore_index = [0,13,15,17,18,19,20,21,22,23,25,26,27,29,30,31,32,35,37,38,40]
    metric = IoU_numpy(num_classes=num_classes, ignore_index=ignore_index)
    metric.reset()
    for scene in scene_list: 
        print('Processing %s...'%(scene))
        mesh_gt = trimesh.load(os.path.join(gt_path, 'mesh', scene + '.ply'), process = False)
        mesh_predicted = trimesh.load(os.path.join(pred_path, 'mesh', scene + '.ply'), process = False)
        file_attributes_predicted = os.path.join(pred_path, 'semseg_data', '%s_attributes.npz'%scene)
        file_attributes_gt = os.path.join(gt_path, 'semseg_data', '%s_attributes.npz'%scene)
        mesh_predicted.vertex_attributes = np.load(file_attributes_predicted)
        semseg_gt = np.load(file_attributes_gt)['semseg']
        semseg_gt[semseg_gt < 0] = 0
        mesh_transfer = project_to_mesh(mesh_predicted, mesh_gt, 'semseg')

        semseg_pred = mesh_transfer.vertex_attributes['semseg']
        metric.add(semseg_pred, semseg_gt)

        transfer_path = os.path.join(pred_path, 'transfer')
        if not os.path.isdir(transfer_path):
            os.makedirs(transfer_path)
        mesh_transfer.export(os.path.join(transfer_path, '%s_transfer.ply'%scene))
    iou,miou = metric.value()
    for label, class_iou in zip(CLASS_LABELS, iou):
        print("{0}: {1:.4f}".format(label, class_iou))
    print('TEST mIoU: %.3f' %(miou))


if __name__ =='__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', help='Path to folder containing GT segmented scene') # /home/trung/segmentation_whole_scenes/gt_scenes
    parser.add_argument('--pred_path', help='Path to folder containing predicted scene, including .ply for mesh and .npz file for semseg data') # /home/trung/segmentation_whole_scenes/semseg_10-09_19-56
    parser.add_argument('--scene_list', help = 'Path to scannetv2_val.txt')

    args = parser.parse_args()
    print(args)
    main(args)