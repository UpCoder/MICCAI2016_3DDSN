import numpy as np
from glob import glob
import os
from data.Generator import Generator
from tools.medicalImage import read_nii, save_mhd_image
from tools.utils import random_crop, random_crop_slice, shuffle
from config import Config

class Reader2:
    '''
    根据事先已经提取出来的ｐａｔｃｈ（perparation.py 的功能）进行数据的准备工作
    '''
    def __init__(self, train_dir, val_dir, batch_size):
        train_image_path＿p, train_gt_path_p = Reader2.get_image_gt_paths(os.path.join(train_dir, 'positive'))
        train_image_path＿n, train_gt_path_n = Reader2.get_image_gt_paths(os.path.join(train_dir, 'negative'))
        val_image_path_p, val_gt_path_p = Reader2.get_image_gt_paths(os.path.join(val_dir, 'positive'))
        val_image_path_n, val_gt_path_n = Reader2.get_image_gt_paths(os.path.join(val_dir, 'negative'))
        self.train_generator_p = Generator([train_image_path＿p, train_gt_path_p], batch_size=batch_size/2).next_batch()
        self.train_generator_n = Generator([train_image_path＿n, train_gt_path_n], batch_size=batch_size/2).next_batch()
        self.val_generator_p = Generator([val_image_path_p, val_gt_path_p], batch_size=batch_size/2).next_batch()
        self.val_generator_n = Generator([val_image_path_n, val_gt_path_n], batch_size=batch_size/2).next_batch()

    @staticmethod
    def get_image_gt_paths(data_dir):
        image_paths = glob(os.path.join(data_dir, 'volume-*'))
        gt_paths = []
        for image_path in image_paths:
            basename = os.path.basename(image_path)
            file_id = basename.split('-')[1]
            gt_paths.append(os.path.join(data_dir, 'segmentation-' + file_id))
        return np.asarray(image_paths), np.asarray(gt_paths)

    @staticmethod
    def static_scalar(image, min_val=-300, max_val=500):
        image = np.asarray(image)
        image[image < min_val] = min_val
        image[image > max_val] = max_val
        interv = max_val - min_val
        image /= (interv / 2)
        return image

    def get_next_batch(self, is_training):
        if is_training:
            p_image_paths, p_gt_paths = self.train_generator_p.__next__()
            n_image_paths, n_gt_paths = self.train_generator_n.__next__()
        else:
            p_image_paths, p_gt_paths = self.val_generator_p.__next__()
            n_image_paths, n_gt_paths = self.val_generator_n.__next__()
        image_paths = np.concatenate([p_image_paths, n_image_paths], axis=0)
        gt_paths = np.concatenate([p_gt_paths, n_gt_paths], axis=0)
        image_paths, gt_paths = shuffle(list(zip(image_paths, gt_paths)), is_single=False)
        batch_images = [read_nii(image_path) for image_path in image_paths]
        batch_gts = [read_nii(gt_path) for gt_path in gt_paths]
        batch_images = Reader2.static_scalar(batch_images)
        batch_gts = np.asarray(np.asarray(batch_gts) >= 1.0, np.float32)
        return batch_images, batch_gts


if __name__ == '__main__':
    reader = Reader2(
        '/home/give/Documents/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training Batch 2_Patch',
        '/home/give/Documents/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training Batch 1_Patch',
        batch_size=2
    )
    for _ in range(100):
        batch_images, batch_gts = reader.get_next_batch(is_training=False)
        print(np.shape(batch_images))