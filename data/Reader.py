import numpy as np
from glob import glob
import os
from data.Generator import Generator
from tools.medicalImage import read_nii, save_mhd_image
from tools.utils import random_crop, random_crop_slice
from config import Config

class Reader:
    def __init__(self, train_dir, val_dir, batch_size):
        train_image_path, train_gt_path = Reader.get_image_gt_paths(train_dir)
        val_image_path, val_gt_path = Reader.get_image_gt_paths(val_dir)
        self.train_generator = Generator([train_image_path, train_gt_path], batch_size=batch_size).next_batch()
        self.val_generator = Generator([val_image_path, val_gt_path], batch_size=batch_size).next_batch()

    @staticmethod
    def get_image_gt_paths(data_dir):
        image_paths = glob(os.path.join(data_dir, 'volume-*'))
        gt_paths = []
        for image_path in image_paths:
            basename = os.path.basename(image_path)
            file_id = basename.split('.')[0].split('-')[1]
            gt_paths.append(os.path.join(data_dir, 'segmentation-' + file_id + '.nii'))
        return image_paths, gt_paths

    @staticmethod
    def static_scalar(image, min_val, max_val):
        image[image < min_val] = min_val
        image[image > max_val] = max_val
        interv = max_val - min_val
        image /= (interv / 2)
        return image


    @staticmethod
    def processing1(image_paths, gt_paths, crop_num):
        # 随机从512*512*XX里面挑选出固定size的patch
        processed_images = []
        processed_gts = []
        for (image_path, gt_path) in zip(image_paths, gt_paths):
            image = read_nii(image_path)
            gt = read_nii(gt_path)
            # print(np.shape(image), image_path)
            # print(np.shape(gt), gt_path)
            crop_results = random_crop([image, gt], pointed_size=Config.vox_size, crop_num=crop_num)
            processed_images.extend(crop_results[:, 0, :, :, :])
            processed_gts.extend(crop_results[:, 1, :, :, :])
        processed_images = np.asarray(processed_images, dtype=np.float32)
        processed_images = Reader.static_scalar(processed_images, -300, 500)
        processed_gts = np.asarray(processed_gts, dtype=np.float32)
        processed_gts[processed_gts >= 1.0] = 1.0
        return processed_images, processed_gts

    @staticmethod
    def processing2(image_paths, gt_paths, slice_num):
        # 从512*512*XX中随机挑选几个slice
        processed_images = []
        processed_gts = []
        for (image_path, gt_path) in zip(image_paths, gt_paths):
            image = read_nii(image_path)
            gt = read_nii(gt_path)
            # print(np.shape(image), image_path)
            # print(np.shape(gt), gt_path)
            crop_results = random_crop_slice([image, gt], slice_num=slice_num)
            processed_images.append(crop_results[0, :, :, :])
            processed_gts.append(crop_results[1, :, :, :])
        # processed_images = np.asarray(processed_images, np.float32)
        # processed_gts = np.asarray(processed_gts, np.float32)
        # processed_result = random_crop_slice([processed_images, processed_gts], slice_num=slic)
        processed_images = np.asarray(processed_images, dtype=np.float32)
        processed_images = Reader.static_scalar(processed_images, -300, 500)
        processed_gts = np.asarray(processed_gts, dtype=np.float32)
        processed_gts[processed_gts >= 1.0] = 1.0
        return processed_images, processed_gts

if __name__ == '__main__':
    reader = Reader(
        '/home/give/Documents/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training Batch 2',
        '/home/give/Documents/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training Batch 1'
    )
    for _ in range(100):
        train_img_patch_batch, train_gt_path_batch = reader.train_generator.__next__()
        print(train_img_patch_batch)
        print(train_gt_path_batch)
        images, gts = Reader.processing(train_img_patch_batch, train_gt_path_batch)
        print(np.shape(images), np.shape(gts))