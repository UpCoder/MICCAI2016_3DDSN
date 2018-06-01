# -*- coding=utf-8 -*-
# 主要是用来从nii格式的文件中提取patch数组
from config import Config
from tools.medicalImage import read_nii, save_mhd_image
from tools.utils import split_array
import numpy as np
import os
from glob import glob
from multiprocessing import Pool


def extract_voxel_patch(image_path, gt_path=None,
                        save_dir='/home/give/Documents/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training Batch 1_Patch'):
    image_volume = read_nii(image_path)
    if gt_path is not None:
        gt_volume = read_nii(gt_path)
        gt_volume = np.asarray(np.asarray(gt_volume) >= 1, np.float32)
    shape = list(np.shape(image_volume))
    batch_image = []
    if gt_path is not None:
        batch_gt = []
    for x in range(0, shape[0], Config.vox_size[0]):
        for y in range(0, shape[1], Config.vox_size[1]):
            for z in range(0, shape[2], Config.vox_size[2]):
                end_x = x + Config.vox_size[0]
                end_y = y + Config.vox_size[1]
                end_z = z + Config.vox_size[2]
                if end_x > shape[0]:
                    x = shape[0] - Config.vox_size[0]
                    end_x = shape[0]
                if end_y > shape[1]:
                    y = shape[1] - Config.vox_size[1]
                    end_y = shape[1]
                if end_z > shape[2]:
                    z = shape[2] - Config.vox_size[2]
                    end_z = shape[2]
                cur_image = image_volume[x:end_x, y:end_y, z:end_z]
                cur_gt = gt_volume[x:end_x, y:end_y, z:end_z]
                batch_image.append(cur_image)
                batch_gt.append(cur_gt)
                if z == shape[2] - Config.vox_size[2]:
                    break
            if y == shape[1] - Config.vox_size[1]:
                break
        if x == shape[0] - Config.vox_size[0]:
            break
    print('ImageBatch shape is ', np.shape(batch_image), 'from ', os.path.basename(image_path))
    print('GTBatch shape is ', np.shape(batch_gt), ' from ', os.path.basename(gt_path))
    for idx, (image, gt) in enumerate(zip(batch_image, batch_gt)):
        basename = os.path.basename(image_path)
        file_id = basename.split('.')[0].split('-')[1]
        gt_mean = np.mean(np.asarray(gt, np.float32))
        # print(idx, ' gt_mean: ', gt_mean, np.shape(image))
        if gt_mean > 0.01:
            image_save_path = os.path.join(save_dir, 'positive')
            gt_save_path = os.path.join(save_dir, 'positive')
        else:
            image_save_path = os.path.join(save_dir, 'negative')
            gt_save_path = os.path.join(save_dir, 'negative')
        image_save_path = os.path.join(image_save_path, 'volume-' + file_id + '_' + str(idx) + '.nii')
        gt_save_path = os.path.join(gt_save_path, 'segmentation-' + file_id + '_' + str(idx) + '.nii')
        save_mhd_image(np.transpose(image, axes=[2, 0, 1]), image_save_path)
        save_mhd_image(np.transpose(gt, axes=[2, 0, 1]), gt_save_path)
    return True


def single_process(batch_image_paths, dir_path, save_dir_path, process_id):
    for idx, image_path in enumerate(batch_image_paths):
        basename = os.path.basename(image_path)
        file_id = basename.split('-')[1].split('.')[0]
        gt_path = os.path.join(dir_path, 'segmentation-' + file_id + '.nii')
        extract_voxel_patch(image_path, gt_path, save_dir_path)
        print('Processed %d / %d at %d' % (idx, len(batch_image_paths), process_id))

def extract_voxel_patch_fromdir(dir_path, save_dir_path):

    image_paths = glob(os.path.join(dir_path, 'volume-*.nii'))
    num_processor = 8
    pool = Pool(processes=num_processor)
    splited_paths = split_array(image_paths, num_processor)
    results = []
    for i in range(num_processor):
        results.append(pool.apply_async(single_process, args=[splited_paths[i], dir_path, save_dir_path, i, ]))
    pool.close()
    pool.join()
    for i in range(num_processor):
        results[i].get()


if __name__ == '__main__':
    # extract_voxel_patch(
    #     '/home/give/Documents/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training Batch 1/volume-0.nii',
    #     '/home/give/Documents/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training Batch 1/segmentation-0.nii',
    #     save_dir='/home/give/Documents/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training Batch 1_Patch'
    # )
    extract_voxel_patch_fromdir(
        '/home/give/Documents/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training Batch 2',
        '/home/give/Documents/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training Batch 2_Patch'
    )