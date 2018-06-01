# -*- coding=utf-8 -*-
import numpy as np


def random_crop(array, pointed_size, crop_num):
    '''
    从一个数组中随机crop出合适的大小
    :param array: 执行crop操作的数组的数组
    :param pointed_size: 指定的大小
    :param crop_num: crop的次数
    :return:
    '''
    shape = np.shape(array)
    if len(shape) != (len(pointed_size) + 1):
        print('The length of dimension is not equal!')
        assert False
    num_arrays = len(array)
    previous_shape = None
    for idx in range(num_arrays):
        if previous_shape is None:
            previous_shape = np.shape(array[idx])
            continue
        else:
            if previous_shape != np.shape(array[idx]):
                print('The shape of the element in the array is not equal!')
                assert False
            previous_shape = np.shape(array[idx])
            continue
    changed_ranges = []
    for idx in range(1, len(shape)):
        # print(shape, idx)
        changed_ranges.append(shape[idx] - pointed_size[idx-1])
    crop_results = []
    for idx in range(crop_num):
        random_start1 = np.random.randint(0, changed_ranges[0])
        random_start2 = np.random.randint(0, changed_ranges[1])
        random_start3 = np.random.randint(0, changed_ranges[2])
        crop_result = []
        for ele_idx in range(num_arrays):
            crop_result.append(array[ele_idx][random_start1: random_start1 + pointed_size[0],
                               random_start2: random_start2 + pointed_size[1],
                               random_start3: random_start3 + pointed_size[2]])
        crop_results.append(crop_result)
    return np.asarray(crop_results, np.float32)


def random_crop_slice(array, slice_num):
    '''
    从一个数组中随机crop连续的指定slice
    :param array: 执行crop操作的数组的数组[512, 512, total_slice_num]
    :param pointed_size: 指定的大小
    :param crop_num: crop的次数
    :return:
    '''
    shape = np.shape(array)
    num_arrays = len(array)
    previous_shape = None
    for idx in range(num_arrays):
        if previous_shape is None:
            previous_shape = np.shape(array[idx])
            continue
        else:
            if previous_shape != np.shape(array[idx]):
                print('The shape of the element in the array is not equal!')
                assert False
            previous_shape = np.shape(array[idx])
            continue
    start_index = np.random.randint(0, shape[-1]-slice_num)
    crop_results = []
    for ele_idx in range(num_arrays):
        crop_results.append(array[ele_idx][:, :, start_index:start_index + slice_num])
    return np.asarray(crop_results, np.float32)


def split_array(array, num_parts):
    '''
    将一个数组拆分成多个数组
    :param array: 待拆分的数组
    :param num_parts: the number of target parts
    :return:
    '''
    array = np.asarray(array)
    num_elements = len(array)
    mod_num = num_elements % num_parts
    mod_data = array[:mod_num]
    array = array[mod_num:]
    split_res = np.split(array, num_parts)
    split_last = split_res[-1]
    split_last = np.concatenate([split_last, mod_data], axis=0)
    split_res[-1] = split_last
    return split_res


def shuffle(array, is_single=True):
    '''
    打乱一个或者是多个数组的顺序
    :param array: 如果是多个数组，则array = list(zip(zrray1, arr2, arr3 ...))
    :param is_single: 如果是多个数组则为False
    :return:
    '''
    np.random.shuffle(array)
    if is_single:
        return array
    else:
        return zip(*array)


def calculate_dicescore(gt_sample, pred_sample):
    '''
    计算两个sample的dice score
    :param gt_sample: [None, 512, 512]
    :param pred_sample: [None, 512, 512]
    :return:
    '''
    print(np.shape(gt_sample))
    print(np.shape(pred_sample))
    gt_shape = np.shape(gt_sample)
    pred_shape = np.shape(pred_sample)
    if gt_shape != pred_shape:
        print('Shape not equal!', gt_shape, pred_shape)
        assert False
    dice_scores = []
    for i in range(len(gt_sample)):
        gt_slice = gt_sample[i]
        pred_slice = pred_sample[i]
        smooth = 1e-7
        intersection = np.sum(gt_slice * pred_slice)
        l = np.sum(gt_slice)
        r = np.sum(pred_slice)
        print('Intersection: {}, l: {}, r: {}'.format(intersection, l, r))
        dice_score = (2.0 * intersection + smooth) / (1.0 * l + 1.0 * r + smooth)
        dice_scores.append(dice_score)
    return np.mean(dice_scores)

if __name__ == '__main__':
    # array = np.random.random([512, 512, 128])
    # crop_reuslt = random_crop([array, array], [160, 160, 70], 10)
    # print(np.shape(crop_reuslt))
    array = range(8)
    array1 = array[::-1]
    a, b = shuffle(list(zip(array, array1)), is_single=False)
    print(a)
    print(b)
    # print(split_array(array, 3))