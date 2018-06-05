import tensorflow as tf
from tensorflow.contrib import slim
import numpy as np


def resize3D(input_layer, width_factor, height_factor, depth_factor):
    shape = input_layer.shape
    print(shape)

    rsz1 = tf.image.resize_images(tf.reshape(input_layer, [-1, shape[1], shape[2], shape[3] * shape[4]]),
                                  [shape[1] * width_factor, shape[2] * height_factor])
    rsz2 = tf.image.resize_images(tf.reshape(tf.transpose(
        tf.reshape(rsz1, [-1, shape[1] * width_factor, shape[2] * height_factor, shape[3], shape[4]]),
        [0, 3, 2, 1, 4]), [-1, shape[3], shape[2] * height_factor, shape[1] * width_factor * shape[4]]),
                                  [shape[3] * depth_factor, shape[2] * height_factor])

    return tf.transpose(tf.reshape(rsz2, [-1, shape[3]*depth_factor, shape[2]*height_factor, shape[1]*width_factor, shape[4]]), [0, 3, 2, 1, 4])

unpool_idx = 0


def unpool(inputs):

    global unpool_idx
    shape = inputs.get_shape().as_list()

    res = resize3D(inputs, 2.0, 2.0, 2.0)
    res = slim.conv3d(res, num_outputs=shape[-1], kernel_size=[3, 3, 3], stride=1, scope='unpool_' + str(unpool_idx),
                      activation_fn=tf.nn.relu)
    res = slim.batch_norm(res, activation_fn=tf.nn.relu)
    unpool_idx += 1
    return res


def random_crop_and_pad_image_and_labels(image, labels, size, axis=2):
  """Randomly crops `image` together with `labels`.

  Args:
    image: A Tensor with shape [D_1, ..., D_K, N]
    labels: A Tensor with shape [D_1, ..., D_K, M]
    size: A Tensor with shape [K] indicating the crop size.
    axis: The dimension index of combination
  Returns:
    A tuple of (cropped_image, cropped_label).
  """
  combined = tf.concat([image, labels], axis=axis)
  print(combined)
  image_shape = tf.shape(image)
  print(image_shape)
  print(size)
  # combined_pad = tf.image.pad_to_bounding_box(
  #     combined, 0, 0,
  #     tf.maximum(size[0], image_shape[0]),
  #     tf.maximum(size[1], image_shape[1]))
  combined_pad = combined
  print('combined_pad: ', combined_pad)
  last_label_dim = tf.shape(labels)[-1]
  last_image_dim = tf.shape(image)[-1]
  print(np.concatenate([size, [last_label_dim, last_image_dim]], axis=0))
  combined_crop = tf.random_crop(
      combined_pad,
      size=tf.concat([size, [last_label_dim + last_image_dim]],
                     axis=0))
  if axis == 2:
      return (combined_crop[:, :, :last_image_dim],
              combined_crop[:, :, last_image_dim:])
  if axis == 3:
      return (combined_crop[:, :, :, :last_image_dim],
              combined_crop[:, :, :, last_image_dim:])
  if axis == 4:
      return (combined_crop[:, :, :, :, :last_image_dim],
              combined_crop[:, :, :, :, last_image_dim:])


def model(input_tensor, weight_decay=1e-5, is_training=True):
    batch_norm_params = {
        'decay': 0.997,
        'epsilon': 1e-5,
        'scale': True,
        'is_training': is_training
    }
    with slim.arg_scope([slim.conv3d, slim.conv3d_transpose],
                        # normalizer_fn=slim.batch_norm,
                        activation_fn=tf.nn.relu,
                        # normalizer_params=batch_norm_params,
                        # weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        # weights_regularizer=slim.l2_regularizer(0.005),
                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        weights_regularizer=slim.l2_regularizer(0.005)):
        with slim.arg_scope([slim.batch_norm], is_training=is_training):
            # `[batch_size] + input_spatial_shape + [in_channels]` if data_format does
            scale1_1 = slim.conv3d(input_tensor, kernel_size=[9, 9, 7], num_outputs=8)
            scale1_1 = slim.batch_norm(scale1_1, activation_fn=tf.nn.relu)
            scale1_2 = slim.conv3d(scale1_1, kernel_size=[9, 9, 7], num_outputs=8)
            scale1_2 = slim.batch_norm(scale1_2, activation_fn=tf.nn.relu)
            down_pooling1 = slim.max_pool3d(scale1_2, kernel_size=[2, 2, 2], stride=2, padding='SAME')

            scale2_1 = slim.conv3d(down_pooling1, kernel_size=[7, 7, 5], num_outputs=16)
            scale2_1 = slim.batch_norm(scale2_1, activation_fn=tf.nn.relu)
            scale2_2 = slim.conv3d(scale2_1, kernel_size=[7, 7, 5], num_outputs=32)
            scale2_2 = slim.batch_norm(scale2_2, activation_fn=tf.nn.relu)
            down_pooling2 = slim.max_pool3d(scale2_2, kernel_size=[2, 2, 2], stride=2, padding='SAME')

            scale3_1 = slim.conv3d(down_pooling2, kernel_size=[5, 5, 3], num_outputs=32)
            scale3_1 = slim.batch_norm(scale3_1, activation_fn=tf.nn.relu)
            scale3_2 = slim.conv3d(scale3_1, kernel_size=[1, 1, 1], num_outputs=32)
            scale3_2 = slim.batch_norm(scale3_2, activation_fn=tf.nn.relu)


            print(scale3_2)
            with tf.variable_scope('predition_last'):
                # up_pooling1_1 = slim.conv3d_transpose(scale3_2, num_outputs=32, kernel_size=[3, 3, 3], stride=[2, 2, 2])
                # up_pooling1_2 = slim.conv3d(up_pooling1_1, num_outputs=2, kernel_size=[3, 3, 3], stride=[2, 2, 2])
                up_pooling1_1 = unpool(scale3_2)
                up_pooling1_2 = unpool(up_pooling1_1)
                pred_last = slim.conv3d(up_pooling1_2, kernel_size=[3, 3, 3], num_outputs=32)
                pred_last = slim.batch_norm(pred_last)
                pred_last = slim.conv3d(pred_last, kernel_size=[1, 1, 1], num_outputs=2, activation_fn=None)

            with tf.variable_scope('prediction_layer6'):
                # up_pooling2_1 = slim.conv3d_transpose(down_pooling2, num_outputs=32, kernel_size=[3, 3, 3],
                #                                       stride=[2, 2, 2])
                # up_pooling2_2 = slim.conv3d_transpose(up_pooling2_1, num_outputs=2, kernel_size=[3, 3, 3], stride=[2, 2, 2])
                up_pooling2_1 = unpool(down_pooling2)
                up_pooling2_2 = unpool(up_pooling2_1)
                pred_6 = slim.conv3d(up_pooling2_2, kernel_size=[3, 3, 3], num_outputs=32)
                pred_6 = slim.batch_norm(pred_6)
                pred_6 = slim.conv3d(pred_6, kernel_size=[1, 1, 1], num_outputs=2, activation_fn=None)

            with tf.variable_scope('prediction_layer3'):
                # up_pooling3_1 = slim.conv3d_transpose(down_pooling1, num_outputs=2, kernel_size=[3, 3, 3], stride=[2, 2, 2])
                up_pooling3_1 = unpool(down_pooling1)
                pred_3 = slim.conv3d(up_pooling3_1, kernel_size=[3, 3, 3], num_outputs=32)
                pred_3 = slim.batch_norm(pred_3)
                pred_3 = slim.conv3d(pred_3, kernel_size=[1, 1, 1], num_outputs=2, activation_fn=None)
            print(pred_last, pred_6, pred_3)

            return pred_last, pred_6, pred_3


def build_loss(gt_tensor, pred_last, pred_6, pred_3, global_step):
    # print(gt_tensor)
    # print(pred_last)
    # print(pred_6)
    # print(pred_3)
    # cross_entropy_last = tf.reduce_mean(
    #     tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_tensor, logits=pred_last + 1e-7))
    # print(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_tensor, logits=pred_last))
    # cross_entropy_6 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_tensor, logits=pred_6))
    # cross_entropy_3 = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_tensor, logits=pred_3))
    # lambda1 = tf.train.exponential_decay(0.4, global_step=global_step, decay_steps=1000, decay_rate=0.95, staircase=True)
    # lambda2 = tf.train.exponential_decay(0.3, global_step=global_step, decay_steps=1000, decay_rate=0.95, staircase=True)
    # tf.summary.scalar('lambda1', lambda1)
    # tf.summary.scalar('lambda2', lambda2)
    # model_loss = cross_entropy_last + lambda1 * cross_entropy_6 + lambda2 * cross_entropy_3
    weights = tf.constant([1, 26.91], dtype=tf.float32, name='weights')
    # cross_entropy_last, _, dice_last = loss_with_binary_dice(pred_last, gt_tensor, weights)
    # cross_entropy_6, _, dice_6 = loss_with_binary_dice(pred_6, gt_tensor, weights)
    # cross_entropy_3, _, dice_3 = loss_with_binary_dice(pred_3, gt_tensor, weights)
    # tf.summary.scalar('dice_last', dice_last)
    # tf.summary.scalar('dice_6', dice_6)
    # tf.summary.scalar('dice_3', dice_3)
    cross_entropy_last = tf.reduce_mean(weights_cross_entropy_loss(pred_last, gt_tensor, weights)[0])
    cross_entropy_6 = tf.reduce_mean(weights_cross_entropy_loss(pred_6, gt_tensor, weights)[0])
    cross_entropy_3 = tf.reduce_mean(weights_cross_entropy_loss(pred_3, gt_tensor, weights)[0])
    lambda1 = tf.train.exponential_decay(0.4, global_step=global_step, decay_steps=1000, decay_rate=0.95, staircase=True)
    lambda2 = tf.train.exponential_decay(0.3, global_step=global_step, decay_steps=1000, decay_rate=0.95, staircase=True)
    tf.summary.scalar('lambda1', lambda1)
    tf.summary.scalar('lambda2', lambda2)

    model_loss = cross_entropy_last + lambda1 * cross_entropy_6 + lambda2 * cross_entropy_3
    # model_loss = cross_entropy_last
    return model_loss, cross_entropy_last, cross_entropy_6, cross_entropy_3

def static_weighted_softmax_cross_entropy_loss(logits, labels, weights, factor=0.5):
    logits = tf.reshape(logits, [-1, tf.shape(logits)[4]], name='flatten_logits')
    labels = tf.reshape(labels, [-1], name='flatten_labels')

    # get predictions from likelihoods
    prediction = tf.argmax(logits, 1, name='predictions')

    # get maps of class_of_interest pixels
    predictions_hit = tf.to_float(tf.equal(prediction, 1), name='predictions_weight_map')
    labels_hit = tf.to_float(tf.equal(labels, 1), name='labels_weight_map')

    predictions_weight_map = predictions_hit * ((factor * weights[1]) - weights[0]) + weights[0]
    labels_weight_map = labels_hit * (weights[1] - weights[0]) + weights[0]

    weight_map = tf.maximum(predictions_weight_map, labels_weight_map, name='combined_weight_map')
    weight_map = tf.stop_gradient(weight_map, name='stop_gradient')

    # compute cross entropy loss
    """
    - new tf version!
    Positional arguments are not allowed anymore. was (logits, labels, name=) instead of (logits=logits, labels=labels, name=)
    """
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels,
                                                                   name='cross_entropy_softmax')

    # apply weights to cross entropy loss
    """
    - new tf version!
    tf.mul -> tf.multiply
    """
    weighted_cross_entropy = tf.multiply(weight_map, cross_entropy, name='apply_weights')

    # get loss scalar
    loss = tf.reduce_mean(weighted_cross_entropy, name='loss')

    # print ("loss", loss)

    return loss, weight_map

def weights_cross_entropy_loss(logits, labels, weights):
    with tf.name_scope('loss'):
        loss, weight_map = static_weighted_softmax_cross_entropy_loss(logits, labels, weights)
    return loss, weight_map


def loss_with_binary_dice(logits, labels, weights, axis=[1, 2, 3], smooth=1e-7):
    weighted_cross_entropy, weight_map = weights_cross_entropy_loss(logits, labels, weights)

    softmaxed = tf.nn.softmax(logits)[:, :, :, :, 1]

    cond = tf.less(softmaxed, 0.5)
    output = tf.where(cond, tf.zeros(tf.shape(softmaxed)), tf.ones(tf.shape(softmaxed)))

    target = labels  # tf.one_hot(labels, depth=2)

    # Make sure inferred shapes are equal during graph construction.
    # Do a static check as shapes won't change dynamically.
    print(output.get_shape())
    print(target.get_shape())
    assert output.get_shape().as_list() == target.get_shape().as_list()

    with tf.name_scope('dice'):
        output = tf.cast(output, tf.float32)
        target = tf.cast(target, tf.float32)
        inse = tf.reduce_sum(output * target, axis=axis)
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
        dice = (2. * inse + smooth) / (l + r + smooth)
        dice = tf.reduce_mean(dice)

    with tf.name_scope('final_cost'):
        final_cost = (weighted_cross_entropy + (1 - dice) + (1 / dice) ** 0.3 - 1) / 3

    return final_cost, weight_map, dice
if __name__ == '__main__':
    # test resize
    input_tensor = tf.placeholder(tf.float32, shape=[None, 160, 160, 72, 1], name='input_tensor')
    output = resize3D(input_tensor, 2.0, 2.0, 2.0)
    print(output)


    # test model function
    # input_tensor = tf.placeholder(tf.float32, shape=[None, 160, 160, 72, 1], name='input_tensor')
    # up_pooling1, up_pooling2, up_pooling3 = model(input_tensor)
    # with tf.Session() as sess:
    #     sess.run(tf.global_variables_initializer())
    #     for i in range(100):
    #         sess.run(
    #             up_pooling1,
    #             feed_dict={
    #                 input_tensor: np.random.random([30, 160, 160, 72, 1])
    #             }
    #         )
    #         print(i)

    # test random_crop_and_pad_image_and_labels function
    # input_tensor = tf.placeholder(tf.float32, shape=[10, 512, 512, None, 1], name='input_tensor')
    # label_tensor = tf.placeholder(tf.float32, shape=[10, 512, 512, None, 1], name='label_tensor')
    # croped_image_tensor, croped_label_tensor = random_crop_and_pad_image_and_labels(input_tensor, label_tensor,
    #                                                                                 [10, 160, 160, 72], axis=4)
    # print(croped_image_tensor)
    # print(croped_label_tensor)
    # with tf.Session() as sess:
    #     # sess.run(tf.global_variables_initializer())
    #     # croped_image_value = sess.run(croped_image_tensor, feed_dict={
    #     #     input_tensor: np.random.random([10, 512, 512, 100, 1]),
    #     #     label_tensor: np.random.random([10, 512, 512, 100, 1])
    #     # })
    #     # print(np.shape(croped_image_value))
    #
    #     cropped_image, cropped_labels = random_crop_and_pad_image_and_labels(
    #         image=tf.reshape(tf.range(2 * 4 * 4 * 3), [2, 4, 4, 3]),
    #         labels=tf.reshape(tf.range(2 * 4 * 4), [2, 4, 4, 1]),
    #         size=[2, 2, 2], axis=3)
    #
    #     with tf.Session() as session:
    #         # print(np.reshape(range(2 * 4 * 4 * 3), [2, 4, 4, 3]))
    #         croped_image, croped_label = session.run([cropped_image, cropped_labels])
    #         print(np.shape(croped_image))
    #         print(np.shape(croped_label))
    #         # print(croped_image)