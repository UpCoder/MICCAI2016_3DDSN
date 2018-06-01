import tensorflow as tf
from tensorflow.contrib import slim
from network.model import model, build_loss
from config import Config
import os
import numpy as np
# from data.Reader import Reader
from data.Reader2 import Reader2
tf.app.flags.DEFINE_float('learning_rate', 0.1, 'the learning rate')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_integer('max_steps', 1000000, '')
tf.app.flags.DEFINE_integer('snap_interval', 100, '')
tf.app.flags.DEFINE_integer('val_interval', 100, '')
tf.app.flags.DEFINE_integer('crop_num', 1, '')
tf.app.flags.DEFINE_integer('slice_num', 72, '')
tf.app.flags.DEFINE_string('model_save_path', '/home/give/PycharmProjects/MICCAI2016_3DDSN/outputs/snaps', '')
tf.app.flags.DEFINE_boolean('restore_flag', False, '')
tf.app.flags.DEFINE_string('model_restore_path', '/home/give/PycharmProjects/MICCAI2016_3DDSN/outputs/snaps', '')
tf.app.flags.DEFINE_string('summary_dir', '/home/give/PycharmProjects/MICCAI2016_3DDSN/outputs/logs', '')
tf.app.flags.DEFINE_integer('print_interval', 5, '')
tf.app.flags.DEFINE_integer('batch_size', 3, '')
FLAGS = tf.app.flags.FLAGS


def train():
    input_image_tensor = tf.placeholder(dtype=tf.float32,
                                        shape=[None, Config.vox_size[0], Config.vox_size[1], Config.vox_size[2], 1],
                                        name='image_tensor')
    input_gt_tensor = tf.placeholder(dtype=tf.int32,
                                     shape=[None, Config.vox_size[0], Config.vox_size[1], Config.vox_size[2]],
                                     name='gt_tensor')
    global_step = tf.train.get_or_create_global_step()
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=10000, decay_rate=0.1,
                                               staircase=True)
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    pred_last, pred_6, pred_3 = model(input_image_tensor)
    pred_last_softmax = tf.nn.softmax(pred_last)
    model_loss, cross_entropy_last, cross_entropy_6, cross_entropy_3 = build_loss(input_gt_tensor, pred_last, pred_6,
                                                                                  pred_3, global_step=global_step)
    total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    tf.summary.image('image', tf.transpose(input_image_tensor[0, :, :, 10:15], perm=[2, 0, 1, 3]), max_outputs=3)
    tf.summary.image('gt',
                     tf.expand_dims(tf.transpose(tf.cast(input_gt_tensor, tf.float32)[0, :, :, 10:15], perm=[2, 0, 1]),
                                    axis=3),
                     max_outputs=3)
    tf.summary.image('pred_last',
                     tf.expand_dims(
                         tf.transpose(tf.cast(tf.argmax(tf.nn.softmax(pred_last)[0, :, :, 10:15, :], axis=3), tf.uint8),
                                      perm=[2, 0, 1]),
                         axis=3),
                     max_outputs=3)
    tf.summary.image('pred_6',
                     tf.expand_dims(
                         tf.transpose(tf.cast(tf.argmax(tf.nn.softmax(pred_6)[0, :, :, 10:15, :], axis=3), tf.uint8),
                                      perm=[2, 0, 1]),
                         axis=3),
                     max_outputs=3)
    tf.summary.image('pred_3',
                     tf.expand_dims(
                         tf.transpose(tf.cast(tf.argmax(tf.nn.softmax(pred_3)[0, :, :, 10:15, :], axis=3), tf.uint8),
                                      perm=[2, 0, 1]),
                         axis=3),
                     max_outputs=3)
    tf.summary.scalar('loss/model_loss', model_loss)
    tf.summary.scalar('loss/total_loss', total_loss)
    tf.summary.scalar('loss/cross_entropy_last', cross_entropy_last)
    tf.summary.scalar('loss/cross_entropy_6', cross_entropy_6)
    tf.summary.scalar('loss/cross_entropy_3', cross_entropy_3)
    # grads = opt.compute_gradients(total_loss)
    # apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
    apply_gradient_op = opt.minimize(total_loss, global_step=global_step)
    variable_averages = tf.train.ExponentialMovingAverage(
        FLAGS.moving_average_decay, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    with tf.control_dependencies([variables_averages_op, apply_gradient_op]):
        train_op = tf.no_op(name='train_op')
    train_summary = tf.summary.FileWriter(os.path.join(FLAGS.summary_dir, 'train'), tf.get_default_graph())
    val_summary = tf.summary.FileWriter(os.path.join(FLAGS.summary_dir, 'val'), tf.get_default_graph())
    summary_op = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # Reader1 Version
        # reader = Reader(
        #     '/home/give/Documents/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training Batch 2',
        #     '/home/give/Documents/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training Batch 1',
        #     batch_size=FLAGS.batch_size
        # )
        # train_generator = reader.train_generator
        # val_generator = reader.val_generator
        reader = Reader2(
            '/home/give/Documents/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training Batch 2_Patch',
            '/home/give/Documents/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training Batch 1_Patch',
            batch_size=FLAGS.batch_size
        )
        start_step = 0
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        if FLAGS.restore_flag:
            ckpt = tf.train.latest_checkpoint(FLAGS.model_restore_path)
            print('continue training from previous checkpoint from %s' % ckpt)
            start_step = int(os.path.basename(ckpt).split('-')[1])
            variable_restore_op = slim.assign_from_checkpoint_fn(ckpt,
                                                                 slim.get_trainable_variables(),
                                                                 ignore_missing_vars=True)
            variable_restore_op(sess)
            sess.run(tf.assign(global_step, start_step))
        for step in range(start_step, FLAGS.max_steps + start_step):
            # train_img_path_batch, train_gt_path_batch = train_generator.__next__()
            # train_image_batch, train_gt_batch = Reader.processing1(train_img_path_batch, train_gt_path_batch,
            #                                                        crop_num=FLAGS.crop_num)
            train_image_batch, train_gt_batch = reader.get_next_batch(is_training=True)
            # print(np.shape(train_image_batch))
            # train_image_batch, train_gt_batch = Reader.processing2(train_img_path_batch, train_gt_path_batch,
            #                                                        slice_num=FLAGS.slice_num)
            # pred_last_value, pred_6_value, pred_3_value = sess.run([pred_last, pred_6, pred_3], feed_dict={
            #     input_image_tensor: np.expand_dims(train_image_batch, axis=4)
            # })
            # print('Pred last, max: {}, min: {}.'.format(np.max(pred_last_value), np.min(pred_last_value)))
            # print('Pred 6, max: {}, min: {}.'.format(np.max(pred_6_value), np.min(pred_6_value)))
            # print('Pred 3, max: {}, min: {}.'.format(np.max(pred_3_value), np.min(pred_3_value)))
            # print('InputImage, max: {}, min: {}'.format(np.max(train_image_batch), np.min(train_image_batch)))
            _, pred_last_softmax_value, total_loss_value, model_loss_value, cross_entropy_last_value, cross_entropy_6_value, cross_entropy_3_value, learning_rate_value, summary_value = sess.run(
                [train_op, pred_last_softmax, total_loss, model_loss, cross_entropy_last, cross_entropy_6, cross_entropy_3, learning_rate,
                 summary_op], feed_dict={
                    input_image_tensor: np.expand_dims(train_image_batch, axis=4),
                    input_gt_tensor: train_gt_batch
                })
            train_summary.add_summary(summary_value, global_step=step)
            if step % FLAGS.print_interval == 0:
                # pred_last_value, pred_6_value, pred_3_value = sess.run([pred_last, pred_6, pred_3], feed_dict={
                #     input_image_tensor: np.expand_dims(train_image_batch, axis=4)
                # })
                # print('Pred last, max: {}, min: {}.'.format(np.max(pred_last_value), np.min(pred_last_value)))
                # print('Pred 6, max: {}, min: {}.'.format(np.max(pred_6_value), np.min(pred_6_value)))
                # print('Pred 3, max: {}, min: {}.'.format(np.max(pred_3_value), np.min(pred_3_value)))
                # print('InputImage, max: {}, min: {}'.format(np.max(train_image_batch), np.min(train_image_batch)))
                print(
                    'Training, Step: {}, total loss: {:.4f}, model loss: {:.04f}, cross_entropy_last: {:.4f}, cross_entropy_6: {:.4f}, cross_entropy_3: {:.4f}, learning rate: {:.7f}'.format(
                        step,
                        total_loss_value,
                        model_loss_value,
                        cross_entropy_last_value,
                        cross_entropy_6_value,
                        cross_entropy_3_value,
                        learning_rate_value))
                print('0, max: ', np.max(pred_last_softmax_value[:, :, :, :, 0]),
                      np.min(pred_last_softmax_value[:, :, :, :, 0]))
                print('1, max: ', np.max(pred_last_softmax_value[:, :, :, :, 1]),
                      np.min(pred_last_softmax_value[:, :, :, :, 1]))

            if step % FLAGS.val_interval == 0:
                # val_img_path_batch, val_gt_path_batch = val_generator.__next__()
                # val_image_batch, val_gt_batch = Reader.processing1(val_img_path_batch, val_gt_path_batch,
                #                                                    crop_num=FLAGS.crop_num)
                val_image_batch, val_gt_batch = reader.get_next_batch(is_training=False)
                # val_image_batch, val_gt_batch = Reader.processing2(val_img_path_batch, val_gt_path_batch,
                #                                                    slice_num=FLAGS.slice_num)
                total_loss_value, model_loss_value, learning_rate_value, summary_value = sess.run(
                    [total_loss, model_loss, learning_rate, summary_op], feed_dict={
                        input_image_tensor: np.expand_dims(val_image_batch, axis=4),
                        input_gt_tensor: val_gt_batch
                    })
                val_summary.add_summary(summary_value, global_step=step)
                print(
                    'Validation, Step: {}, total loss: {:.4f}, model loss: {:.04f}, learning rate: {:.7f}'.format(step,
                                                                                                                  total_loss_value,
                                                                                                                  model_loss_value,
                                                                                                                  learning_rate_value))
            if step % FLAGS.snap_interval == 0:
                print('model saving in ', os.path.join(FLAGS.model_save_path, 'model.ckpt'))
                saver.save(sess, os.path.join(FLAGS.model_save_path, 'model.ckpt'), global_step=global_step)
        train_summary.close()
        val_summary.close()

if __name__ == '__main__':
    train()
