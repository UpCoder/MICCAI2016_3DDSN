import tensorflow as tf
from tensorflow.contrib import slim
from network.model import model, build_loss
from config import Config
import os
import numpy as np
from data.Reader import Reader
from glob import glob
from tools.medicalImage import read_nii, save_mhd_image
from tools.utils import calculate_dicescore
tf.app.flags.DEFINE_float('learning_rate', 0.1, 'the learning rate')
tf.app.flags.DEFINE_float('moving_average_decay', 0.997, '')
tf.app.flags.DEFINE_integer('max_steps', 1000000, '')
tf.app.flags.DEFINE_integer('snap_interval', 100, '')
tf.app.flags.DEFINE_integer('val_interval', 100, '')
tf.app.flags.DEFINE_integer('batch_size', 2, '')
tf.app.flags.DEFINE_integer('crop_num', 5, '')
tf.app.flags.DEFINE_integer('slice_num', 72, '')
tf.app.flags.DEFINE_string('model_save_path', '/home/give/PycharmProjects/MICCAI2016_3DDSN/outputs/snaps', '')
tf.app.flags.DEFINE_boolean('restore_flag', True, '')
tf.app.flags.DEFINE_string('model_restore_path', '/home/give/PycharmProjects/MICCAI2016_3DDSN/outputs/snaps', '')
tf.app.flags.DEFINE_string('summary_dir', '/home/give/PycharmProjects/MICCAI2016_3DDSN/outputs/logs', '')
tf.app.flags.DEFINE_integer('print_interval', 5, '')
FLAGS = tf.app.flags.FLAGS


def compute_onefile(sess, input_image_tensor, pred_last_tensor, image_path, save_path):
    volume = read_nii(image_path)
    volume = Reader.static_scalar(volume, -300, 500)
    print(np.max(volume), np.min(volume))
    shape = list(np.shape(volume))
    # volume = np.transpose(volume, axes=[2, 0, 1])
    batch_data = []
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
                cur_data = volume[x:end_x, y:end_y, z:end_z]
                batch_data.append(cur_data)
                if z == shape[2] - Config.vox_size[2]:
                    break
            if y == shape[1] - Config.vox_size[1]:
                break
        if x == shape[0] - Config.vox_size[0]:
            break
    batch_size = 10
    pred_result = []
    idx = 0
    while idx < len(batch_data):
        end = idx + batch_size
        if end > len(batch_data):
            end = len(batch_data)
        cur_data = batch_data[idx: end]
        [pred_last_value] = sess.run([pred_last_tensor], feed_dict={
            input_image_tensor: np.expand_dims(cur_data, axis=4)
        })
        print('0, max: ', np.max(pred_last_value[:, :, :, :, 0]), np.min(pred_last_value[:, :, :, :, 0]))
        print('1, max: ', np.max(pred_last_value[:, :, :, :, 1]), np.min(pred_last_value[:, :, :, :, 1]))
        print('finished ', np.shape(pred_last_value))
        pred_result.extend(pred_last_value)
        idx = end
    print('batch_data shape is ', np.shape(batch_data))
    print('pred_result shape is ', np.shape(pred_result))
    # pred_result = np.random.random(np.shape(batch_data))
    pred_mask = np.zeros(np.shape(volume))
    index = 0
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
                # pred_mask[x:end_x, y:end_y, z:end_z] = np.logical_and(volume[x:end_x, y:end_y, z:end_z] > 0,
                #                                                       volume[x:end_x, y:end_y, z:end_z] < 100)

                # print(np.max(np.argmax(pred_result[index], axis=3)), np.min(np.argmax(pred_result[index], axis=3)))
                # pred_mask[x:end_x, y:end_y, z:end_z] = np.argmax(pred_result[index], axis=3)
                pred_mask[x:end_x, y:end_y, z:end_z] =  np.argmax(pred_result[index], axis=3)
                index += 1
                if z == shape[2] - Config.vox_size[2]:
                    break
            if y == shape[1] - Config.vox_size[1]:
                break
        if x == shape[0] - Config.vox_size[0]:
            break
    save_mhd_image(np.transpose(pred_mask, axes=[2, 1, 0]), save_path)
    return pred_mask

def validation_dice(val_dir):
    input_image_tensor = tf.placeholder(dtype=tf.float32,
                                        shape=[None, Config.vox_size[0], Config.vox_size[1], Config.vox_size[2], 1],
                                        name='image_tensor')
    input_gt_tensor = tf.placeholder(dtype=tf.int32,
                                     shape=[None, Config.vox_size[0], Config.vox_size[1], Config.vox_size[2]],
                                     name='gt_tensor')
    pred_last, pred_6, pred_3 = model(input_image_tensor)
    global_step = tf.train.get_or_create_global_step()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        ckpt = tf.train.latest_checkpoint(FLAGS.model_restore_path)
        print('continue training from previous checkpoint from %s' % ckpt)
        start_step = int(os.path.basename(ckpt).split('-')[1])
        variable_restore_op = slim.assign_from_checkpoint_fn(ckpt,
                                                             slim.get_trainable_variables(),
                                                             ignore_missing_vars=True)
        variable_restore_op(sess)
        sess.run(tf.assign(global_step, start_step))

        # obtain the data of validation
        image_paths = glob(os.path.join(val_dir, 'volume-*.nii'))
        for image_path in image_paths:
            basename = os.path.basename(image_path)
            file_id = basename.split('-')[1].split('.')[0]
            gt_path = os.path.join(val_dir, 'segmentation-' + file_id + '.nii')
            pred_result = compute_onefile()


        # for step in range(start_step, FLAGS.max_steps + start_step):
        #     train_img_path_batch, train_gt_path_batch = train_generator.__next__()
        #     train_image_batch, train_gt_batch = Reader.processing1(train_img_path_batch, train_gt_path_batch,
        #                                                            crop_num=FLAGS.crop_num)
        #     # train_image_batch, train_gt_batch = Reader.processing2(train_img_path_batch, train_gt_path_batch,
        #     #                                                        slice_num=FLAGS.slice_num)
        #     # pred_last_value, pred_6_value, pred_3_value = sess.run([pred_last, pred_6, pred_3], feed_dict={
        #     #     input_image_tensor: np.expand_dims(train_image_batch, axis=4)
        #     # })
        #     # print('Pred last, max: {}, min: {}.'.format(np.max(pred_last_value), np.min(pred_last_value)))
        #     # print('Pred 6, max: {}, min: {}.'.format(np.max(pred_6_value), np.min(pred_6_value)))
        #     # print('Pred 3, max: {}, min: {}.'.format(np.max(pred_3_value), np.min(pred_3_value)))
        #     # print('InputImage, max: {}, min: {}'.format(np.max(train_image_batch), np.min(train_image_batch)))
        #     _, total_loss_value, model_loss_value, cross_entropy_last_value, cross_entropy_6_value, cross_entropy_3_value, learning_rate_value, summary_value = sess.run(
        #         [train_op, total_loss, model_loss, cross_entropy_last, cross_entropy_6, cross_entropy_3, learning_rate,
        #          summary_op], feed_dict={
        #             input_image_tensor: np.expand_dims(train_image_batch, axis=4),
        #             input_gt_tensor: train_gt_batch
        #         })
        #     train_summary.add_summary(summary_value, global_step=step)
        #     if step % FLAGS.print_interval == 0:
        #         # pred_last_value, pred_6_value, pred_3_value = sess.run([pred_last, pred_6, pred_3], feed_dict={
        #         #     input_image_tensor: np.expand_dims(train_image_batch, axis=4)
        #         # })
        #         # print('Pred last, max: {}, min: {}.'.format(np.max(pred_last_value), np.min(pred_last_value)))
        #         # print('Pred 6, max: {}, min: {}.'.format(np.max(pred_6_value), np.min(pred_6_value)))
        #         # print('Pred 3, max: {}, min: {}.'.format(np.max(pred_3_value), np.min(pred_3_value)))
        #         # print('InputImage, max: {}, min: {}'.format(np.max(train_image_batch), np.min(train_image_batch)))
        #         print(
        #             'Training, Step: {}, total loss: {:.4f}, model loss: {:.04f}, cross_entropy_last: {:.4f}, cross_entropy_6: {:.4f}, cross_entropy_3: {:.4f}, learning rate: {:.7f}'.format(
        #                 step,
        #                 total_loss_value,
        #                 model_loss_value,
        #                 cross_entropy_last_value,
        #                 cross_entropy_6_value,
        #                 cross_entropy_3_value,
        #                 learning_rate_value))
        #
        #     if step % FLAGS.val_interval == 0:
        #         val_img_path_batch, val_gt_path_batch = val_generator.__next__()
        #         val_image_batch, val_gt_batch = Reader.processing1(val_img_path_batch, val_gt_path_batch,
        #                                                            crop_num=FLAGS.crop_num)
        #         # val_image_batch, val_gt_batch = Reader.processing2(val_img_path_batch, val_gt_path_batch,
        #         #                                                    slice_num=FLAGS.slice_num)
        #         total_loss_value, model_loss_value, learning_rate_value, summary_value = sess.run(
        #             [total_loss, model_loss, learning_rate, summary_op], feed_dict={
        #                 input_image_tensor: np.expand_dims(val_image_batch, axis=4),
        #                 input_gt_tensor: val_gt_batch
        #             })
        #         val_summary.add_summary(summary_value, global_step=step)
        #         print(
        #             'Validation, Step: {}, total loss: {:.4f}, model loss: {:.04f}, learning rate: {:.7f}'.format(step,
        #                                                                                                           total_loss_value,
        #                                                                                                           model_loss_value,
        #                                                                                                           learning_rate_value))
        #     if step % FLAGS.snap_interval == 0:
        #         print('model saving in ', os.path.join(FLAGS.model_save_path, 'model.ckpt'))
        #         saver.save(sess, os.path.join(FLAGS.model_save_path, 'model.ckpt'), global_step=global_step)
        # train_summary.close()
        # val_summary.close()

if __name__ == '__main__':
    input_image_tensor = tf.placeholder(dtype=tf.float32,
                                        shape=[None, Config.vox_size[0], Config.vox_size[1], Config.vox_size[2], 1],
                                        name='image_tensor')
    pred_last, pred_6, pred_3 = model(input_image_tensor)
    global_step = tf.train.get_or_create_global_step()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        ckpt = tf.train.latest_checkpoint(FLAGS.model_restore_path)
        print('continue training from previous checkpoint from %s' % ckpt)
        start_step = int(os.path.basename(ckpt).split('-')[1])
        variable_restore_op = slim.assign_from_checkpoint_fn(ckpt,
                                                             slim.get_trainable_variables(),
                                                             ignore_missing_vars=True)
        variable_restore_op(sess)
        sess.run(tf.assign(global_step, start_step))

        pred_mask = compute_onefile(sess, input_image_tensor, tf.nn.softmax(pred_last),
                                    '/home/give/PycharmProjects/MICCAI2016_3DDSN/tmp/volume-55.nii',
                                    '/home/give/PycharmProjects/MICCAI2016_3DDSN/tmp/pred-55.nii')
        gt_path = '/home/give/PycharmProjects/MICCAI2016_3DDSN/tmp/segmentation-55.nii'
        gt_mask = read_nii(gt_path)
        print('GT mask shape: ', np.shape(gt_mask))
        print('Pred mask shape: ', np.shape(pred_mask))
        dice_score = calculate_dicescore(gt_mask, pred_mask)
        print('dice_score: ', dice_score)






        # test effective on train
        # input_image_tensor = tf.placeholder(dtype=tf.float32,
        #                                     shape=[None, Config.vox_size[0], Config.vox_size[1], Config.vox_size[2], 1],
        #                                     name='image_tensor')
        # input_gt_tensor = tf.placeholder(dtype=tf.int32,
        #                                  shape=[None, Config.vox_size[0], Config.vox_size[1], Config.vox_size[2]],
        #                                  name='gt_tensor')
        # global_step = tf.train.get_or_create_global_step()
        # learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step, decay_steps=4000, decay_rate=0.1,
        #                                            staircase=True)
        # opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
        # pred_last, pred_6, pred_3 = model(input_image_tensor)
        # model_loss, cross_entropy_last, cross_entropy_6, cross_entropy_3 = build_loss(input_gt_tensor, pred_last,
        #                                                                               pred_6,
        #                                                                               pred_3, global_step=global_step)
        # total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        # tf.summary.image('image', tf.transpose(input_image_tensor[0, :, :, 10:15], perm=[2, 0, 1, 3]), max_outputs=3)
        # tf.summary.image('gt',
        #                  tf.expand_dims(
        #                      tf.transpose(tf.cast(input_gt_tensor, tf.float32)[0, :, :, 10:15], perm=[2, 0, 1]),
        #                      axis=3),
        #                  max_outputs=3)
        # tf.summary.image('pred_last',
        #                  tf.expand_dims(tf.transpose(tf.nn.softmax(pred_last)[0, :, :, 10:15, 1], perm=[2, 0, 1]),
        #                                 axis=3),
        #                  max_outputs=3)
        # tf.summary.image('pred_6',
        #                  tf.expand_dims(tf.transpose(tf.nn.softmax(pred_6)[0, :, :, 10:15, 1], perm=[2, 0, 1]), axis=3),
        #                  max_outputs=3)
        # tf.summary.image('pred_3',
        #                  tf.expand_dims(tf.transpose(tf.nn.softmax(pred_3)[0, :, :, 10:15, 1], perm=[2, 0, 1]), axis=3),
        #                  max_outputs=3)
        # tf.summary.scalar('loss/model_loss', model_loss)
        # tf.summary.scalar('loss/total_loss', total_loss)
        # tf.summary.scalar('loss/cross_entropy_last', cross_entropy_last)
        # tf.summary.scalar('loss/cross_entropy_6', cross_entropy_6)
        # tf.summary.scalar('loss/cross_entropy_3', cross_entropy_3)
        # # grads = opt.compute_gradients(total_loss)
        # # apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        # apply_gradient_op = opt.minimize(total_loss, global_step=global_step)
        # variable_averages = tf.train.ExponentialMovingAverage(
        #     FLAGS.moving_average_decay, global_step)
        # variables_averages_op = variable_averages.apply(tf.trainable_variables())
        #
        # with tf.control_dependencies([variables_averages_op, apply_gradient_op]):
        #     train_op = tf.no_op(name='train_op')
        # train_summary = tf.summary.FileWriter(os.path.join(FLAGS.summary_dir, 'train'), tf.get_default_graph())
        # val_summary = tf.summary.FileWriter(os.path.join(FLAGS.summary_dir, 'val'), tf.get_default_graph())
        # summary_op = tf.summary.merge_all()
        # with tf.Session() as sess:
        #     sess.run(tf.global_variables_initializer())
        #     sess.run(tf.local_variables_initializer())
        #
        #     reader = Reader(
        #         '/home/give/Documents/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training Batch 2',
        #         '/home/give/Documents/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training Batch 1',
        #         batch_size=FLAGS.batch_size
        #     )
        #     train_generator = reader.train_generator
        #     val_generator = reader.val_generator
        #     start_step = 0
        #     saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)
        #
        #     ckpt = tf.train.latest_checkpoint(FLAGS.model_restore_path)
        #     print('continue training from previous checkpoint from %s' % ckpt)
        #     start_step = int(os.path.basename(ckpt).split('-')[1])
        #     variable_restore_op = slim.assign_from_checkpoint_fn(ckpt,
        #                                                          slim.get_trainable_variables(),
        #                                                          ignore_missing_vars=True)
        #     variable_restore_op(sess)
        #     sess.run(tf.assign(global_step, start_step))
        #     reader = Reader(
        #         '/home/give/Documents/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training Batch 2',
        #         '/home/give/Documents/dataset/ISBI2017/media/nas/01_Datasets/CT/LITS/Training Batch 1',
        #         batch_size=FLAGS.batch_size
        #     )
        #     train_generator = reader.train_generator
        #     val_generator = reader.val_generator
        #     train_img_path_batch, train_gt_path_batch = train_generator.__next__()
        #     train_image_batch, train_gt_batch = Reader.processing1(train_img_path_batch, train_gt_path_batch,
        #                                                            crop_num=FLAGS.crop_num)
        #     model_loss, cross_entropy_last, cross_entropy_6, cross_entropy_3 = build_loss(input_gt_tensor, pred_last,
        #                                                                                   pred_6,
        #                                                                                   pred_3,
        #                                                                                   global_step=global_step)
        #     total_loss = tf.add_n([model_loss] + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        #     [pred_last_value, model_loss_value, total_loss_value] = sess.run(
        #         [tf.nn.softmax(pred_last), model_loss, total_loss], feed_dict={
        #             input_image_tensor: np.expand_dims(train_image_batch, axis=4),
        #             input_gt_tensor: train_gt_batch
        #         })
        #     print('0, max: ', np.max(pred_last_value[:, :, :, :, 0]), np.min(pred_last_value[:, :, :, :, 0]))
        #     print('1, max: ', np.max(pred_last_value[:, :, :, :, 1]), np.min(pred_last_value[:, :, :, :, 1]))
        #     print('finished ', np.shape(pred_last_value))
        #     print('model loss: {}, total loss: {}'.format(model_loss_value, total_loss_value))







