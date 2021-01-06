from __future__ import print_function

import os
import sys
import time
import tensorflow as tf
import numpy as np
import argparse
import imageio

from model import PSPNet101, PSPNet50
from tools import *

ADE20k_param = {'crop_size': [473, 473],
                'num_classes': 150,
                'model': PSPNet50}  #ADE20k数据集参数
cityscapes_param = {'crop_size': [720, 720],
                    'num_classes': 15,
                    'model': PSPNet101} #cityScape数据集参数

SAVE_DIR = './output/'
SNAPSHOT_DIR = './model/'


def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced PSPNet")
    parser.add_argument("--img-path", type=str, default='',
                        help="Path to the RGB image file.")
    parser.add_argument("--checkpoints", type=str, default=SNAPSHOT_DIR,
                        help="Path to restore weights.")
    parser.add_argument("--save-dir", type=str, default=SAVE_DIR,
                        help="Path to save output.")
    parser.add_argument("--flipped-eval", action="store_true",
                        help="whether to evaluate with flipped img.")
    parser.add_argument("--dataset", type=str, default='',
                        choices=['ade20k', 'cityscapes'],
                        required=True)

    return parser.parse_args()


def save(saver, sess, logdir, step):

    model_name = 'model.ckpt-55000'
    checkpoint_path = os.path.join(logdir, model_name)

    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')


def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


def main():
    args = get_arguments()

    # load parameters
    if args.dataset == 'ade20k':
        param = ADE20k_param
    elif args.dataset == 'cityscapes':
        param = cityscapes_param

    crop_size = param['crop_size']
    num_classes = param['num_classes']
    PSPNet = param['model']

    # preprocess images
    img, filename = load_img(args.img_path)  #load_img调用的tool.py
    img_shape = tf.shape(img)
    h, w = (tf.maximum(crop_size[0], img_shape[0]), tf.maximum(crop_size[1], img_shape[1]))
    img = preprocess(img, h, w)  #preprocess 调用tool.py

    # Create network.
    """tf.squeeze()从张量中移除大小为1的维度"""
    net = PSPNet({'data': img}, is_training=False, num_classes=num_classes)
    with tf.variable_scope('', reuse=True):        #用于定义创建变量(层)的操作的上下文管理器.
        flipped_img = tf.image.flip_left_right(tf.squeeze(img))
        flipped_img = tf.expand_dims(flipped_img, dim=0)
        net2 = PSPNet({'data': flipped_img}, is_training=False, num_classes=num_classes)

    raw_output = net.layers['conv6']
    
    # Do flipped eval or not
    if args.flipped_eval:
        flipped_output = tf.image.flip_left_right(tf.squeeze(net2.layers['conv6']))
        flipped_output = tf.expand_dims(flipped_output, dim=0)
        raw_output = tf.add_n([raw_output, flipped_output])

    # Predictions.
    raw_output_up = tf.image.resize_bilinear(raw_output, size=[h, w], align_corners=True)
    raw_output_up = tf.image.crop_to_bounding_box(raw_output_up, 0, 0, img_shape[0], img_shape[1]) #左上角的坐标和目标坐标
    raw_output_up = tf.argmax(raw_output_up, axis=3)  #axis=0、1，即返回每行或者每列最大值的位置索引，axis= 2、3、4时，即多维张量
    pred = decode_labels(raw_output_up, img_shape, num_classes)
    
    # Init tf Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)
    
    restore_var = tf.global_variables()
    
    ckpt = tf.train.get_checkpoint_state(args.checkpoints)  #通过checkpoint文件找到模型文件名
    if ckpt and ckpt.model_checkpoint_path:
        loader = tf.train.Saver(var_list=restore_var)
        load_step = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1])
        load(loader, sess, ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found.')
    
    preds = sess.run(pred)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    # misc.imsave(args.save_dir + filename, preds[0])
    imageio.imsave(args.save_dir + filename, preds[0])

if __name__ == '__main__':
    main()