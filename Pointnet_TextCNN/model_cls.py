import os
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../tf_ops/sampling'))
sys.path.append(os.path.join(BASE_DIR, '../tf_ops/grouping'))
sys.path.append(os.path.join(BASE_DIR, '../tf_ops/3d_interpolation'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point
from tf_interpolate import three_nn, three_interpolate
import tensorflow as tf
import numpy as np
import tf_util
from pointnet_util import pointnet_sa_module, pointnet_sa_module_msg

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl

def get_textcnn_model(text_embedding, filter_sizes, filter_num, is_training, bn_decay=None):
    '''
    Input:
        text_embedding: [batch, sequence_length, embedding_size]
        filter_sizes: []
        filter_num: int32
    '''
    batch_size = text_embedding.get_shape()[0].value
    sequence_length = text_embedding.get_shape()[1].value
    embedding_size = text_embedding.get_shape()[2].value
    pooled_output = []
    # [batch_size, sequence_length, embedding_size, 1]
    text_embedding = tf.expand_dims(text_embedding, -1)
    for i, filter_size in enumerate(filter_sizes):
        # [batch_size, sequence_length - filter_size + 1, 1, filter_num]
        conved = tf_util.conv2d(text_embedding, filter_num, [filter_size, embedding_size],
                                scope = 'textcnn_conv_%d'%(i), stride = [1, 1],
                                padding = 'VALID', bn = True, bn_decay = bn_decay,
                                is_training = is_training, data_format = 'NHWC')
        # [batch_size, 1, 1, filter_num]
        conved = tf.reduce_max(conved, axis = [1], keep_dims = True, name = 'textcnn_maxpool')
        # [batch_size, filter_num]
        conved = tf.reshape(conved, [batch_size, -1])
        pooled_output.append(conved)
    
    # [batch_size, filter_num * len(filter_sizes)]
    concated = tf.concat(pooled_output, 1)
    return concated


def get_model(sample_num, sample_scale, point_cloud, is_training, filter_sizes, filter_num, bn_decay=None):
    '''
    Input: 
        sample_num: int32; sample M points from originally N points.
        sample_scale: []; find Ki points from sampled points' neighbours
    '''
    batch_size = point_cloud.get_shape()[0].value
    feature_collection = []
    channels = [32, 64, 128]
    M_sampled_points = farthest_point_sample(sample_num, point_cloud)
    # [batch, sample_num, 3]
    new_xyz = gather_point(point_cloud, M_sampled_points)
    for i, scale in enumerate(sample_scale):
        # [batch, sample_num, scale]
        _, idx = knn_point(scale, point_cloud, new_xyz)
        # [batch, sample_num, scale, 3]
        points_features = group_point(point_cloud, idx)
        for j, channel in enumerate(channels):
            # [batch, sample_num, scale, channel]
            points_features = tf_util.conv2d(points_features, channel, [1, 1],
                                             padding = 'VALID', stride = [1, 1],
                                             bn = True, is_training = is_training,
                                             scope='conv_%d_%d'%(i, j), bn_decay = bn_decay,
                                             data_format = 'NHWC')
        # [batch, sample_num, 1, 128]
        points_features = tf.reduce_max(points_features, axis = [2], keep_dims = True, name = 'maxpool')
        # [batch, sample_num, 128]
        points_features = tf.squeeze(points_features, [2])
        # [batch, sample_num, 1, 128]
        points_features = tf.expand_dims(points_features, 2)
        # [batch * sample_num, 1, 128]
        points_features = tf.reshape(points_features, [batch_size * sample_num, 1, channels[-1]])
        feature_collection.append(points_features)

    # [batch * sample_num, len(sample_scale), 128]
    textcnn_embedding = tf.concat(feature_collection, 1)
    # [batch * sample_num, feature_size = 128]
    textcnn_encoded = get_textcnn_model(textcnn_embedding, filter_sizes, filter_num, is_training, bn_decay)
    # [batch, sample_num, feature_size]
    textcnn_encoded = tf.reshape(textcnn_encoded, [batch_size, sample_num, -1])
    # [batch, sample_num, 1, feature_size]
    global_feature = tf.expand_dims(textcnn_encoded, 2)
    channels = [256, 512, 1024]
    for i, channel in enumerate(channels):
        # [batch, sample_num, 1, channel]
        global_feature = tf_util.conv2d(global_feature, channel, [1, 1],
                                        padding = 'VALID', stride = [1, 1],
                                        bn = True, is_training = is_training,
                                        scope='feature_aggregation_conv_%d'%(i), bn_decay = bn_decay,
                                        data_format = 'NHWC')
    # [batch, 1, 1, 1024]
    global_feature = tf.reduce_max(global_feature, axis = [1], keep_dims = True, name = 'global_feature_maxpool')
    # [batch, 1024]
    global_feature = tf.reshape(global_feature, [batch_size, -1])
    classify_feature = tf_util.fully_connected(global_feature, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    classify_feature = tf_util.dropout(classify_feature, keep_prob=0.4, is_training=is_training, scope='dp1')
    classify_feature = tf_util.fully_connected(classify_feature, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    classify_feature = tf_util.dropout(classify_feature, keep_prob=0.4, is_training=is_training, scope='dp2')
    classify_feature = tf_util.fully_connected(classify_feature, 40, activation_fn=None, scope='fc3')
    return classify_feature

def get_loss(pred, label):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)
    tf.add_to_collection('losses', classify_loss)
    return classify_loss
