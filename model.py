import tensorflow as tf
import numpy as np

# 参考官网cifar10 model的结构
def inference(images, batch_size, n_classes, reuse=None):
    '''Build the model
    Args:
        images: image batch, 4D tensor, tf.float32, [batch_size, width, height, channels]
    Returns:
        output tensor with the computed logits, float, [batch_size, n_classes]
    '''

    # conv1, shape = [kernel size, kernel size, channels, kernel numbers]

    with tf.variable_scope('conv1', reuse=reuse) as scope:
        # 使用tf.Variable时，如果检测到命名冲突，系统会自己处理。使用tf.get_variable()时，系统不会处理冲突，而会报错
        # 基于这两个函数的特性，当我们需要共享变量的时候，需要使用tf.get_variable()。在其他情况下，这两个的用法是一样的
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 3, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(images, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    # pool1
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pooling1')

    # conv2
    with tf.variable_scope('conv2', reuse=reuse) as scope:
        weights = tf.get_variable('weights',
                                  shape=[3, 3, 16, 16],
                                  dtype=tf.float32,
                                  initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        biases = tf.get_variable('biases',
                                 shape=[16],
                                 dtype=tf.float32,
                                 initializer=tf.constant_initializer(0.1))
        conv = tf.nn.conv2d(pool1, weights, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name='conv2')

    # pool2
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1],
                               padding='SAME', name='pooling2')

    # local3
    with tf.variable_scope('local3', reuse=reuse):
        reshape = tf.reshape(pool2, shape=[batch_size, np.prod(pool2.get_shape()[1:])])
        local3 = tf.layers.dense(inputs=reshape,
                                 units=128,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32),
                                 bias_initializer=tf.constant_initializer(0.1),
                                 activation=tf.nn.relu,
                                 name='local3')

    # local4
    with tf.variable_scope('local4', reuse=reuse):
        local4 = tf.layers.dense(inputs=local3,
                                 units=128,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32),
                                 bias_initializer=tf.constant_initializer(0.1),
                                 activation=tf.nn.relu,
                                 name='local4')

    # 由于我们是多标签 所以把softmax层 变成sigmoid层
    with tf.variable_scope('logits', reuse=reuse) as scope:
        logits = tf.layers.dense(inputs=local4,
                                 units=n_classes,
                                 kernel_initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32),
                                 bias_initializer=tf.constant_initializer(0.),
                                 activation=None,
                                 name='None')

    return logits


def losses(logits, labels):

    with tf.variable_scope('loss') as scope:
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels, logits=logits)
        loss = tf.reduce_mean(cross_entropy, name='loss')
        tf.summary.scalar(scope.name + '/loss', loss)
    return loss


def training(loss, learning_rate):

    with tf.name_scope('optimizer'):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        global_step = tf.Variable(0, name='global_step', trainable=False)
        train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op


def evaluation(logits, labels):

    with tf.variable_scope('accuracy') as scope:
        final_tensor = tf.nn.sigmoid(logits)
        correct_prediction = tf.equal(tf.round(final_tensor), labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar(scope.name + '/accuracy', accuracy)
    return accuracy
