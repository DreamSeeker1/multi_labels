# 项目的主要目的是利用TensorFlow训练自己的数据集
#    - input_data.py:  读入数据并生成batch
#    - model.py: 建立模型结构
#    - training.py: 训练及测试模型

import os
import numpy as np
import tensorflow as tf
import input_data
import model

#参数设置

N_CLASSES = 5   # 数据集标签数 desert,mountains, sea,sunset,trees
IMG_W = 208     # 图片resize，图片越大，训练越慢
IMG_H = 208
BATCH_SIZE = 16  # 一般是2^n
MAX_STEP = 1000  # 当前参数下，建议训练步数最好>10000
learning_rate = 0.0001  # 当前参数下，建议学习率<0.0001


# 训练模型

def run_training():
    # 模型参数保存路径
    logs_train_dir = 'logs/train/'

    # 读取训练集与测试集数据
    data_resource = input_data.DataProcessor('images', 'image_labels_dir', 'labels.txt')
    training_data_set = data_resource.get_dataset(IMG_W, IMG_H, BATCH_SIZE, 'training')
    training_iterator = training_data_set.make_initializable_iterator()
    testing_data_set = data_resource.get_dataset(IMG_W, IMG_H, BATCH_SIZE, 'testing')
    testing_iterator = testing_data_set.make_initializable_iterator()

    # 生成batch
    test_batch, test_label_batch = testing_iterator.get_next()
    train_batch, train_label_batch = training_iterator.get_next()

    train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)
    train_loss = model.losses(train_logits, train_label_batch)
    train_op = model.training(train_loss, learning_rate)
    train__acc = model.evaluation(train_logits, train_label_batch)

    # log汇总
    summary_op = tf.summary.merge_all()
    # 产生会话
    sess = tf.Session()
    # 产生一个writer写log文件
    train_writer = tf.summary.FileWriter(logs_train_dir, sess.graph)
    # 产生saver来存储训练好的模型
    saver = tf.train.Saver()
    # 所有节点初始化
    sess.run(tf.global_variables_initializer())
    sess.run(training_iterator.initializer)

    for step in np.arange(MAX_STEP):
        # 执行MAX_STEP步训练，一步一个batch
        try:
            _, tra_loss, tra_acc, summary_str = sess.run([train_op, train_loss, train__acc, summary_op])
        except tf.errors.OutOfRangeError:  # 训练完一个epoch会到这里
            sess.run(training_iterator.initializer)
            _, tra_loss, tra_acc, summary_str = sess.run([train_op, train_loss, train__acc, summary_op])
        if step % 20 == 0:
            print('Step %d, train loss = %.6f, train accuracy = %.2f%%' % (step, tra_loss, tra_acc * 100.0))
            train_writer.add_summary(summary_str, step)
        if step % 2000 == 0 or (step + 1) == MAX_STEP:
            checkpoint_path = os.path.join(logs_train_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=step)

    # 验证模型，只测了一个batch
    test_logits = model.inference(test_batch, BATCH_SIZE, N_CLASSES, True)
    test_loss = model.losses(test_logits, test_label_batch)
    test__acc = model.evaluation(test_logits, test_label_batch)
    sess.run(testing_iterator.initializer)
    print(sess.run([test_loss, test__acc]))


    sess.close()


run_training()


# 训练完后，取消注释以下代码
# 评估一张图片（任意一张与数据集类似的图片）

# from PIL import Image
#
# def get_one_image():
#    image = Image.open('images/57.jpg')
#    image = image.resize([208, 208])
#    image = np.array(image)
#    return image
#
# def evaluate_one_image():
#    '''Test one image against the saved models and parameters
#    '''
#    label_lines = [line.rstrip() for line in tf.gfile.GFile("labels.txt")]
#    # you need to change the directories to yours.
#
#    image_array = get_one_image()
#
#    with tf.Graph().as_default():
#        BATCH_SIZE = 1
#        N_CLASSES = 5
#
#        image = tf.cast(image_array, tf.float32)
#        image = tf.image.per_image_standardization(image)
#        image = tf.reshape(image, [1, 208, 208, 3])
#        logit = model.inference(image, BATCH_SIZE, N_CLASSES)
#
#        logit = tf.nn.sigmoid(logit)
#
#
#        x = tf.placeholder(tf.float32, shape=[208, 208, 3])
#
#        # you need to change the directories to yours.
#        logs_train_dir = 'logs/train/'
#
#        saver = tf.train.Saver()
#
#        with tf.Session() as sess:
#
#            print("Reading checkpoints...")
#            ckpt = tf.train.get_checkpoint_state(logs_train_dir)
#            if ckpt and ckpt.model_checkpoint_path:
#                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
#                saver.restore(sess, ckpt.model_checkpoint_path)
#                print('Loading success, global_step is %s' % global_step)
#            else:
#                print('No checkpoint file found')
#
#            prediction = sess.run(logit, feed_dict={x: image_array})
#            print(sess.run(logit))  # [[0.22, 0.9, 0.55, 0.7, 0.8]]
#
#            top_k = prediction[0].argsort()[::-1]  # 输出预测值从大到小的序号。
#                                                   # x = np.array([3, 1, 2]) np.argsort(x）---》array([1, 2, 0])

#            for node_id in top_k:
#                human_string = label_lines[node_id]
#                score = prediction[0][node_id]
#                print('%s (score = %.5f)' % (human_string, score))
#
# evaluate_one_image()
