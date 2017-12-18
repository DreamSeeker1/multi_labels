import os

import tensorflow as tf


class DataProcessor(object):

    def __init__(self, image_dir, label_dir, label_file):
        self.image_list = []  # 图片文件路径列表
        self.label_list = []  # 图片所对应的标签列表
        self.all_labels = {}  # labels文件中的label所对应的下标（第N个），下标从0开始

        # 初始化参数
        self._image_dir = image_dir  # 图片文件夹路径
        self._label_dir = label_dir  # 标签文件夹路径
        self._all_labels_path = label_file  # "labels.txt" 储存所有标签的文件路径

        # init the data
        self.init_image_labels()

    # 将数据集分成训练集和测试集 20%作为测试集
    def _category_data(self, category):
        test = 20
        test_counts = test * len(self.image_list) // 100

        if category == 'testing':
            return self.image_list[:test_counts], self.label_list[:test_counts]
        elif category == 'training':
            return self.image_list[test_counts:], self.label_list[test_counts:]

        else:
            print("category param wrong")
            exit(1)

    # 将filename对应的图片文件读入，并缩放到统一的大小并标准化
    def _parse(self, filename, image_W, image_H):
        img = tf.read_file(filename)
        img_decoded = tf.image.decode_jpeg(img, channels=3)
        img_resized = tf.image.resize_image_with_crop_or_pad(img_decoded, image_W, image_H)
        image_normalized = tf.image.per_image_standardization(img_resized)
        return image_normalized

    # 生成batch
    def get_dataset(self, image_W, image_H, batch_size, category):
        image_list, label_list = self._category_data(category)
        label_list = tf.cast(label_list, tf.float32)

        # 此时dataset_image 中的一个元素是（image）
        dataset_image = tf.data.Dataset.from_tensor_slices(image_list)

        # 此时dataset_image 中的一个元素是(image_resized)
        dataset_image = dataset_image.map(lambda x: self._parse(x, image_W, image_H))

        dataset_label = tf.data.Dataset.from_tensor_slices(label_list)

        # [(image)(image)...], [(label) (label),..)  == > [(image, label) (image, label)...]
        dataset = tf.data.Dataset.zip((dataset_image, dataset_label))

        return dataset.batch(batch_size)  # 返回（image_batchsize, label_batchsize）

    def _get_img_label(self, img_path):
        label_path = self._label_dir + os.path.sep + os.path.basename(
            img_path) + '.txt'  # os.path.sep，为路径分隔符  得到label_dir/img_name.txt 这样的格式
        img_label = [0] * len(self.all_labels)  # img_label 维度为跟总的标签分类数，0表示图片标签中不存在这个标签，1表示存在
        for label in open(label_path, 'r').readlines():  # 读取当前的图片所对应的标签
            label = label.strip()  # 去掉收尾空格字符串
            if label in self.all_labels:
                img_label[self.all_labels[label]] = 1  # 存在，则把label所对应的编号位置置为1。实际上应该是True，False对，但为了简单，使用0，1对
        return img_label

    def init_image_labels(self):
        """generate image and labels"""

        index = 0
        for val in open(self._all_labels_path).readlines():
            val = val.strip()
            if val:
                self.all_labels[val] = index  # 对所有的标签，按照文件中的顺序，对应上相应的编号，下标从0开始
                index += 1

        for img_path in os.listdir(self._image_dir):
            self.image_list.append(self._image_dir + os.path.sep + img_path)  # 获取图片路径
            self.label_list.append(self._get_img_label(img_path))  # 图片所对应的标签[[0,1,0,0,1],...] == [[第二个和第五个标签存在], ...]


