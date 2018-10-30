import os
import re
import json
from os.path import expanduser

import zipfile
import datetime
import tensorflow as tf
import numpy as np
from deepdrive_dataset.dataset_util import *
from deepdrive_dataset.deepdrive_versions import DEEPDRIVE_LABELS
from deepdrive_dataset.tf_features import *
from PIL import Image


import errno    


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

class DeepdriveDatasetWriter(object):

    def __init__(self):
        self.input_path = os.path.join(expanduser('~'), '.deepdrive')

    def createTFExample(self, height, width, encoded_image_data, image_format, filename, xmins, xmaxs, ymins, ymaxs, classes_text, classes):
        return tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(height),
            'image/width': int64_feature(width),
            'image/encoded': bytes_feature(encoded_image_data),
            'image/format': bytes_feature(image_format),
            'image/filename': bytes_feature(filename),
            'image/source_id': bytes_feature(filename),
            'image/object/bbox/xmin': float_list_feature(xmins),
            'image/object/bbox/xmax': float_list_feature(xmaxs),
            'image/object/bbox/ymin': float_list_feature(ymins),
            'image/object/bbox/ymax': float_list_feature(ymaxs),
            'image/object/class/text': bytes_list_feature(classes_text),
            'image/object/class/label': int64_list_feature(classes),}))

    def get_image_label_folder(self, fold_type=None, version=None):
        """
        Returns the folder containing all images and the folder containing all label information
        :param fold_type:
        :param version:
        :return: Raises BaseExceptions if expectations are not fulfilled
        """
        assert(fold_type in ['train', 'test', 'val'])
        version = '100k' if version is None else version
        assert(version in ['100k', '10k'])
        

        expansion_images_folder = os.path.join(self.input_path, 'images')
        expansion_labels_folder = os.path.join(self.input_path, 'labels')

        full_labels_path = os.path.join(expansion_labels_folder, 'bdd100k', 'labels')
        full_images_path = os.path.join(expansion_images_folder, 'bdd100k', 'images', version, fold_type)

        if fold_type == 'test':
            return full_images_path, None
        return full_images_path, full_labels_path

    def _get_label_data(self, pictureEntry):
        xmins, xmaxs, ymins, ymaxs, classes_text, classes = [], [], [], [], [], []
        for i in pictureEntry["labels"]:
            if i["category"] in DEEPDRIVE_LABELS.keys():
              classes_text.append(str.encode(i["category"]))
              classes.append(DEEPDRIVE_LABELS[i["category"]])
              xmins.append(i["box2d"]["x1"])
              xmaxs.append(i["box2d"]["x2"])
              ymins.append(i["box2d"]["y1"])
              ymaxs.append(i["box2d"]["y2"])
        return (xmins, xmaxs, ymins, ymaxs, classes_text, classes)
        


    def _get_tf_feature(self, image_path, image_format, annotations):
        height, width = Image.open(image_path).size
        xmins, xmaxs, ymins, ymaxs, classes_text, classes = self._get_label_data(annotations)
        with open(image_path, 'rb') as f:
            return self.createTFExample(height, width, f.read(), str.encode(image_format), str.encode(annotations["name"]), xmins, xmaxs, ymins, ymaxs, classes_text, classes)

    def write_tfrecord(self, fold_type=None, version=None, max_elements_per_file=1000, write_masks=False):
        output_path = os.path.join(self.input_path, 'tfrecord', version if version is not None else '100k', fold_type)
        if not os.path.exists(output_path):
            mkdir_p(output_path)

        full_images_path, full_labels_path = self.get_image_label_folder(fold_type, version)

        def get_jsonObject():
            if full_labels_path is None:
                return None
            with open(os.path.join(full_labels_path, 'bdd100k_labels_images_' + fold_type + '.json'), 'r') as f:
                return json.loads(f.read())

        labels = get_jsonObject()
        image_filename_regex = re.compile('^(.*)\.(jpg)$')
        tfrecord_file_id, writer = 0, None
        tfrecord_filename_template = os.path.join(output_path, 'output_{version}_{{iteration:06d}}.tfrecord'.format(
            version=fold_type + ('100k' if version is None else version)
        ))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for file_counter, detections in enumerate(labels):
                if file_counter % max_elements_per_file == 0:
                    if writer is not None:
                        writer.close()
                        tfrecord_file_id += 1
                    tmp_filename_tfrecord = tfrecord_filename_template.format(iteration=tfrecord_file_id)
                    print('{0}: Create TFRecord filename: {1} after processing {2}/{3} files'.format(
                        str(datetime.datetime.now()), tmp_filename_tfrecord, file_counter, len(labels)
                    ))
                    writer = tf.python_io.TFRecordWriter(tmp_filename_tfrecord)
                elif file_counter % 250 == 0:
                    print('\t{0}: Processed file: {1}/{2}'.format(
                        str(datetime.datetime.now()), file_counter, len(labels)))
                # match the filename with the regex
                m = image_filename_regex.search(detections['name'])
                if m is None:
                    print('Filename did not match regex: {0}'.format(detections['name']))
                    continue

                feature = self._get_tf_feature(
                    os.path.join(full_images_path, detections['name']),
                    m.group(2), detections)
                writer.write(feature.SerializeToString())

            # Close the last files
            if writer is not None:
                writer.close()
