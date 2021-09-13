import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers as layers_lib


def gen_temp_mask(reference, basic_index, aim_index):
	result = np.ones_like(reference)
	result[basic_index, aim_index, :] = 0.
	return result

def gen_angle_comp(reference, index_y, index_x):
	result = np.zeros_like(reference)
	result[index_y, index_x] = 1.
	return result


class CoordPredictor():
	def __init__(self, size=7):
		self.size = size
		self.basic_label = tf.convert_to_tensor(self.build_basic_label(self.size), dtype=tf.float32)

	def build_basic_label(self, size):
		return np.array([[(i, j) for j in range(size)] for i in range(size)])

	def forword(self, x, mask):
		N, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
		reduce_x = tf.reshape(x, [N, H * W, C])
		mask = tf.reshape(mask, [N, H * W])

		thresholds = tf.math.reduce_mean(mask, axis=1, keep_dims=True)
		binary_mask = tf.cast(tf.math.less(thresholds, mask), tf.float32)
		binary_mask = tf.reshape(binary_mask, [N, H, W])
		binary_mask = tf.tile(tf.expand_dims(binary_mask, axis=-1), [1, 1, 1, C])

		masked_x = tf.reshape(x * binary_mask, [N, H * W, C])
		reduce_x_max_index = tf.cast(tf.argmax(tf.math.reduce_mean(masked_x, axis=-1), dimension=-1), dtype=tf.int64) # [N]
		basic_index = tf.cast(tf.range(N), dtype=tf.int64)
		indic = tf.stack([basic_index, reduce_x_max_index], axis=1)

		max_features = tf.gather_nd(reduce_x, indic)
		max_features_concat = tf.tile(tf.expand_dims(max_features, axis=1), [1, H * W, 1])

		discriminative_feature = tf.concat([reduce_x, max_features_concat], -1)
		discriminative_feature = tf.reshape(discriminative_feature, [N, H, W, 2 * C])
		preds_coord = layers_lib.conv2d(discriminative_feature, num_outputs=2, kernel_size=[1, 1])
		preds_coord = tf.reshape(preds_coord, [N, H * W, 2])

		temp_mask = tf.py_func(gen_temp_mask, [preds_coord, basic_index, reduce_x_max_index], tf.float32)
		preds_coord = tf.math.multiply(preds_coord, temp_mask)
		preds_coord = tf.reshape(preds_coord, [N, H, W, 2])

		label = self.basic_label
		label = tf.tile(tf.expand_dims(label, axis=0), [N, 1, 1, 1])
		label = tf.reshape(label, [N, H * W, 2])
		basic_anchor = tf.expand_dims(tf.gather_nd(label, indic), axis=1)

		relative_coord = (label - basic_anchor) / self.size
		relative_dist = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(relative_coord), axis=-1))
		relative_angle = tf.math.atan2(relative_coord[:, :, 1], relative_coord[:, :, 0])
		relative_angle = (relative_angle / np.pi + 1) / 2

		preds_dist, preds_angle = preds_coord[:, :, :, 0], preds_coord[:, :, :, 1]

		preds_dist = tf.reshape(preds_dist, [N, H, W])
		relative_dist = tf.reshape(relative_dist, [N, H, W])
		dist_loss = tf.math.square(tf.math.subtract(preds_dist, relative_dist))

		preds_angle = tf.reshape(preds_angle, [N, H * W])
		gap_angle = preds_angle - relative_angle
		angle_indic = tf.transpose(tf.where(tf.math.less(gap_angle, 0)), perm=[1, 0])
		comp = tf.py_func(gen_angle_comp, [gap_angle, angle_indic[0], angle_indic[1]], tf.float32)
		gap_angle = gap_angle + comp
		gap_angle = gap_angle - tf.math.reduce_mean(gap_angle, axis=1, keep_dims=True)
		gap_angle = tf.reshape(gap_angle, [N, H, W])
		angle_loss = tf.math.square(gap_angle)

		return tf.math.add(dist_loss, angle_loss)

class SCLModule():
	def __init__(self, size, structure_dim, avg):
		self.structure_dim = structure_dim
		self.avg = avg
		self.coord_predictor = CoordPredictor(size)

	def forword(self, features):
		with tf.compat.v1.variable_scope("scl_lrx"):
			if self.avg:
				features = layers_lib.avg_pool2d(features, [2, 2], stride=2)

			mask = tf.nn.relu(features)
			mask = layers_lib.conv2d(mask, num_outputs=1, kernel_size=[1, 1])

			structure_map = tf.nn.relu(features)
			structure_map = layers_lib.conv2d(structure_map, self.structure_dim, kernel_size=[1, 1])

			coord_loss = self.coord_predictor.forword(structure_map, mask)

		return features, mask, coord_loss