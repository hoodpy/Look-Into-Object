import tensorflow as tf
import time
import tensorflow.contrib.slim as slim
import tensorflow.contrib.layers as layers_lib
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib.framework.python.ops import arg_scope
from scl_module import SCLModule


class Timer():
	def __init__(self):
		self.total_time = 0
		self.calls = 0
		self.start_time = 0
		self.diff = 0
		self.average_time = 0

	def tic(self):
		self.start_time = time.time()

	def toc(self):
		self.calls += 1
		self.diff = time.time() - self.start_time
		self.total_time += self.diff
		self.average_time = self.total_time / self.calls

class ResnetImproved():
	def __init__(self, feature_size, is_training, num_classes):
		self._feature_size = feature_size
		self._is_training = is_training
		self._num_classes = num_classes
		if self._is_training:
			self._scl_module = SCLModule(size=self._feature_size, structure_dim=1024, avg=True)

	def forword(self, inputs):
		with slim.arg_scope(resnet_v1.resnet_arg_scope(batch_norm_decay=0.9997)):
			logits, end_points = resnet_v1.resnet_v1_50(inputs, self._num_classes)
			logits_scores = tf.squeeze(logits, axis=[1, 2]) # [batch_size, num_classes]
			end = end_points["resnet_v1_50/block4"] # [batch_size, 14, 14, 2048] for input: [batch_size, 448, 448, 3]

			if self._is_training:
				features, mask, coord_loss = self._scl_module.forword(end)
				return logits_scores, features, mask, coord_loss

		return logits_scores