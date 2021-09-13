import tensorflow as tf
import numpy as np
import cv2
import os
import tensorflow.contrib.slim as slim
from model import Timer, ResnetImproved


class Trainer():
	def __init__(self):
		self.file_path = "D:/program/garbage_fusion/data_fusion/train/"
		self.pre_trained = "D:/program/look_into_object/checkpoint/resnet_v1_50.ckpt"
		self.log_dir = "D:/program/look_into_object/log"
		self.model_path = "D:/program/look_into_object/model"
		self.epochs = 50
		self.batch_size = 5
		self.num_positive = 3
		self.image_size = 448
		self.feature_size = 7
		self.feature_channal = 2048
		self.num_classes = 5
		self.learning_rate = 0.001
		self.momentum = 0.9
		self.image_paths, self.label_list = self.gen_img_with_label()
		self.samples_num = len(self.image_paths)
		self.timer = Timer()
		self.network = ResnetImproved(feature_size=self.feature_size, is_training=True, num_classes=self.num_classes)
		self.loss_weights = {"mask": tf.convert_to_tensor(0.1, dtype=tf.float32), "coord": tf.convert_to_tensor(0.1, dtype=tf.float32)}

	def gen_img_with_label(self):
		image_paths, label_list, label = [], [], 0
		for category in os.listdir(self.file_path):
			images_path = self.file_path + category
			for name in os.listdir(images_path):
				image_paths.append(os.path.join(images_path, name))
				label_list.append(label)
			label += 1
		return image_paths, label_list

	def data_loader(self, image_paths, label_list, start, end):
		imgs, labels = [], []
		for i in range(start, end):
			index = i % self.samples_num
			image = cv2.resize(cv2.imread(image_paths[index]), (self.image_size, self.image_size))[:, :, ::-1] / 255.
			label = label_list[index]
			imgs.append(image)
			labels.append(label)
			temp_image_paths = np.array(image_paths[:index] + image_paths[index+1:])
			temp_label_list = np.array(label_list[:index] + label_list[index+1:])
			temp_image_paths = temp_image_paths[np.where(temp_label_list == label)]
			temp_image_paths = np.random.choice(temp_image_paths, size=self.num_positive, replace=False)
			for path in temp_image_paths:
				image = cv2.resize(cv2.imread(path), (self.image_size, self.image_size))[:, :, ::-1] / 255.
				imgs.append(image)
				labels.append(label)
		return imgs, labels

	def mask_to_binary(self, inputs):
		mask = tf.squeeze(inputs, axis=-1)
		mask = tf.reshape(mask, [self.batch_size*(self.num_positive+1), self.feature_size*self.feature_size])
		thresholds = tf.math.reduce_mean(mask, axis=1, keepdims=True)
		binary_x = tf.cast(tf.math.less(thresholds, mask), tf.float32)
		binary_x = tf.reshape(binary_x, [self.batch_size*(self.num_positive+1), self.feature_size, self.feature_size])
		return binary_x

	def calc_mask(self, object_1, object_2):
		object_1 = tf.reshape(tf.tile(tf.expand_dims(object_1, axis=1), [1, self.num_positive, 1, 1]), [
			self.batch_size * self.num_positive, self.feature_size * self.feature_size, self.feature_channal])
		relation = tf.linalg.matmul(object_1, tf.transpose(object_2, perm=[0, 2, 1])) / self.feature_channal
		object_1_target = tf.reshape(tf.math.reduce_max(relation, axis=2), [self.batch_size, self.num_positive,
			self.feature_size * self.feature_size])
		object_1_target = tf.math.reduce_mean(object_1_target, axis=1) # [batch_size, feature_size * feature_size]
		return object_1_target

	def get_mask(self, all_features):
		all_features = tf.reshape(all_features, [self.batch_size*(self.num_positive+1), self.feature_size*self.feature_size,
			self.feature_channal])
		all_features = tf.reshape(all_features, [self.batch_size, self.num_positive + 1, self.feature_size * self.feature_size,
			self.feature_channal])
		all_masks = []

		for i in range(self.num_positive + 1):
			main_features = tf.gather(all_features, i, axis=1)
			sub_indic = [k for k in range(self.num_positive + 1) if k != i]
			sub_features = tf.gather(all_features, sub_indic, axis=1)
			sub_features = tf.reshape(sub_features, [self.batch_size * self.num_positive, self.feature_size * self.feature_size,
				self.feature_channal])
			main_mask = self.calc_mask(main_features, sub_features) # [batch_size, feature_size * feature_size]
			main_mask = tf.reshape(main_mask, [self.batch_size, self.feature_size, self.feature_size])
			all_masks.append(tf.expand_dims(main_mask, axis=1))

		all_masks = tf.concat(all_masks, axis=1) # [batch_size, num_positive + 1, feature_size, feature_size]
		all_masks = tf.reshape(all_masks, [self.batch_size*(self.num_positive+1), self.feature_size, self.feature_size])

		return all_masks

	def calc_rela(self, all_features, pred_masks):
		all_masks = self.get_mask(all_features)
		pred_masks = tf.squeeze(pred_masks, axis=-1)
		rela_loss = tf.math.reduce_mean(tf.math.square(tf.math.subtract(all_masks, pred_masks)))
		return rela_loss

	def get_load_variables(self):
		added, exclusion = "resnet_v1_50", "resnet_v1_50/logits"
		variables_to_restore = []
		for variable in slim.get_model_variables():
			if (added in variable.op.name) and (exclusion not in variable.op.name):
				variables_to_restore.append(variable)
		return variables_to_restore

	def train(self):
		config = tf.compat.v1.ConfigProto()
		config.allow_soft_placement = True
		config.gpu_options.allow_growth = True
		with tf.compat.v1.Session(config=config) as sess:
			global_step = tf.Variable(0, trainable=False)
			images_input = tf.placeholder(tf.float32, [self.batch_size*(self.num_positive+1), self.image_size, self.image_size, 3], 
				name="images_input")
			images_label = tf.placeholder(tf.int32, [self.batch_size*(self.num_positive+1)], name="images_label")
			logits_scores, features, mask, coord_loss = self.network.forword(images_input)

			cls_loss = tf.math.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=images_label, logits=logits_scores),
				name="cross_entropy")
			coord_loss = tf.math.multiply(tf.math.reduce_mean(coord_loss * self.mask_to_binary(mask)), self.loss_weights["coord"],
				name="coord_loss")
			mask_reg_loss = self.calc_rela(features, mask) * self.loss_weights["mask"]

			total_loss = cls_loss + coord_loss + mask_reg_loss

			tf.compat.v1.summary.scalar("cls_loss", cls_loss)
			tf.compat.v1.summary.scalar("coord_loss", coord_loss)
			tf.compat.v1.summary.scalar("mask_reg_loss", mask_reg_loss)
			tf.compat.v1.summary.scalar("total_loss", total_loss)

			update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
			optimizer = tf.compat.v1.train.MomentumOptimizer(self.learning_rate, self.momentum)
			with tf.control_dependencies(update_ops):
				grads_with_vars = optimizer.compute_gradients(total_loss, var_list=tf.compat.v1.trainable_variables())
				final_grads_with_vars = []
				for grad, var in grads_with_vars:
					if "/logits" in var.name or "lrx" in var.name:
						grad = tf.math.multiply(grad, 10.0)
					final_grads_with_vars.append((grad, var))
				train_op = optimizer.apply_gradients(final_grads_with_vars, global_step=global_step)

			self.saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables(), max_to_keep=5)
			load_fn = slim.assign_from_checkpoint_fn(self.pre_trained, self.get_load_variables(), ignore_missing_vars=True)
			merged = tf.compat.v1.summary.merge_all()
			summary_writer = tf.compat.v1.summary.FileWriter(self.log_dir, sess.graph)

			sess.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()])
			load_fn(sess)
			print("Load network: " + self.pre_trained)

			for epoch in range(self.epochs):
				image_paths, label_list = self.image_paths, self.label_list
				state = np.random.get_state()
				np.random.shuffle(image_paths)
				np.random.set_state(state)
				np.random.shuffle(label_list)
				start, end = 0, self.batch_size

				while start < self.samples_num:
					self.timer.tic()
					images, labels = self.data_loader(image_paths, label_list, start, end)
					feed_dict = {images_input: images, images_label: labels}
					_, _total_loss, _cls_loss, _coord_loss, _mask_reg_loss, steps, summary = sess.run([train_op, total_loss,
						cls_loss, coord_loss, mask_reg_loss, global_step, merged], feed_dict = feed_dict)
					summary_writer.add_summary(summary, steps)
					start += self.batch_size
					end += self.batch_size
					self.timer.toc()

					if (steps + 1) % 560 == 0:
						print(">>>total_loss: %.6f\n>>>cls_loss: %.6f\n>>>coord_loss: %.6f\n>>>mask_reg_loss: %.6f" % (_total_loss,
							_cls_loss, _coord_loss, _mask_reg_loss))
						print(">>>average_speed: %.6fs/step\n" % (self.timer.average_time))

				if (epoch + 1) % 10 == 0:
					self.snap_shot(sess, epoch + 1)

	def snap_shot(self, sess, iter):
		file_name = os.path.join(self.model_path, "model" + str(iter) + ".ckpt")
		self.saver.save(sess, file_name)
		print("Wrote snapshot to: " + file_name + "\n")


if __name__ == "__main__":
	trainer = Trainer()
	trainer.train()