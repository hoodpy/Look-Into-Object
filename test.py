import tensorflow as tf
import numpy as np
import cv2
import os
import pandas as pd
import tensorflow.contrib.slim as slim
from model import Timer, ResnetImproved


def calc_acc_t(confusion_matrix):
	num_classes = np.shape(confusion_matrix)[0]
	correct = [confusion_matrix[i, i] for i in range(num_classes)]
	return sum(correct) / np.sum(confusion_matrix)

def calc_acc_f1(confusion_matrix):
	num_classes = np.shape(confusion_matrix)[0]
	acc_list, f1_list = [], []
	for label in range(num_classes):
		current_matrix = np.zeros((2, 2))
		current_matrix[0, 0] = confusion_matrix[label, label]
		current_matrix[0, 1] = np.sum(confusion_matrix[label, :]) - current_matrix[0, 0]
		current_matrix[1, 0] = np.sum(confusion_matrix[:, label]) - current_matrix[0, 0]
		current_matrix[1, 1] = np.sum(confusion_matrix) - current_matrix[0, 0] - current_matrix[0, 1] - current_matrix[1, 0]
		precision = current_matrix[0, 0] / (current_matrix[0, 0] + current_matrix[1, 0])
		recall = current_matrix[0, 0] / (current_matrix[0, 0] + current_matrix[0, 1])
		acc_list.append(current_matrix[0, 0] / (current_matrix[0, 0] + current_matrix[0, 1]))
		f1_list.append(2 * precision * recall / (precision + recall))
	acc_list.append(sum(acc_list) / num_classes)
	f1_list.append(sum(f1_list) / num_classes)
	return acc_list, f1_list

def calc_auc(data_frame, num_classes):
	auc_list = []
	for label in range(num_classes):
		TPR_list, FPR_list, area = [], [], 0
		probabilities = list(data_frame.iloc[:, label])
		acc_labels = list(data_frame.iloc[:, -1])
		acc_labels = [1 - int(value==label) for value in acc_labels]
		threshold_list = list(set(probabilities))
		threshold_list.sort(reverse=True)
		for threshold in threshold_list:
			pre_labels = [1 - int(value > threshold) for value in probabilities]
			current_matrix = np.zeros((2, 2))
			for i in range(len(acc_labels)):
				current_matrix[acc_labels[i], pre_labels[i]] += 1
			TPR = current_matrix[0, 0] / (current_matrix[0, 0] + current_matrix[0, 1])
			FPR = current_matrix[1, 0] / (current_matrix[1, 0] + current_matrix[1, 1])
			TPR_list.append(TPR)
			FPR_list.append(FPR)
		for i in range(len(TPR_list)-1):
			area += 0.5 * (FPR_list[i+1] - FPR_list[i]) * (TPR_list[i+1] + TPR_list[i])
		auc_list.append(area)
	auc_list.append(sum(auc_list) / num_classes)
	return auc_list


file_path = "D:/program/garbage_fusion/data_fusion/train/"
model_path = "D:/program/look_into_object/model"
content_csv = "ACC,AUC,F1_Score,FPS,Acc0,Acc1,Acc2,Acc3,Acc4,Acc_average,F0,F1,F2,F3,F4,AUC0,AUC1,AUC2,AUC3,AUC4"
categories = os.listdir(file_path)
network = ResnetImproved(feature_size=7, is_training=False, num_classes=len(categories))
timer = Timer()

images_path, labels_list, label = [], [], 0
for category in categories:
	files_path = file_path + category
	for name in os.listdir(files_path):
		images_path.append(os.path.join(files_path, name))
		labels_list.append(label)
	label += 1

confusion_matrix = np.zeros((len(categories), len(categories)), dtype=np.float32)
data_frame = pd.DataFrame(columns=["P0", "P1", "P2", "P3", "P4"])


if __name__ == "__main__":
	config = tf.compat.v1.ConfigProto()
	config.allow_soft_placement = True
	config.gpu_options.allow_growth = True
	with tf.compat.v1.Session(config=config) as sess:
		images_input = tf.placeholder(tf.float32, [448, 448, 3], name="images_input")
		logits_scores = network.forword(tf.expand_dims(images_input, axis=0))
		pred_prob = tf.nn.softmax(logits_scores, axis=1, name="pred_prob") # [None, num_classes]
		pred_labels = tf.argmax(logits_scores, dimension=1, name="pred_labels") # [None]

		load_fn = slim.assign_from_checkpoint_fn(model_path, tf.compat.v1.global_variables(), ignore_missing_vars=True)
		sess.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()])
		load_fn(sess)
		print("Load network: " + model_path)

		for i in range(len(images_path)):
			timer.tic()
			image = cv2.resize(cv2.imread(images_path[i]), (448, 448))[:, :, ::-1] / 255.
			prob, result = sess.run([pred_prob, pred_labels], feed_dict={images_input: image})
			timer.toc()
			prob, result = prob[0], result[0]
			data_frame.loc[i] = prob
			confusion_matrix[labels_list[i], result] += 1.0
		data_frame["Class"] = labels_list

	if model_path.split("/")[-2] == "model1":
		content_csv = content_csv + "\n"
	else:
		content_csv = ""

	content_csv+=str(round(acc_t,4))+","+str(round(auc[5],4))+","+str(round(f1[5],4))+","+str(round(1./timer.average_time,2))+","
	content_csv+=str(round(acc[0],4))+","+str(round(acc[1],4))+","+str(round(acc[2],4))+","+str(round(acc[3],4))+","
	content_csv+=str(round(acc[4],4))+","+str(round(acc[5],4))+","+str(round(f1[0],4))+","+str(round(f1[1],4))+","
	content_csv+=str(round(f1[2],4))+","+str(round(f1[3],4))+","+str(round(f1[4],4))+","+str(round(auc[0],4))+","
	content_csv+=str(round(auc[1],4))+","+str(round(auc[2],4))+","+str(round(auc[3],4))+","+str(round(auc[4],4))+"\n"

	with open("E:/paper/garbage-classification/statistic/kaggle/salinet_inception/Test(MSRA-C).txt", "a+") as f:
		f.write(content_csv)

	content = "Test (MSRA-C)\n" + model_path + "\n"
	for category in categories:
		content += category + " "
	content += "\n"
	content += "Total accuracy: " + str(round(acc_t, 4)) + "  FPS: " + str(round(1. / timer.average_time, 2)) + "\n"
	content += "acc in 0  acc in 1  acc in 2  acc in 3  acc in 4  average acc \n"
	content += str(round(acc[0], 4)) + "  " + str(round(acc[1], 4)) + "  " + str(round(acc[2], 4)) + "  " + \
	str(round(acc[3], 4)) + "  " + str(round(acc[4], 4)) + "  " + str(round(acc[5], 4)) + "\n"
	content += "f1 in 0  f1 in 1  f1 in 2  f1 in 3  f1 in 4  average f1 \n"
	content += str(round(f1[0], 4)) + "  " + str(round(f1[1], 4)) + "  " + str(round(f1[2], 4)) + "  " + \
	str(round(f1[3], 4)) + "  " + str(round(f1[4], 4)) + "  " + str(round(f1[5], 4)) + "\n"
	content += "auc in 0  auc in 1  auc in 2  auc in 3  auc in 4  average auc \n"
	content += str(round(auc[0], 4)) + "  " + str(round(auc[1], 4)) + "  " + str(round(auc[2], 4)) + "  " + \
	str(round(auc[3], 4)) + "  " + str(round(auc[4], 4)) + "  " + str(round(auc[5], 4)) + "\n"
	print(content)