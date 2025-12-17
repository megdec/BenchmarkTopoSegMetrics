####################################################################################################
# Author: Meghane Decroocq
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# provided license.
####################################################################################################

from ErrorGenerator import ErrorGenerator
import glob
import os
import numpy as np
from PIL import Image
from utils import *
import random

"""
This code provide an example of how you can create a dataset of synthetic images from some labels.
This code was used to generate the benchmark dataset itself.
"""

# Where to write the benchmark generator
output_folder = "../output/dataset/"

label_folder = "../data/playground/" # Path to the folder containing the labels to use
dataset_names = ["dataset1", "dataset2"] # Datasets in the label folder

# Error types that we want to write in the benchmark generator
error_types = ["false_component", "component_merging", "missing_component", "disconnection", "hole_merging", "merging", "hole"]
error_types += ["cycle_disconnection", "branch_merging", "missing_branch", "radius_dilation", "radius_erosion", "deformation", "false_branch"]
error_types += ["missing_terminal", "false_terminal", "self_merging"]

nb_img = 100 # Target number of images to generate for each error
target = 20 # Number of images to keep in the final dataset
min_lim = 1 # Minimum number of images

# Generate errors one by one
for error_type in error_types:
	for dataset_name in dataset_names:
		list_files = glob.glob(label_folder + dataset_name + "/*")
		print("Generating error type \"" + error_type + "\" for dataset \"" + dataset_name + "\".")

		img_id = -1
		for file in list_files:
			img_id += 1
	
			# Folder to write the images 
			write_folder = output_folder + error_type + "/" + dataset_name + "/img_" + str(img_id) + "/" 
			os.makedirs(write_folder, exist_ok=True)

			# Open image
			if file[-4:] == ".nii" or file[-7:] == ".nii.gz":
				img = nib.load(file)
				img = img.get_fdata().astype(bool)
			else:
				img = np.array(Image.open(file).convert('L')).copy().astype(bool)

			generator = ErrorGenerator(img, error_types = [error_type])
			save_image(write_folder + "gt", generator.get_label())
			colnames = [k for k in generator.get_scores()[error_type].keys()]
			gt_metrics = [generator.get_scores()[error_type][k] for k in generator.get_scores()[error_type].keys()]

			fail = False
			nb = 0
			metrics = None 

			while nb < nb_img and not fail:

				fail = generator.generate_error(error_type)

				if not fail:
					save_image(write_folder + "tmp_" + str(nb), generator.pred)
					if metrics is None:
						metrics = np.zeros((nb_img, len(colnames)))
					metrics[nb, :] = [generator.get_scores()[error_type][k] for k in generator.get_scores()[error_type].keys()]
					nb += 1


			# Re-write folder to have less images 
			sep = "\t"
			ext = ".png"

			metric_file = open(write_folder + "degradation_score.csv", "w")

			# Write column names
			line1 = "filename"
			for col in colnames:
				line1 += sep + col
			metric_file.write(line1 +"\n")

			# Write label metrics
			string = "gt.png"
			for k in range(len(gt_metrics)):
				string += sep + str(gt_metrics[k])
			metric_file.write(string + "\n")


			if nb < min_lim:
				to_keep = []
			elif nb >= min_lim and nb<=target:
				to_keep = np.arange(0, nb).tolist()
			else:
				to_keep = np.round(np.linspace(0, nb-1, target + 1)).astype(int).tolist()[1:]

			c = 0
			for i in range(nb):

				if i in to_keep:

					# Write metrics in file
					string = "pred_" + str(c) + ".png"
					for k in range(metrics.shape[1]):
						string += sep + str(metrics[i, k])
					metric_file.write(string + "\n")

					# Rename file
					os.rename(write_folder + "tmp_" + str(i) + ext, write_folder + "pred_" + str(c) + ext)
					c+=1

				else:
					# Delete other files
					os.remove(write_folder + "tmp_" + str(i) + ext)
			metric_file.close()


# Generate several errors at the same time
error_types = ["missing_component", "false_component"]

for dataset_name in dataset_names:
	list_files = glob.glob(label_folder + dataset_name + "/*")

	img_id = -1
	for file in list_files:
		img_id += 1
	
		# Folder to write the images 
		write_folder = output_folder + str(error_types) + "/" + dataset_name + "/img_" + str(img_id) + "/" 
		os.makedirs(write_folder, exist_ok=True)

		# Open image
		if file[-4:] == ".nii" or file[-7:] == ".nii.gz":
			img = nib.load(file)
			img = img.get_fdata().astype(bool)
		else:
			img = np.array(Image.open(file).convert('L')).copy().astype(bool)

		generator = ErrorGenerator(img, error_types = error_types)
		save_image(write_folder + "gt", generator.get_label())
		colnames = [k for k in generator.get_scores()[error_types[0]].keys()]
		gt_metrics = [generator.get_scores()[error_types[0]][k] for k in generator.get_scores()[error_types[0]].keys()]

		fail = False
		nb = 0
		metrics = None 

		while nb < nb_img and not fail:
			# Choose an error randomly 
			category = random.choice(error_types)
			fail = generator.generate_error(category)

			if not fail:
				print("Writing synthetic image " + str(nb))
				save_image(write_folder + "tmp_" + str(nb), generator.get_pred())
				if metrics is None:
					metrics = np.zeros((nb_img, len(colnames)))

				# Combine metrics from both categories
				scores = generator.combine_scores(error_types)
				metrics[nb, :] = [scores[k] for k in generator.get_scores()[error_types[0]].keys()]
				nb += 1

		# Re-write folder to have less images 
		sep = "\t"
		ext = ".png"

		metric_file = open(write_folder + "degradation_score.csv", "w")

		# Write column names
		line1 = "filename"
		for col in colnames:
			line1 += sep + col
		metric_file.write(line1 +"\n")

		# Write label metrics
		string = "gt.png"
		for k in range(len(gt_metrics)):
			string += sep + str(gt_metrics[k])
		metric_file.write(string + "\n")

		if nb < min_lim:
			to_keep = []
		elif nb >= min_lim and nb<=target:
			to_keep = np.arange(0, nb).tolist()
		else:
			to_keep = np.round(np.linspace(0, nb-1, target + 1)).astype(int).tolist()[1:]

		c = 0
		for i in range(nb):

			if i in to_keep:

				# Write metrics in file
				string = "pred_" + str(c) + ".png"
				for k in range(metrics.shape[1]):
					string += sep + str(metrics[i, k])
				metric_file.write(string + "\n")

				# Rename file
				os.rename(write_folder + "tmp_" + str(i) + ext, write_folder + "pred_" + str(c) + ext)
				c+=1

			else:
					# Delete other files
				os.remove(write_folder + "tmp_" + str(i) + ext)
		metric_file.close()		







