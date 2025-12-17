####################################################################################################
# Author: Meghane Decroocq
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# provided license.
####################################################################################################

import numpy as np
import pandas as pd
import os
import glob
import time
from PIL import Image

"""
This code provide an example of how you can add your own metric to the benchmark.
"""

def your_metric(pred, label): # The function of the metric you want to test
	return 1

# Compute your metric for each image of the dataset
def compute_metric(dataset_path, metric_name):
	metric_dict = {"filename" : [], metric_name : []}
	
	pred_files = [x for x in os.walk(dataset_path)][0][2]
	pred_files = [f for f in pred_files if "pred" in f] 

	gt_file = dataset_path + "/gt.png"
	gt = np.array(Image.open(gt_file).convert('L')).copy().astype(bool)
		
	for pred_file in ["gt.png"] + pred_files:

		print("Computing metric for : ", category, dataset, image, pred_file)
		pred = np.array(Image.open(dataset_path + pred_file).convert('L')).copy().astype(bool)
		metric_dict["filename"].append(pred_file)
			
		# Compute your metric
		val = your_metric(pred, gt)
		metric_dict[metric_name].append(val)
		
	data = pd.DataFrame.from_dict(metric_dict)
	data.to_csv(dataset_path + metric_name + ".csv", sep='\t')



output_folder = "" # Where to write the complete metrics table
dataset_path = "../data/benchmark_dataset/"


categories = ["false_component", "missing_component", "component_merging", "disconnection", "cycle_disconnection", "hole_merging", "merging", "hole"]
categories  += [ "missing_branch", "false_branch", "branch_merging", "radius_erosion", "radius_dilation", "deformation", "radius_dilation", "radius_erosion"] 
categories += ["missing_component_and_false_component"] #, "self_merging", "false_terminal", "missing_terminal"]

datasets = ["Roads", "Colon_cells", "CREMI", "LES-AV", "Minivess", "NeuroMorpho"]
metric_name = "my_metric"

# Compute your metric for each image in the benchmark and write the result in each folder
for category in categories:
	for dataset in datasets:
		images = [x for x in os.walk(dataset_path + category + "/" + dataset + "/")][0][1]
		for image in images:
			compute_metric(dataset_path + category + "/" + dataset + "/" + image + "/", metric_name)


# Write all metrics to a single file 
first = True
for category in categories:
	for dataset in datasets:
		images = [x for x in os.walk(dataset_path + category + "/" + dataset + "/")][0][1]
		for image in images:
			# Read degradation scores 
			dg = pd.read_csv(dataset_path + category + "/" + dataset + "/" + image + "/" + metric_name +".csv", sep="\t")
			
			# Read metrics
			mt = pd.read_csv(dataset_path + category + "/" + dataset + "/" + image + "/" + "degradation_score.csv", sep="\t")

			join = mt.join(dg.set_index('filename'), on='filename')
			join.insert(0, "image", [image] * join.shape[0], True)
			join.insert(0, "dataset", [dataset] * join.shape[0], True)
			join.insert(0, "category", [category] * join.shape[0], True)

			if first:
				df = join 
				first = False
			else:
				df = pd.concat([df, join], axis=0, ignore_index=True)

df.to_csv(output_folder + metric_name + ".csv", sep='\t', index = False)
