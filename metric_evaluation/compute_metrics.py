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
from quality_metrics import *

"""
This code computes the quality metrics included in the benchmark on all the images of the benchmark dataset. 
The results are written and output as a csv file.
"""

def compute_metrics(dataset_path, eval_metrics):
	
	colnames = ["filename"]

	for metric in eval_metrics:
		if metric in ["Betti_error", "Betti_matching"]:
			colnames += [metric + "_0", metric + "_1",  metric + "_time"]
		else:
			colnames += [metric, metric + "_time"]

	metric_dict = {}
	for col in colnames:
		metric_dict[col] = []


	pred_files = [x for x in os.walk(dataset_path)][0][2]
	pred_files = [f for f in pred_files if "pred" in f] #glob.glob(dataset_path + category + "/" + dataset + "/" + image + "/pred_*.png")

	gt_file = dataset_path + "/gt.png"
	gt = np.array(Image.open(gt_file).convert('L')).copy().astype(bool)
		
	for pred_file in ["gt.png"] + pred_files:

		print("Computing metrics for : ", category, dataset, image, pred_file)
		pred = np.array(Image.open(dataset_path + pred_file).convert('L')).copy().astype(bool)
		metric_dict["filename"].append(pred_file)

					
		# Compute evaluation metrics
		for metric in eval_metrics:
		
			t1 = time.time()
			if metric == "clDice":
				val = clDice(pred, gt)
				t = time.time() - t1
				metric_dict["clDice"].append(val)
				metric_dict["clDice_time"].append(t)

			elif metric == "Dice":
				val = Dice(pred, gt)
				t = time.time() - t1
				metric_dict["Dice"].append(val)
				metric_dict["Dice_time"].append(t)
				
			elif metric == "Betti_matching":
				val = Betti_matching(pred, gt, relative=False, comparison='union', filtration='superlevel', construction='V')
				t = time.time() - t1
				metric_dict["Betti_matching_0"].append(val[0])
				metric_dict["Betti_matching_1"].append(val[1])
				metric_dict["Betti_matching_time"].append(t)

			elif metric == "Betti_error":
				val = Betti_error(pred, gt)
				t = time.time() - t1
				metric_dict["Betti_error_0"].append(val[0])
				metric_dict["Betti_error_1"].append(val[1])
				metric_dict["Betti_error_time"].append(t)

			elif metric == "ccDice_0":
				val = ccDice(pred, gt, alpha=0.5)
				t = time.time() - t1
				metric_dict["ccDice_0"].append(val)
				metric_dict["ccDice_0_time"].append(t)

			elif metric == "ccDice_1":
				val = ccDice_1(pred, gt, alpha=0.5)
				t = time.time() - t1
				metric_dict["ccDice_1"].append(val)
				metric_dict["ccDice_1_time"].append(t)

			if metric == "cbDice":
				val = cbDice(pred, gt, method = "srimb_norm")
				t = time.time() - t1
				metric_dict["cbDice"].append(val)
				metric_dict["cbDice_time"].append(t)

	data = pd.DataFrame.from_dict(metric_dict)
	data.to_csv(dataset_path + "metrics.csv", sep='\t')



output_folder = "" # Where to write the complete metrics table
dataset_path = "../outputs/benchmark_dataset/"


categories = ["false_component", "missing_component", "component_merging", "disconnection", "cycle_disconnection", "hole_merging", "merging", "hole"]
categories  += [ "missing_branch", "false_branch", "branch_merging", "radius_erosion", "radius_dilation", "deformation", "radius_dilation", "radius_erosion"] 
categories += ["missing_component_and_false_component"] #, "self_merging", "false_terminal", "missing_terminal"]

eval_metrics = ["Dice", "cbDice", "clDice", "ccDice_0", "ccDice_1", "Betti_error", "Betti_matching"]
datasets = ["Roads", "Colon_cells", "CREMI", "LES-AV", "Minivess", "NeuroMorpho"]


# Compute the evaluation metrics for each image in the benchmark and write the result in each folder
for category in categories:
	for dataset in datasets:
		images = [x for x in os.walk(dataset_path + category + "/" + dataset + "/")][0][1]
		for image in images:
			compute_metrics(dataset_path + category + "/" + dataset + "/" + image + "/", eval_metrics)


# Write all metrics to a single file 
first = True
for category in categories:
	for dataset in datasets:
		images = [x for x in os.walk(dataset_path + category + "/" + dataset + "/")][0][1]
		for image in images:
			# Read degradation scores 
			dg = pd.read_csv(dataset_path + category + "/" + dataset + "/" + image + "/" + "evaluation_metrics.csv", sep="\t")
			
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

df.to_csv(output_folder + "metric_data.csv", sep='\t', index = False)
