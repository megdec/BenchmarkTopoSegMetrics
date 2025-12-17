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
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from quality_metrics import *
import math
from matplotlib.lines import Line2D

"""
This code reproduces the weight and correlation maps of the paper, based on the metric_data.csv table provided that contains the metric values.
"""

def Pearson_correlation(x, y):
	n = len(x)
	return (n*np.sum(x*y) - np.sum(x)*np.sum(y)) / (np.sqrt((n*np.sum(x**2) - np.sum(x)**2)*(n*np.sum(y**2) - np.sum(y)**2)))

# Normalization strategy
def normalize(tp, fp, fn, function = "dice"):

	# Dice
	if function == "dice":
		return 2*tp / (2*tp + fp + fn)

	# 1 - Dice
	if function == "rev_dice":
		return 1- (2*tp / (2*tp + fp + fn))

	# IoU
	elif function == "IoU":
		return tp / (tp + fp + fn)

	elif function == "frac":
		return (tp + fn) / (tp + fn + tp)

	# No normalization 
	elif function == "no_norm":
		return fp + fn

	else:
		raise NameError("Undefined function.")


def scale_weights_for_visu(array, axis=0):

	for i in range(array.shape[axis]):
		if axis == 0:
			if np.nanmax(np.abs(array[i,:])) != 0:
				array[i,:] = array[i,:] / np.nanmax(np.abs(array[i,:]))
		else:
			if np.nanmax(np.abs(array[:,i])) != 0:
				array[:,i] = array[:,i] / np.nanmax(np.abs(array[:,i]))
	return array


def plot_array(array, legendx, legendy, title, save_plot, output_file = "", std = None, color = None):

	fig, ax = plt.subplots() 
	if color is None:
		ax.matshow(np.abs(array))
	else:
		ax.matshow(np.abs(color))

	for i in range(array.shape[0]):
		for j in range(array.shape[1]):
			label_color = "black"
			if math.isnan(array[i,j]):
				val = "NA"
			else:
				if std is None:
					val = str(round(array[i,j], 2))
				else:
					val = str(round(array[i,j], 2)) + "\n"
					std_val ="\n" + r"$\pm$" + str(round(std[i,j], 2))

				if abs(round(array[i,j], 2)) < 0.9:
					label_color = "white"
				
			if std is None:
				plt.text(j, i, val, ha='center', va='center', fontweight="bold", fontsize = 9, color = label_color)
			else:
				plt.text(j, i, val, ha='center', va='center', fontweight="bold", fontsize = 9, color = label_color)
				plt.text(j, i, std_val, ha='center', va='center', fontsize = 6, color = label_color)



	ax.set_xticks(np.arange(0,len(legendx),1), legendx, fontsize=12, rotation = 90)
	ax.set_yticks(np.arange(0,len(legendy),1), legendy, fontsize=12)

	lw = 2
	for axis in ['top','bottom','left','right']:
		ax.spines[axis].set_linewidth(lw)
		ax.spines[axis].set_color("white")


	ax.set_title(title)
	ax.tick_params(top=True, bottom=False, left=True, right=False)

	if save_plot:
		plt.savefig(output_file, format='svg', bbox_inches='tight')
		plt.close()
	else:
		plt.show()



def get_scores(df, normalization, prop):

	if prop == "volume":
		x = normalize(df["tp_volume"].values, df["fp_volume"].values, df["fn_volume"].values, function = normalization)
	elif prop == "length":
		x = normalize(df["tp_length"].values, df["fp_length"].values, df["fn_length"].values, function = normalization)
	elif prop == "count":
		x = normalize(df["tp_count"].values, df["fp_count"].values, df["fn_count"].values, function = normalization)
	elif prop == "distance":
		x = df["distance"].values
	else:
		raise ValueError("Unknown property")

	return x


def correlation_array(categories, datasets, eval_metrics, normalization, prop, save_plot = False, output_folder = "", filename = "array", show_std = False):

	array = np.zeros((len(eval_metrics), len(categories), len(datasets)))
	for c in range(len(categories)): #category in categories:
		# Get all the images in the category
		df = data.loc[data["category"] == categories[c]]
		
		for m in range(len(eval_metrics)):
			
			for d in range(len(datasets)):
				sub_df = df.loc[df["dataset"] == datasets[d]]
				if (categories[c] in ["radius_erosion", "radius_dilation", "deformation"] and prop[m] in ["length", "count"]) or (categories[c] in ["self_merging", "false_terminal", "missing_terminal"] and prop[m] == "count"):
					x = get_scores(sub_df, normalization[m], "volume")
					#x = np.full(x.shape, np.nan)
				else:
					x = get_scores(sub_df, normalization[m], prop[m])

				if len(x)>= 10 and np.isnan(x).sum() < len(x):
					y = sub_df[eval_metrics[m]].values
					# Sort 
					sort = np.argsort(x)
					x = x[sort]
					y = y[sort]

					# Correlation
					rho = Pearson_correlation(x, y)
					if math.isnan(rho):
						rho = 0
					array[m, c, d] = abs(rho)

				else:
					array[m, c, d] = np.nan


	if show_std:
		std = np.nanstd(array, axis = 2) 
	else:
		std = None

	plot_array(np.nanmean(array, axis = 2), legendx =  [tex_names_error[c] for c in categories], legendy = [tex_names[m] for m in eval_metrics], title = "",
		save_plot = save_plot, output_file = output_folder + filename, std = std)

	


def weight_array(categories, datasets, eval_metrics, normalization, prop, save_plot, output_folder = "", filename = "array", show_std = False):

	array = np.zeros((len(eval_metrics), len(categories), len(datasets)))
	for c in range(len(categories)): #category in categories:
		# Get all the images in the category
		df = data.loc[data["category"] == categories[c]]
		
		for m in range(len(eval_metrics)):
			
			for d in range(len(datasets)):
				sub_df = df.loc[df["dataset"] == datasets[d]]
				if (categories[c] in ["radius_erosion", "radius_dilation", "deformation"] and prop[m] in ["length", "count"]) or (categories[c] in ["self_merging", "false_terminal", "missing_terminal"] and prop[m] == "count"):
					x = get_scores(sub_df, normalization[m], "volume")
				else:
					x = get_scores(sub_df, normalization[m], prop[m])

				if len(x)>= 10 and np.isnan(x).sum() < len(x):
					y = sub_df[eval_metrics[m]].values
					if eval_metrics[m] in ["Betti_error_0", "Betti_error_1"]:
						y = np.abs(y)
					# Sort 
					sort = np.argsort(x)
					x = x[sort]
					y = y[sort]

					# Weights
					try:
						coef = np.polyfit(x, y, 1)
						array[m, c, d] = coef[0]
					except:
						array[m, c, d] = np.nan
				else:
					array[m, c, d] = np.nan

	if show_std:
		std = np.nanstd(array, axis = 2) 
	else:
		std = None
	plot_array(np.nanmean(array, axis = 2), legendx = [tex_names_error[c] for c in categories], legendy = [tex_names[m] for m in eval_metrics], title = "",
		save_plot = save_plot, output_file = output_folder + filename, std = std, color = scale_weights_for_visu(np.nanmean(array, axis = 2), axis=0))



def time_table(eval_metrics):

	time = np.zeros((len(eval_metrics),))

	for i in range(len(eval_metrics)):
		
		t = data[eval_metrics[i] + "_time"].values
		time[i] = np.mean(t)


# Latex for plot
tex_names = {"Dice" : r"$Dice$", "cbDice" : r"$cbDice$", "clDice" : r"$clDice$", "ccDice_0" : r"$ccDice$", "ccDice_1": r"$ccDice_1$",
	"Betti_error_0" : r"$\beta^{err}_0$", "Betti_error_1" : r"$\beta^{err}_1$", "Betti_matching_0" : r"$\mu^{err}_0$", "Betti_matching_1" : r"$\mu^{err}_1$"}

tex_names_error = {"radius_erosion" : "rad. erosion", "radius_dilation" : "rad. dilation", "deformation" :  "deformation", "false_component" : "false cc", "missing_component" :  "missing cc",
	"component_merging" : "cc merging", "disconnection" : "disc.", "cycle_disconnection" : "cycle disc.", "hole_merging" : "hole merging", "merging" : "merging", "hole":"hole", "self_merging":"self merging", 
	"branch_merging" : "branch merging", "missing_terminal" : "missing term.", "false_terminal" : "false term.", "false_branch" : "false branch", "missing_branch" : "missing branch", 
	"missing_component_and_false_component" : "missing cc\n   +false cc"}



save_plot = False

os.makedirs("../output/images/", exist_ok = True)
output_folder = "../output/images/"

categories = ["false_component", "missing_component", "missing_component_and_false_component", "component_merging", "disconnection", "cycle_disconnection", "hole_merging", "merging", "hole"]
categories  += ["missing_branch", "false_branch", "branch_merging"] 
categories  += ["radius_erosion", "radius_dilation", "deformation"] 
#categories  += ["self_merging", "false_terminal", "missing_terminal", "radius_erosion", "radius_dilation"] 

data = pd.read_csv("metric_data.csv", sep="\t")


eval_metrics = ["Dice", "cbDice", "clDice", "ccDice_0", "Betti_error_0", "Betti_error_1", "Betti_matching_0", "Betti_matching_1"]
normalization = ["rev_dice", "rev_dice", "rev_dice", "rev_dice", "no_norm", "no_norm", "no_norm", "no_norm"]
prop = ["volume", "length", "length", "count", "count", "count", "count", "count"]


datasets = [ "CREMI", "Roads","NeuroMorpho", "Colon_cells", "LES-AV", "Minivess"]

# Visualize correlation array
correlation_array(categories, datasets, eval_metrics, normalization, prop, save_plot = save_plot, output_folder = output_folder, filename = "correlation_array", show_std = False)

# Visualize weight array
weight_array(categories, datasets, eval_metrics, normalization, prop, save_plot = save_plot, filename = "weight_array", output_folder = output_folder, show_std = False)

# Weight array for one dataset (CREMI)
weight_array(categories, [ "CREMI"], eval_metrics, normalization, prop, save_plot = save_plot, filename = "weight_array_CREMI", output_folder = output_folder, show_std = False)

# Table of the average computing time 
eval_metrics = ["Dice", "cbDice", "clDice", "ccDice_0", "Betti_error", "Betti_matching"]
time_table(eval_metrics)
