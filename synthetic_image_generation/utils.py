####################################################################################################
# Author: Meghane Decroocq
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# provided license.
####################################################################################################

import numpy as np
import networkx as nx
from numpy.linalg import norm 
from math import pi, sin, cos, tan, atan, acos, asin, sqrt, floor, ceil, exp, isnan
from numpy import dot, cross, arctan2
from skimage.measure import label, regionprops, euler_number
from scipy.ndimage import distance_transform_edt
from PIL import Image
import nibabel as nib

import matplotlib.pyplot as plt
import os
import glob
import time
import copy
from random import shuffle, choice

from skimage.morphology import dilation, erosion, area_closing
from skimage.measure import label, regionprops
from scipy.stats import multivariate_normal
from skimage.morphology import skeletonize



def deterioration_score(pred, gt, c, p):

	FN = np.logical_and(gt, np.logical_not(pred))
	FP = np.logical_and(pred, np.logical_not(gt))
	TP = np.logical_and(pred, gt)

	if p == "volume":
		fp = np.sum(FP)
		fn = np.sum(FN)
		tp = np.sum(TP)
		return tp, fp, fn

	elif p == "length":
		ske_gt = skeletonize(gt)
		ske_fp = skeletonize(FP)
		ske_fn = skeletonize(FN)

		if c in ["false_component", "merging", "component_merging", "false_branch", "false_terminal", "self_merging", "hole_merging", "branch_merging"]:
			fp = np.sum(ske_fp)
			fn = 0
			tp = np.sum(ske_gt)
			
		elif c in ["hole"]:
			fp = np.sum(ske_fn)
			fn = np.sum(ske_gt * FN)
			tp = np.sum(ske_gt * TP)

		elif c in ["missing_component", "disconnection", "cycle_disconnection", "missing_branch", "missing_terminal"]:

			fp = 0
			fn = np.sum(ske_gt * FN)
			tp = np.sum(ske_gt * TP)
		else:
			fp, tp, fn = np.nan, np.nan, np.nan
		return tp, fp, fn

	elif p == "count":
		ncc_gt = label(gt, connectivity=1).max()
		ncc_pred = label(pred, connectivity=1).max()
		nh_gt = ncc_gt - euler_number(gt, connectivity=1)
		nh_pred = ncc_pred - euler_number(pred, connectivity=1)

		nb_gt = branch_number(gt)
		nb_pred = branch_number(pred)
	
		if c in ["false_component", "disconnection"]:
			fp = ncc_pred - ncc_gt
			tp = ncc_gt
			fn = 0

		elif c in ["merging", "hole"]:
			fp = nh_pred - nh_gt
			tp = nh_gt
			fn = 0

		elif c in ["missing_component", "component_merging"]:
			fn = ncc_gt - ncc_pred
			tp = ncc_pred
			fp = 0

		elif c in ["cycle_disconnection", "hole_merging"]:
			fn = nh_gt - nh_pred
			tp = nh_pred
			fp = 0

		elif c in ["branch_merging", "missing_branch"]:
			fp = 0
			tp = nb_pred
			fn = nb_gt - nb_pred

		elif c in ["false_branch"]:
			fp = nb_pred - nb_gt
			tp = nb_pred
			fn = 0
		else:
			fp, tp, fn = np.nan, np.nan, np.nan

		return tp, fp, fn

	elif p == "distance":
		# Radius difference

		if c in ["radius_erosion", "radius_dilation"]:
			pass

		elif c in ["deformation"]:
			pass
		return np.nan


def mismatch(img1, img2):
	rgb_visu = np.zeros((img1.shape[0], img2.shape[1], 3), dtype = bool)
	rgb_visu[:,:,0] = img1
	rgb_visu[:,:,1] = np.logical_and(img1, img2)
	rgb_visu[:,:,2] = img2
	return rgb_visu.astype(float)

def Betti_error(pred, gt, connectivity = 1):
	# 1-connectivity = 4 connected pixels
	# 2-connectivity = 8 connected pixels

	# In 2D 
	if len(pred.shape) == 2:

		euler_pred = euler_number(pred, connectivity=connectivity)
		object_pred = label(pred, connectivity=connectivity).max()
		holes_pred = object_pred - euler_pred

		euler_gt = euler_number(gt, connectivity=connectivity)
		object_gt = label(gt, connectivity=connectivity).max()
		holes_gt = object_gt - euler_gt

		cavity_gt = 0
		cavity_pred = 0

	else:

		euler_pred = euler_number(pred, connectivity=connectivity)
		object_pred = label(pred, connectivity=connectivity).max()
		cavity_pred = label(np.invert(pred), connectivity=connectivity).max()
		holes_pred = euler_pred - object_pred + cavity_pred


		euler_gt = euler_number(gt, connectivity=connectivity)
		object_gt = label(gt, connectivity=connectivity).max()
		cavity_gt = label(np.invert(gt), connectivity=connectivity).max()
		holes_gt = euler_gt - object_gt + cavity_gt

	return object_pred - object_gt, holes_pred - holes_gt, cavity_pred - cavity_gt

def branch_number(img):

	G = segmentation_to_graph(img)
	T = full_to_topo(G)
	nb_branch = T.number_of_edges()
	return nb_branch


def terminal_number(img):

	G = segmentation_to_graph(img)
	T = full_to_topo(G)
	nb_term = len([e for e in T.edges(keys= True) if ((T.degree(e[0]) == 1) or (T.degree(e[1]) == 1))])
	return nb_term


def branch_number_error(pred, gt):

	# Create ground-truth graph

	nb_branch_pred= branch_number(pred)
	nb_branch_gt= branch_number(gt)

	return nb_branch_pred - nb_branch_gt



def four_connect(img): 

	# 4-connect the image 
	configurations = [np.array([[0,1],[1,0]]), np.array([[1,0],[0,1]])]

	if len(img.shape) == 2:
		for i in range(img.shape[0]-1):
			for j in range(img.shape[1]-1):
				N = img[i:i+2, j:j+2]
				for C in configurations:
					if np.all(N == C):
						img[i:i+2, j:j+2] = np.ones((2,2))

	elif len(img.shape) == 3:
		#Diagonals
		diagonals = [np.array([[[1,0],[0,0]], [[0,0],[0,1]]]), np.array([[[0,1],[0,0]], [[0,0],[1,0]]]), np.array([[[0,0],[1,0]], [[0,1],[0,0]]]),  np.array([[[0,0],[0,1]], [[1,0],[0,0]]])]

		#Z slices
		for i in range(img.shape[0]-1):
			for j in range(img.shape[1]-1):
				for k in range(img.shape[2]):
					N = img[i:i+2, j:j+2, k]
					for C in configurations:
						if np.all(N == C):
							img[i:i+2, j:j+2, k] = np.ones((2,2))
		# Y slices
		for i in range(img.shape[0]-1):
			for j in range(img.shape[1]):
				for k in range(img.shape[2]-1):
					N = img[i:i+2, j, k:k+2]
					for C in configurations:
						if np.all(N == C):
							img[i:i+2, j, k:k+2] = np.ones((2,2))
		# X slices
		for i in range(img.shape[0]):
			for j in range(img.shape[1]-1):
				for k in range(img.shape[2]-1):
					N = img[i, j:j+2, k:k+2]
					for C in configurations:
						if np.all(N == C):
							img[i, j:j+2, k:k+2] = np.ones((2,2))

		# Diagonals
			for C in diagonals:
				if np.all(N == C):
					img[i:i+2, j:j+2, k:k+2] = np.ones((2,2,2))		

	else:
		print("Image must be 2d or 3d.")
	return img


def rotate_vector(v, axis, theta):

	"""
	Return the rotation matrix associated with counterclockwise rotation about
	the given axis by theta radians.

	Keywords arguments:
	v -- vector to rotate
	axis -- axis of rotation
	theta -- angle
	"""
	
	axis = axis / norm(axis)
	a = cos(theta / 2.0)
	b, c, d = -axis * sin(theta / 2.0)
	aa, bb, cc, dd = a * a, b * b, c * c, d * d
	bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d

	R = np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
					 [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
					 [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


	return dot(R, v)


def full_to_topo(G, add_to_branching = []):

	""" Convert an undirected graph to the corresponding topo graph.
	Keyword arguments :
	G -- networkx undirected Graph
	"""

	# Remove loops if there are
	G.remove_edges_from(nx.selfloop_edges(G))

	if nx.is_directed(G):
		T = nx.MultiDiGraph()
	else:
		T = nx.MultiGraph()

	for n in G.nodes():
		T.add_node(n, coords = G.nodes[n]["coords"])
	for e in G.edges():
		T.add_edge(e[0], e[1])

	nx.set_edge_attributes(T, [], name="nodes")
	nx.set_edge_attributes(T, [], name="edges")
	nx.set_edge_attributes(T, [], name="coords")

	G_nodes_list = [n for n in G.nodes()]
	for n in G_nodes_list:

		# If regular nodes
		regular = False
		if nx.is_directed(G) and G.in_degree(n) == 1 and G.out_degree(n) == 1 and n not in add_to_branching:
			regular = True
		if not nx.is_directed(G) and G.degree(n) == 2 and n not in add_to_branching:
			regular = True

		if regular:
			if nx.is_directed(G):
				nb_list = [nb for nb in T.predecessors(n)] + [nb for nb in T.successors(n)]
				edg = [e for e in T.in_edges(n, keys=True)] + [e for e in T.out_edges(n, keys=True)] 
			else:
				nb_list = [nb for nb in T.neighbors(n)]
				edg = [e for e in T.edges(n, keys=True)] 
				
			if len(nb_list) == 2:

				edges = T.edges[edg[0]]["edges"] + T.edges[edg[1]]["edges"]
				if len(T.edges[edg[0]]["edges"]) == 0:
					edges = [edg[0]] + edges
				if len(T.edges[edg[1]]["edges"]) == 0:
					edges += [edg[1]]

				nodes = T.edges[edg[0]]["nodes"] + [n] + T.edges[edg[1]]["nodes"]
				coords = T.edges[edg[0]]["coords"] + [G.nodes[n]['coords']] + T.edges[edg[1]]["coords"]
						
				# Create new edge by merging the 2 edges of regular point
				T.add_edge(nb_list[0], nb_list[1], coords = coords, edges = edges, nodes = nodes)
				# Remove regular point
				T.remove_node(n)

	# Reorder nodes along the branch
	for e in T.edges(keys=True):

		nds_list = T.edges[e]['nodes']
		edg_list = T.edges[e]['edges']
		coords = T.edges[e]['coords']
			
		ordered_nds = [e[0]]
		ordered_edg = []
		order_coords = []
		if len(nds_list) == 0:
			ordered_edg = [e]
			ordered_nds = [e[0], e[1]]
		else:

			for i in range(len(nds_list) + 1):
				for edg in edg_list:
					if edg[0] == ordered_nds[-1] and edg[1] not in ordered_nds:
						nnd = edg[1]
						ordered_edg.append((edg[0], edg[1]))
					elif edg[1] == ordered_nds[-1] and edg[0] not in ordered_nds:
						nnd = edg[0]
						ordered_edg.append((edg[1], edg[0]))
				if nnd in nds_list:
					order_coords.append(nds_list.index(nnd))
				ordered_nds.append(nnd)

		T.edges[e]['nodes'] = ordered_nds[1:-1]
		T.edges[e]['edges'] = ordered_edg
		T.edges[e]['coords'] = [coords[i] for i in order_coords]

	return T


def topo_to_full(G, transfer_edg_att = []):

	""" Convert a topo graph back to an undirected graph.
	Keyword arguments :
	G -- networkx topo graph 
	"""

	G_nds_list = [n for n in G.nodes()]

	if nx.is_directed(G):
		H = nx.DiGraph()
	else:
		H = nx.Graph()

	for att in transfer_edg_att:
		nx.set_edge_attributes(H, 0, name=att)

	for n in G.nodes():
		H.add_node(n, coords = G.nodes[n]["coords"])
		
	for e in G.edges(keys=True):
		nds_list = G.edges[e]['nodes']
		edg_list = G.edges[e]['edges']
		coords = G.edges[e]['coords']

		for i in range(len(nds_list)):
			H.add_node(nds_list[i], coords = coords[i])

		for edg in edg_list:
			H.add_edge(edg[0], edg[1])

			for att in transfer_edg_att:
				H.edges[(edg[0], edg[1])][att] = G.edges[e][att]

	return H


def segmentation_to_graph_org(img): 

	""" Create a graph based on the skeleton of a binary image 
	Keyword arguments:
	img -- binary image as a boolean numpy matrix """

	# Extract radius
	if len(img.shape) == 2:
		rad_map = distance_transform_edt(img, sampling = [1,1]) - 0.2
	else:
		rad_map = distance_transform_edt(img, sampling = [1,1,1]) - 0.2

	# Skeletonize 
	ske = skeletonize(img.astype(bool), method = "lee").astype(bool)

	G = nx.Graph()

	# Mark bifurcation
	branch_map = np.zeros(img.shape)

	# 2D case
	if len(img.shape) == 2:
		for i in range(img.shape[0]-1):
			for j in range(img.shape[1]-1):
				N = ske[i:i+2, j:j+2]
				if np.sum(N) > 2:
					for c in np.argwhere(N):
						branch_map[i + c[0], j + c[1]] = 1
	# 3D case
	else:
		for i in range(img.shape[0]-1):
			for j in range(img.shape[1]-1):
				for k in range(img.shape[2]-1):
					N = ske[i:i+2, j:j+2, k:k+2]
					if np.sum(N) > 2:
						for c in np.argwhere(N):
							branch_map[i + c[0], j + c[1], k + c[2]] = 1

	branch_map = label(branch_map, connectivity = 1)
	
	bg_id = branch_map[tuple(np.argwhere(ske == 0)[0,:])]

	# Add branching nodes
	for i in range(branch_map.max() + 1):
		if i != bg_id:
			coords = np.argwhere(branch_map == i)
			full_coords = np.array([0,0,0,0])
			full_coords[:coords.shape[1]] = np.median(coords, axis = 0) #TO DO : Replace by something else
			full_coords[-1] = rad_map[tuple(full_coords[:coords.shape[1]])]
			G.add_node(i, coords = full_coords)

	# Add non branching node
	node_coords = np.argwhere(ske)
	start_id = branch_map.max() + 1

	for coords in node_coords:
		if branch_map[tuple(coords)] == bg_id:
			branch_map[tuple(coords)] = start_id

			full_coords = np.array([0,0,0,0])
			full_coords[:node_coords.shape[1]] = coords
			full_coords[-1] = rad_map[tuple(full_coords[:node_coords.shape[1]])]

			G.add_node(start_id, coords = full_coords)

		start_id += 1

	list_of_nodes = [n for n in G.nodes()]

	# Add all edges
	for coords in node_coords:
		n1 = branch_map[tuple(coords)]

		# Get 8-neighborhood (or 26 in 3D)
		if len(img.shape) == 2:
			N = branch_map[max([0, coords[0] - 1]):min([img.shape[0] - 1, coords[0] + 2]), max([0, coords[1] - 1]):min([img.shape[1] - 1, coords[1] + 2])]
		else:
			N = branch_map[max([0, coords[0] - 1]):min([img.shape[0] - 1, coords[0] + 2]), max([0, coords[1] - 1]):min([img.shape[1] - 1, coords[1] + 2]), max([0, coords[2] - 1]):min([img.shape[2] - 1, coords[2] + 2])]

		# Get index of other neighbor nodes 
		index = np.unique(N.flatten()).tolist()

		# Add edges 
		for n2 in index:
			if n2 != bg_id and n2 != n1:
				G.add_edge(n1, n2)

	return G



def segmentation_to_graph(img):

	""" Create a graph based on the skeleton of a binary image 
	Keyword arguments:
	img -- binary image as a boolean numpy matrix """

	# Extract radius
	if len(img.shape) == 2:
		rad_map = distance_transform_edt(img, sampling = [1,1]) * 0.8
	else:
		rad_map = distance_transform_edt(img, sampling = [1,1,1]) *0.8

	# Skeletonize 
	ske = skeletonize(img.astype(bool), method = "lee").astype(bool)

	# Graph creation
	idx_img = np.zeros(ske.shape) - 1
	G = nx.Graph()
	node_coords = np.argwhere(ske)

	for i in range(node_coords.shape[0]):
		r = rad_map[node_coords[i,0], node_coords[i,1]]# Correction for voxelization

		G.add_node(i, coords = np.array([node_coords[i,0], node_coords[i,1], 0, r]))
		idx_img[node_coords[i,0], node_coords[i,1]] = i

	for i in range(node_coords.shape[0]):
		for r in [-1,0,1]:
			for c in [-1,0,1]:
				x, y = node_coords[i,0] + r, node_coords[i,1] + c
				if x > 0 and y > 0 and x<idx_img.shape[0] and y < idx_img.shape[1] and idx_img[x,y] != -1 and not (r == 0 and c==0):
					if r != 0 and c !=0: # Diagonal positive 
						if not ske[node_coords[i,0], y] and not ske[x, node_coords[i,1]]:
							G.add_edge(i, int(idx_img[node_coords[i,0] + r, node_coords[i,1] + c]))
					else:
						G.add_edge(i, int(idx_img[node_coords[i,0] + r, node_coords[i,1] + c]))

	return G



def prune_small_branches(G, thres = 0.5):

	T = full_to_topo(G)
	for e in T.edges(keys=True):
		if (T.degree(e[0]) == 1 and T.degree(e[1]) > 1) or (T.degree(e[1]) == 1 and T.degree(e[0]) > 1):

			end = e[0]
			if T.degree(e[1]) == 1:
				end = e[1] 

			# Get length 
			polyline = np.vstack([T.nodes[e[0]]["coords"]] + T.edges[e]["coords"] + [T.nodes[e[1]]["coords"]])
			if length_polyline(polyline[:,:3])[-1] < (thres + 1) * T.nodes[end]["coords"][-1]: 
				# If too small, remove branch
				G.remove_node(end)
				for n in T.edges[e]["nodes"]:
					G.remove_node(n)
	return G


def merge_branching_graph(G):

	""" Merge close branching points """
	T = full_to_topo(G)

	collapse = nx.Graph()
	for e in T.edges(keys=True):
		if T.degree(e[0]) > 2 and T.degree(e[1]) > 2:
			polyline = np.vstack([T.nodes[e[0]]["coords"]] + T.edges[e]["coords"] + [T.nodes[e[1]]["coords"]])

			if length_polyline(polyline[:,:3])[-1] < T.nodes[e[0]]["coords"][3] + T.nodes[e[1]]["coords"][3]:
				for edg in T.edges[e]["edges"]:
					collapse.add_edge(edg[0], edg[1])
			
	max_id = max([n for n in G.nodes]) + 1
	for group in nx.connected_components(collapse):

		# Collapse the branch in G
		G.add_node(max_id, coords = np.mean(np.vstack([G.nodes[n]["coords"] for n in group]), axis = 0))
		for n in group:
			for neigh in G.neighbors(n):
				if neigh not in group:
					G.add_edge(neigh, max_id)
		for n in group:
			G.remove_node(n)
		max_id += 1

	return G


def length_polyline(coords):

	""" Return the length of a polyLine

	Keyword arguments:
	coords -- list of node coordinates
	"""
	if coords.shape[0] == 0:
		length = [0.0]
	else:
		length = np.zeros((coords.shape[0],))
		length[0] = 0.0

	for i in range(1, len(coords)):
		length[i] = length[i-1] + norm(coords[i] - coords[i-1])

	return length


def write_image(G, shape):

	if len(shape) == 2:
		mode = "2d"
	elif len(shape) == 3:
		mode = "3d"
	else:
		print("Wrong shape")

	# Create mask of positives and negatives
	img = np.zeros(shape, dtype = bool)
	for n in G.nodes():
		center = G.nodes[n]["coords"][:3]
		rad = max([1, G.nodes[n]["coords"][3]])

		if mode == "2d":
			img = write_circle(img, center, rad)
		elif mode == "3d":
			img = write_sphere(img, center, rad)

	for e in G.edges():
		
		c1 = G.nodes[e[0]]["coords"]
		c2 = G.nodes[e[1]]["coords"]
				
		if norm(c2[:3] - c1[:3]) >= 1:
					
			# Interpolate edge and write nodes
			new_nodes = np.hstack([np.linspace(c1[i], c2[i], int(norm(c2[:2] - c1[:2])) + 2)[1:-1].reshape(-1, 1) for i in range(4)])
			for i in range(new_nodes.shape[0]):
				rad =  max([1,new_nodes[i, 3]])
				center = new_nodes[i, :3]

				if mode == "2d":
					img = write_circle(img, center, rad)
				elif mode == "3d":
					img = write_sphere(img, center, rad)
	return img



def pad_image(img):

	padded_img = np.zeros(np.array(img.shape) + 2).astype(bool)
	if len(img.shape) == 2:
		padded_img[1:-1, 1:-1] = img
	elif len(img.shape) == 3:
		padded_img[1:-1, 1:-1, 1:-1] = img
	else:
		raise ValueError("Img must have shape (x, x, n) where n = 2 or 3.")

	return padded_img
	



def write_circle(img, center, rad, val = 1):
	if int(center[0]) >= 0 and int(center[1]) >= 0 and int(center[0]) < img.shape[0] and int(center[1]) < img.shape[1]:
		img[int(center[0]), int(center[1])] = val

	for x in range(int(center[0]) - int(rad), int(center[0]) + int(rad) + 1):
		for y in range(int(center[1]) - int(rad), int(center[1]) + int(rad) + 1):
			if norm(np.array([x,y]) - center[:2]) <= rad and x >= 0 and y >= 0 and x < img.shape[0] and y < img.shape[1]:
				img[x, y] = val
	return img


def write_sphere(img, center, rad, val = 1):

	img[int(center[0]), int(center[1]), int(center[2])] = val

	for x in range(int(center[0]) - int(rad), int(center[0]) + int(rad) + 1):
		for y in range(int(center[1]) - int(rad), int(center[1]) + int(rad) + 1):
			for z in range(int(center[2]) - int(rad), int(center[2]) + int(rad) + 1):
				if norm(np.array([x,y,z]) - center[:3]) <= rad and x >= 0 and y >= 0 and z >=0 and x < img.shape[0] and y < img.shape[1] and z < img.shape[2]:
					img[x, y, z] = val
	return img


def save_image(filename, img):

	if len(img.shape) == 2:
		im = Image.fromarray((img*255).astype(np.uint8)).convert("L")
		im.save(filename + ".png")
	else:
		# Save as nifti 
		affine = np.eye(4)
		nifti_file = nib.Nifti1Image(img.astype(float), affine)
		nib.save(nifti_file, filename + ".nii")

