import numpy as np
from BettiMatching.BettiMatching import *
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops, euler_number
from scipy import ndimage


"""
Here we provide the code to compute the quality metrics included in the benchmark. 
The code is based on the original implementation provided by the authors of the metrics. 
We have modified some of the code for our purpose of our study, according to the provided license.
"""

# cbDice : Code copied and modified from https://github.com/PengchengShi1220/cbDice. The original code is licensed under the Apache License 2.0.
def combine_arrays_test(A, B, C):
	A_C = A * C
	B_C = B * C
	D = B_C.copy()
	mask_AC = (A != 0) & (B == 0)
	D[mask_AC] = A_C[mask_AC]
	return D

def get_weights_2d(mask, skel):
	# For CPU:
	dist_map = ndimage.distance_transform_edt(mask)

	dist_map[mask == 0] = 0
	skel_radius = np.zeros_like(skel, dtype=np.float32)
	skel_radius[skel == 1] = dist_map[skel == 1]

	if skel_radius.max() == 0 or skel_radius.min() == skel_radius.max():
		return mask, skel.clone(), skel.clone(), skel.clone(), skel.clone(), skel.clone()

	smooth = 1e-7
	skel_radius_max = skel_radius.max()
	dist_map[dist_map > skel_radius_max] = skel_radius_max

	skel_R = skel_radius
	skel_R_norm = skel_radius / skel_radius_max

	skel_1_R = np.zeros_like(skel, dtype=np.float32)
	skel_1_R_norm = np.zeros_like(skel, dtype=np.float32)
	skel_1_R[skel == 1] = (1 + smooth) / (skel_R[skel == 1] + smooth)
	skel_1_R_norm[skel == 1] = (1 + smooth) / (skel_R_norm[skel == 1] + smooth)
	dist_map_norm = dist_map / skel_radius_max

	return dist_map, dist_map_norm, skel_R, skel_R_norm, skel_1_R, skel_1_R_norm


def cbDice(vp, vl, method = "srimb_norm"):
	smooth = 1e-3
	if len(vp.shape)==2:
		sp = skeletonize(vp)
		sl = skeletonize(vl)
		vl_dist_map, vl_dist_map_norm, sl_R, sl_R_norm, sl_1_R, sl_1_R_norm = get_weights_2d(vl, sl)
		vp_dist_map, vp_dist_map_norm, sp_R, sp_R_norm, sp_1_R, sp_1_R_norm = get_weights_2d(vp, sp)

		if method == "sr":
			q_sl = sl_R
			q_sp = sp_R
			q_vl = vl
			q_vp = vp
			q_slvl = sl
			q_spvp = sp
	   
		elif method == "mb":
			q_sl = sl
			q_sp = sp
			q_vl = vl_dist_map
			q_vp = vp_dist_map
			q_slvl = sl_R
			q_spvp = sp_R

		elif method == "srmb":
			q_sl = sl_R
			q_sp = sp_R
			q_vl = vl_dist_map
			q_vp = vp_dist_map
			q_slvl = sl_R
			q_spvp = sp_R

		elif method == "srimb":
			q_sl = sl_1_R
			q_sp = sp_1_R
			q_vl = vl_dist_map
			q_vp = vp_dist_map
			q_slvl = sl_R
			q_spvp = sp_R

		elif method == "sr_norm":
			q_sl = sl_R_norm
			q_sp = sp_R_norm
			q_vl = vl
			q_vp = vp
			q_slvl = sl
			q_spvp = sp

		elif method == "mb_norm":
			q_sl = sl
			q_sp = sp
			q_vl = vl_dist_map_norm
			q_vp = vp_dist_map_norm
			q_slvl = sl_R_norm
			q_spvp = sp_R_norm

		elif method == "srmb_norm":
			q_sl = sl_R_norm
			q_sp = sp_R_norm
			q_vl = vl_dist_map_norm
			q_vp = vp_dist_map_norm
			q_slvl = sl_R_norm
			q_spvp = sp_R_norm

		elif method == "srimb_norm":
			q_sl = sl_1_R_norm
			q_sp = sp_1_R_norm
			q_vl = vl_dist_map_norm
			q_vp = vp_dist_map_norm
			q_slvl = sl_R_norm
			q_spvp = sp_R_norm
		else:
			print("not implemented")

		weighted_tprec = (np.sum(q_sp*q_vl)+smooth)/(np.sum(combine_arrays_test(q_spvp, q_slvl, q_sp))+smooth)
		weighted_tsens = (np.sum(q_sl*q_vp)+smooth)/(np.sum(combine_arrays_test(q_slvl, q_spvp, q_sl))+smooth) 
		
	return 2.0 * (weighted_tprec * weighted_tsens) / (weighted_tprec + weighted_tsens)

# Dice 
def Dice(prediction, ground_truth):
	dice = np.sum(prediction[ground_truth==1])*2.0 / (np.sum(prediction) + np.sum(ground_truth))
	return dice

# Betti matching : Code copied and modified from https://github.com/nstucki/Betti-matching. The code is licensed under the MIT License. Copyright (c) 2024 nstucki
def Betti_matching(prediction, ground_truth, relative=False, comparison='union', filtration='superlevel', construction='V'):
	BM = BettiMatching(prediction, ground_truth, relative=relative, comparison=comparison, filtration=filtration, construction=construction)
	return BM.loss(dimensions=[0]), BM.loss(dimensions=[1]) #BM.loss(dimensions=[0,1]),BM.Betti_number_error(threshold=0.5, dimensions=[0,1]), BM.Betti_number_error(threshold=0.5, dimensions=[0]), BM.Betti_number_error(threshold=0.5, dimensions=[1])

def Betti_error(pred, gt, connectivity = 1):

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

	return abs(object_pred - object_gt), abs(holes_pred - holes_gt), abs(cavity_pred - cavity_gt)

# clDice : Code copied and modified from https://github.com/jocpae/clDice. The code is licensed under the MIT License. Copyright (c) 2021 Johannes C. Paetzold and Suprosanna Shit 
def cl_score(v, s):
	"""[this function computes the skeleton volume overlap]
	Args:
		v ([bool]): [image]
		s ([bool]): [skeleton]
	Returns:
		[float]: [computed skeleton volume intersection]
	"""
	return np.sum(v*s)/np.sum(s)


def clDice(v_p, v_l):
	"""[this function computes the cldice metric]
	Args:
		v_p ([bool]): [predicted image]
		v_l ([bool]): [ground truth image]
	Returns:
		[float]: [cldice metric]
	"""

	if len(v_p.shape)==2:
		tprec = cl_score(v_p,skeletonize(v_l))
		tsens = cl_score(v_l,skeletonize(v_p))
	elif len(v_p.shape)==3:
		tprec = cl_score(v_p,skeletonize_3d(v_l))
		tsens = cl_score(v_l,skeletonize_3d(v_p))
	return 2*tprec*tsens/(tprec+tsens)


# ccDice : Code copied and modified from https://github.com/PierreRouge/ccDice.
def S(y1, y2):
	return np.sum(y1 * y2) / np.sum(y1)

def ccDice(y_pred, y_true, alpha=0.5):
	
	y_pred_label, cc_pred = label(y_pred, return_num=True)
	y_true_label, cc_true = label(y_true, return_num=True)
	
	y_true_label[y_true_label != 0] = y_true_label[y_true_label != 0] + cc_pred

	list_s = []
	indices_cc = []
	for a in range(1, cc_pred + 1):
		for b in range(cc_pred + 1, cc_pred + cc_true + 1):
			
			y1 = np.zeros(y_pred_label.shape)
			y1[y_pred_label == a] = 1
			
			y2 = np.zeros(y_true_label.shape)
			y2[y_true_label == b] = 1
			
			s_ab = S(y1, y2)
			s_ba = S(y2, y1)
			
			list_s.append(s_ab)
			list_s.append(s_ba)
			
			indices_cc.append((a, b))
			indices_cc.append((b, a))
		
	if alpha <= 0.5:
		# Sort the list
		list_s = np.array(list_s)
		indices = np.argsort(-list_s)
		indices_cc = np.array(indices_cc)
		
		list_s = np.array(list_s)
		list_s = list_s[indices]
		indices_cc = indices_cc[indices]
	
	if len(list_s) > 0:
		left_list = []
		right_list = []
		tp = 0
		i = 0
		s = list_s[0]
	
		coor = indices_cc[0]
		while s >= alpha and i < len(list_s):
			
			if (coor[0] not in left_list) and (coor[1] not in right_list):
			
				left_list.append(coor[0])
				right_list.append(coor[1])
				tp += 1
				
			i += 1
			if i < len(list_s):
				s = list_s[i]
				coor = indices_cc[i]
			  
		ccdice = tp / (cc_pred + cc_true)
		
		return ccdice
	else:
		return np.nan



def ccDice_1(y_pred, y_true, alpha=0.5):

	# Pad images
	y_pred = np.pad(y_pred, 1, mode = "constant")
	y_true = np.pad(y_true, 1, mode = "constant")

	# Invert image
	y_pred = np.invert(y_pred)
	y_true = np.invert(y_true)

	# Remove background (?)
	lbl_pred = label(y_pred)
	y_pred[lbl_pred == 1] = 0

	lbl_true = label(y_true)
	y_true[lbl_true == 1] = 0

	return ccDice(y_pred, y_true, alpha = 0.5)

