####################################################################################################
# Author: Meghane Decroocq
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# provided license.
####################################################################################################

from utils import *

class ErrorGenerator:

	def __init__(self, img, error_types = []):

		if len(error_types) == 0:
			self.categories = ["false_component", "missing_component", "component_merging", "disconnection", "cycle_disconnection", "hole_merging", "merging", "hole", "missing_branch","false_branch", "branch_merging",
			"self_merging", "missing_terminal", "false_terminal", "radius_dilation", "radius_erosion", "deformation"] 
		else:
			self.categories = error_types

		if len(img.shape) == 2:
			self.mode = "2d"
		elif len(img.shape) == 3:
			self.mode = "3d"

		#img = four_connect(img)

		# Create ground-truth graph
		G = segmentation_to_graph(img)
		G = merge_branching_graph(G)
		G = prune_small_branches(G)
		T = full_to_topo(G)

		self.Ggt = G
		self.Gpred = copy.deepcopy(G)


		# Write new (tubular) image from graph 
		self.img = write_image(G, img.shape)
		self.ske = skeletonize(img)
		self.pred = np.copy(self.img)
		self.decomp = np.repeat(np.copy(self.img)[:, :, np.newaxis], len(self.categories), axis=2)

		# Initialize scores
		self.scores = {}

		# Compute initial metric values
		for c in self.categories:
			self.scores[c] = {}
			self.update_scores(c)

		self.terminal_list = []

	def get_label(self):
		return self.img

	def get_pred(self):
		return self.pred

	def get_scores(self):
		return self.scores

	def update_scores(self, category):
		pred = self.decomp[:,:,self.categories.index(category)]
		self.scores[category]["tp_volume"], self.scores[category]["fp_volume"], self.scores[category]["fn_volume"] = deterioration_score(pred, self.img, category, "volume")
		self.scores[category]["tp_length"], self.scores[category]["fp_length"], self.scores[category]["fn_length"] = deterioration_score(pred, self.img, category, "length")
		self.scores[category]["tp_count"], self.scores[category]["fp_count"], self.scores[category]["fn_count"] = deterioration_score(pred, self.img, category, "count")
		self.scores[category]["distance"] = deterioration_score(pred, self.img, category, "distance")

	def combine_scores(self, categories):

		combined_scores = {}
		# Check if the scores can be combined (= Same tp + score is defined)
		gt_tp, gt_fp, gt_fn = deterioration_score(self.img, self.img, categories[0], "volume")
		combined_scores["fp_volume"] = sum([self.scores[c]["fp_volume"] for c in categories])
		combined_scores["fn_volume"] = sum([self.scores[c]["fn_volume"] for c in categories])
		combined_scores["tp_volume"] = gt_tp - combined_scores["fn_volume"]
		

		gt_tp, gt_fp, gt_fn = deterioration_score(self.img, self.img, categories[0], "length")
		combined_scores["fp_length"] = sum([self.scores[c]["fp_length"] for c in categories])
		combined_scores["fn_length"] = sum([self.scores[c]["fn_length"] for c in categories])
		combined_scores["tp_length"] = gt_tp - combined_scores["fn_length"]

		gt_tp, gt_fp, gt_fn = deterioration_score(self.img, self.img, categories[0], "count")
		combined_scores["fp_count"] = sum([self.scores[c]["fp_count"] for c in categories])
		combined_scores["fn_count"] = sum([self.scores[c]["fn_count"] for c in categories])
		combined_scores["tp_count"] = gt_tp - combined_scores["fn_count"]

		combined_scores["distance"] = np.nan
		return combined_scores

	def combine_errors(self):
		for k in range(self.decomp.shape[2]):
			FN = np.logical_and(self.img, np.logical_not(self.decomp[:,:,k]))
			FP = np.logical_and(self.decomp[:,:,k], np.logical_not(self.img))
			self.pred[FN] = 0
			self.pred[FP] = 1

	def add_to_decomposition(self, category, new_pred):
		FN = np.logical_and(self.pred, np.logical_not(new_pred))
		FP = np.logical_and(new_pred, np.logical_not(self.pred))
		self.decomp[:,:,self.categories.index(category)][FN] = 0
		self.decomp[:,:,self.categories.index(category)][FP] = 1


	def generate_error(self, category):
		if category == "missing_component":
			fail, pred = self.missing_component()
		elif category == "disconnection":
			fail, pred = self.disconnection()
		elif category == "cycle_disconnection":
			fail, pred = self.cycle_disconnection()
		elif category == "hole":
			fail, pred = self.hole()
		elif category == "false_component":
			fail, pred = self.false_component()
		elif category == "merging":
			fail, pred = self.merging()
		elif category == "hole_merging":
			fail, pred = self.hole_merging()
		elif category == "component_merging":
			fail, pred = self.component_merging()
		elif category == "missing_branch":
			fail, pred = self.missing_branch()
		elif category == "false_branch":
			fail, pred = self.false_branch()
		elif category == "missing_terminal":
			fail, pred = self.missing_terminal()
		elif category == "false_terminal":
			fail, pred = self.false_terminal()
		elif category == "self_merging":
			fail, pred = self.self_merging()
		elif category == "branch_merging":
			fail, pred = self.branch_merging()
		elif category == "radius_erosion":
			fail, pred = self.radius_erosion()
		elif category == "radius_dilation":
			fail, pred = self.radius_dilation()
		elif category == "deformation":
			fail, pred = self.deformation()

		if not fail:
			
			self.add_to_decomposition(category, pred)
			self.combine_errors()
			#self.decomp[:,:,self.categories.index(category)] = pred
			#self.pred = pred 
			self.update_scores(category)
			
		return fail


	# TOPOLOGY
	def missing_component(self):
		# Remove a connected component

		fail = True
		pred = np.copy(self.pred)
		
		lbl = label(np.logical_and(self.pred, self.img), connectivity = 2)
		nb_cc = lbl.max()
		bg_id = lbl[tuple(np.argwhere(self.img == 0)[0,:])]

		cc_list = [i for i in range(lbl.max() + 1) if i!=bg_id]
		shuffle(cc_list)

		if nb_cc > 1:
			for i in range(len(cc_list)):
			
				pred = np.copy(self.pred)
				pred[lbl == cc_list[i]] = 0

				b0, b1, b2 = Betti_error(pred, self.pred, connectivity = 2)
				if b0 == -1 and b1 == 0 and b2 == 0:
					fail = False
					# Remove node from graph
					for nd in [n for n in self.Gpred.nodes()]:
						if not self.pred[tuple(self.Gpred.nodes[nd]["coords"][:len(self.img.shape)].astype(int))]:
							self.Gpred.remove_node(nd)
					break

		return fail, pred


	def disconnection(self):
		# Disconnect branch

		fail = True
		pred = np.copy(self.pred)
		T = full_to_topo(self.Gpred)
		edge_list = [e for e in T.edges(keys=True) if len(T.edges[e]["nodes"]) > 0]
		shuffle(edge_list)

		for e in edge_list:
			nds = T.edges[e]["nodes"]
			max_rad = max([self.Gpred.nodes[n]["coords"][-1] for n in nds])
			min_size = max([1, int(max_rad*2) + 2])
			max_size = max([min_size + 1, len(nds) // 2])

			if len(nds) > min_size:

				size = np.random.choice(np.arange(min_size, max_size))
				start = np.random.choice(np.arange(0, len(nds) - size))
				to_remove = nds[start:start + size]

				Gtmp = copy.deepcopy(self.Gpred)		
				for n in to_remove:
					Gtmp.remove_node(n)

				pred = write_image(Gtmp, self.img.shape)

				b0, b1, b2 = Betti_error(pred, self.pred, connectivity = 1)
				b0_2, b1_2, b2_2 = Betti_error(pred, self.pred, connectivity = 2)

				if b0 == 1 and b1 == 0 and b2 == 0 and b0_2 == 1 and b1_2 == 0 and b2_2 == 0:
					for n in to_remove:
						self.Gpred.remove_node(n)

					fail = False
					break

		return fail, pred


	def cycle_disconnection(self):
		# Disconnect cycle

		fail = True
		pred = np.copy(self.pred)
		T = full_to_topo(self.Gpred)
		edge_list = [e for e in T.edges(keys=True) if T.degree(e[0]) > 1 and T.degree(e[1]) > 1 and len(T.edges[e]["nodes"]) > 0]
		shuffle(edge_list)

		for e in edge_list:
			nds = T.edges[e]["nodes"]
			max_rad = max([self.Gpred.nodes[n]["coords"][-1] for n in nds])
			min_size = max([1, int(max_rad*2) + 2])

			if len(nds) > min_size:

				size = np.random.choice(np.arange(min_size, len(nds)))
				start = np.random.choice(np.arange(0, len(nds) - size))
				
				to_remove = nds[start:start + size]

				Gtmp = copy.deepcopy(self.Gpred)		
				for n in to_remove:
					Gtmp.remove_node(n)

				pred = write_image(Gtmp, self.img.shape)

				b0, b1, b2 = Betti_error(pred, self.pred, connectivity = 1)
				b0_2, b1_2, b2_2 = Betti_error(pred, self.pred, connectivity = 2)
				if b0 == 0 and b1 == -1 and b2 == 0 and b0_2 == 0 and b1_2 == -1 and b2_2 == 0:
					
					for n in to_remove:
						self.Gpred.remove_node(n)

					fail = False
					break

		return fail, pred


	def hole(self):
		# Add a hole in 2d images, add a cavity in 3d images

		pred = np.copy(self.pred)
		edt = distance_transform_edt(np.logical_and(self.img, self.pred))
		min_size = 1
		margin = 2

		# Get a seed
		seeds = np.argwhere(edt > min_size + margin)
		np.random.shuffle(seeds)

		fail = True
		for seed in seeds:

			# Choose radius
			rad = np.random.uniform(min_size, edt[tuple(seed)] - margin)

			direction = np.random.uniform(-1, 1, 3)
			direction = direction / norm(direction)

			# Create object
			pred = np.copy(self.pred)

			if self.mode == "2d":
				pred = write_circle(pred, seed, rad, val = 0)
				correct_betti = [0, +1, 0]
			else:
				pred = write_sphere(pred, seed, rad, val = 0)
				correct_betti = [0, 0, +1]

			b0, b1, b2 = Betti_error(pred, self.pred, connectivity = 1)
			b0_2, b1_2, b2_2 = Betti_error(pred, self.pred, connectivity = 2)

			if b0 == correct_betti[0] and b1 == correct_betti[1] and b2 == correct_betti[2] and b0_2 == correct_betti[0] and b1_2 == correct_betti[1] and b2_2 == correct_betti[2]:
				pred2 = np.copy(pred)
				fail = False
				b0, b1, b2, b0_2, b1_2, b2_2 = 0, 0, 0, 0, 0, 0
				while b0 == 0 and b1 == 0 and b2 == 0 and b0_2 == 0 and b1_2 == 0 and b2_2 == 0:
					pred = np.copy(pred2)

					# Next coordinate
					seed = seed + direction[:len(self.img.shape)]
					angle = np.random.normal(loc=0.0, scale=0.1)

					if self.mode == "2d":
						axis = np.array([0, 0, 1])
					else:
						axis = np.random.uniform(-1, 1, 3)
						axis = axis / norm(axis)
				
					direction = rotate_vector(direction, axis = axis, theta=angle)

					if edt[tuple(seed.astype(int))] - margin * 2 > min_size:
						# Next rad
						rad = np.random.uniform(max([min_size, rad - 1]), min([edt[tuple(seed.astype(int))] - margin, rad + 1]))

						if self.mode == "2d":
							pred2 = write_circle(pred2, seed, rad, val = 0)
						else:
							pred2 = write_sphere(pred2, seed, rad, val = 0)

						b0, b1, b2 = Betti_error(pred2, pred, connectivity = 1)
						b0_2, b1_2, b2_2 = Betti_error(pred2, pred, connectivity = 2)

					else:
						b0 = 1 # Stop the hole

				break

		return fail, pred


	def false_component(self):
		# Add a false component

		pred = np.copy(self.pred)
		edt = distance_transform_edt(np.invert(np.logical_or(self.img, self.pred)))
		min_size = 1
		max_size = max([self.Ggt.nodes[n]["coords"][-1] for n in self.Ggt.nodes()])
		margin = 2

		T = full_to_topo(self.Gpred)
		size_branch = np.array([len(T.edges[e]["nodes"]) for e in T.edges(keys=True)])

		# Get a seed
		seeds = np.argwhere(edt > min_size + margin * 2)
		np.random.shuffle(seeds)

		fail = True
		for seed in seeds:

			# Choose radius
			rad = np.random.uniform(min_size, min([max_size, edt[tuple(seed)] - margin * 2]))

			direction = np.random.uniform(-1, 1, 3)
			direction = direction / norm(direction)

			# Create object
			pred = np.copy(self.pred)
			if self.mode == "2d":
				pred = write_circle(pred, seed, rad)
			else:
				pred = write_sphere(pred, seed, rad)

			b0, b1, b2 = Betti_error(pred, self.pred, connectivity = 1)
			b0_2, b1_2, b2_2 = Betti_error(pred, self.pred, connectivity = 2)

			if b0 == 1 and b1 == 0 and b2 == 0 and b0_2 == 1 and b1_2 == 0 and b2_2 == 0:
				pred2 = np.copy(pred)
				length = 1
				max_length = np.random.normal(size_branch.mean(), size_branch.mean() / 4)
				fail = False

				while b0 == 1 and b1 == 0 and b2 == 0 and b0_2 == 1 and b1_2 == 0 and b2_2 == 0:
					pred = np.copy(pred2)

					# Next coordinate
					seed = seed + direction[:len(self.img.shape)]
					angle = np.random.normal(loc=0.0, scale=0.1)

					if self.mode == "2d":
						axis = np.array([0, 0, 1])
					else:
						axis = np.random.uniform(-1, 1, 3)
						axis = axis / norm(axis)
				
					direction = rotate_vector(direction, axis = axis, theta=angle)
					if np.all(seed.astype(int) >= np.zeros(len(self.img.shape))) and np.all(seed.astype(int) < np.array(self.img.shape)):
						if edt[tuple(seed.astype(int))] - margin * 2 > min_size and length < max_length:
							# Next rad
							diff_range = np.random.normal(loc=0.0, scale=0.1)
							rad = np.random.uniform(max([min_size, rad - diff_range]), min([edt[tuple(seed.astype(int))] - margin * 2, rad + diff_range]))

							if self.mode == "2d":
								pred2 = write_circle(pred2, seed, rad)
							else:
								pred2 = write_sphere(pred2, seed, rad)

							b0, b1, b2 = Betti_error(pred2, self.pred, connectivity = 1)
							b0_2, b1_2, b2_2 = Betti_error(pred2, self.pred, connectivity = 2)
							length +=1

						else:
							break
					else:
						break

				break
					
		return fail, pred

	def hole_merging(self):

		# Fill a hole
		rad = np.array([self.Ggt.nodes[n]["coords"][-1] for n in self.Ggt.nodes()])
		max_rad = 4*np.mean(rad)

		fail = True
		pred = np.copy(self.pred)

		lbl = label(np.invert(self.img), connectivity = 2)
		rad_lbl = distance_transform_edt(lbl)

		nb_cc = lbl.max()
		bg_id = lbl[tuple(np.argwhere(self.img == 0)[0,:])]

		cc_list = [i for i in range(lbl.max() + 1) if i!=bg_id and np.max(rad_lbl * (lbl == i)) < max_rad]
		shuffle(cc_list)

		if nb_cc > 1:
			for i in range(len(cc_list)):
			
				pred = np.copy(self.pred)
				pred[lbl == cc_list[i]] = 1

				b0, b1, b2 = Betti_error(pred, self.pred, connectivity = 2)
				if b0 == 0 and b1 == -1 and b2 == 0:
					fail = False
					break

		return fail, pred


	def merging(self):
		
		# Add foregroung-foreground connection creating a hole 
		fail = True
		pred = np.copy(self.pred)
		max_try = 10
		# Average branch radius
		rad = [self.Ggt.nodes[n]["coords"][-1] for n in self.Ggt.nodes()]
		mean_rad = sum(rad) / len(rad)
		max_N = int(mean_rad * 8)
		t = 0

		while fail is True and t < max_try:
			N = np.random.choice(np.arange(1, max_N))

			cls_img = np.copy(self.pred)
			for i in range(N):
				cls_img = dilation(cls_img)
			for i in range(N):
				cls_img = erosion(cls_img)

			cls_img[self.pred] = 0
			cc = label(cls_img)

			region_list = list(regionprops(cc))
			shuffle(region_list)

			for region in region_list:
				pred = np.copy(self.pred)

				for coord in region.coords:
					pred[tuple(coord)] = 1

				b0, b1, b2 = Betti_error(pred, self.pred)
				b0_2, b1_2, b2_2 = Betti_error(pred, self.pred, connectivity = 2)
				if b0 == 0 and b1 == 1 and b2 == 0 and b0_2 == 0 and b1_2 == 1 and b2_2 == 0:
					fail = False
					break
			t += 1

		return fail, pred


	def component_merging(self):
		# Merge connected components 

		fail = True
		pred = np.copy(self.pred)
		max_try = 5

		rad = [self.Ggt.nodes[n]["coords"][-1] for n in self.Ggt.nodes()]
		mean_rad = sum(rad) / len(rad)
		max_N = int(mean_rad * 5)
		t = 0

		while fail is True and t < max_try:
			N = np.random.choice(np.arange(1, max_N))

			cls_img = np.copy(self.pred)
			for i in range(N):
				cls_img = dilation(cls_img)
			for i in range(N):
				cls_img = erosion(cls_img)

			cls_img[self.pred] = 0
			cc = label(cls_img)

			region_list = list(regionprops(cc))
			shuffle(region_list)
			for region in region_list:
				pred = np.copy(self.pred)

				for coord in region.coords:
					pred[tuple(coord)] = 1

				b0, b1, b2 = Betti_error(pred, self.pred, connectivity = 1)
				b0_2, b1_2, b2_2 = Betti_error(pred, self.pred, connectivity = 2)
				if b0 == -1 and b1 == 0 and b2 == 0 and b0_2 == -1 and b1_2 == 0 and b2_2 == 0:
					fail = False
					break
			t += 1

		return fail, pred


	# MORPHOLOGY
	def missing_branch(self):
		# Remove a branch

		T = full_to_topo(self.Gpred)
		pred = np.copy(self.pred)
		# Get list of terminal branches
		terminal_branches = [e for e in T.edges(keys=True) if (T.degree(e[0]) == 1 and T.degree(e[1]) > 1) or (T.degree(e[1]) == 1 and T.degree(e[0]) > 1)]
		shuffle(terminal_branches)
		fail = True

		for e in terminal_branches:

			nds = T.edges[e]["nodes"]
			reverse = False
			if T.degree(e[0]) == 1:
				reverse = True

			# Leave enough length to form a branch
			rad_branch, rad_end = int(self.Ggt.nodes[e[0]]["coords"][-1]), int(self.Ggt.nodes[e[1]]["coords"][-1])
			if reverse:
				rad_branch, rad_end = int(self.Ggt.nodes[e[1]]["coords"][-1]), int(self.Ggt.nodes[e[0]]["coords"][-1])
					
			max_remove = len(nds) - rad_branch * 4

			to_remove = nds[:]
			if reverse:
				to_remove += [e[0]]
			else:
				to_remove += [e[1]]

			# Write new image
			Gtmp = copy.deepcopy(self.Gpred)		
			for n in to_remove:
				Gtmp.remove_node(n)

			pred = write_image(Gtmp, self.img.shape)

			b0, b1, b2 = Betti_error(pred, self.pred, connectivity = 1)
			if b0 == 0 and b1 == 0 and b2 == 0:
				for n in to_remove:
					self.Gpred.remove_node(n)
				fail = False

				break 

		return fail, pred


	def false_branch(self):
		# Add a false branch (changes the morphology but not the topology)

		pred = np.copy(self.pred)
		edt = distance_transform_edt(np.invert(np.logical_or(self.img, self.pred)))
		min_size = 1
		max_size = max([self.Ggt.nodes[n]["coords"][-1] for n in self.Ggt.nodes()])
		margin = 2

		T = full_to_topo(self.Gpred)
		size_branch = np.array([len(T.edges[e]["nodes"]) for e in T.edges(keys=True)])
		start_rad = np.random.uniform(min_size, max_size)

		# Get a seed
		seeds = np.argwhere((edt > start_rad + margin) & (edt < start_rad + margin + 2))
		np.random.shuffle(seeds)

		fail = True
		for seed in seeds:
			if np.all(seed.astype(int) > np.zeros(len(self.img.shape))) and np.all(seed.astype(int) < np.array(self.img.shape) - 1):

				rad = start_rad
				# Direct toward increasing gradient of distance map 
				direction = np.random.uniform(-1, 1, 3)
				max_grad = 0
				for x in [0, 1, -1]:
					for y in [0, 1, -1]:
						for z in [0, 1, -1]:
							shift = np.array([x, y, z])
						grad = edt[tuple(seed + shift[:len(seed.shape)])] - edt[tuple(seed)] 
						if grad > max_grad:
							max_grad = grad 
							direction = shift
				direction = direction / norm(direction)

				# Create object
				pred = np.copy(self.pred)
				if self.mode == "2d":
					pred = write_circle(pred, seed, rad)
				else:
					pred = write_sphere(pred, seed, rad)

				b0, b1, b2 = Betti_error(pred, self.pred)
				if b0 == 1 and b1 == 0 and b2 == 0:
					length = 1
					max_length = max([round(start_rad), np.random.normal(size_branch.mean(), size_branch.mean() / 4)])

					while b0 == 1 and b1 == 0 and b2 == 0:

						# Next coordinate
						seed = seed + direction[:len(self.img.shape)]
						angle = np.random.normal(loc=0.0, scale=0.1)

						if self.mode == "2d":
							axis = np.array([0, 0, 1])
						else:
							axis = np.random.uniform(-1, 1, 3)
							axis = axis / norm(axis)
					
						direction = rotate_vector(direction, axis = axis, theta=angle)
						if np.all(seed.astype(int) >= np.zeros(len(self.img.shape))) and np.all(seed.astype(int) < np.array(self.img.shape)):
							if edt[tuple(seed.astype(int))] - margin > min_size and length < max_length:

								# Next rad
								diff_range = np.random.normal(loc=0.0, scale=0.1)
								rad = np.random.uniform(max([min_size, rad - diff_range]), min([edt[tuple(seed.astype(int))] - margin, rad + diff_range]))
								if self.mode == "2d":
									pred = write_circle(pred, seed, rad)
								else:
									pred = write_sphere(pred, seed, rad)

								b0, b1, b2 = Betti_error(pred, self.pred)
								length +=1
							else:
								break
						else:
							seed = seed - direction[:len(self.img.shape)]
							break

					if length > start_rad:

						# Connect to foreground
						cls_img = np.copy(pred)
						for i in range(round(start_rad)):
							cls_img = dilation(cls_img)
						for i in range(round(start_rad)):
							cls_img = erosion(cls_img)

						cls_img[pred] = 0
						cc = label(cls_img).astype(int)

						# Keep only cc close to the new component
						new_comp = np.logical_and(pred, np.logical_not(self.pred))
						new_comp = dilation(new_comp)
						new_comp = dilation(new_comp)

						cc_keep = np.unique(cc[new_comp].flatten())

						for cc_id in cc_keep: 

							pred_test = np.copy(pred)
							pred_test[cc == cc_id] = 1

							b0, b1, b2 = Betti_error(pred_test, self.pred)
							ske_err = branch_number_error(pred_test, self.pred)
							if b0 == 0 and b1 == 0 and b2 == 0 and ske_err == 2:
								fail = False
								break 

						if not fail:
							break
		return fail, pred_test


	# GEOMETRY
	def missing_terminal(self):
		# Remove a branch terminal
		pred = np.copy(self.pred)
		T = full_to_topo(self.Gpred)
		# Get list of terminal branches
		terminal_branches = [e for e in T.edges(keys=True) if (self.Ggt.degree(e[0]) == 1 and self.Ggt.degree(e[1]) > 1) or (self.Ggt.degree(e[1]) == 1 and self.Ggt.degree(e[0]) > 1)]
		fail = True

		for e in terminal_branches:

			nds = T.edges[e]["nodes"]
			reverse = False
			if T.degree(e[0]) == 1:
				reverse = True

			# Leave enough length to form a branch
			rad_branch, rad_end = int(self.Ggt.nodes[e[0]]["coords"][-1]), int(self.Ggt.nodes[e[1]]["coords"][-1])
			if reverse:
				rad_branch, rad_end = int(self.Ggt.nodes[e[1]]["coords"][-1]), int(self.Ggt.nodes[e[0]]["coords"][-1])
					
			max_remove = len(nds) - rad_branch * 4

			# Remove part of the branch
			prob = np.random.uniform()
			if max_remove > rad_end * 4:

				# Remove a random number of nodes
				nb_remove = np.random.choice(np.arange(rad_end*4, max_remove))
				if reverse:
					to_remove = nds[0:nb_remove] + [e[0]]
				else:
					to_remove = nds[::-1][0:nb_remove] + [e[1]]
						
				# Write new image
				Gtmp = copy.deepcopy(self.Gpred)		
				for n in to_remove:
					Gtmp.remove_node(n)

				pred = write_image(Gtmp, self.img.shape)

				b0, b1, b2 = Betti_error(pred, self.pred)
				ske_err = branch_number_error(pred, self.pred)
				if b0 == 0 and b1 == 0 and b2 == 0 and ske_err == 0:
					for n in to_remove:
						self.Gpred.remove_node(n)

					fail = False

					break 
		return fail, pred



	def false_terminal(self):

		pred = np.copy(self.pred)
		# Add false terminal
		T = full_to_topo(self.Ggt)
		size_branch = np.array([len(T.edges[e]["nodes"]) for e in T.edges(keys=True)])
		# Get list of terminal branches
		terminal_branches = [e for e in T.edges(keys=True) if ((T.degree(e[0]) == 1 and T.degree(e[1]) > 1) or (T.degree(e[1]) == 1 and T.degree(e[0]) > 1))
			and e[1] not in self.terminal_list and e[0] not in self.terminal_list]

		fail = True
		min_size = 2

		for e in terminal_branches:

			if T.degree(e[0]) == 1:
				n = e[0]
			else:
				n = e[1]

			# Choose radius
			rad = max([min_size, self.Ggt.nodes[n]["coords"][-1]])
			coord = self.Ggt.nodes[n]["coords"][:len(self.img.shape)]
			min_length = rad * 4

			direction = self.Ggt.nodes[n]["coords"][:3] - self.Ggt.nodes[list(self.Ggt.neighbors(n))[0]]["coords"][:3]
			direction = direction / norm(direction)

			# Create object
			pred = np.copy(self.pred)
			if self.mode == "2d":
				pred = write_circle(pred, coord, rad)
			else:
				pred = write_sphere(pred, coord, rad)

			b0, b1, b2 = Betti_error(pred, self.pred)
			length = 1

			if b0 == 0 and b1 == 0 and b2 == 0:

				max_length = max([min_length + 1, np.random.normal(size_branch.mean(), size_branch.mean() / 4)])
				while b0 == 0 and b1 == 0 and b2 == 0:
					pred2 = np.copy(pred)
					# Next coordinate
					coord = coord + direction[:len(self.img.shape)]
					angle = np.random.normal(loc=0.0, scale=0.1)

					if self.mode == "2d":
						axis = np.array([0, 0, 1])
					else:
						axis = np.random.uniform(-1, 1, 3)
						axis = axis / norm(axis)
				
					direction = rotate_vector(direction, axis = axis, theta=angle)
					if np.all(coord.astype(int) >= np.zeros(len(self.img.shape))) and np.all(coord.astype(int) < np.array(self.img.shape)) and length < max_length:
					
						# Next rad
						diff_range = np.random.normal(loc=0.0, scale=0.1)
						rad = np.random.uniform(max([min_size, rad - diff_range]), rad + diff_range)

						if self.mode == "2d":
							pred2 = write_circle(pred2, coord, rad)
						else:
							pred2 = write_sphere(pred2, coord, rad)

						b0, b1, b2 = Betti_error(pred2, self.pred)
						if b0 == 0 and b1 == 0 and b2 == 0:
							pred = np.copy(pred2)
						length +=1

					else:
						b0 = 1

			ske_err = branch_number_error(pred, self.pred)

			if length > min_length and ske_err == 0:
				fail = False
				self.terminal_list.append(n)
				break
					
		return fail, pred


	def branch_merging(self):
		# Add foregroung-foreground connection than does not change the topology and the morphology

		pred = np.copy(self.pred)
		fail = True
		max_try = 10

		rad = [self.Ggt.nodes[n]["coords"][-1] for n in self.Ggt.nodes()]
		mean_rad = sum(rad) / len(rad)
		max_N = int(mean_rad * 5)
		min_N = 1
		N = np.random.choice(np.arange(1, max_N))
		ncc = label(np.logical_and(self.pred, np.logical_not(self.img))).max()

		t = 0
		min_vol = pi * mean_rad**2 * 2

		while fail is True and t < max_try:
			N = np.random.choice(np.arange(min_N, max_N))

			cls_img = np.copy(self.img)
			for i in range(N):
				cls_img = dilation(cls_img)
			for i in range(N):
				cls_img = erosion(cls_img)

			cls_img[self.img] = 0
			cc = label(cls_img)
			region_list = list(regionprops(cc))
			shuffle(region_list)

			for region in region_list:
				if region.area > min_vol:
					# Try out the region
					pred = np.copy(self.pred)

					for coord in region.coords:
						pred[tuple(coord)] = 1

					# Check that a new error is created
					ncc2 = label(np.logical_and(pred, np.logical_not(self.img))).max()
					if ncc2 > ncc:

						# Check Betti numbers and number of branches
						b0, b1, b2 = Betti_error(pred, self.pred)
						b0_2, b1_2, b2_2 = Betti_error(pred, self.pred, connectivity = 2)
						ske_err = branch_number_error(pred, self.pred)
		
						if b0 == 0 and b1 == 0 and b2 == 0 and b0_2 == 0 and b1_2 == 0 and b2_2 == 0 and ske_err < 0:
							fail = False
							break
			t += 1

		return fail, pred


	def self_merging(self):
		# Add foregroung-foreground connection than does not change the topology and the morphology

		fail = True
		pred = np.copy(self.pred)
		max_try = 10

		rad = [self.Ggt.nodes[n]["coords"][-1] for n in self.Ggt.nodes()]
		mean_rad = sum(rad) / len(rad)
		max_N = int(mean_rad * 5)
		N = np.random.choice(np.arange(1, max_N))
		ncc = label(np.logical_and(self.pred, np.logical_not(self.img))).max()

		t = 0

		min_vol = pi * mean_rad**2 * 2

		while fail is True and t < max_try:
			N = np.random.choice(np.arange(1, max_N))

			cls_img = np.copy(self.img)
			for i in range(N):
				cls_img = dilation(cls_img)
			for i in range(N):
				cls_img = erosion(cls_img)

			cls_img[self.img] = 0
			cc = label(cls_img)
			region_list = list(regionprops(cc))
			shuffle(region_list)

			for region in region_list:
				if region.area > min_vol:
					# Try out the region
					pred = np.copy(self.pred)

					for coord in region.coords:
						pred[tuple(coord)] = 1

					# Check that a new error is created
					ncc2 = label(np.logical_and(pred, np.logical_not(self.img))).max()
					if ncc2 > ncc:

						# Check Betti numbers and number of branches
						b0, b1, b2 = Betti_error(pred, self.pred)
						b0_2, b1_2, b2_2 = Betti_error(pred, self.pred, connectivity = 2)
						ske_err = branch_number_error(pred, self.pred)
					
						if b0 == 0 and b1 == 0 and b2 == 0 and b0_2 == 0 and b1_2 == 0 and b2_2 == 0 and ske_err == 0:
							fail = False
							break
			t += 1

		return fail, pred


	def radius_erosion(self):
		# Erode radius without changing the topology or morphology

		fail = True
		pred = np.copy(self.pred)
		T = full_to_topo(self.Gpred)
		edge_list = [e for e in T.edges(keys=True)]
		shuffle(edge_list)

		min_diff = 1
		min_rad = 1
		
		for e in edge_list:

			nds = [e[0]] + T.edges[e]["nodes"] + [e[1]]
			rad = [self.Gpred.nodes[n]["coords"][-1] for n in nds]

			min_size = min([len(nds)-1, 2*round(sum(rad)/len(rad))])
			
			if len(nds) > min_size and sum(rad)/len(rad) > min_rad + min_diff:

				size = np.random.choice(np.arange(min_size, len(nds)))
				i1 = np.random.choice(np.arange(0, len(nds)))
				nds = nds[i1:i1+size]
				rad = rad[i1:i1+size]
				
				diff = np.random.uniform(min_diff, max([min_diff + 1, max(rad) - min_rad]))

				# Gradually increase radius if not terminal
				coef = 0.25
				slope = np.ones((2, len(nds))) * diff
					
				if T.degree(e[0]) > 2:
					lin = np.linspace(0, diff, int(diff // coef)).tolist()
					for i in range(min([len(lin), len(nds)])):
						slope[0,i] = lin[i]

				if T.degree(e[1]) > 2:
					lin = np.linspace(0, diff, int(diff // coef)).tolist()
					for i in range(min([len(lin), len(nds)])):
						slope[1, len(nds) - 1 - i] = lin[i]
				diff_list = np.min(slope, axis = 0)


				Gtmp = copy.deepcopy(self.Gpred)
				for i in range(len(nds)):
					old_rad = Gtmp.nodes[nds[i]]["coords"][-1]
					new_rad = max([1, old_rad - diff_list[i]])
					Gtmp.nodes[nds[i]]["coords"][-1] = new_rad

				pred = write_image(Gtmp, self.img.shape)
				b0, b1, b2 = Betti_error(pred, self.pred, connectivity = 1)
				ske_err = branch_number_error(pred, self.pred)
			
				if b0 == 0 and b1 == 0 and b2 == 0 and ske_err == 0:
					for i in range(len(nds)):
						old_rad = self.Gpred.nodes[nds[i]]["coords"][-1]
						new_rad = max([1, old_rad - diff_list[i]])
						self.Gpred.nodes[nds[i]]["coords"][-1] = new_rad

					fail = False
					break

		return fail, pred


	def radius_dilation(self):
		# Dilate the radius without changing the topology and morphology

		fail = True
		pred = np.copy(self.pred)
		T = full_to_topo(self.Gpred)
		edge_list = [e for e in T.edges(keys=True)]
		shuffle(edge_list)
		min_diff = 1

		for e in edge_list:

			nds = [e[0]] + T.edges[e]["nodes"] + [e[1]]
			rad = [self.Gpred.nodes[n]["coords"][-1] for n in nds]
			org_rad = [self.Ggt.nodes[n]["coords"][-1] for n in nds]

			min_size = max([1, 2*round(sum(rad)/len(rad))])

			if len(nds) > min_size:

				size = np.random.choice(np.arange(min_size, len(nds)))
				i1 = np.random.choice(np.arange(0, len(nds) - size))
				nds = nds[i1:i1+size]
				rad = np.array(rad[i1:i1+size])
				org_rad = np.array(org_rad[i1:i1+size])

				max_diff = np.min(2*org_rad - rad)
				if max_diff >= min_diff:
					
					diff = np.random.uniform(min_diff, max([min_diff + 1, max_diff]))

					# Gradually increase radius if not terminal
					coef = 0.5
					slope = np.ones((2, len(nds))) * diff
					
					if T.degree(e[0]) > 2:
						lin = np.linspace(0, diff, int(diff // coef)).tolist()
						for i in range(min([len(lin), len(nds)])):
							slope[0,i] = lin[i]

					if T.degree(e[1]) > 2:
						lin = np.linspace(0, diff, int(diff // coef)).tolist()
						for i in range(min([len(lin), len(nds)])):
							slope[1, len(nds) - 1 - i] = lin[i]

					diff_list = np.min(slope, axis = 0)

					pred = np.copy(self.pred)
					for i in range(len(nds)):
						rad = self.Gpred.nodes[nds[i]]["coords"][-1] + diff_list[i]
						coord = self.Gpred.nodes[nds[i]]["coords"][:3].astype(int)

						if self.mode == "2d":
							pred = write_circle(pred, coord, rad)
						else:
							pred = write_sphere(pred, coord, rad)

					b0, b1, b2 = Betti_error(pred, self.pred)
					ske_err = branch_number_error(pred, self.pred)

					if b0 == 0 and b1 == 0 and b2 == 0 and ske_err == 0:
						rad_map = distance_transform_edt(pred)
						for i in range(len(nds)):
							new_rad = rad_map[tuple(self.Gpred.nodes[nds[i]]["coords"][:len(self.img.shape)].astype(int))]
							self.Gpred.nodes[nds[i]]["coords"][-1] = new_rad

						fail = False
						break
		return fail, pred


	def deformation(self, constraint = "none"):

		fail = True
		pred = np.copy(self.pred)

		#max_try = 1000
		min_int = 0.5
		max_int = 2

		T = full_to_topo(self.Gpred)
		edge_list = [e for e in T.edges(keys=True)]
		shuffle(edge_list)

		size_branch = np.array([len(T.edges[e]["nodes"]) for e in T.edges(keys=True)])
		ncc = label(self.pred).max()

		for e in edge_list:

			# Write deform dataset
			d_field = np.zeros(list(self.img.shape) + [3])

			# Choose random position 
			nds = [e[0]] + T.edges[e]["nodes"] + [e[1]]

			n = choice(nds)
			rad = self.Gpred.nodes[n]["coords"][-1]
			mu = self.Gpred.nodes[n]["coords"][:2].astype(int)

			# Deformation size = mean branch length
			size = len(nds) * 10
			s = max([0, np.random.normal(size, size/2)])

			# Choose direction 
			direction = np.random.uniform(-1, 1, 2)
			direction = direction / norm(direction)

			try:

				# Choose intensity
				mag = np.random.uniform(low=min_int, high=max([min_int + 0.1, min([rad, max_int])])) 
	
				# Write deformation
				x, y = np.mgrid[0:self.img.shape[0]:1, 0:self.img.shape[1]:1]
				pos = np.dstack((x, y))
				rv = multivariate_normal(mu, np.identity(2)*s)
				mag_field = rv.pdf(pos) * (1/rv.pdf(mu)) * mag

				d_field[:,:,0] = d_field[:,:,0] + (np.ones(self.img.shape) * direction[0]) * mag_field
				d_field[:,:,1] = d_field[:,:,1] + (np.ones(self.img.shape) * direction[1]) * mag_field

				# Deform graph
				Gtmp = copy.deepcopy(self.Gpred)
				for n in Gtmp.nodes():
					coords = Gtmp.nodes[n]["coords"]
					coords[:3] = np.round(coords[:3] + d_field[int(coords[0]), int(coords[1]), :])
					Gtmp.nodes[n]["coords"] = coords

				pred = write_image(Gtmp, self.img.shape)

				b0, b1, b2 = Betti_error(pred, self.pred, connectivity = 1)
				b0_2, b1_2, b2_2 = Betti_error(pred, self.pred, connectivity = 2)

				if b0 == 0 and b1 == 0 and b2 == 0 and b0_2 == 0 and b1_2 == 0 and b2_2 == 0:
					# Check that we didn't create any new cc
					for n in self.Gpred.nodes():
						coords = self.Gpred.nodes[n]["coords"]
						coords[:3] = np.round(coords[:3] + d_field[int(coords[0]), int(coords[1]), :])
						self.Gpred.nodes[n]["coords"] = coords

					fail = False
					break
			except:
				pass
		
		return fail, pred







