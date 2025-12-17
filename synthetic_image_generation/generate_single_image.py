####################################################################################################
# Author: Meghane Decroocq
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# provided license.
####################################################################################################

from ErrorGenerator import ErrorGenerator
import numpy as np
from PIL import Image
from utils import *

"""
This code provide an example of how you can add an error to an image.
"""

# Load label 
file = "../data/playground/dataset1/img1.png"
label = np.array(Image.open(file).convert('L')).copy().astype(bool)

error_type = "disconnection"
# Other error types : "false_component", "component_merging", "missing_component", "disconnection", "hole_merging", "merging", "hole", 
# "cycle_disconnection", "branch_merging", "missing_branch", "radius_dilation", "radius_erosion", "deformation", "false_branch", "missing_terminal", "false_terminal", "self_merging"

generator = ErrorGenerator(label, error_types = [error_type])
new_label = generator.get_label() # Get modified version of the label
save_image("../output/single_image/label", new_label)

# Add error
fail = generator.generate_error(error_type)
if not fail:
	print("Error successfully generated.")
	pred = generator.get_pred()
	save_image("../output/single_image/pred", pred)
else:
	print("The error required cannot be generated on this image.")
