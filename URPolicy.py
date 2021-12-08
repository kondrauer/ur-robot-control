# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 13:09:41 2021

@author: Student
"""

import json
import os
import time
import logging
import colorlog

import numpy as np

from autolab_core import YamlConfig
from perception import (BinaryImage, CameraIntrinsics, ColorImage, DepthImage,
                        RgbdImage)

from gqcnn.grasping import (RobustGraspingPolicy,
                            CrossEntropyRobustGraspingPolicy, RgbdImageState,
                            FullyConvolutionalGraspingPolicyParallelJaw,
                            FullyConvolutionalGraspingPolicySuction)
from gqcnn.utils import GripperMode

from visualization import Visualizer2D as vis

from RealsenseInterface import RealsenseInterface
from HandEyeCalibrator import HandEyeCalibrator
#TODO: make paths relative (model and pictures in ur robot control)

if __name__ == "__main__":
	root_logger = logging.getLogger()
	for hdlr in root_logger.handlers:
		if isinstance(hdlr, logging.StreamHandler):
			root_logger.removeHandler(hdlr)
	
	logging.root.name = 'Robotiklabor'
	logging.getLogger().setLevel(logging.DEBUG)
	
	handler = colorlog.StreamHandler()
	formatter = colorlog.ColoredFormatter("%(purple)s%(name)-10s "
						"%(log_color)s%(levelname)-8s%(reset)s "
						"%(white)s%(message)s",
						reset=True,
						log_colors={
							"DEBUG": "cyan",
							"INFO": "green",
							"WARNING": "yellow",
							"ERROR": "red",
							"CRITICAL": "red,bg_white",
						},)
	handler.setFormatter(formatter)
	logger = colorlog.getLogger()
	logger.addHandler(handler)

	hec = HandEyeCalibrator()
	model_path = 'C:/Users/Student/Desktop/gqcnn-master/models/GQCNN-4.0-PJ'
	model_config = json.load(open(os.path.join(model_path, "config.json"),"r"))

	try:
		gqcnn_config = model_config["gqcnn"]
		gripper_mode = gqcnn_config["gripper_mode"]
	except KeyError:
		gqcnn_config = model_config["gqcnn_config"]
		input_data_mode = gqcnn_config["input_data_mode"]
		if input_data_mode == "tf_image":
			gripper_mode = GripperMode.LEGACY_PARALLEL_JAW
		elif input_data_mode == "tf_image_suction":
			gripper_mode = GripperMode.LEGACY_SUCTION
		elif input_data_mode == "suction":
			gripper_mode = GripperMode.SUCTION
		elif input_data_mode == "multi_suction":
			gripper_mode = GripperMode.MULTI_SUCTION
		elif input_data_mode == "parallel_jaw":
			gripper_mode = GripperMode.PARALLEL_JAW
		else:
			raise ValueError("Input data mode {} not supported!".format(input_data_mode))
			
	config_filename = 'C:/Users/Student/Desktop/gqcnn-master/cfg/examples/gqcnn_pj.yaml'
	
	config = YamlConfig(config_filename)
	inpaint_rescale_factor = config["inpaint_rescale_factor"]
	policy_config = config["policy"]

    # Make relative paths absolute.
	if "gqcnn_model" in policy_config["metric"]:
		policy_config["metric"]["gqcnn_model"] = model_path
		if not os.path.isabs(policy_config["metric"]["gqcnn_model"]):
			policy_config["metric"]["gqcnn_model"] = os.path.join(
				os.path.dirname(os.path.realpath(__file__)), "..",
				policy_config["metric"]["gqcnn_model"])

    # Setup sensor.
	camera_intr = CameraIntrinsics.load('C:/Users/Student/Desktop/gqcnn-master/data/calib/realsense/realsense.intr')
	
	re = RealsenseInterface(align=True, decimation=False)
	re.start()
	index = re.saveImageSet(iterationsDilation = 0, filter=True)
	
	# Read images.
	depth_data = np.load(f'img/depth_{index}.npy')
	depth_im = DepthImage(depth_data, frame=camera_intr.frame)
	color_im = ColorImage(np.zeros([depth_im.height, depth_im.width,3]).astype(np.uint8),frame=camera_intr.frame)
	
	segmask = BinaryImage.open(f'img/segmask_{index}.png')
	depth_im = depth_im.inpaint(rescale_factor=inpaint_rescale_factor)
	
	rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
	state = RgbdImageState(rgbd_im, camera_intr, segmask=segmask)
	
	policy_type = "cem"
	if "type" in policy_config:
		policy_type = policy_config["type"]
	if policy_type == "ranking":
		policy = RobustGraspingPolicy(policy_config)
	elif policy_type == "cem":
		policy = CrossEntropyRobustGraspingPolicy(policy_config)
	else:
		raise ValueError("Invalid policy type: {}".format(policy_type))
		
	# Query policy.
	policy_start = time.time()
	action = policy(state)
	logger.info("Planning took %.3f sec" % (time.time() - policy_start))
	
	if policy_config["vis"]["final_grasp"]:
		vis.figure(size=(10, 10))
		vis.imshow(rgbd_im.depth,
			       vmin=policy_config["vis"]["vmin"],
				   vmax=policy_config["vis"]["vmax"])
		vis.grasp(action.grasp, scale=2.5, show_center=False, show_axis=True)
		vis.title("Planned grasp at depth {0:.3f}m with Q={1:.3f}".format(
			action.grasp.depth, action.q_value))
		vis.show()
	
	tfGraspToBase = hec.graspTransformer(action.grasp.pose())
	hec.moveToPoint(tfGraspToBase)