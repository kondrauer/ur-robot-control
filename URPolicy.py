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
from autolab_core import (BinaryImage, CameraIntrinsics, ColorImage, DepthImage,
                        RgbdImage)

from gqcnn.grasping import (RobustGraspingPolicy,
                            CrossEntropyRobustGraspingPolicy, RgbdImageState,
                            FullyConvolutionalGraspingPolicyParallelJaw,
                            FullyConvolutionalGraspingPolicySuction)
from gqcnn.utils import GripperMode
from gqcnn.utils.policy_exceptions import NoValidGraspsException

from visualization import Visualizer2D as vis

from RealsenseInterface import RealsenseInterface
from HandEyeCalibrator import HandEyeCalibrator
#TODO: make paths relative (model and pictures in ur robot control)

class URPolicy:
	
	def __init__(self, debug: bool = False, fully_conv: bool = False, policy_type: str = "cem",
				  model_path:str = 'C:/Users/Student/Desktop/gqcnn-master/models/GQCNN-2.0', 
				  re: RealsenseInterface=None):
		
		self.fully_conv = fully_conv
		self.policy_type = policy_type
		self.model_path = model_path
		self.model_config = json.load(open(os.path.join(model_path, "config.json"),"r"))
		
		try:
			self.gqcnn_config = self.model_config["gqcnn"]
			self.gripper_mode = self.gqcnn_config["gripper_mode"]
		except KeyError:
			self.gqcnn_config = self.model_config["gqcnn_config"]
			self.input_data_mode = self.gqcnn_config["input_data_mode"]
			if self.input_data_mode == "tf_image":
				self.gripper_mode = GripperMode.LEGACY_PARALLEL_JAW
			elif self.input_data_mode == "tf_image_suction":
				self.gripper_mode = GripperMode.LEGACY_SUCTION
			elif self.input_data_mode == "suction":
				self.gripper_mode = GripperMode.SUCTION
			elif self.input_data_mode == "multi_suction":
				self.gripper_mode = GripperMode.MULTI_SUCTION
			elif self.input_data_mode == "parallel_jaw":
				self.gripper_mode = GripperMode.PARALLEL_JAW
			else:
				raise ValueError("Input data mode {} not supported!".format(self.input_data_mode))
		
		if self.fully_conv:
			self.config_filename = 'C:/Users/Student/Desktop/gqcnn-master/cfg/examples/fc_gqcnn_pj.yaml'
		else:
			self.config_filename = 'C:/Users/Student/Desktop/gqcnn-master/cfg/examples/gqcnn_pj.yaml'
		
		self.config = YamlConfig(self.config_filename)
		self.inpaint_rescale_factor = self.config["inpaint_rescale_factor"]
		self.policy_config = self.config["policy"]
		
		self.policy_config["metric"]["gqcnn_model"] = self.model_path
		
		self.camera_intr = CameraIntrinsics.load('C:/Users/Student/Desktop/gqcnn-master/data/calib/realsense/realsense.intr')
		
		self.hec = HandEyeCalibrator(realsense=False)
		self.tfGraspToBase = None
		
		if not re:
 			self.re = RealsenseInterface(align=True, decimation=False)
 			self.re.start()
		else:
			self.re = re   
		
		logging.root.name = 'Robotiklabor'
		
		if debug:
			logging.getLogger().setLevel(logging.DEBUG)
		else:
			logging.getLogger().setLevel(logging.INFO)
		
	def planGrasp(self):
		
		self.index = self.re.saveImageSet(iterationsDilation = 1, filter=True)
		
		# Read images.
		depth_data = np.load(f'img/depth_{self.index}.npy')
		depth_im = DepthImage(depth_data, frame=self.camera_intr.frame)
		color_im = ColorImage(np.zeros([depth_im.height, depth_im.width,3]).astype(np.uint8),frame=self.camera_intr.frame)
		
		segmask = BinaryImage.open(f'img/segmask_{self.index}.png')
		depth_im = depth_im.inpaint(rescale_factor=self.inpaint_rescale_factor)
		
		self.rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
		state = RgbdImageState(self.rgbd_im, self.camera_intr, segmask=segmask)
		
		if self.fully_conv:
			self.policy_config["metric"]["fully_conv_gqcnn_config"][
	            "im_height"] = depth_im.shape[0]
			self.policy_config["metric"]["fully_conv_gqcnn_config"][
	            "im_width"] = depth_im.shape[1]
		
		logging.info(self.policy_config["metric"]["gqcnn_model"])
		
		
		
		if self.fully_conv:
			policy = FullyConvolutionalGraspingPolicyParallelJaw(self.policy_config)
		elif self.policy_type == 'ranking':
			policy = RobustGraspingPolicy(self.policy_config)
		elif self.policy_type == 'cem':
			policy = CrossEntropyRobustGraspingPolicy(self.policy_config)
			
		# Query policy.
		policy_start = time.time()
		try:
			self.action = policy(state)
		except NoValidGraspsException:
			logging.error("No valid graps found")
		logging.info("Planning took %.3f sec" % (time.time() - policy_start))
			
		self.tfGraspToBase = self.hec.graspTransformer(self.action.grasp.pose())
		
	def executeGrasp(self):
		
		self.hec.moveToPoint(self.tfGraspToBase)

if __name__ == "__main__":
 	root_logger = logging.getLogger()
 	for hdlr in root_logger.handlers:
		 if isinstance(hdlr, logging.StreamHandler):
 			root_logger.removeHandler(hdlr)
 	
 	logging.root.name = 'Robotiklabor'
# 	logging.getLogger().setLevel(logging.DEBUG)
 	logging.getLogger().setLevel(logging.INFO)
 	
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
 	
 	fully_conv = False
 	policy_type = 'cem'
 	
 	hec = HandEyeCalibrator()
 	model_path = 'C:/Users/Student/Desktop/gqcnn-master/models/GQCNN-2.0'
 	# model_path = 'C:/Users/Student/Desktop/gqcnn-master/models/GQCNN4.0-robotiq_3dnet'
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
 	
 	if fully_conv:
		 config_filename = 'C:/Users/Student/Desktop/gqcnn-master/cfg/examples/fc_gqcnn_pj.yaml'
 	else:
		 config_filename = 'C:/Users/Student/Desktop/gqcnn-master/cfg/examples/gqcnn_pj.yaml'
 	
 	config = YamlConfig(config_filename)
 	inpaint_rescale_factor = config["inpaint_rescale_factor"]
 	policy_config = config["policy"]

    # Make relative paths absolute.
 	if "gqcnn_model" in policy_config["metric"]:
		 policy_config["metric"]["gqcnn_model"] = model_path
# 		if not os.path.isabs(policy_config["metric"]["gqcnn_model"]):
# 			policy_config["metric"]["gqcnn_model"] = os.path.join(
# 				os.path.dirname(os.path.realpath(__file__)), "..",
# 				policy_config["metric"]["gqcnn_model"])

    # Setup sensor.
 	camera_intr = CameraIntrinsics.load('C:/Users/Student/Desktop/gqcnn-master/data/calib/realsense/realsense.intr')
 	
 	re = RealsenseInterface(align=True, decimation=False)
 	re.start()
 	index = re.saveImageSet(iterationsDilation = 1, filter=True)
 	
 	# Read images.
 	depth_data = np.load(f'img/depth_{index}.npy')
 	depth_im = DepthImage(depth_data, frame=camera_intr.frame)
 	color_im = ColorImage(np.zeros([depth_im.height, depth_im.width,3]).astype(np.uint8),frame=camera_intr.frame)
 	
 	segmask = BinaryImage.open(f'img/segmask_{index}.png')
 	depth_im = depth_im.inpaint(rescale_factor=inpaint_rescale_factor)
 	
 	rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
 	state = RgbdImageState(rgbd_im, camera_intr, segmask=segmask)
 	
 	if fully_conv:
		 policy_config["metric"]["fully_conv_gqcnn_config"][
            "im_height"] = depth_im.shape[0]
		 policy_config["metric"]["fully_conv_gqcnn_config"][
            "im_width"] = depth_im.shape[1]
 	
 	logging.info(policy_config["metric"]["gqcnn_model"])
 	
 	
 	
 	if fully_conv:
		 policy = FullyConvolutionalGraspingPolicyParallelJaw(policy_config)
 	elif policy_type == 'ranking':
		 policy = RobustGraspingPolicy(policy_config)
 	elif policy_type == 'cem':
		 policy = CrossEntropyRobustGraspingPolicy(policy_config)

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