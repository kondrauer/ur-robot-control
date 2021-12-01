import numpy as np
import cv2
import logging
import colorlog
import time

from autolab_core import RigidTransform
from pathlib import Path

import PoseEstimator as pe
import Server 

class HandEyeCalibrator:
	"""Class for solving the Hand-Eye-Calibration problem with an Intel Realsense L515
	
	Attributes:
		tfBaseToWorld (RigidTransform): Transformation from Base-Frame to World-Frame
		tfGripperToCam (RigidTransform): Transformation from Gripper-Frame to Cam-Frame
		tfDepthToWorld (RigidTransform): Transformation from cameras Depth-Frame to World-Frame
		
		poseEstimator (pe.CharucoPoseEstimator): Class for Pose Estimation of the Camera
		
		server (Server.Server): Wrapper-Class of socket built-in
	
	"""
	
	
	def __init__(self, numberOfMeasurements: int = 25) -> None:
		"""Constructor for HandEyeCalibrator
		
			Args:
				numberOfMeasurements (int): Determines how many poses the robot should approach (this has to be in line with the programm on the robot itself)
			
		"""
		
		if numberOfMeasurements <= 3:
		
			# Number of Measurements can't be smaller than 3
			# see https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga41b1a8dd70eae371eba707d101729c36 under notes; it is a requirement by the method chosen 
			raise ValueError('Number of measurements needs to be > 0')
		
		else:
		
			# any number greater three is accepted
			self.numberOfMeasurements = numberOfMeasurements
		
		self.tfBaseToWorld = RigidTransform(from_frame='base', to_frame='world')
		self.tfGripperToCam = RigidTransform(from_frame='gripper', to_frame='cam')
		
		self.poseEstimator: pe.CharucoPoseEstimator = pe.CharucoPoseEstimator(depth=True)
		self.tfDepthToWorld: RigidTransform = None
		
		self.server = Server.Server()
		
		# check if the calibration was already done and load it
		self.loadCalibrationFromFile()
		
	def calibrateHandEye(self):
		"""Executes the Hand-Eye-Calibration
		
		First a connection with the robot is established. After that the server starts to receive i poses of the Tool-Center-Point from the robot. After receiving a single pose the
		corresponding computation of the camera pose takes place. Both the TCP-pose and the camera-pose are then append to a list. The server sends back an acknowledgement to the robot,
		which in turn approaches the next pose.
		
		After numberOfMeasurements steps the lists of the recorded poses are temporarly saved and cv2.calibrateRobotWorldHandEye() with the lists casted to np.arrays.
		
		The returned transforms are stored and written to the class attributes and the connection to the robot is closed.
		
		Args:
			None
		
		"""
		try:
		
			self.server.commSocket, self.server.commAddress = self.server.establishConnection()
			
			# Buffer variables
			listRMatBaseToGripper: list = []
			listTVecBaseToGripper: list = []
			listRMatWorldToCamera: list = []
			listTVecWorldToCamera: list = []
			
			for i in range(self.numberOfMeasurements):
				
				logging.info(f'Step {i+1} of {self.numberOfMeasurements}')
				
				# get the TCP-pose from the robot
				baseToGripper = self.server.receiveData(1024)
				baseToGripper = baseToGripper[2:len(baseToGripper)]
				
				# str format is 'p[tVecX, tVecY, tVecZ, rVecX, rVecY, rVecZ]'
				baseToGripper = np.fromstring(baseToGripper, dtype = np.float64, sep = ',')
				rVecBaseToGripper = baseToGripper[3:6].reshape(3,1)
				tVecBaseToGripper = baseToGripper[0:3].reshape(3,1)
				
				# convert rotation vector to rotation matrix
				rMatBaseToGripper, _ = cv2.Rodrigues(rVecGripperToBase)
				
				logging.info('Gripper to base of robot transformation determined')
				logging.debug(f'Translation: {tVecGripperToBase}')
				logging.debug(f'Rotation: {rMatGripperToBase}')
				
				# estimate camera pose with regards to charuco board (which is the world frame)
				success, rMatWorldToCamera, tVecWorldToCamera = self.poseEstimator.estimatePose()
				
				# only save results of this step if the pose estimation of the camera was successful
				if success:
					logging.info('World to camera transformation determined')
					logging.debug(f'Translation: {tVecWorldToCamera}')
					logging.debug(f'Rotation: {rMatWorldToCamera}')
					logging.info('Saving transformations')
					
					listRMatBaseToGripper.append(rMatBaseToGripper)
					listTVecBaseToGripper.append(tVecBaseToGripper)
					
					listRMatWorldToCamera.append(rMatWorldToCamera)
					listTVecWorldToCamera.append(tVecWorldToCamera)
				else:
					logging.error('Pose estimation of camera failed, skipping this step')
				
				# tell robot to approach next pose
				input('press key to continue')
				communication = 1
				self.server.sendData(f'{communication}')
			
			# check if folder exists		
			Path('handEyeCalibration/').mkdir(exist_ok=True)
			
			# save all determined poses
			with open('handEyeCalibration/rMatGripperToBase.npy', 'wb') as f:
				np.save(f, np.array(listRMatBaseToGripper)) 
			with open('handEyeCalibration/tVecGripperToBase.npy', 'wb') as f:
				np.save(f, np.array(listTVecBaseToGripper)) 
			with open('handEyeCalibration/rMatWorldToCamera.npy', 'wb') as f:
				np.save(f, np.array(listRMatWorldToCamera)) 
			with open('handEyeCalibration/tVecWorldToCamera.npy', 'wb') as f:
				np.save(f, np.array(listTVecWorldToCamera)) 
				
			logging.info('Calculating transformations from given poses')
			
			start = time.time()
			
			rMatBaseToWorld, tVecBaseToWorld, rMatGripperToCam, tVecGripperToCam = cv2.calibrateRobotWorldHandEye(np.array(listRMatWorldToCamera),
																	np.array(listTVecWorldToCamera),
																	np.array(listRMatBaseToGripper), 
																	np.array(listTVecBaseToGripper),
																	method = cv2.CALIB_ROBOT_WORLD_HAND_EYE_LI)
																	
																			
			logging.info(f'Calculation took {time.time()-start} seconds')
			
			self.tfBaseToWorld: RigidTransform = RigidTransform(rotation = rMatBaseToWorld, translation = tVecBaseToWorld, from_frame = 'base', to_frame = 'world')
			self.tfGripperToCam: RigidTransform = RigidTransform(rotation = rMatGripperToCam, translation = tVecGripperToCam, from_frame = 'gripper', to_frame = 'cam')
																		 
			self.tfBaseToWorld.save('handEyeCalibration/baseToWorld.tf')
			self.tfGripperToCam.save('handEyeCalibration/gripperToCam.tf')
			
			logging.info('Saved calibration to file!')
			
			self.server.closeConnection()
			
			return True
			
		# TODO: Better error handling
		except Exception as e:
			
			print(e)
			return False
	
	def loadCalibrationFromFile(self, path: str = 'handEyeCalibration/'):
		"""Loads Hand-Eye-Calibration if it exists in path.
		
		   Args:
		   	path (str): determines where to look for the files
		   	
		"""
			
		if Path(path).exists():
		
			if Path(path + 'baseToWorld.tf').exists() and Path(path + 'gripperToCam.tf').exists():
			
				logging.info('Loading...')
				self.tfBaseToWorld = RigidTransform.load('handEyeCalibration/baseToWorld.tf')
				self.tfGripperToCam = RigidTransform.load('handEyeCalibration/gripperToCam.tf')
				logging.info('Hand-Eye-Calibration loaded')
			
		else:
			logging.warning('No saved Hand-Eye-Calibration found')
			
	def graspTransformer(self, tfGraspToDepth: RigidTransform):
		"""Transforms a grasp to the base of the robot.
		
		This method takes a grasp calculated by the GQCNN (from_frame = grasp, to_frame = depth) and sets the reference frame to the robots base coordinate system
		by multiplying the respective transformations.
		
		In the end it returns a new transform tfGraspToBase with from_frame = grasp and to_frame = base 
		
		Args:
			tfGraspToDepth (RigidTransform): transformation from grasp-frame to depth-frame of image sensor
			
		Returns:
			
			tfGraspToBase (RigidTransform): transformation from grasp-frame to base-frame of robot
		"""
		
		#TODO: determine if the tfGraspToBase needs to be inversed
		
		if (self.tfBaseToWorld.rotation == np.eye(3)).all():
			logging.warning('Base to World seems to be default matrix')
		
		self.tfDepthToWorld = self.poseEstimator.getDepthToWorldTransform()
		tfGraspToBase =  self.tfBaseToWorld.inverse() * self.tfDepthToWorld * tfGraspToDepth 
		logging.info(tfGraspToBase)
		
		return tfGraspToBase
			
			
if __name__ == '__main__':
	
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
	
	#hec = HandEyeCalibrator()
	#hec.graspTransformer(RigidTransform(from_frame='grasp', to_frame='depth'))	
	#hec.calibrateHandEye()		
	
	listRMatBaseToGripper = np.load('handEyeCalibration/rMatBaseToWorld.npy')
	listTVecBaseToGripper = np.load('handEyeCalibration/tVecBaseToWorld.npy')
	listRMatWorldToCamera = np.load('handEyeCalibration/rMatWorldToCamera.npy')
	listTVecWorldToCamera = np.load('handEyeCalibration/tVecWorldToCamera.npy')
	rMatBaseToWorld, tVecBaseToWorld, rMatGripperToCam, tVecGripperToCam = cv2.calibrateRobotWorldHandEye(np.array(listRMatWorldToCamera),
																	np.array(listTVecWorldToCamera),
																	np.array(listRMatBaseToGripper), 
																	np.array(listTVecBaseToGripper))	
	
