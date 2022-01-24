import cv2 
import numpy as np
import logging

from RealsenseInterface import RealsenseInterface
from autolab_core import RigidTransform

from pathlib import Path

class CharucoPoseEstimator:
	"""Class for estimating camera pose of a realsense L515 with regards to a charuco board
	
	Attributes:
		squareX (int): number of squares on short side
		squareY (int): number of squares on long side
		squareLength (float): length of squares in meters
		squareWidth (float): width of squares in meters
		
		arucoDict: predefined aruco dict from opencv
		charucoBoard: board representation from opencv
		boardSize: holding number of corners
		
		realsense (RealsenseInterface): class object to interface with intel realsense
	"""


	def __init__(self, squareX: int = 5, squareY: int = 7, squareLength: float = 0.035, squareWidth: float = 0.023, depth: bool = False, realsense: bool = True) -> None:
		"""Constructor for CharucoPoseEstimator
		
		Args:
			squareX (int): number of squares on short side
			squareY (int): number of squares on long side
			squareLength (float): length of squares in meters
			squareWidth (float): width of squares in meters
			depth (bool): determines if depth-sensor should be enabled or not
			realsense(bool): wether or not to start realsense
			
		"""
		
		self.squareX: int = squareX
		self.squareY: int = squareY
		self.squareLength: float = squareLength
		self.squareWidth: float  = squareWidth
		
		self.arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
		self.charucoBoard = cv2.aruco.CharucoBoard_create(self.squareX, self.squareY, self.squareLength, self.squareWidth, self.arucoDict)
		self.boardSize = self.charucoBoard.getChessboardSize()
		
		if realsense:
			self.realsense = RealsenseInterface(align=False, depth=depth)
			self.realsense.start()
	
	def estimatePose(self, correction_z: float = 0.0, debug: bool = False):
		"""Estimates the camera pose with regards to a charuco board
		
		Args:
			correction_z (float): correction constant for z-translation
			debug (bool): shows information about estimated pose if true	
		"""
		
		# Take a picture with realsense and get the colorFrame as np.array
		self.realsense.getFrames()
		
		colorFrame = self.realsense.colorFrame
		colorArray = np.array(colorFrame.get_data())
		
		# Find the Board markers
		markerCorners, markerIds, _ = cv2.aruco.detectMarkers(colorArray, self.arucoDict, cameraMatrix = self.realsense.colorCameraMatrix, distCoeff = self.realsense.colorCameraDistortion)
		
		# Check if markers are detected
		if markerIds is not None and len(markerIds) > 0:
			
			# Interpolate the charuco corners from markers
			cornerCount, corners, ids = cv2.aruco.interpolateCornersCharuco(markerCorners, markerIds, colorArray, self.charucoBoard,
												 self.realsense.colorCameraMatrix, self.realsense.colorCameraDistortion)
			# Check if any corner is found								 
			if cornerCount > 0 and len(ids) > 0:
			
				logging.info(f'Detected Corners: {cornerCount}, IDs: {len(ids)}')
				
				if debug:
					cv2.aruco.drawDetectedCornersCharuco(colorArray, corners, ids)
				
				# Estimate camera pose
				success, rotationVectorWorldToCamera, translationVectorWorldToCamera = cv2.aruco.estimatePoseCharucoBoard(corners, ids, self.charucoBoard, 														self.realsense.colorCameraMatrix, self.realsense.colorCameraDistortion, None, None, False)
				
				if success:
					colorArray = cv2.aruco.drawAxis(colorArray, self.realsense.colorCameraMatrix, self.realsense.colorCameraDistortion, rotationVectorWorldToCamera, 		
										translationVectorWorldToCamera, 100.)
					
				minCorners = int((self.boardSize[0] - 1) * (self.boardSize[1] - 1) * 0.5) # as % of total
				
				# At least 50% of the corners need to be detected, otherwise pose is not legit
				if cornerCount >= minCorners and ids.size >= minCorners:
					
					rotationMatrixWorldToCamera, _ = cv2.Rodrigues(rotationVectorWorldToCamera)
					
					# There seems to be a systematic error with the z-coordinate, so we add a constant
					translationVectorWorldToCamera[2] = translationVectorWorldToCamera[2] + correction_z
					
					if debug:
						logging.info('Rotation: ')
						logging.info(rotationMatrixWorldToCamera)
						logging.info('Translation: ')
						logging.info(translationVectorWorldToCamera)
						logging.info(f'Distance: {np.linalg.norm(translationVectorWorldToCamera)}')
					
						cv2.imshow('Estimated Pose', colorArray)
						cv2.waitKey(0)
						
					return True, rotationMatrixWorldToCamera, translationVectorWorldToCamera
					
				else:
					logging.error('Not enough corners detected try again!')
					
					return False, None, None
					
	def saveNPoses(self, n: int = 20) -> None:
		"""Saves n poses to file for lates usage or calibration
		
		Args:
			n (int): Number of poses to capture	
		"""
		
		rMat = []
		tVec = []
		
		for i in range(n):
		
			input('Press enter to capture next camera pose...')
			
			success, rotationMatrix, translationVector = self.estimatePose()
			
			if success:
				rMat.append(rotationMatrix)
				tVec.append(translationVector)
		
		Path('poses/').mkdir(exist_ok=True)
			
		with open(f'poses/worldToCameraRotation_{n}.npy', 'wb') as f:
			np.save(f, np.array(rMat))

		with open(f'poses/worldToCameraTranslation_{n}.npy', 'wb') as f:
			np.save(f, np.array(tVec))
	
	def getDepthToWorldTransform(self, debug: bool = False) -> RigidTransform:
		"""Computes transform from depth-frame of camera to world-frame
		
		Args:
			debug (bool): shows information about computed transformations if true	
			
		Returns:
			
			RigidTransform: Transformation from depth to world
			
		"""	
		
		#TODO: maybe change z_correction to 0 and add 0.035/0.05 to translation afterwards
		
		success, rMatWorldToCamera, tVecWorldToCamera = self.estimatePose(debug=debug)
		
		if success:
			
			tfWorldToCamera = RigidTransform(rotation = rMatWorldToCamera, translation = tVecWorldToCamera, from_frame = 'world', to_frame = 'realsense_color')
			tfDepthToCamera = RigidTransform(rotation = self.realsense.depthToColorRotation, translation = self.realsense.depthToColorTranslation, 
								from_frame = 'realsense_depth', to_frame = 'realsense_color')
								
			tfDepthToWorld =  tfWorldToCamera.inverse() * tfDepthToCamera
			
			if debug:
				logging.info(tfDepthToWorld)
			
			return tfDepthToWorld
		else:
			logging.error('Failed to estimate pose, try again')
			return None
		
	def getWorldToCameraTransform(self, debug: bool = False) -> RigidTransform:
		"""Computes transform from depth-frame of camera to world-frame
		
		Args:
			debug (bool): shows information about computed transformations if true	
			
		Returns:
			
			RigidTransform: Transformation from depth to world
			
		"""	
			
		success, rMatWorldToCamera, tVecWorldToCamera = self.estimatePose(debug=debug)
		
		if success:
			
			tfWorldToCamera = RigidTransform(rotation = rMatWorldToCamera, translation = tVecWorldToCamera, from_frame = 'world', to_frame = 'cam')
			
			if debug:
				logging.info(tfWorldToCamera)
			
			return tfWorldToCamera
		else:
			logging.error('Failed to estimate pose, try again')
			return None
		
	def savePoseAsDepthToWorldTransform(self, path: str = '/media/max/gqcnn/gqcnn/data/calib/realsense/realsense_to_world.tf'):
		"""Saves current pose as depth to world transform for usage with gqcnn package
		
		Args:
			path (str): path to save the transform to
		"""	
			
		tfDepthToWorld = self.getDepthToWorldTransform()
		
		if tfDepthToWorld:
			
			tfDepthToWorld.save(path)	
			logging.info('Saved Depth to World transform to ' + path)
	
	def savePoseAsWorldToCameraTransform(self, path: str = '/media/max/gqcnn/gqcnn/data/calib/realsense/realsense_to_world.tf'):
		"""Saves current pose as depth to world transform for usage with gqcnn package
		
		Args:
			path (str): path to save the transform to
		"""	
			
		tfWorldToCamera = self.getWorldToCameraTransform()
		
		if tfWorldToCamera:
			
			tfWorldToCamera.save(path)	
			logging.info('Saved World to camera transform to ' + path)
		
if __name__ == '__main__':
	pass
# 	print('test')
# 	logging.getLogger().setLevel(logging.INFO)
# 	cpe = CharucoPoseEstimator(depth=True)
# 	_, rMat, tVec = cpe.estimatePose(debug=True)
	
 	# cpe.savePoseAsDepthToWorldTransform()	
	 
