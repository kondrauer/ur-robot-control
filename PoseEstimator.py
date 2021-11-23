import cv2 
import RealsenseInterface
import numpy as np

from autolab_core import RigidTransform

from pathlib import Path

class CharucoPoseEstimator:

	def __init__(self, showDetectedAruco: bool = True, showDetectedCharucoBoard: bool = True, showPoseEstimationCharuco: bool = True, 
			squareX: int = 5, squareY: int = 7, squareLength: float = 0.035, squareWidth: float = 0.023, depth: bool = False) -> None:
		
		self.showDetectedAruco = showDetectedAruco
		self.showDetectedCharucoBoard = showDetectedCharucoBoard
		self.showPoseEstimationCharuco = showPoseEstimationCharuco
		
		self.squareX: int = squareX
		self.squareY: int = squareY
		self.squareLength: float = squareLength
		self.squareWidth: float  = squareWidth
		
		self.arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
		self.charucoBoard = cv2.aruco.CharucoBoard_create(self.squareX, self.squareY, self.squareLength, self.squareWidth, self.arucoDict)
		self.boardSize = self.charucoBoard.getChessboardSize()
		
		self.realsense = RealsenseInterface.RealsenseInterface(depth=depth)
		self.realsense.start()
	
	def estimatePose(self, correction_z: int = 0.035, debug: bool = False):
		
		# code from RoboDK Python API Tutorials, slightly altered
		# https://robodk.com/doc/en/PythonAPI/examples.html#camera-pose
		
		self.realsense.getFrames()
		
		colorFrame = self.realsense.colorFrame
		colorArray = np.array(colorFrame.get_data())
		
		# Find the Board markers
		markerCorners, markerIds, _ = cv2.aruco.detectMarkers(colorArray, self.arucoDict, cameraMatrix = self.realsense.colorCameraMatrix, distCoeff = self.realsense.colorCameraDistortion)
		
		if markerIds is not None and len(markerIds) > 0:
			
			# Interpolate the charuco corners from markers
			cornerCount, corners, ids = cv2.aruco.interpolateCornersCharuco(markerCorners, markerIds, colorArray, self.charucoBoard,
												 self.realsense.colorCameraMatrix, self.realsense.colorCameraDistortion)
												 
			if cornerCount > 0 and len(ids) > 0:
				print(f'Detected Corners: {cornerCount}, IDs: {len(ids)}')
				
				if self.showDetectedAruco:
					cv2.aruco.drawDetectedCornersCharuco(colorArray, corners, ids)
				
				success, rotationVectorWorldToCamera, translationVectorWorldToCamera = cv2.aruco.estimatePoseCharucoBoard(corners, ids, self.charucoBoard, 														self.realsense.colorCameraMatrix, self.realsense.colorCameraDistortion, None, None, False)
				
				if success:
					colorArray = cv2.aruco.drawAxis(colorArray, self.realsense.colorCameraMatrix, self.realsense.colorCameraDistortion, rotationVectorWorldToCamera, 		
										translationVectorWorldToCamera, 100.)
					
				minCorners = int((self.boardSize[0] - 1) * (self.boardSize[1] - 1) * 0.5) # as % of total
				
				if cornerCount >= minCorners and ids.size >= minCorners:
					
					rotationMatrixWorldToCamera, _ = cv2.Rodrigues(rotationVectorWorldToCamera)
					
					translationVectorWorldToCamera[2] = translationVectorWorldToCamera[2] + correction_z
					
					if debug:
						print('Rotation: ')
						print(rotationMatrixWorldToCamera)
						print('Translation: ')
						print(translationVectorWorldToCamera)
						print(f'Distance: {np.linalg.norm(translationVectorWorldToCamera)}')
					
					if self.showPoseEstimationCharuco:
						cv2.imshow('Estimated Pose', colorArray)
						key = cv2.waitKey(0)
						
					return True, rotationMatrixWorldToCamera, translationVectorWorldToCamera
					
				else:
					print('Not enough corners detected try again!')
					
					return False, None, None
					
	def saveNPoses(self, n: int = 20):
	
		rMat = []
		tVec = []
		
		for i in range(n):
		
			input('Press enter to capture next camera pose...')
			
			success, rotationMatrix, translationVector = cpe.estimatePose()
			
			if success:
				rMat.append(rotationMatrix)
				tVec.append(translationVector)
		
		Path('poses/').mkdir(exist_ok=True)
			
		with open(f'poses/worldToCameraRotation_{n}.npy', 'wb') as f:
			np.save(f, np.array(rMat))

		with open(f'poses/worldToCameraTranslation_{n}.npy', 'wb') as f:
			np.save(f, np.array(tVec))
	
	def getDepthToWorldTransform(self, debug: bool = False) -> RigidTransform:
		
		success, rMatWorldToCamera, tVecWorldToCamera = self.estimatePose(debug=False, correction_z = 0.050)
		
		if success:
			
			tfWorldToCamera = RigidTransform(rotation = rMatWorldToCamera, translation = tVecWorldToCamera, from_frame = 'world', to_frame = 'realsense_color')
			tfCameraToDepth = RigidTransform(rotation = self.realsense.depthToColorRotation, translation = self.realsense.depthToColorTranslation, 
								from_frame = 'realsense_color', to_frame = 'realsense_depth')
								
			tfWorldToDepth = tfCameraToDepth * tfWorldToCamera
			tfDepthToWorld = tfWorldToDepth.inverse()
			
			if debug:
				print(tfDepthToWorld)
			
			return tfDepthToWorld
		else:
			print('Failed to estimate pose, try again')
			return None
		
	def savePoseAsDepthToWorldTransform(self, path: str = '/media/max/gqcnn/gqcnn/data/calib/realsense/realsense_to_world.tf'):
		
		tfDepthToWorld = self.getDepthToWorldTransform()
		
		if tfDepthToWorld:
			
			tfDepthToWorld.save(path)	
			print('Saved Depth to World transform to ' + path)	
			
		else:
			print('Failed to estimate pose, try again')

		
if __name__ == '__main__':

	cpe = CharucoPoseEstimator(showPoseEstimationCharuco=False, depth=True)
	#_, rMat, tVec = cpe.estimatePose(debug=True)
	
	cpe.savePoseAsDepthToWorldTransform()	
