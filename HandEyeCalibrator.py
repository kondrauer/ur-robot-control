import numpy as np
import cv2

import PoseEstimator as pe
import Server 

from autolab_core import RigidTransform

from pathlib import Path

class HandEyeCalibrator:

	def __init__(self, numberOfMeasurements: int = 20):
		
		if numberOfMeasurements <= 0:
		
			raise ValueError('Number of measurements needs to be > 0')
		
		elif numberOfMeasurements < 3:
		
			print('At least three measurements are required. Set it to 3')
			self.numberOfMeasurements = 3
			
		else:
			self.numberOfMeasurements = numberOfMeasurements
		
		self.tfBaseToWorld = None
		self.tfGripperToCam = None
		
		self.loadCalibrationFromFile()
		
		self.poseEstimator: pe.CharucoPoseEstimator = pe.CharucoPoseEstimator(showPoseEstimationCharuco=False)
		self.tfDepthToWorld: RigidTransform = self.poseEstimator.getDepthToWorldTransform()
		self.server = Server.Server()
		
	def calibrateHandEye(self):
		try:
		
			self.server.commSocket, self.server.commAddress = self.server.establishConnection()
			
			listRMatGripperToBase: list = []
			listTVecGripperToBase: list = []
			listRMatWorldToCamera: list = []
			listTVecWorldToCamera: list = []
			
			for i in range(self.numberOfMeasurements):
				
				print(f'Step {i+1} of {numberOfMeasurements}')
				
				gripperToBase = self.server.receiveData(1024)
				gripperToBase = gripperToBase[2:len(gripperToBase)]
				
				gripperToBase = np.fromstring(gripperToBase, dtype = np.float64, sep = ',')
				rVecGripperToBase = gripperToBase[3:6].reshape(3,1)
				tVecGripperToBase = gripperToBase[0:3].reshape(3,1)
				
				rMatGripperToBase, _ = cv2.Rodrigues(rVecGripperToBase)
				
				print('Gripper to base of robot transformation determined')
				
				success, rMatWorldToCamera, tVecWorldToCamera = self.poseEstimator.estimatePose()
				
				if success:
					print('World to camera transformation determined')
					print('Saving transformations')
					
					listRMatGripperToBase.append(rMatGripperToBase)
					listTVecGripperToBase.append(tVecGripperToBase)
					
					listRMatWorldToCamera.append(rMatWorldToCamera)
					listTVecWorldToCamera.append(tVecWorldToCamera)
				else:
					print('Pose estimation of camera failed, skipping this step')
					
				communication = 1
                		self.serverData.send(str(communication))
					
			Path('handEyeCalibration/').mkdir(exist_ok=True)
			
			with open(f'handEyeCalibration/rMatGripperToBase.npy', 'wb') as f:
				np.save(f, np.array(listRMatGripperToBase)) 
			with open(f'handEyeCalibration/tVecGripperToBase.npy', 'wb') as f:
				np.save(f, np.array(listTVecGripperToBase)) 
			with open(f'handEyeCalibration/rMatWorldToCamera.npy', 'wb') as f:
				np.save(f, np.array(listRMatWorldToCamera)) 
			with open(f'handEyeCalibration/tVecWorldToCamera.npy', 'wb') as f:
				np.save(f, np.array(listTVecWorldToCamera)) 
				
			
			rMatBaseToWorld, tVecBaseToWorld, rMatGripperToCam, tVecGripperToCam = cv2.calibrateRobotWorldHandEye(np.array(listRMatGripperToBase), 
																	np.array(listTVecGripperToBase),
																	np.array(listRMatWorldToCamera),
																	np.array(listTVecWorldToCamera))		
			temp: np.array = np.array([0,0,0,1])
			
			self.tfBaseToWorld: RigidTransform = RigidTransform(rotation = rMatBaseToWorld, translation = tVecBaseToWorld, from_frame = 'base', to_frame = 'world')
			self.tfGripperToCam: RigidTransform = RigidTransform(rotation = rMatGripperToCam, translation = tVecGripperToCam, from_frame = 'gripper', to_frame = 'cam')
																		 
			self.tfBaseToWorld.save('handEyeCalibration/baseToWorld.tf')
			self.tfGripperToCam.save('handEyeCalibration/GripperToCam.tf')
			
			self.server.closeConnection()
			
			return True
			
		except Exception as e:
			
			print(e)
			return False
	
	def loadCalibrationFromFile(self, path: str = 'handEyeCalibration/'):
		
		p = Path(path)
		
		if p.exists():
			
			print('Loading...')
			self.tfBaseToWorld = RigidTransform.load('handEyeCalibration/baseToWorld.tf')
			self.tfGripperToCam = RigidTransform.load('handEyeCalibration/GripperToCam.tf')
			print('Hand-Eye-Calibration loaded')
			
		else:
			print('No saved Hand-Eye-Calibration found')
			
	def graspTransformer(self, tfGraspToDepth: RigidTransform):

		if self.tfBaseToWorld:
		
			tfGraspToBase =  self.tfBaseToWorld.inverse() * self.tfDepthToWorld * tfGraspToDepth 
			print(tfGraspToBase)
			
		else:
			print('Hand-Eye-Calibration needs to be done first!')
			
if __name__ == '__main__':

	hec = HandEyeCalibrator()
	hec.graspTransformer(None)			
		
		
	
