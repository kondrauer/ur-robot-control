import pyrealsense2 as rs

import numpy as np
import cv2
import time
import glob

from pathlib import Path

class RealsenseInterface:

	def __init__(self, color: bool = True, depth: bool = True, align: bool = False, decimation: bool = False, colorWidth: int = 1280, 
			colorHeight: int = 720, colorFPS: int = 30, depthWidth: int = 1024, depthHeight: int = 768, depthFPS: int = 30):
		
		# Stream settings
		self.color: bool = color
		self.depth: bool = depth
		self.align: bool = align
		self.decimation: bool = decimation
		
		# Color settings
		self.colorWidth: int = colorWidth
		self.colorHeight: int = colorHeight
		self.colorFPS: int = colorFPS
		self.colorFormat: rs.format = rs.format.bgr8
		
		# Depth settings
		self.depthWidth: int = depthWidth
		self.depthHeight: int = depthHeight
		self.depthFPS: int = depthFPS
		self.depthFormat: rs.format = rs.format.z16
		
		# Pipeline object
		self.pipeline: rs.pipeline = None
		
		# Config object
		self.config: rs.config = rs.config()
		
		# Profile objects
		self.profile: rs.profile = None
		self.depthProfile: rs.video_stream_profile = None
		self.colorProfile: rs.video_stream_profile = None
		
		
		# Filter objects
		self.filterHoleFilling: rs.hole_filling_filter = rs.hole_filling_filter()
		self.filterTemporal: rs.temporal_filter = rs.temporal_filter()
		
		self.filterSpatial: rs.spatial_filter = rs.spatial_filter()
		self.filterSpatial.set_option(rs.option.filter_magnitude, 3)
		self.filterSpatial.set_option(rs.option.filter_smooth_alpha, 1)
		self.filterSpatial.set_option(rs.option.filter_smooth_delta, 50)
		
		self.depthToDisparity: rs.disparity_transform = rs.disparity_transform(True)
		self.disparityToDepth: rs.disparity_transform = rs.disparity_transform(False)
		
		if self.decimation:
			self.filterDecimation: rs.decimation_filter = rs.decimation_filter()
			self.filterDecimation.set_option(rs.option.filter_magnitude, 2)
		
		if self.align:
			self.align: rs.align = rs.align(rs.stream.color)
		
		# Intrinsics
		self.depthIntrinsics: rs.intrinsics = None
		self.colorIntrinsics: rs.intrinsics = None
		
		# Extrinsics between color and depth
		self.depthToColorExtrinsics: rs.extrinsics = None
		self.colorToDepthExtrinsics: rs.extrinsics = None
		
		self.depthToColorRotation: np.array = None
		self.depthToColorTranslation: np.array = None
		
		self.colorToDepthRotation: np.array = None
		self.colorToDepthTranslation: np.array = None
		
		# Setup config
		if self.color:
			self.config.enable_stream(rs.stream.color, width = self.colorWidth, height = self.colorHeight, format = self.colorFormat, framerate = self.colorFPS)
		if self.depth:
			self.config.enable_stream(rs.stream.depth, width = self.depthWidth, height = self.depthHeight, format = self.depthFormat, framerate = self.depthFPS)
		
	def __del__(self):
		
		cv2.destroyAllWindows()
		self.pipeline.stop()
		ctx = rs.context()
		devices = ctx.query_devices()
		for dev in devices:
			dev.hardware_reset()
		
	def start(self, filter: bool = False) -> None:
		
		while True:
			try:
				self.pipeline = rs.pipeline()
				self.profile = self.pipeline.start(self.config)
				
				depth_sensor = self.profile.get_device().first_depth_sensor()
				preset_range = depth_sensor.get_option_range(rs.option.visual_preset)

				for i in range(int(preset_range.max)):
					visulpreset = depth_sensor.get_option_value_description(rs.option.visual_preset,i)
					print('%02d: %s' %(i,visulpreset))
					if visulpreset == "Low Ambient Light":
						depth_sensor.set_option(rs.option.visual_preset, i)
				
				for _ in range(15):
					self.pipeline.wait_for_frames()
				
				self.getProfiles()
				self.getIntrinsics()
				
				if filter:
					depth_sensor: rs.depth_sensor = self.profile.get_device().first_depth_sensor()
					depth_sensor.set_option(rs.option.confidence_threshold, 1)
					depth_sensor.set_option(rs.option.noise_filtering, 4)
				
				if self.color and self.depth:
					self.getExtrinsics()
				
				print('Camera start successfull! :-)')
				break
				
			except Exception as e:
			
				print(e)
				print('Error occured, stopping pipeline and resetting Hardware now...')
				# Stopping pipeline
				self.pipeline.stop()
				# Hardware reset the device(s) if anything doesn't work
				ctx = rs.context()
				devices = ctx.query_devices()
				for dev in devices:
					dev.hardware_reset()
				# Wait some time for USB to reconnect
				time.sleep(4)
	
	def getProfiles(self) -> None:
		
		if self.depth:
			self.depthProfile: rs.video_stream_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.depth))	
			
		if self.color:
			self.colorProfile: rs.video_stream_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.color))
	
	def getIntrinsics(self) -> None:
		
		if self.depth:
			self.depthIntrinsics = self.depthProfile.get_intrinsics()
			self.depthCameraMatrix: np.array = np.array([[self.depthIntrinsics.fx, 0, self.depthIntrinsics.ppx], 
									[0, self.depthIntrinsics.fy, self.depthIntrinsics.ppy],
									[0, 0, 1]])
			self.depthCameraDistortion: np.array = np.array(self.depthIntrinsics.coeffs).reshape(1, -1)
		if self.color:
			self.colorIntrinsics = self.colorProfile.get_intrinsics()
			self.colorCameraMatrix: np.array = np.array([[self.colorIntrinsics.fx, 0, self.colorIntrinsics.ppx], 
									[0, self.colorIntrinsics.fy, self.colorIntrinsics.ppy],
									[0, 0, 1]])
			self.colorCameraDistortion: np.array = np.array(self.colorIntrinsics.coeffs).reshape(1, -1)
	
		
				    
	def getExtrinsics(self) -> None:
		
		if self.depth and self.color:
			self.depthToColorExtrinsics = self.depthProfile.get_extrinsics_to(self.colorProfile)
			self.depthToColorRotation = np.array(self.depthToColorExtrinsics.rotation).reshape(3,3)
			self.depthToColorTranslation = np.array(self.depthToColorExtrinsics.translation).reshape(3,1)
			
			self.colorToDepthExtrinsics = self.depthProfile.get_extrinsics_to(self.depthProfile)
			self.colorToDepthRotation = np.array(self.colorToDepthExtrinsics.rotation).reshape(3,3)
			self.colorToDepthTranslation = np.array(self.colorToDepthExtrinsics.translation).reshape(3,1)

	
	def printIntrinsics(self) -> None:
		
		if self.color:
			print('Camera matrix:')
			print(self.colorCameraMatrix)
			print('Distortion model:')
			print(self.colorCameraDistortion)
		
		if self.depth:
			print('Depth Intrinsics:')
			print(self.depthCameraMatrix)
		
	
	def printExtrinsics(self) -> None:	
		
		if self.color and self.depth:
			print('Rotation depth -> color:') 
			print(self.depthToColorRotation)
			print('Translation depth -> color:') 
			print(self.depthToColorTranslation*1000)
		else:
			print('This only works with color and depth both enabled!')
		
	def saveIntrinsics(self) -> None:
		
		Path('calibration/').mkdir(exist_ok=True)
		
		with open('calibration/cameraMatrix.npy', 'wb') as f:
			np.save(f, self.colorCameraMatrix)
		with open('calibration/cameraDistortion.npy', 'wb') as f:
			np.save(f, self.colorCameraDistortion)
		
		print('Intrinsics saved!')
	
	def saveExtrinsics(self) -> None:
		
		Path('calibration/').mkdir(exist_ok=True)
		
		with open('calibration/rotationDepthToColor.npy', 'wb') as f:
			np.save(f, self.depthToColorRotation)
		with open('calibration/translationDepthToColor.npy', 'wb') as f:
			np.save(f, self.depthToColorTranslation)
		
		print('Extrinsics saved!')
	
	def getFrames(self, filter: bool = False, frameCount: int = 5) -> None:
		
		if filter:
			frames: list = []

			for x in range(frameCount):
			    frameset = self.pipeline.wait_for_frames()
			    frames.append(frameset.get_depth_frame())

			self.colorFrame = frameset.get_color_frame()
			
			for x in range(frameCount):
				self.frame = frames[x]
				
				if self.decimation:
					self.frame = self.filterDecimation.process(self.frame)
				self.frame = self.depthToDisparity.process(self.frame)
				self.frame = self.filterSpatial.process(self.frame)
				self.frame = self.filterTemporal.process(self.frame)
				self.frame = self.disparityToDepth.process(self.frame)
				self.frame = self.filterHoleFilling.process(self.frame)
			
			if self.align:
			    	self.frame = self.align.process(frameset)
			    	self.depthFrame = self.frame.get_depth_frame()
			else:   	
				self.depthFrame = self.frame
		else:
			self.frame: rs.frame = self.pipeline.wait_for_frames()
			
			if self.align:
				self.frame = self.align.process(self.frame)
			if self.color:
				self.colorFrame = self.frame.get_color_frame()
			if self.depth:
				self.depthFrame = self.frame.get_depth_frame()
	
	def showStreams(self) -> None:
		
		realsense.getFrames()
		
		if self.color:
			cv2.imshow('Color frame', np.array(self.colorFrame.get_data()))
		if self.depth:
			coloredDepth = cv2.applyColorMap(cv2.convertScaleAbs(np.array(self.depthFrame.get_data()), alpha=0.03), cv2.COLORMAP_JET)
			cv2.imshow('Depth frame', coloredDepth)
		
		cv2.waitKey(0)
	
	def openViewer(self, filter: bool = False) -> None:
		while True:
	
			self.getFrames(filter = filter)
			
			if self.color:
				colorImage = np.array(self.colorFrame.get_data())
			if self.depth:
				depthArray = np.array(self.depthFrame.get_data())
				coloredDepth = cv2.applyColorMap(cv2.convertScaleAbs(depthArray, alpha=255/depthArray.max()), cv2.COLORMAP_JET)
			
			images = np.hstack((colorImage, coloredDepth))
			
			# Show images
			cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
			cv2.imshow('RealSense', images)
			key = cv2.waitKey(1)
			if key & 0xFF == ord('q') or key == 27:
				cv2.destroyAllWindows()
				break
	
	def saveImageSet(self, iterationsDilation: int = 3, filter: bool = False) -> None:
	
		Path('img/').mkdir(exist_ok=True)
		self.getFrames(filter = filter)
		
		pngCount = int(len(glob.glob1('img/',"*.png"))/3)
		colorArray = np.array(self.colorFrame.get_data())
		
		cv2.imwrite(f'img/color_{pngCount}.png', colorArray)
		
		depthArray = np.array(self.depthFrame.get_data())
		coloredDepth = cv2.applyColorMap(cv2.convertScaleAbs(depthArray, alpha=255/depthArray.max()), cv2.COLORMAP_JET)
		
		cv2.imwrite(f'img/depth_{pngCount}.png', coloredDepth)
		with open(f'img/depth_{pngCount}.npy', 'wb') as f:
    			np.save(f, np.array(self.depthFrame.get_data()))
    			
		lowerGreen = np.array([0, 40, 0])
		upperGreen = np.array([120, 255, 100])
		
		midX = int(colorArray.shape[1]/2)
		midY = int(colorArray.shape[0]/2)
		
		mask = cv2.inRange(colorArray[midY-250:midY+250, midX-250:midX+250], lowerGreen, upperGreen)
		
		kernel = np.ones((3,3), np.uint8)
		mask = cv2.dilate(mask, kernel, iterations = iterationsDilation)
		mask = cv2.medianBlur(mask, 3)
		
		binary = np.zeros(colorArray.shape[:2])
		binary[midY-250:midY+250, midX-250:midX+250] = ~mask
		cv2.imwrite(f'img/segmask_{pngCount}.png', binary)
				
if __name__ == '__main__':

	realsense = RealsenseInterface(align=True, decimation=False)
	realsense.start()
	
	realsense.printIntrinsics()
	realsense.openViewer(filter=True)
	realsense.saveImageSet(iterationsDilation = 0, filter=True)	
