import pyrealsense2 as rs

import numpy as np
import cv2
import time
import glob

import logging

from pathlib import Path

class RealsenseInterface:
	"""Class for interfacing with an intel realsense L515
	
	Attributes:
		color(bool): activate color stream if true
		depth(bool): activate depth stream if true
		align(bool): align color and depth streams if true
		decimation(bool): activate decimation filter (cuts image size to size/decimation_magnitude. i.e. 1280*720 becomes 640*360 with magnitude 2)
		
		colorWidth(int): width of color stream
		colorHeight(int): height of color stream
		colorFPS(int): frames per second of color stream
		colorFormat(rs.format): format of color stream
		
		depthWidth(int): width of depth stream
		depthHeight(int): height of depth stream
		depthFPS(int): frames per second of depth stream
		depthFormat(rs.format): format of depth stream
		
		pipeline(rs.pipeline): pipeline object for receiving frames
		config(rs.config): configuration for depth and color stream
		
		profile(rs.profile): helper object to get information about depth sensor (i.e. available presets)
		depthProfile(rs.video_stream_profile): holds information about depth stream
		colorProfile(rs.video_stream_profile): holds information about color stream
	
		filterHoleFilling(rs.hole_filling_filter): hole filling filter
		filterTemporal(rs.temporal_filter): temporal filter
		filterSpatial(rs.spatial_filter): spatial filter
		filterDecimation(rs.decimation_filter): decimation filter if decimation=True
		
		depthToDisparity(rs.disparity_transform): converts depth-frame to disparity
		disparityToDepth(rs.disparity_transform): converts disparity to depth-frame
		
		align(rs.align): object to align depth and color stream if align = True
		
		depthIntrinsics(rs.intrinsics): structure that holds intrinsic parameters of depth stream
		colorIntrinsics(rs.intrinsics): structure that holds intrinsic parameters of color stream
		
		depthToColorExtrinsics(rs.extrinsics): structure that holds transformation from depth to color stream
		colorToDepthExtrinsics(rs.extrinsics): structure that holds transformation from color to depth stream
		
		depthToColorRotation(np.array): rotation from depth to color
		depthToColorTranslation(np.array): translation from depth to color
		
		colorToDepthRotation(np.array): rotation from color to depth
		colorToDepthTranslation(np.array): translation from color to depth		
	"""
	
	def __init__(self, color: bool = True, depth: bool = True, align: bool = False, decimation: bool = False, colorWidth: int = 1280, 
			colorHeight: int = 720, colorFPS: int = 30, depthWidth: int = 640, depthHeight: int = 480, depthFPS: int = 30):
		"""Constructor for RealsenseInterface
		
		Args:
			color(bool): activate color stream if true
			depth(bool): activate depth stream if true
			align(bool): align color and depth streams if true
			decimation(bool): activate decimation filter (cuts image size to size/decimation_magnitude. i.e. 1280*720 becomes 640*360 with magnitude 2)
			
			colorWidth(int): width of color stream
			colorHeight(int): height of color stream
			colorFPS(int): frames per second of color stream
			
			depthWidth(int): width of depth stream
			depthHeight(int): height of depth stream
			depthFPS(int): frames per second of depth stream
			
		"""
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
		
		if self.decimation:
			self.filterDecimation: rs.decimation_filter = rs.decimation_filter()
			self.filterDecimation.set_option(rs.option.filter_magnitude, 2)
		
		self.depthToDisparity: rs.disparity_transform = rs.disparity_transform(True)
		self.disparityToDepth: rs.disparity_transform = rs.disparity_transform(False)

		if self.align:
			self.align: rs.align = rs.align(rs.stream.depth)
		
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
		"""Destructor for Realsense interface
		
		Makes sure the pipeline is stopped and hardware resets the device, seems to be the only way to take multiple captures without rebooting the system with the specific camera used.
		
		Note:
			Doesn't always work :-)
		"""	
		
		# Destroy any open opencv windows
		cv2.destroyAllWindows()
		
		# Stop pipeline
		self.pipeline.stop()
		
		# create context object and query devices connected to pc
		ctx = rs.context()
		devices = ctx.query_devices()
		
		# reset every connected realsense device (typically only 1)
		for dev in devices:
			dev.hardware_reset()
		
	def start(self, preprocessing: bool = False) -> None:
		"""Setup and start the cameras pipeline.
		
		Args:
			preprocessing(bool): whether or not to preprocess the depth stream
		
		Note:
			Doesnt always work :-)
		"""		
		while True:
			try:
				# create pipeline object and get its profile
				self.pipeline = rs.pipeline()
				self.profile = self.pipeline.start(self.config)
				
				# get depth sensor and its available presets
				depth_sensor = self.profile.get_device().first_depth_sensor()
				preset_range = depth_sensor.get_option_range(rs.option.visual_preset)
				
				# iterate over presets and select 'Low Ambient Light' one
				for i in range(int(preset_range.max)):
					visulpreset = depth_sensor.get_option_value_description(rs.option.visual_preset,i)
					#print('%02d: %s' %(i,visulpreset))
					if visulpreset == "Low Ambient Light":
						depth_sensor.set_option(rs.option.visual_preset, i)
				
				# wait some time for auto exposure
				for _ in range(15):
					self.pipeline.wait_for_frames()
				
				# get stream profiles and intrinsics
				self.getProfiles()
				self.getIntrinsics()
				
				# some more setup if wanted
				if preprocessing:
					depth_sensor: rs.depth_sensor = self.profile.get_device().first_depth_sensor()
					depth_sensor.set_option(rs.option.confidence_threshold, 1)
					depth_sensor.set_option(rs.option.noise_filtering, 4)
				
				# works only if both streams are activated
				if self.color and self.depth:
					self.getExtrinsics()
				
				# everything worked
				logging.info('Camera start successfull! :-)')
				break
				
			except Exception as e:
			
				print(e)
				logging.error('Error occured, stopping pipeline and resetting Hardware now...')
				
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
		"""Gets stream profile of activated streams

		"""			
		if self.depth:
			self.depthProfile: rs.video_stream_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.depth))	
			
		if self.color:
			self.colorProfile: rs.video_stream_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.color))
	
	def getIntrinsics(self) -> None:
		"""Gets intrinsics profile of activated streams and save it to respective attributes

		"""	
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
		"""Gets extrinsic transformations between activated streams and save it to respective attributes
		
		Note:
			Only works if color and depth are activated, otherwise there are no transformations necessary

		"""
		if self.depth and self.color:
			self.depthToColorExtrinsics = self.depthProfile.get_extrinsics_to(self.colorProfile)
			self.depthToColorRotation = np.array(self.depthToColorExtrinsics.rotation).reshape(3,3)
			self.depthToColorTranslation = np.array(self.depthToColorExtrinsics.translation).reshape(3,1)
			
			self.colorToDepthExtrinsics = self.depthProfile.get_extrinsics_to(self.depthProfile)
			self.colorToDepthRotation = np.array(self.colorToDepthExtrinsics.rotation).reshape(3,3)
			self.colorToDepthTranslation = np.array(self.colorToDepthExtrinsics.translation).reshape(3,1)
	
	def printIntrinsics(self) -> None:
		"""Prints intrinsics profile of activated streams to console

		"""
		if self.color:
			logging.info('Camera matrix:')
			logging.info(self.colorCameraMatrix)
			logging.info('Distortion model:')
			logging.info(self.colorCameraDistortion)
		
		if self.depth:
			logging.info('Depth Intrinsics:')
			logging.info(self.depthCameraMatrix)
		
	
	def printExtrinsics(self) -> None:	
		"""Prints extrinsic transformations between activated streams to console
		
		Note:
			Only works if color and depth are activated, otherwise there are no transformations necessary

		"""
		if self.color and self.depth:
			logging.info('Rotation depth -> color:') 
			logging.info(self.depthToColorRotation)
			logging.info('Translation depth -> color:') 
			logging.info(self.depthToColorTranslation)
		else:
			logging.warning('This only works with color and depth both enabled!')
		
	def saveIntrinsics(self) -> None:
		"""Saves intrinsics profile of activated streams to file

		"""
		Path('calibration/').mkdir(exist_ok=True)
		
		with open('calibration/cameraMatrix.npy', 'wb') as f:
			np.save(f, self.colorCameraMatrix)
		with open('calibration/cameraDistortion.npy', 'wb') as f:
			np.save(f, self.colorCameraDistortion)
		
		logging.info('Intrinsics saved!')
	
	def saveExtrinsics(self) -> None:
		"""Saves extrinsic transformations between activated streams to file
		
		Note:
			Only works if color and depth are activated, otherwise there are no transformations necessary

		"""
		if self.color and self.depth:
		
			Path('calibration/').mkdir(exist_ok=True)
			
			with open('calibration/rotationDepthToColor.npy', 'wb') as f:
				np.save(f, self.depthToColorRotation)
			with open('calibration/translationDepthToColor.npy', 'wb') as f:
				np.save(f, self.depthToColorTranslation)
			
			logging.info('Extrinsics saved!')
	
	def getFrames(self, filter: bool = False, frameCount: int = 5) -> None:
		"""Gets a set of frames from the pipeline and saves it to corresponding class attributes
		
		Args:
			filter(bool): whether or not to filter the depth frameset
			frameCount(int): how many depth-frames are taking for filtering
		
		"""
		if filter:
			frames: list = []
			
			# take frameCount frames
			for x in range(frameCount):
			    frameset = self.pipeline.wait_for_frames()
			    frames.append(frameset.get_depth_frame())

			# apply filters successively
			for x in range(frameCount):
				self.frame = frames[x]
				
				if self.decimation:
					self.frame = self.filterDecimation.process(self.frame)
					
				self.frame = self.depthToDisparity.process(self.frame)
				self.frame = self.filterSpatial.process(self.frame)
				self.frame = self.filterTemporal.process(self.frame)
				self.frame = self.disparityToDepth.process(self.frame)
				self.frame = self.filterHoleFilling.process(self.frame)
			
			# align the streams if true
			if self.align:
			    	self.frame = self.align.process(frameset)
			    	self.colorFrame = self.frame.get_color_frame()
			    	self.depthFrame = self.frame.get_depth_frame()
			else:   	
				self.depthFrame = self.frame
		else:
			# just save one frame set if no filters are applied
			self.frame: rs.frame = self.pipeline.wait_for_frames()
			
			if self.align:
				self.frame = self.align.process(self.frame)
			if self.color:
				self.colorFrame = self.frame.get_color_frame()
			if self.depth:
				self.depthFrame = self.frame.get_depth_frame()
	
	def showStreams(self) -> None:
		"""Shows a picture of the activated streams and waits for user input to close it/them.

		"""
		realsense.getFrames()
		
		if self.color:
			cv2.imshow('Color frame', np.array(self.colorFrame.get_data()))
		if self.depth:
			coloredDepth = cv2.applyColorMap(cv2.convertScaleAbs(np.array(self.depthFrame.get_data()), alpha=0.03), cv2.COLORMAP_JET)
			cv2.imshow('Depth frame', coloredDepth)
		
		cv2.waitKey(0)
	
	def openViewer(self, filter: bool = False) -> None:
		"""Shows the activated streams in an opencv window as video.
		
		Args:
			filter(bool): filters the depth stream if true

		"""
		
		if self.align:
			while True:
		
				self.getFrames(filter = filter)
				
				if self.color:
					colorImage = np.array(self.colorFrame.get_data())
				if self.depth:
					depthArray = np.array(self.depthFrame.get_data())
					# apply colormap to depth frame so it looks pretty
					coloredDepth = cv2.applyColorMap(cv2.convertScaleAbs(depthArray, alpha=255/depthArray.max()), cv2.COLORMAP_JET)
				
				# stack the pictures next to each other
				images = np.hstack((colorImage, coloredDepth))
				
				# show images
				cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
				cv2.imshow('RealSense', images)
				key = cv2.waitKey(1)
				
				# close if q or esc are pressed
				if key & 0xFF == ord('q') or key == 27:
					cv2.destroyAllWindows()
					break
		else:
			logging.error('Streams need to be aligned for the viewer to work!')
	
	def saveImageSet(self, iterationsDilation: int = 3, filter: bool = False) -> None:
		"""Saves an image set for later use with a gqcnn.
		
		This function saves 3 images: The color frame for reference, a depth frame in .npy and .png format and a segmask.
		
		Note:
			A Green background is required so that this function works properly. Otherwise the function needs to be altered to work.
		
		Args:
			iterationsDilation(int): how often the dilation operation takes place
			filter(bool): filters the depth image if true

		"""
		# create folder to save the pictures
		Path('img/').mkdir(exist_ok=True)
		pngCount = int(len(glob.glob1('img/',"*.png"))/3)
		
		self.getFrames(filter = filter)
		
		# turn color frame to array and save it
		colorArray = np.array(self.colorFrame.get_data())
		cv2.imwrite(f'img/color_{pngCount}.png', colorArray)
		
		# turn depth frame into array and apply color map
		depthArray = np.array(self.depthFrame.get_data())
		coloredDepth = cv2.applyColorMap(cv2.convertScaleAbs(depthArray, alpha=255/depthArray.max()), cv2.COLORMAP_JET)
		
		# saved colored depth as .png and .npy
		cv2.imwrite(f'img/depth_{pngCount}.png', coloredDepth)
		with open(f'img/depth_{pngCount}.npy', 'wb') as f:
    			np.save(f, np.array(self.depthFrame.get_data()))
    		
    		# this can be tuned, depends on lighting conditions	
		lowerGreen = np.array([0, 40, 0])
		upperGreen = np.array([120, 255, 100])
		
		# get mid coordinates of picture
		midX = int(colorArray.shape[1]/2)
		midY = int(colorArray.shape[0]/2)
		
		# mask around the middle of the picture, makes it easier if the table is not filling the whole picture
		# can be altered if the whole background is green
		mask = cv2.inRange(colorArray[midY-250:midY+250, midX-250:midX+250], lowerGreen, upperGreen)
		
		# dilation and median blur to smooth mask and fill holes
		kernel = np.ones((3,3), np.uint8)
		mask = cv2.dilate(mask, kernel, iterations = iterationsDilation)
		mask = cv2.medianBlur(mask, 3)
		
		# close any remaining holes by just ignoring them if they are to far from the center of the picture
		binary = np.zeros(colorArray.shape[:2])
		binary[midY-250:midY+250, midX-250:midX+250] = ~mask
		cv2.imwrite(f'img/segmask_{pngCount}.png', binary)
				
if __name__ == '__main__':

	realsense = RealsenseInterface(align=True, decimation=False)
	realsense.start()
	
	realsense.printIntrinsics()
	realsense.openViewer(filter=True)
	realsense.saveImageSet(iterationsDilation = 0, filter=True)	
