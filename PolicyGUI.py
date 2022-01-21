# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 10:18:41 2022

@author: Maximilian Schlosser
"""

import tkinter as tk
import numpy as np

from PIL import ImageTk, Image

from RealsenseInterface import RealsenseInterface

class Toolbar(tk.Frame):
	
	def __init__(self, parent, *args, **kwargs):
		tk.Frame.__init__(self, parent, *args, **kwargs)
		self.parent = parent
		
		menubar = tk.Menu(self.parent.parent)
		self.parent.parent.config(menu=menubar)
		
		fileMenu = tk.Menu(menubar, tearoff=0)
		fileMenu.add_command(label="Exit", command=self.parent.onExit)
		menubar.add_cascade(label="File", menu=fileMenu)
		


class VideoFeed(tk.Frame):
	
	def __init__(self, parent, *args, **kwargs):
		tk.Frame.__init__(self, parent, *args, **kwargs)
		self.parent = parent
		
		self.realsense = RealsenseInterface(align=True)
		self.realsense.start()
		
		self.videoStream = tk.Label(self)
		self.videoStream.pack(padx=5, pady=5)
		
		self.getNewFrames(self.realsense)
			
	def getNewFrames(self, rs: RealsenseInterface):
		
		rs.getFrames()
		
		colorImage = np.array(rs.colorFrame.get_data())
		
		#convert from bgr to rgb
		colorImage = colorImage[:,:,::-1]
		
		img = Image.fromarray(colorImage)
		img = img.resize((320, 240), Image.ANTIALIAS)
		imgtk = ImageTk.PhotoImage(image=img)
		self.videoStream.imgtk = imgtk
		self.videoStream.configure(image=imgtk)
		self.afterID = self.after(1, lambda: self.getNewFrames(self.realsense))
		
			
class GraspViewer(tk.LabelFrame):
	
	def __init__(self, parent, *args, **kwargs):
			tk.LabelFrame.__init__(self, parent, *args, **kwargs)
			self.parent = parent
			
			self.graspCanvas = tk.Canvas(self)
			self.informationLabel
			self.graspCanvas.pack(fill="both")
			
			self.img = Image.open(r'img\color_0.png')
			self.img = self.img.resize((320, 240), Image.ANTIALIAS)
			self.img = ImageTk.PhotoImage(self.img)
			self.graspCanvas.create_image((0, 0), anchor=tk.NW, image=self.img)
			
class Actions(tk.Frame):
	
	def __init__(self, parent, *args, **kwargs):
			tk.Frame.__init__(self, parent, *args, **kwargs)
			self.parent = parent
			
			test = tk.Label(self, text="Actions")
			test.pack()
			
class DepthSegmentationViewer(tk.Frame):
	
	def __init__(self, parent, *args, **kwargs):
			tk.Frame.__init__(self, parent, *args, **kwargs)
			self.parent = parent
			
			test = tk.Label(self, text="DepthSegmentationViewer")
			test.pack()
			
class DebugLogger(tk.Frame):
	
	def __init__(self, parent, *args, **kwargs):
			tk.Frame.__init__(self, parent, *args, **kwargs)
			self.parent = parent
			
			test = tk.Label(self, text="DebugLogger")
			test.pack()

class UpperFrame(tk.Frame):
	
	def __init__(self, parent, *args, **kwargs):
			tk.Frame.__init__(self, parent, *args, **kwargs)
			self.parent = parent
			
			self.videoFeed = VideoFeed(self)
			self.graspViewer = GraspViewer(self, text="Predicted Grasp")
			
			self.videoFeed.pack(side = "left", fill=tk.X)
			self.graspViewer.pack(side = "right", fill=tk.X)
	
class LowerFrame(tk.Frame):
	
	def __init__(self, parent, *args, **kwargs):
			tk.Frame.__init__(self, parent, *args, **kwargs)
			self.parent = parent
			
			self.actions = Actions(self)
			self.depthSegmentationViewer = DepthSegmentationViewer(self)
			
			self.actions.pack(side="left")
			self.depthSegmentationViewer.pack(side="right")

class PolicyApp(tk.Frame):
	
		def __init__(self, parent, *args, **kwargs):
			tk.Frame.__init__(self, parent, *args, **kwargs)
			self.parent = parent
			self.parent.title("Robotiklabor Dex-Net GUI")
			self.parent.geometry("1000x600")
			
			self.toolbar = Toolbar(self)
			self.upperFrame = UpperFrame(self)
			self.lowerFrame = LowerFrame(self)
			self.debugLogger = DebugLogger(self)
			
			
			self.toolbar.pack(side = "top")
			self.upperFrame.pack(side = "top", fill=tk.X, padx=5)
			self.debugLogger.pack(side = "bottom")
			self.lowerFrame.pack(side = "bottom", fill=tk.X)
			
		def onExit(self):
			
			self.upperFrame.videoFeed.after_cancel(self.upperFrame.videoFeed.afterID)
			self.parent.destroy()
			
			
if __name__ == "__main__":
	root = tk.Tk()
	pa = PolicyApp(root)
	pa.pack(side="top", fill="both", expand=True)
	root.protocol("WM_DELETE_WINDOW", pa.onExit)
	root.mainloop()