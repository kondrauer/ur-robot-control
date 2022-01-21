# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 10:18:41 2022

@author: Maximilian Schlosser
"""

import tkinter as tk

class Toolbar(tk.Frame):
	
	def __init__(self, parent, *args, **kwargs):
			tk.Frame.__init__(self, parent, *args, **kwargs)
			self.parent = parent
			
			test = tk.Label(self, text="Toolbar")
			test.pack()

class VideoFeed(tk.Frame):
	
	def __init__(self, parent, *args, **kwargs):
			tk.Frame.__init__(self, parent, *args, **kwargs)
			self.parent = parent
			
			test = tk.Label(self, text="VideoFeed")
			test.pack()
			
class GraspViewer(tk.Frame):
	
	def __init__(self, parent, *args, **kwargs):
			tk.Frame.__init__(self, parent, *args, **kwargs)
			self.parent = parent
			
			test = tk.Label(self, text="GraspViewer")
			test.pack()

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
			self.graspViewer = GraspViewer(self)
			
			self.videoFeed.pack(side = "left")
			self.graspViewer.pack(side = "right")
	
class LowerFrame(tk.Frame):
	
	def __init__(self, parent, *args, **kwargs):
			tk.Frame.__init__(self, parent, *args, **kwargs)
			self.parent = parent
			
			self.actions = Actions(self)
			self.depthSegmentationViewer = DepthSegmentationViewer(self)
			
			self.actions.pack(side = "left")
			self.depthSegmentationViewer.pack(side = "right")

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
			self.upperFrame.pack(side = "top")
			self.debugLogger.pack(side = "bottom")
			self.lowerFrame.pack(side = "bottom")
			
			
			
if __name__ == "__main__":
	root = tk.Tk()
	PolicyApp(root).pack(side="top", fill="both", expand=True)
	root.mainloop()