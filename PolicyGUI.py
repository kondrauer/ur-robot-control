# -*- coding: utf-8 -*-
"""
Created on Fri Jan 21 10:18:41 2022

@author: Maximilian Schlosser
"""

import tkinter as tk
import numpy as np
import glob
import logging
import queue

import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg)

from tkinter.scrolledtext import ScrolledText
from PIL import ImageTk, Image

from visualization import Visualizer2D as vis

from URPolicy import URPolicy
from RealsenseInterface import RealsenseInterface

root_logger = logging.getLogger()

class QueueHandler(logging.Handler):
    """Class to send logging records to a queue

    It can be used from different threads
    """

    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(record)

class Toolbar(tk.Frame):
	
	def __init__(self, parent, *args, **kwargs):
		tk.Frame.__init__(self, parent, *args, **kwargs)
		self.parent = parent
		
		menubar = tk.Menu(self.parent.parent)
		self.parent.parent.config(menu=menubar)
		
		fileMenu = tk.Menu(menubar, tearoff=0)
		fileMenu.add_command(label="Exit", command=self.parent.onExit)
		menubar.add_cascade(label="File", menu=fileMenu)
	
class VideoFeed(tk.LabelFrame):
	
	def __init__(self, parent, *args, **kwargs):
		tk.LabelFrame.__init__(self, parent, *args, **kwargs)
		self.parent = parent

		self.videoStream = tk.Label(self)
		self.videoStream.pack(padx=5, pady=5)
		
		self.getNewFrames(self.parent.parent.realsense)
			
	def getNewFrames(self, rs: RealsenseInterface):
		
		rs.getFrames()
		
		colorImage = np.array(rs.unalignedColor.get_data())
		
		#convert from bgr to rgb
		colorImage = colorImage[:,:,::-1]
		
		img = Image.fromarray(colorImage)
		img = img.resize((480, 270), Image.ANTIALIAS)
		imgtk = ImageTk.PhotoImage(image=img)
		self.videoStream.imgtk = imgtk
		self.videoStream.configure(image=imgtk)
		self.afterID = self.after(50, lambda: self.getNewFrames(self.parent.parent.realsense))
		
			
class GraspViewer(tk.LabelFrame):
	
	def __init__(self, parent, *args, **kwargs):
		tk.LabelFrame.__init__(self, parent, *args, **kwargs)
		self.parent = parent
		
		fig = plt.figure(figsize=(4,3))
		self.graspCanvas = FigureCanvasTkAgg(fig, master=self)
		self.graspCanvas.get_tk_widget().pack()
		self.informationLabel = tk.Label(self, text="Depth: 0.000m Quality: 0.00",
 								         bg="white", borderwidth=2, relief="groove")
		
		self.informationLabel.pack(side="bottom")
		
# 		self.img = Image.open(r'img\color_0.png')
# 		self.img = self.img.resize((320, 240), Image.ANTIALIAS)
# 		self.img = ImageTk.PhotoImage(self.img)
# 		self.graspCanvas.update()
# 		self.graspCanvas.create_image(self.graspCanvas.winfo_reqwidth()/2, self.graspCanvas.winfo_reqheight()/2, anchor="center", image=self.img)
			
	def update(self):
		
		self.img = self.parent.parent.urPolicy.rgbd_im.depth
		self.action = self.parent.parent.urPolicy.action
		
		plt.clf()
		plt.axis("off")
		plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
		plt.imshow(self.img.data, vmin=0.5, vmax=0.8, cmap=plt.cm.gray_r)
		vis.grasp(self.action.grasp, scale=2.5, show_center=False, show_axis=True)
		self.graspCanvas.draw()
		
		self.informationLabel.config(text=f"Depth: {self.action.grasp.depth:.3f}m Quality: {self.action.q_value:.2f}")
		
		
		
	
class Actions(tk.LabelFrame):
	
	def __init__(self, parent, *args, **kwargs):
		tk.LabelFrame.__init__(self, parent, *args, **kwargs)
		self.parent = parent
		
		self.planButton = tk.Button(self, text="Plan Grasp", width=20, command=self.planGrasp)
		self.executeButton = tk.Button(self, text="Execute Grasp", width=20, command=self.executeGrasp)
		
		self.planButton.pack(side="top", padx=15)
		self.executeButton.pack(side="top", padx=10)
		

		
	def planGrasp(self):
		self.parent.parent.urPolicy.planGrasp()
		self.parent.parent.upperFrame.graspViewer.update()
		self.parent.parent.upperFrame.depthSegmentationViewer.showSelected()

	def executeGrasp(self):
		self.parent.parent.urPolicy.executeGrasp()

		
class DepthSegmentationViewer(tk.LabelFrame):
	
	def __init__(self, parent, *args, **kwargs):
		tk.LabelFrame.__init__(self, parent, *args, **kwargs)
		self.parent = parent
		self.imageCanvas = tk.Canvas(self, bg="black")
		self.radioButtonFrame = tk.Frame(self)
		
		self.choices = ["color", "segmask", "depth"]
		
		self.selected = tk.IntVar()
		self.selected.set(0)
		
		for i, choice in enumerate(self.choices):
			tk.Radiobutton(self.radioButtonFrame, text=choice, variable=self.selected, command=self.showSelected, value=i).pack(side="left")
		
		self.radioButtonFrame.pack(side="bottom", anchor="center")
		self.imageCanvas.pack(side="top")
			
	def showSelected(self):
		i = self.selected.get()
		pngCount = int(len(glob.glob1('img/',"*.png"))/3) - 1 
		
		if i==0:
			self.img = Image.open(fr'img\color_{pngCount}.png')
		elif i==1:
			self.img = Image.open(fr'img\segmask_{pngCount}.png')
		elif i==2:
			self.img = Image.open(fr'img\depth_{pngCount}.png')
		
		self.updateCanvas()
		
	def updateCanvas(self):
		self.img = self.img.resize((320, 240), Image.ANTIALIAS)
		self.img = ImageTk.PhotoImage(self.img)
		self.imageCanvas.update()
		self.imageCanvas.create_image(self.imageCanvas.winfo_reqwidth()/2, self.imageCanvas.winfo_reqheight()/2, anchor="center", image=self.img)
	
			
class DebugLogger(tk.Frame):
	
	def __init__(self, parent, *args, **kwargs):
		tk.Frame.__init__(self, parent, *args, **kwargs)
		self.parent = parent
		
		self.scrolledText = ScrolledText(self, state="disabled", height=10, bg="black")
		self.scrolledText.tag_config('INFO', foreground='green')
		self.scrolledText.tag_config('DEBUG', foreground='cyan')
		self.scrolledText.tag_config('WARNING', foreground='yellow')
		self.scrolledText.tag_config('ERROR', foreground='red')
		self.scrolledText.tag_config('CRITICAL', foreground='red', underline=1)


		for hdlr in root_logger.handlers:
			if isinstance(hdlr, logging.StreamHandler):
				root_logger.removeHandler(hdlr)
		
		logging.root.name = 'Robotiklabor'
		
		self.logQueue = queue.Queue()
		self.queueHandler = QueueHandler(self.logQueue)
		self.formatter = logging.Formatter('%(name)s: %(levelname)-8s %(message)s')
		
		self.queueHandler.setFormatter(self.formatter)
		root_logger.addHandler(self.queueHandler)
		self.after(100, self.pollLogQueue)
		
		self.scrolledText.pack()
	
	def appendMessage(self, message):
		
		msg = self.queueHandler.format(message)
		self.scrolledText.configure(state="normal")
		self.scrolledText.insert(tk.END, msg + "\n", message.levelname)
		self.scrolledText.configure(state="disabled")
		self.scrolledText.yview(tk.END)
		
	def pollLogQueue(self):
		
		while True:
			try:
				message = self.logQueue.get(block=False)
			except queue.Empty:
				break
			else:
				self.appendMessage(message)
		
		self.afterID = self.after(100, self.pollLogQueue)
			
class UpperFrame(tk.Frame):
	
	def __init__(self, parent, *args, **kwargs):
		tk.Frame.__init__(self, parent, *args, **kwargs)
		self.parent = parent
		
		self.videoFeed = VideoFeed(self, text="Liveview Realsense")
		self.graspViewer = GraspViewer(self, text="Predicted Grasp")
		self.depthSegmentationViewer = DepthSegmentationViewer(self.videoFeed, text="Image Viewer")
		
		self.videoFeed.pack(side="left", fill="both")
		self.graspViewer.pack(side="left", fill="x", anchor=tk.N, padx=10, ipadx=10)
		self.depthSegmentationViewer.pack(side="bottom", fill="both", padx=10, pady=5)
	
class LowerFrame(tk.Frame):
	
	def __init__(self, parent, *args, **kwargs):
		tk.Frame.__init__(self, parent, *args, **kwargs)
		self.parent = parent
		
		self.actions = Actions(self, text="Actions")
		self.debugLogger = DebugLogger(self)
		
		self.actions.pack(side="left", anchor="center", padx=5, ipadx=10, ipady=20)
		self.debugLogger.pack(side="left", fill="x")
			
			

class PolicyApp(tk.Frame):
	
	def __init__(self, parent, *args, **kwargs):
		tk.Frame.__init__(self, parent, *args, **kwargs)
		self.parent = parent
		self.parent.title("Robotiklabor Dex-Net GUI")
		#self.parent.geometry("1280x720")
		
		self.toolbar = Toolbar(self)
		self.lowerFrame = LowerFrame(self)
		
		self.realsense = RealsenseInterface(align=True, decimation=False)
		self.realsense.start()
		
		self.urPolicy = URPolicy(re = self.realsense)
		
		self.upperFrame = UpperFrame(self)
		
		
		self.toolbar.pack(side = "top")
		
		self.upperFrame.pack(side = "top", anchor="center")
		self.lowerFrame.pack(side = "bottom", fill="y", anchor="center", pady=10)
		
	def onExit(self):
		self.upperFrame.videoFeed.after_cancel(self.upperFrame.videoFeed.afterID)
		self.lowerFrame.debugLogger.after_cancel(self.lowerFrame.debugLogger.afterID)
		self.parent.quit()
			
			
if __name__ == "__main__":
	
# 	logging.getLogger().setLevel(logging.DEBUG)
	logging.getLogger().setLevel(logging.INFO)
	
	root = tk.Tk()
	pa = PolicyApp(root)
	pa.pack(side="top", fill="both", expand=True)
	root.protocol("WM_DELETE_WINDOW", pa.onExit)
	root.mainloop()