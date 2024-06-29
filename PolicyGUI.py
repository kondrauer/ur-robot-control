import tkinter as tk
import numpy as np
import glob
import logging
import queue

import os

import matplotlib.pyplot as plt

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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

        # convert from bgr to rgb
        colorImage = colorImage[:, :, ::-1]

        img = Image.fromarray(colorImage)
        img = img.resize((480, 270), Image.ANTIALIAS)
        imgtk = ImageTk.PhotoImage(image=img)
        self.videoStream.imgtk = imgtk
        self.videoStream.configure(image=imgtk)
        self.afterID = self.after(
            50, lambda: self.getNewFrames(self.parent.parent.realsense)
        )


class GraspViewer(tk.LabelFrame):
    def __init__(self, parent, *args, **kwargs):
        tk.LabelFrame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        fig = plt.figure(figsize=(4, 3))
        self.graspCanvas = FigureCanvasTkAgg(fig, master=self)
        self.graspCanvas.get_tk_widget().pack(pady=(6, 0))
        self.informationLabel = tk.Label(
            self,
            text="Depth: 0.000m Quality: 0.00",
            bg="white",
            borderwidth=2,
            relief="groove",
        )

        self.informationLabel.pack(side="bottom", pady=5)

    def update(self):
        self.urPolicy = self.parent.parent.parent.lowerFrame.actions.urPolicy
        self.img = self.urPolicy.rgbd_im.depth
        self.action = self.urPolicy.action

        plt.clf()
        plt.axis("off")
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        plt.imshow(self.img.data, vmin=0.5, vmax=0.8, cmap=plt.cm.gray_r)
        vis.grasp(self.action.grasp, scale=2.5, show_center=False, show_axis=True)
        self.graspCanvas.draw()

        self.informationLabel.config(
            text=f"Depth: {self.action.grasp.depth:.3f}m Quality: {self.action.q_value:.2f}"
        )


class Actions(tk.LabelFrame):
    def __init__(self, parent, *args, **kwargs):
        tk.LabelFrame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        # 		self.calibrationButton = tk.Button(self, text="Hand-Eye-Calibration", width=20, command=self.calibrateHandEye)
        self.planButton = tk.Button(
            self, text="Plan Grasp", width=20, command=self.planGrasp
        )
        self.executeButton = tk.Button(
            self, text="Execute Grasp", width=20, command=self.executeGrasp
        )

        # 		self.calibrationButton.pack(side="top", padx=15)
        self.planButton.pack(side="top", padx=15)
        self.executeButton.pack(side="top", padx=10)

        self.urPolicy = None

    # 		self.hec = None

    # 	def calibrateHandEye(self):
    #
    # 		self.hec = self.parent.parent.urPolicy.hec
    # 		self.hec.calibrateHandEye()

    def planGrasp(self):
        self.urPolicy = self.parent.parent.urPolicy

        self.urPolicy.planGrasp()
        self.parent.parent.upperFrame.rightFrame.graspViewer.update()
        self.parent.parent.upperFrame.depthSegmentationViewer.showSelected()

    def executeGrasp(self):
        if not self.urPolicy:
            logging.warning("You have to Plan a Grasp first!")
        else:
            self.options = self.parent.parent.upperFrame.rightFrame.options

            rotMat = self.options.rotMatEntry.get("1.0", "end-1c")

            rotMat = rotMat.replace("[", "")
            rotMat = rotMat.replace("]", "")
            rotMat = rotMat.replace("\n", " ")
            rotMat = rotMat.replace(" ", "")

            rotMat = np.array(rotMat.split(","), dtype=np.float64).reshape(3, 3)

            self.parent.parent.urPolicy.executeGrasp(
                corrX=float(self.options.corrXVar.get()),
                corrY=float(self.options.corrYVar.get()),
                corrZ=float(self.options.corrZVar.get()),
                rotMat=rotMat,
                safetyVal=float(self.options.safetyValVar.get()),
            )


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
            tk.Radiobutton(
                self.radioButtonFrame,
                text=choice,
                variable=self.selected,
                command=self.showSelected,
                value=i,
            ).pack(side="left")

        self.radioButtonFrame.pack(side="bottom", anchor="center")
        self.imageCanvas.pack(side="top")

    def showSelected(self):
        i = self.selected.get()
        pngCount = int(len(glob.glob1("img/", "*.png")) / 3) - 1

        if i == 0:
            self.img = Image.open(rf"img\color_{pngCount}.png")
        elif i == 1:
            self.img = Image.open(rf"img\segmask_{pngCount}.png")
        elif i == 2:
            self.img = Image.open(rf"img\depth_{pngCount}.png")

        self.updateCanvas()

    def updateCanvas(self):
        self.img = self.img.resize((320, 240), Image.ANTIALIAS)
        self.img = ImageTk.PhotoImage(self.img)
        self.imageCanvas.update()
        self.imageCanvas.create_image(
            self.imageCanvas.winfo_reqwidth() / 2,
            self.imageCanvas.winfo_reqheight() / 2,
            anchor="center",
            image=self.img,
        )


class DebugLogger(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.scrolledText = ScrolledText(self, state="disabled", height=10, bg="black")
        self.scrolledText.tag_config("INFO", foreground="green")
        self.scrolledText.tag_config("DEBUG", foreground="cyan")
        self.scrolledText.tag_config("WARNING", foreground="yellow")
        self.scrolledText.tag_config("ERROR", foreground="red")
        self.scrolledText.tag_config("CRITICAL", foreground="red", underline=1)

        for hdlr in root_logger.handlers:
            if isinstance(hdlr, logging.StreamHandler):
                root_logger.removeHandler(hdlr)

        logging.root.name = "Robotiklabor"

        self.logQueue = queue.Queue()
        self.queueHandler = QueueHandler(self.logQueue)
        self.formatter = logging.Formatter("%(name)s: %(levelname)-8s %(message)s")

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


class Options(tk.LabelFrame):
    def __init__(self, parent, *args, **kwargs):
        tk.LabelFrame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.corrX = tk.Label(self, text="Correction X-Axis (in mm): ")
        self.corrY = tk.Label(self, text="Correction Y-Axis (in mm): ")
        self.corrZ = tk.Label(self, text="Correction Z-Axis (in mm): ")
        self.rotMat = tk.Label(self, text="Rotation Matrix (3x3): ")
        self.safetyVal = tk.Label(self, text="Safety Z-Value (in mm): ")
        self.modelPath = tk.Label(self, text="Model Path: ")
        self.cfgPath = tk.Label(self, text="Policy Config Path: ")
        self.intrPath = tk.Label(self, text="Camera Intrinsics Path: ")
        self.hecPath = tk.Label(self, text="Hand Eye Calibration Path: ")
        self.ip = tk.Label(self, text="IP: ")
        self.port = tk.Label(self, text="Port: ")

        self.corrXVar = tk.StringVar(self, value="0.000")
        self.corrXEntry = tk.Entry(self, textvariable=self.corrXVar, width=5)
        self.corrYVar = tk.StringVar(self, value="0.05")
        self.corrYEntry = tk.Entry(self, textvariable=self.corrYVar, width=5)
        self.corrZVar = tk.StringVar(self, value="0.03")
        self.corrZEntry = tk.Entry(self, textvariable=self.corrZVar, width=5)

        self.rotMatEntry = tk.Text(self, height=3, width=20)
        self.rotMatEntry.insert(tk.END, "[[0, 0, -1],\n[0, 1, 0],\n[1, 0, 0]]")

        self.safetyValVar = tk.StringVar(self, value="0.105")
        self.safetyValEntry = tk.Entry(self, textvariable=self.safetyValVar, width=5)
        self.modelPathVar = tk.StringVar(self, value="J:/Labor/gqcnn/models/GQCNN-2.0")
        self.modelPathEntry = tk.Entry(self, textvariable=self.modelPathVar, width=50)
        self.cfgPathVar = tk.StringVar(
            self, value="J:/Labor/gqcnn/cfg/examples/gqcnn_pj.yaml"
        )
        self.cfgPathEntry = tk.Entry(self, textvariable=self.cfgPathVar, width=50)
        self.intrPathVar = tk.StringVar(
            self, value="J:/Labor/gqcnn/data/calib/realsense/realsense.intr"
        )
        self.intrPathEntry = tk.Entry(self, textvariable=self.intrPathVar, width=50)
        self.hecPathVar = tk.StringVar(self, value="handEyeCalibration/")
        self.hecPathEntry = tk.Entry(self, textvariable=self.hecPathVar)
        self.ipVar = tk.StringVar(self, value="10.83.2.1")
        self.ipEntry = tk.Entry(self, textvariable=self.ipVar, width=15)
        self.portVar = tk.StringVar(self, value="2000")
        self.portEntry = tk.Entry(self, textvariable=self.portVar, width=5)

        self.reloadConfig = tk.Button(
            self, text="Reload Policy", command=self.reloadPolicy, width=20
        )

        self.corrX.grid(row=0, column=0, sticky="w", padx=(5, 0))
        self.corrXEntry.grid(row=0, column=1, sticky="w")
        self.corrY.grid(row=1, column=0, sticky="w", padx=(5, 0))
        self.corrYEntry.grid(row=1, column=1, sticky="w")
        self.corrZ.grid(row=2, column=0, sticky="w", padx=(5, 0))
        self.corrZEntry.grid(row=2, column=1, sticky="w")

        self.rotMat.grid(row=3, column=0, sticky="w", padx=(5, 0))
        self.rotMatEntry.grid(row=3, column=1, sticky="w")
        self.safetyVal.grid(row=4, column=0, sticky="w", padx=(5, 0))
        self.safetyValEntry.grid(row=4, column=1, sticky="w")
        self.modelPath.grid(row=5, column=0, sticky="w", padx=(5, 0))
        self.modelPathEntry.grid(row=5, column=1, sticky="w")
        self.cfgPath.grid(row=6, column=0, sticky="w", padx=(5, 0))
        self.cfgPathEntry.grid(row=6, column=1, sticky="w")
        self.intrPath.grid(row=7, column=0, sticky="w", padx=(5, 0))
        self.intrPathEntry.grid(row=7, column=1, sticky="w")

        self.hecPath.grid(row=8, column=0, sticky="w", padx=(5, 0))
        self.hecPathEntry.grid(row=8, column=1, sticky="w")
        self.ip.grid(row=9, column=0, sticky="w", padx=(5, 0))
        self.ipEntry.grid(row=9, column=1, stick="w")
        self.port.grid(row=9, column=2, sticky="w")
        self.portEntry.grid(row=9, column=3, sticky="w")

        self.reloadConfig.grid(row=10, column=0, sticky="w", pady=(5, 0), padx=5)

    def reloadPolicy(self):
        # self.parent.parent.parent.urPolicy = None
        if os.path.isdir(self.modelPathVar.get()):
            self.parent.parent.parent.urPolicy.hec.server.closeConnection()
            del self.parent.parent.parent.urPolicy
            self.parent.parent.parent.urPolicy = URPolicy(
                re=self.parent.parent.parent.realsense,
                model_path=self.modelPathVar.get(),
                config_filename=self.cfgPathVar.get(),
                intr_path=self.intrPathEntry.get(),
                ip=self.ipVar.get(),
                port=int(self.portVar.get()),
                hec_path=self.hecPathVar.get(),
            )
        else:
            logging.error("Model directory does not exist")


class RightFrame(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.graspViewer = GraspViewer(self, text="Predicted Grasp")
        self.options = Options(self, text="Options")

        self.graspViewer.pack(side="top", fill="x", anchor=tk.N)
        self.options.pack(side="bottom", fill="x", ipadx=10, ipady=5)


class UpperFrame(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        self.videoFeed = VideoFeed(self, text="Liveview Realsense")
        self.rightFrame = RightFrame(self)
        self.depthSegmentationViewer = DepthSegmentationViewer(
            self.videoFeed, text="Image Viewer"
        )

        self.videoFeed.pack(side="left", fill="both")
        self.rightFrame.pack(side="left", fill="both", padx=10)
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
        # self.parent.geometry("1280x720")

        self.toolbar = Toolbar(self)
        self.lowerFrame = LowerFrame(self)

        self.realsense = RealsenseInterface(align=True, decimation=False)
        self.realsense.start()

        self.upperFrame = UpperFrame(self)

        self.urPolicy = URPolicy(re=self.realsense)

        self.toolbar.pack(side="top")
        self.upperFrame.pack(side="top", anchor="center", padx=10)
        self.lowerFrame.pack(
            side="bottom", fill="both", anchor="center", pady=10, padx=10
        )

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
