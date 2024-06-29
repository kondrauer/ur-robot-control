import json
import os
import time
import logging
import colorlog

import numpy as np

from autolab_core import YamlConfig
from autolab_core import (
    BinaryImage,
    CameraIntrinsics,
    ColorImage,
    DepthImage,
    RgbdImage,
)

from gqcnn.grasping import (
    RobustGraspingPolicy,
    CrossEntropyRobustGraspingPolicy,
    RgbdImageState,
    FullyConvolutionalGraspingPolicyParallelJaw,
)
from gqcnn.utils import GripperMode
from gqcnn.utils.policy_exceptions import NoValidGraspsException

from visualization import Visualizer2D as vis

from RealsenseInterface import RealsenseInterface
from HandEyeCalibrator import HandEyeCalibrator


class URPolicy:
    """Class to execute a CEM grasping policy.
    Adapted from https://github.com/BerkeleyAutomation/gqcnn/blob/master/tools/run_policy.py

        Attributes:
                fully_conv(boolean): determines wether to expect a FC-GQ-CNN
                policy_type(str): 'cem' or 'ranking'
                model_path(str): path to gqcnn-model
            input_data_mode(str): only needed if gqcnn config is not found
            gripper_mode(GripperMode): only needed if gqcnn config is not found

            inpaint_rescale_factor(float): factor to scale depth image

            model_config(json): loaded gqcnn-model config
            gqcnn_config(json): loaded gqcnn-model config
            config(YamlConfig): policy config
            policy_config(json): policy config


            rgbd_im(RgbdImage): RGB and depth image of scene
            index(int): index of image in img/ folder
            action(GraspAction): planned grasp action

            camera_intr(CameraIntrinsics): camera intrinsics for used camera
            hec(HandEyeCalibrator): hand eye calibrator for grasp execution
            tfGraspToBase(RigidTransform): transform from grasp to base
            re(RealsenseInterface): realsense control

    """

    def __init__(
        self,
        debug: bool = False,
        fully_conv: bool = False,
        policy_type: str = "cem",
        model_path: str = r"J:\Labor\gqcnn\models\GQCNN-4.0-PJ",
        config_filename: str = "J:/Labor/gqcnn/cfg/examples/gqcnn_pj.yaml",
        intr_path: str = "J:/Labor/gqcnn/data/calib/realsense/realsense.intr",
        re: RealsenseInterface = None,
        ip="10.83.2.1",
        port=2000,
        hec_path: str = "handEyeCalibration/",
    ):
        """Constructor for URPolicy

            Args:
                    debug: set output level to logger
        fully_conv: true if fc-gq-cnn is used
        policy_type: 'cem' or 'ranking'
        model_path: path to gq-cnn
        config_filename: path to policy config
        intr_path: path to camera intrinsics
        re: RealsenseInterface object
        ip: ip to bind server to
        port: port to bind server to
        hec_path: path to saved hand-eye-calibration from a HandEyeCalibrator

        """

        self.fully_conv = fully_conv
        self.policy_type = policy_type
        self.model_path = model_path
        self.model_config = json.load(
            open(os.path.join(model_path, "config.json"), "r")
        )

        try:
            self.gqcnn_config = self.model_config["gqcnn"]
            self.gripper_mode = self.gqcnn_config["gripper_mode"]
        except KeyError:
            self.gqcnn_config = self.model_config["gqcnn_config"]
            self.input_data_mode = self.gqcnn_config["input_data_mode"]
            if self.input_data_mode == "tf_image":
                self.gripper_mode = GripperMode.LEGACY_PARALLEL_JAW
            elif self.input_data_mode == "tf_image_suction":
                self.gripper_mode = GripperMode.LEGACY_SUCTION
            elif self.input_data_mode == "suction":
                self.gripper_mode = GripperMode.SUCTION
            elif self.input_data_mode == "multi_suction":
                self.gripper_mode = GripperMode.MULTI_SUCTION
            elif self.input_data_mode == "parallel_jaw":
                self.gripper_mode = GripperMode.PARALLEL_JAW
            else:
                raise ValueError(
                    "Input data mode {} not supported!".format(self.input_data_mode)
                )

        self.config = YamlConfig(config_filename)
        self.inpaint_rescale_factor = self.config["inpaint_rescale_factor"]
        self.policy_config = self.config["policy"]

        self.policy_config["metric"]["gqcnn_model"] = self.model_path

        self.camera_intr = CameraIntrinsics.load(intr_path)
        self.hec = HandEyeCalibrator(realsense=False, ip=ip, port=port, path=hec_path)
        self.tfGraspToBase = None

        if not re:
            self.re = RealsenseInterface(align=True, decimation=False)
            self.re.start(preprocessing=True)
        else:
            self.re = re

        logging.root.name = "Robotiklabor"

        if debug:
            logging.getLogger().setLevel(logging.DEBUG)
        else:
            logging.getLogger().setLevel(logging.INFO)

    def planGrasp(self):
        """Plans grasp and stores resulting transformation from grasp to base

        Returns:
                None
        """
        self.index = self.re.saveImageSet(iterationsDilation=1, filter=False)

        # Read images.
        depth_data = np.load(f"img/depth_{self.index}.npy")
        depth_im = DepthImage(depth_data, frame=self.camera_intr.frame)
        color_im = ColorImage(
            np.zeros([depth_im.height, depth_im.width, 3]).astype(np.uint8),
            frame=self.camera_intr.frame,
        )

        segmask = BinaryImage.open(f"img/segmask_{self.index}.png")
        depth_im = depth_im.inpaint(rescale_factor=self.inpaint_rescale_factor)

        self.rgbd_im = RgbdImage.from_color_and_depth(color_im, depth_im)
        state = RgbdImageState(self.rgbd_im, self.camera_intr, segmask=segmask)

        if self.fully_conv:
            self.policy_config["metric"]["fully_conv_gqcnn_config"]["im_height"] = (
                depth_im.shape[0]
            )
            self.policy_config["metric"]["fully_conv_gqcnn_config"]["im_width"] = (
                depth_im.shape[1]
            )

        logging.info(self.policy_config["metric"]["gqcnn_model"])

        if self.fully_conv:
            policy = FullyConvolutionalGraspingPolicyParallelJaw(self.policy_config)
        elif self.policy_type == "ranking":
            policy = RobustGraspingPolicy(self.policy_config)
        elif self.policy_type == "cem":
            policy = CrossEntropyRobustGraspingPolicy(self.policy_config)

        # Query policy.
        policy_start = time.time()
        try:
            self.action = policy(state)
        except NoValidGraspsException:
            logging.error("No valid graps found")
        logging.info("Planning took %.3f sec" % (time.time() - policy_start))

        self.tfGraspToBase = self.hec.graspTransformer(self.action.grasp.pose())

    def showGrasp(self):
        """Renders grasp and shows it

        Returns:
                None
        """
        if self.action:
            vis.figure(size=(10, 10))
            vis.imshow(
                self.rgbd_im.depth,
                vmin=self.policy_config["vis"]["vmin"],
                vmax=self.policy_config["vis"]["vmax"],
            )
            vis.grasp(self.action.grasp, scale=2.5, show_center=False, show_axis=True)
            vis.title(
                "Planned grasp at depth {0:.3f}m with Q={1:.3f}".format(
                    self.action.grasp.depth, self.action.q_value
                )
            )
            vis.show()
        else:
            logging.warning("You have to plan a grasp first!")

    def executeGrasp(
        self,
        corrX: float = 0,
        corrY: float = 0,
        corrZ: float = 0,
        rotMat: np.array = None,
        safetyVal: float = 0.105,
    ):
        """Executes grasp with regards to hand-eye-calibration

        Returns:
                None
        """
        self.hec.moveToPoint(
            self.tfGraspToBase,
            corrX=corrX,
            corrY=corrY,
            corrZ=corrZ,
            rotMat=rotMat,
            safetyVal=safetyVal,
        )


if __name__ == "__main__":
    # setting up logger
    root_logger = logging.getLogger()
    for hdlr in root_logger.handlers:
        if isinstance(hdlr, logging.StreamHandler):
            root_logger.removeHandler(hdlr)

    logging.root.name = "Robotiklabor"
    #     logging.getLogger().setLevel(logging.DEBUG)
    logging.getLogger().setLevel(logging.INFO)

    handler = colorlog.StreamHandler()
    formatter = colorlog.ColoredFormatter(
        "%(purple)s%(name)-10s "
        "%(log_color)s%(levelname)-8s%(reset)s "
        "%(white)s%(message)s",
        reset=True,
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )
    handler.setFormatter(formatter)
    logger = colorlog.getLogger()
    logger.addHandler(handler)

    # exectue policy

    urp = URPolicy()
    urp.planGrasp()
    urp.executeGrasp()
