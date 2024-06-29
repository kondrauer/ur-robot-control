import numpy as np
import cv2
import logging
import colorlog
import time

from autolab_core import RigidTransform
from pathlib import Path

import PoseEstimator as pe
import Server


class HandEyeCalibrator:
    """Class for solving the Hand-Eye-Calibration problem with an Intel Realsense L515

    Attributes:
            tfBaseToWorld (RigidTransform): Transformation from Base-Frame to World-Frame
            tfGripperToCam (RigidTransform): Transformation from Gripper-Frame to Cam-Frame
            tfDepthToWorld (RigidTransform): Transformation from cameras Depth-Frame to World-Frame

            poseEstimator (pe.CharucoPoseEstimator): Class for Pose Estimation of the Camera

            server (Server.Server): Wrapper-Class of socket built-in

    """

    def __init__(
        self,
        numberOfMeasurements: int = 25,
        realsense: bool = True,
        ip: str = "10.83.2.1",
        port: int = 2000,
        path: str = "handEyeCalibration/",
    ) -> None:
        """Constructor for HandEyeCalibrator

        Args:
                numberOfMeasurements (int): Determines how many poses the robot should approach (this has to be in line with the programm on the robot itself)

        """

        if numberOfMeasurements <= 3:
            # Number of Measurements can't be smaller than 3
            # see https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga41b1a8dd70eae371eba707d101729c36 under notes; it is a requirement by the method chosen
            raise ValueError("Number of measurements needs to be > 0")

        else:
            # any number greater three is accepted
            self.numberOfMeasurements = numberOfMeasurements

        self.tfBaseToWorld = RigidTransform(from_frame="base", to_frame="world")
        self.tfGripperToCam = RigidTransform(from_frame="gripper", to_frame="cam")
        self.tfGripperCamToGripper = RigidTransform(
            from_frame="cam", to_frame="gripper"
        )

        self.poseEstimator: pe.CharucoPoseEstimator = pe.CharucoPoseEstimator(
            depth=True, realsense=realsense
        )
        self.tfDepthToWorld: RigidTransform = None
        self.tfWorldToCamera: RigidTransform = None

        self.server = Server.Server(ip=ip, port=port)

        # check if the calibration was already done and load it
        self.loadCalibrationFromFile(path=path)

    def calibrateHandEye(self):
        """Executes the Hand-Eye-Calibration

        First a connection with the robot is established. After that the server starts to receive i poses of the Tool-Center-Point from the robot. After receiving a single pose the
        corresponding computation of the camera pose takes place. Both the TCP-pose and the camera-pose are then append to a list. The server sends back an acknowledgement to the robot,
        which in turn approaches the next pose.

        After numberOfMeasurements steps the lists of the recorded poses are temporarly saved and cv2.calibrateRobotWorldHandEye() with the lists casted to np.arrays.

        The returned transforms are stored and written to the class attributes and the connection to the robot is closed.

        Args:
                None

        """
        try:
            self.server.commSocket, self.server.commAddress = (
                self.server.establishConnection()
            )

            # Buffer variables
            listRMatBaseToGripper: list = []
            listTVecBaseToGripper: list = []
            listRMatGripperToBase: list = []
            listTVecGripperToBase: list = []
            listRMatWorldToCamera: list = []
            listTVecWorldToCamera: list = []

            for i in range(self.numberOfMeasurements):
                logging.info(f"Step {i+1} of {self.numberOfMeasurements}")

                # get the TCP-pose from the robot from gripper to base(!!!)
                gripperToBase = self.server.receiveData(1024)
                gripperToBase = gripperToBase[2 : len(gripperToBase)]

                # str format is 'p[tVecX, tVecY, tVecZ, rVecX, rVecY, rVecZ]'
                gripperToBase = np.fromstring(gripperToBase, dtype=np.float64, sep=",")
                rVecGripperToBase = gripperToBase[3:6].reshape(3, 1)
                tVecGripperToBase = gripperToBase[0:3].reshape(3, 1)

                # convert rotation vector to rotation matrix
                rMatGripperToBase, _ = cv2.Rodrigues(rVecGripperToBase)

                # invert to get Base to Gripper
                rMatBaseToGripper = rMatGripperToBase.T
                tVecBaseToGripper = -rMatBaseToGripper @ tVecGripperToBase

                logging.info("Base to gripper of robot transformation determined")
                logging.debug(f"Translation: {tVecBaseToGripper}")
                logging.debug(f"Rotation: {rMatBaseToGripper}")

                # estimate camera pose with regards to charuco board (which is the world frame)
                success, rMatWorldToCamera, tVecWorldToCamera = (
                    self.poseEstimator.estimatePose()
                )

                # only save results of this step if the pose estimation of the camera was successful
                if success:
                    logging.info("World to camera transformation determined")
                    logging.debug(f"Translation: {tVecWorldToCamera}")
                    logging.debug(f"Rotation: {rMatWorldToCamera}")
                    logging.info("Saving transformations")

                    listRMatGripperToBase.append(rMatGripperToBase)
                    listTVecGripperToBase.append(tVecGripperToBase)

                    listRMatBaseToGripper.append(rMatBaseToGripper)
                    listTVecBaseToGripper.append(tVecBaseToGripper)

                    listRMatWorldToCamera.append(rMatWorldToCamera)
                    listTVecWorldToCamera.append(tVecWorldToCamera)
                else:
                    logging.error(
                        "Pose estimation of camera failed, skipping this step"
                    )

                # tell robot to approach next pose
                # input('press key to continue')
                communication = 1
                self.server.sendData(f"{communication}")

            # check if folder exists
            Path("handEyeCalibration/").mkdir(exist_ok=True)

            # save all determined poses
            with open("handEyeCalibration/rMatBaseToGripper.npy", "wb") as f:
                np.save(f, np.array(listRMatBaseToGripper))
            with open("handEyeCalibration/tVecBaseToGripper.npy", "wb") as f:
                np.save(f, np.array(listTVecBaseToGripper))
            with open("handEyeCalibration/rMatWorldToCamera.npy", "wb") as f:
                np.save(f, np.array(listRMatWorldToCamera))
            with open("handEyeCalibration/tVecWorldToCamera.npy", "wb") as f:
                np.save(f, np.array(listTVecWorldToCamera))

            logging.info("Calculating transformations from given poses")

            start = time.time()

            rMatBaseToWorld, tVecBaseToWorld, rMatGripperToCam, tVecGripperToCam = (
                cv2.calibrateRobotWorldHandEye(
                    np.array(listRMatWorldToCamera),
                    np.array(listTVecWorldToCamera),
                    np.array(listRMatBaseToGripper),
                    np.array(listTVecBaseToGripper),
                    method=cv2.CALIB_ROBOT_WORLD_HAND_EYE_SHAH,
                )
            )

            rMatCamToGripper, tVecCamToGripper = cv2.calibrateHandEye(
                np.array(listRMatGripperToBase),
                np.array(listTVecGripperToBase),
                np.array(listRMatWorldToCamera),
                np.array(listTVecWorldToCamera),
            )

            logging.info(f"Calculation took {time.time()-start} seconds")

            logging.debug(rMatBaseToWorld)
            logging.debug(tVecBaseToWorld)
            logging.debug(f"Distance Base To World: {np.linalg.norm(tVecBaseToWorld)}")
            logging.debug(rMatGripperToCam)
            logging.debug(tVecGripperToCam)
            logging.debug(
                f"Distance Gripper To Cam: {np.linalg.norm(tVecGripperToCam)}"
            )
            logging.debug(rMatCamToGripper)
            logging.debug(tVecCamToGripper)
            logging.debug(
                f"Distance Cam to Gripper: {np.linalg.norm(tVecCamToGripper)}"
            )

            self.tfBaseToWorld: RigidTransform = RigidTransform(
                rotation=rMatBaseToWorld,
                translation=tVecBaseToWorld,
                from_frame="base",
                to_frame="world",
            )
            self.tfGripperToCam: RigidTransform = RigidTransform(
                rotation=rMatGripperToCam,
                translation=tVecGripperToCam,
                from_frame="gripper",
                to_frame="cam",
            )

            self.tfCamToGripper: RigidTransform = RigidTransform(
                rotation=rMatCamToGripper,
                translation=tVecCamToGripper,
                from_frame="cam",
                to_frame="gripper",
            )

            self.tfBaseToWorld.save("handEyeCalibration/baseToWorld.tf")
            self.tfGripperToCam.save("handEyeCalibration/gripperToCam.tf")
            self.tfCamToGripper.save("handEyeCalibration/camToGripper.tf")

            logging.info("Saved calibration to file!")

            self.server.closeConnection()

            return True

        # TODO: Better error handling
        except Exception as e:
            print(e)
            return False

    def loadCalibrationFromFile(self, path: str = "handEyeCalibration/"):
        """Loads Hand-Eye-Calibration if it exists in path.

        Args:
             path (str): determines where to look for the files

        """

        if Path(path).exists():
            if (
                Path(path + "baseToWorld.tf").exists()
                and Path(path + "gripperToCam.tf").exists()
            ):
                logging.info("Loading...")
                self.tfBaseToWorld = RigidTransform.load(
                    "handEyeCalibration/baseToWorld.tf"
                )
                self.tfGripperToCam = RigidTransform.load(
                    "handEyeCalibration/gripperToCam.tf"
                )
                self.tfCamToGripper = RigidTransform.load(
                    "handEyeCalibration/camToGripper.tf"
                )
                logging.info("Hand-Eye-Calibration loaded")

        else:
            logging.warning("No saved Hand-Eye-Calibration found")

    def graspTransformer(self, tfGraspToDepth: RigidTransform):
        """Transforms a grasp to the base of the robot.

        This method takes a grasp calculated by the GQCNN (from_frame = grasp, to_frame = depth) and sets the reference frame to the robots base coordinate system
        by multiplying the respective transformations.

        In the end it returns a new transform tfGraspToBase with from_frame = grasp and to_frame = base

        Args:
                tfGraspToDepth (RigidTransform): transformation from grasp-frame to depth-frame of image sensor

        Returns:

                tfGraspToBase (RigidTransform): transformation from grasp-frame to base-frame of robot
        """

        # TODO: determine if the tfGraspToBase needs to be inversed

        if (self.tfBaseToWorld.rotation == np.eye(3)).all():
            logging.warning("Base to World seems to be default matrix")

        self.tfDepthToWorld = RigidTransform.load("handEyeCalibration/depthToWorld.tf")
        tfGraspToBase = (
            self.tfBaseToWorld.inverse() * self.tfDepthToWorld * tfGraspToDepth
        )
        logging.debug(tfGraspToDepth)
        logging.info(tfGraspToBase)

        return tfGraspToBase

    def graspTransformerGripper(self, tfGraspToDepth: RigidTransform):
        self.server.commSocket, self.server.commAddress = (
            self.server.establishConnection()
        )

        # get the TCP-pose from the robot from gripper to base(!!!)
        gripperToBase = self.server.receiveData(1024)
        gripperToBase = gripperToBase[2 : len(gripperToBase)]

        # str format is 'p[tVecX, tVecY, tVecZ, rVecX, rVecY, rVecZ]'
        gripperToBase = np.fromstring(gripperToBase, dtype=np.float64, sep=",")
        rVecGripperToBase = gripperToBase[3:6].reshape(3, 1)
        tVecGripperToBase = gripperToBase[0:3].reshape(3, 1)

        # convert rotation vector to rotation matrix
        rMatGripperToBase, _ = cv2.Rodrigues(rVecGripperToBase)

        tfGripperToBase = RigidTransform(
            rotation=rMatGripperToBase,
            translation=tVecGripperToBase,
            from_frame="gripper",
            to_frame="base",
        )

        self.tfWorldToCamera = RigidTransform.load(
            "handEyeCalibration/worldToCamera.tf"
        )
        tfWorldToBase = tfGripperToBase * self.tfCamToGripper * self.tfWorldToCamera

        self.tfDepthToWorld = RigidTransform.load("handEyeCalibration/depthToWorld.tf")
        self.tfDeü
        tfGraspToBase = tfWorldToBase * self.tfDepthToWorld * tfGraspToDepth

        logging.info(tfGraspToBase)

        return tfGraspToBase

    def moveToPoint(
        self,
        tfPointToMove: RigidTransform,
        rotationXAxis: bool = False,
        corrX: float = 0,
        corrY: float = 0,
        corrZ: float = 0,
        rotMat: np.array = None,
        safetyVal: float = 0.105,
    ):
        self.server.commSocket, self.server.commAddress = (
            self.server.establishConnection()
        )

        if rotationXAxis:
            rMatXAxis = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            tfPointToMove.rotation = np.transpose(
                rMatXAxis @ np.transpose(tfPointToMove.rotation)
            )

            # 			tfPointToMove.translation[0] = tfPointToMove.translation[0] - 0.01
            tfPointToMove.translation[1] = tfPointToMove.translation[1] + 0.01

        else:
            tfPointToMove.translation[0] += corrX
            tfPointToMove.translation[1] += corrY
            tfPointToMove.translation[2] += corrZ

            tfPointToMove.rotation = np.transpose(
                rotMat @ np.transpose(tfPointToMove.rotation)
            )
        # 			rotYAxis90 = np.array([[0, 0, -1],
        # 	 						       [0, 1, 0],
        # 								   [1, 0, 0]], dtype=np.float32)

        # 			rotZAxis180 = np.array([[-1, 0, 0],
        # 							        [0, -1, 0],
        # 									[0, 0, 1]], dtype=np.float32)

        # 			rotXAxis5 = np.array([[1, 0, 0],
        #  							        [0, 0.99, -0.1392],
        #  									[0, 0.1392, 0.99]], dtype=np.float32)

        # 			tfPointToMove.rotation = np.transpose(rotYAxis90 @ np.transpose(tfPointToMove.rotation))
        # 			tfPointToMove.rotation = np.transpose(rotZAxis180 @ np.transpose(tfPointToMove.rotation))
        # 			tfPointToMove.rotation = np.transpose(rotXAxis5 @ np.transpose(tfPointToMove.rotation))

        # 		tfPointToMove.translation *= 1000
        # 		logging.debug(tfPointToMove.matrix)

        if tfPointToMove.translation[2] < safetyVal:
            tfPointToMove.translation[2] = safetyVal

        strTVecMarkerToBase = (
            str(tfPointToMove.translation[0])
            + ","
            + str(tfPointToMove.translation[1])
            + ","
            + str(tfPointToMove.translation[2])
        )
        strRVecMarkerToBase = (
            str(tfPointToMove.axis_angle[0])
            + ","
            + str(tfPointToMove.axis_angle[1])
            + ","
            + str(tfPointToMove.axis_angle[2])
        )
        strMarkerToBase = "( " + strTVecMarkerToBase + "," + strRVecMarkerToBase + " )"

        logging.debug(strMarkerToBase)
        logging.info(tfPointToMove.matrix)

        self.server.sendData(strMarkerToBase)


if __name__ == "__main__":
    pass
    root_logger = logging.getLogger()
    for hdlr in root_logger.handlers:
        if isinstance(hdlr, logging.StreamHandler):
            root_logger.removeHandler(hdlr)

    logging.root.name = "Robotiklabor"
    # 	logging.getLogger().setLevel(logging.DEBUG)
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

    hec = HandEyeCalibrator()

    hec.calibrateHandEye()

# 	hec.poseEstimator.savePoseAsDepthToWorldTransform(path='handEyeCalibration/depthToWorld.tf')
# 	hec.poseEstimator.savePoseAsWorldToCameraTransform(path='handEyeCalibration/worldToCamera.tf')

# hec.moveToPoint(hec.tfBaseToWorld.inverse(), rotationXAxis=True)

# hec.moveToPoint()

# hec.poseEstimator.realsense.saveImageSet(iterationsDilation = 1, filter=True)
# hec.poseEstimator.realsense.openViewer(filter=True)
