from robotcontrol import RobotErrorType
from robotcontrol import Auboi5Robot,logger_init
import logging
from my_api import get_current_joints,get_current_xyz_rxryrz

logger = logging.getLogger('main.robotcontrol')
logger_init()
Auboi5Robot.initialize()
robot = Auboi5Robot()
handle = robot.create_context()
logger.info("robot.rshd={0}".format(handle))
try:
    ip = '192.168.1.101'
    port = 8899
    result = robot.connect(ip, port)
    if result != RobotErrorType.RobotError_SUCC:
        logger.info("connect server{0}:{1} failed.".format(ip, port))
    else:
        robot.robot_startup()
        robot.enable_robot_event()
        robot.init_profile()
        joint_maxvelc = (1.0,1.0,1.0,1.0,1.0,1.0)
        joint_maxacc = (1.0,1.0,1.0,1.0,1.0,1.0)
        j=get_current_joints(robot)
        print(j)
        x,y,z,rx,ry,rz = get_current_xyz_rxryrz(robot)
        print(x,y,z,rx,ry,rz)
        robot.disconnect()
finally:
    if robot.connected:
        robot.disconnect()
    Auboi5Robot.uninitialize()
    print("run end-------------------------")

