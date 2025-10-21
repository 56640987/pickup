import time
import libpyauboi5
import logging 
from logging.handlers import RotatingFileHandler
from multiprocessing import Process, Queue
from time import sleep
import os
from math import pi
from robotcontrol import logger_init
from robotcontrol import Auboi5Robot
from robotcontrol import RobotErrorType
import numpy as np
import cv2
from math import degrees

# 初始化 logger
logger = logging.getLogger('main.robotcontrol')
logger_init()

# 相机内参


# fx, fy, cx, cy = 908.800537109375, 908.79345703125,638.7694091796875,384.8724365234375
#10组
fx,fy,cx,cy = 921.07471859,919.91720227,639.302913,378.86558034

#6组
# fx,fy,cx,cy = 921.92756454,921.03205236,640.79801198,379.52144969

camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])

# 畸变系数
# dist_coeffs = np.array([0.163418, -0.448028, -0.015510, 0.018532, 0.000000])

#10组
dist_coeffs = np.array([8.53842454e-02 , 1.67726232e-03, -1.56849113e-04,  2.46366555e-04,-6.17566112e-01])

#6组
# dist_coeffs = np.array([8.66327864e-02, -2.57222368e-02 ,-4.90210702e-04 , 8.40818004e-04 ,-5.27218272e-01])
# 手眼矩阵
hand_eye_matrix = np.array([

#10组数据
 [-0.9998005518,  0.0177087282, -0.0092335045 , 0.0343838741],
 [-0.017835186,  -0.999745729 ,  0.0137979538 , 0.0894847901],
 [-0.0089868125,  0.0139598831 , 0.9998621699 , 0.0182836322],

#6组数据（剔除异常值）
#  [-0.999833086  , 0.016847731 , -0.0070678262 , 0.0349779039],
#  [-0.0169513081 ,-0.9997458824,  0.0148601438,  0.0918757811],
#  [-0.0068156704 , 0.0149774724 , 0.9998646018 , 0.0207470741],
    [0, 0, 0, 1]
])

# 去畸变处理
def undistort_point(point, camera_matrix, dist_coeffs):
    point = np.array([point], dtype=np.float32).reshape(1, 1, 2)
    undistorted_point = cv2.undistortPoints(point, camera_matrix, dist_coeffs, P=camera_matrix)
    return undistorted_point[0][0]

# 将像素坐标转换为相机坐标系下的3D点
def pixel_to_camera(pixel_coords, camera_matrix):
    u, v, z = pixel_coords
    x = (u - camera_matrix[0, 2]) * z / camera_matrix[0, 0]
    y = (v - camera_matrix[1, 2]) * z / camera_matrix[1, 1]
    return np.array([x, y, z, 1])

# 使用手眼矩阵将相机坐标系下的3D点转换为机器人末端坐标系下的3D点
def transform_to_end_effector(camera_coords, hand_eye_matrix):
    end_effector_coords = np.dot(hand_eye_matrix, camera_coords)
    return end_effector_coords[:3]

# 将末端坐标系下的3D点转换为基坐标系下的3D点
def transform_to_base(end_effector_coords, base_to_end_matrix):
    end_effector_coords_tf = np.append(end_effector_coords, 1)
    base_coords = np.dot(base_to_end_matrix, end_effector_coords_tf)
    return base_coords[:3]

def get_base_to_end_matrix(robot):
    current_waypoint = robot.get_current_waypoint()
    pos = current_waypoint['pos']
    ori = current_waypoint['ori']
    rotation_matrix = quaternion_to_matrix(ori)
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = pos
    return transform_matrix

def quaternion_to_matrix(quat):
    w, x, y, z = quat
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

def move_to_target(robot, target_pos, target_ori):
    current_waypoint = robot.get_current_waypoint()
    joint_radian = current_waypoint['joint']
    result = robot.inverse_kin(joint_radian, target_pos, target_ori)
    if result is not None:
        target_joint = result['joint']
        robot.move_joint(target_joint)
        return True
    else:
        logger.error("逆解失败")
        return False

def project_reset():
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
            robot.move_to_target_in_cartesian((-0.3, -0.12, 0.42), (180, 0, -90))
            robot.disconnect()
    finally:
        if robot.connected:
            robot.disconnect()
        Auboi5Robot.uninitialize()
        print("run end-------------------------")


def project_detect():
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
            # robot.move_to_target_in_cartesian((-0.15, -0.1, 0.5), (90, 0, 90))
            robot.move_to_target_in_cartesian((0.26,-0.12,0.44),(90,0,90))
            robot.disconnect()
    finally:
        if robot.connected:
            robot.disconnect()
        Auboi5Robot.uninitialize()
        print("run end-------------------------")

def move_realse():
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
            cur_jop = robot.get_current_waypoint()
            cur_j = cur_jop['joint']
            cur_o = cur_jop['ori']
            cur_p = cur_jop['pos']
            pos_jop = robot.inverse_kin(joint_radian=cur_j, pos=((cur_p[0]-0.1), cur_p[1], cur_p[2]), ori=(0.5, 0.5, 0.5, 0.5))
            joint_radian = pos_jop['joint']
            robot.move_joint(joint_radian, True)
            robot.move_to_target_in_cartesian((-0.125, -0.2, 0.45), (90, 0, 0))
            robot.disconnect()
    finally:
        if robot.connected:
            robot.disconnect()
        Auboi5Robot.uninitialize()
        print("run end-------------------------")


def apply_gripper_offset(end_pos, offset_z):
    """
    end_pos : 3×1，物体在末端系的位置
    offset_z: 抓手比末端长多少米（负号表示沿末端Ze负方向）
    return  : 3×1，抓手中心在末端系的位置
    """
    return end_pos + np.array([0, 0, offset_z])

def pick_up(robot, pixel_coords):
    try:
        undistorted_pixel = undistort_point(pixel_coords[:2], camera_matrix, dist_coeffs)
        undistorted_pixel_coords = np.array([undistorted_pixel[0], undistorted_pixel[1], pixel_coords[2]])

        camera_coords = pixel_to_camera(undistorted_pixel_coords, camera_matrix)

        end_effector_coords = transform_to_end_effector(camera_coords, hand_eye_matrix)
        
        #减去手爪长度
        gripper_center_end = apply_gripper_offset(end_effector_coords, offset_z=-0.12)

        base_to_end_matrix = get_base_to_end_matrix(robot)
        base_coords = transform_to_base(gripper_center_end, base_to_end_matrix)

        logger.info(f"目标物体在基坐标系下的坐标: {base_coords}")

        current_waypoint = robot.get_current_waypoint()
        target_ori = current_waypoint['ori']

        success = move_to_target(robot, base_coords, target_ori)
        if success:
            logger.info("机械臂已移动到目标位置")
            return True
        else:
            logger.error("机械臂移动失败")
            return False
    except Exception as e:
        logger.error(f"发生错误: {e}")
        return False
    

def test_reset(x,y,z,rx,ry,rz):
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
            robot.move_to_target_in_cartesian((x,y,z), (rx,ry,rz))
            robot.disconnect()
    finally:
        if robot.connected:
            robot.disconnect()
        Auboi5Robot.uninitialize()
        print("run end-------------------------")


def get_current_joints(robot):
    """返回 6 个关节的当前角度，单位 rad"""
    return robot.get_current_waypoint()['joint']

def get_current_xyz_rxryrz(robot):
    waypoint = robot.get_current_waypoint()
    xyz = waypoint['pos']                       # 列表，3 个元素，单位 m
    quat = waypoint['ori']                      # 四元数 [w, x, y, z]
    rpy_rad = robot.quaternion_to_rpy(quat)     # 返回 [rx, ry, rz]，单位弧度
    rpy_deg = [degrees(angle) for angle in rpy_rad]
    return xyz + rpy_deg   



if __name__ == '__main__':
    logger.info("test completed")
    # move_realse()
    project_detect()