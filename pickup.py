import numpy as np
import cv2
from robotcontrol import Auboi5Robot, RobotErrorType, logger_init
import logging
from time import sleep
from math import pi,radians
from my_api import get_current_joints,get_current_xyz_rxryrz

# 初始化 logger
logger = logging.getLogger('main.robotcontrol')

# 相机内参
# fx, fy, cx, cy = 908.800537109375, 908.79345703125,638.7694091796875,384.8724365234375
fx,fy,cx,cy = 921.07471859,919.91720227,639.302913,378.86558034



camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])

# 畸变系数
# dist_coeffs = np.array([0.166056, -0.163080, -0.029346, 0.007306, 0.0000000])
dist_coeffs = np.array([8.53842454e-02 , 1.67726232e-03, -1.56849113e-04,  2.46366555e-04,-6.17566112e-01])
# 手眼矩阵
hand_eye_matrix = np.array([

 [-0.9998005518,  0.0177087282, -0.0092335045 , 0.0343838741],
 [-0.017835186,  -0.999745729 ,  0.0137979538 , 0.0894847901],
 [-0.0089868125,  0.0139598831 , 0.9998621699 , 0.0182836322],
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
    """获取基坐标系到末端坐标系的变换矩阵"""
    current_waypoint = robot.get_current_waypoint()
    pos = current_waypoint['pos']
    ori = current_waypoint['ori']
    rotation_matrix = quaternion_to_matrix(ori)
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = pos
    return transform_matrix

def quaternion_to_matrix(quat):
    """将四元数转化为旋转矩阵"""
    w, x, y, z = quat
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

def move_to_target(robot, target_pos, target_ori):
    """移动机械臂到目标位置和姿态"""
    # 求逆解
    current_waypoint = robot.get_current_waypoint()
    joint_radian = current_waypoint['joint']
    result = robot.inverse_kin(joint_radian, target_pos, target_ori)
    if result is not None:
        target_joint = result['joint']
        # 移动机械臂
        robot.move_joint(target_joint)
        return True
    else:
        logger.error("逆解失败")
        return False

def apply_gripper_offset(end_pos, offset_z):
    """
    end_pos : 3×1，物体在末端系的位置
    offset_z: 抓手比末端长多少米（负号表示沿末端Ze负方向）
    return  : 3×1，抓手中心在末端系的位置
    """
    return end_pos + np.array([0, 0, offset_z])


def main():
    # 初始化 logger
    logger_init()

    # 创建机械臂控制类
    robot = Auboi5Robot()
    robot.create_context()

    try:
        # 连接到机械臂服务器
        ip = '192.168.1.101'
        port = 8899
        result = robot.connect(ip, port)
        if result != RobotErrorType.RobotError_SUCC:
            logger.error(f"连接服务器 {ip}:{port} 失败")
            return

        # 上电
        robot.robot_startup()

        # 初始化全局运动属性
        robot.init_profile()

        # 设置运动参数
        joint_maxvelc = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        joint_maxacc = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
        robot.set_joint_maxvelc(joint_maxvelc)
        robot.set_joint_maxacc(joint_maxacc)

        # 假设目标物体的像素坐标和深度值 
        pixel_coords = np.array([721,106,0.423])  # 示例数据


        # 去畸变处理
        undistorted_pixel = undistort_point(pixel_coords[:2], camera_matrix, dist_coeffs)
        undistorted_pixel_coords = np.array([undistorted_pixel[0], undistorted_pixel[1], pixel_coords[2]])

        # 转换为相机坐标系下的3D点
        camera_coords = pixel_to_camera(undistorted_pixel_coords, camera_matrix)

        # 转换为末端坐标系下的3D点
        end_effector_coords = transform_to_end_effector(camera_coords, hand_eye_matrix)
        
        #减去手爪长度

        gripper_center_end = apply_gripper_offset(end_effector_coords, offset_z=-0.15)

        # 获取基坐标系到末端坐标系的变换矩阵
        base_to_end_matrix = get_base_to_end_matrix(robot)

        # 转换为基坐标系下的3D点
        base_coords = transform_to_base(gripper_center_end, base_to_end_matrix)

        logger.info(f"目标物体在基坐标系下的坐标: {base_coords}")

        # 设置目标姿态（示例：保持当前姿态）
        current_waypoint = robot.get_current_waypoint()
        target_ori = current_waypoint['ori']

        # 移动机械臂到目标位置
        # success = move_to_target(robot, base_coords, target_ori)

        # j = get_current_joints(robot)   # 6 个弧度
        # j4_new = j[3] + radians(20)
        # #移动到下方
        # robot.move_joint((j[0], j[1], j[2], j4_new, j[4], j[5])) 
        # x,y,z,rx,ry,rz= get_current_xyz_rxryrz(robot)
        # robot.move_to_target_in_cartesian((x+0.05,y,z),(rx,ry,rz))
        # sleep(2)

        if success:
            logger.info("机械臂已移动到目标位置")
        else:
            logger.error("机械臂移动失败")

    except Exception as e:
        logger.error(f"发生错误: {e}")

    finally:
        # 断开连接
        if robot.connected:
            robot.disconnect()
        # 释放资源
        Auboi5Robot.uninitialize()
        logger.info("程序结束")

if __name__ == '__main__':
    main()

