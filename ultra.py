from PyQt5 import QtCore,QtGui,QtWidgets,uic
from PyQt5.QtWidgets import QApplication, QMainWindow,QDesktopWidget
import sys
import cv2
from robotcontrol import RobotErrorType
from robotcontrol import Auboi5Robot,logger_init
import pyrealsense2 as rs
import numpy as np
from Serial import port_open, port_send_open, port_send_close, port_close
from my_api import project_detect,move_realse,project_reset,pick_up,get_current_joints,get_current_xyz_rxryrz
from camera import init_detect_model, detect_object
from time import sleep
import logging 
import time
from math import radians
logger = logging.getLogger('main.robotcontrol')



#登陆界面
def center(UI):
    screen = QDesktopWidget().screenGeometry()
    size = UI.geometry()
    UI.move((screen.width() - size.width()) // 2, (screen.height() - size.height()) // 2)

class LoginWindow():
    def __init__(self):
        self.ui = uic.loadUi('./UI/login.ui')
        self.user_name = self.ui.lineEdit_name          #用户名
        self.user_password = self.ui.lineEdit_pwd  #密码
        self.login_btn = self.ui.pushButton_log         #登录按钮
        self.quit_btn = self.ui.pushButton_quit
        self.init_win_slots()
        center(self.ui)

    #信号连接
    def init_win_slots(self):
        self.user_password.returnPressed.connect(self.login)
        self.login_btn.clicked.connect(self.login)
        self.quit_btn.clicked.connect(app.quit)

    def login(self):
        print("You pressed sign in")
        name = self.user_name.text().strip().strip()
        password = self.user_password.text().strip()
        if name == "123" and password == "1":
            self.win_new =MainWindow ()
            self.ui.close()
        else:
            print("用户名与密码输入错误")

#主界面
class MainWindow():
    def __init__(self):
        self.ui = uic.loadUi('./UI/main_new.ui')
        self.pipeline = None  # 用于存储相机 pipeline 对象
        self.align = None     # 用于对齐深度图和彩色图像
        self.session = None   # 用于存储模型会话
        self.model_inputs = None
        self.input_width = 640
        self.input_height = 480


        #功能按钮
        self.all_reset = self.ui.pushButton_reset             #复位
        self.auto_pickup = self.ui.pushButton_pickup               #自动采摘        
        self.recognize = self.ui.pushButton_recognize               #打开相机
        self.open_claw = self.ui.pushButton_claw_open         #夹爪打开
        self.close_claw = self.ui.pushButton_claw_close       #夹爪闭合        
        self.lock_claw = self.ui.pushButton_lock                         #锁定
        self.unlock_claw = self.ui.pushButton_unlock                  #解锁
        self.move_recognize = self.ui.pushButton_recognize_location        #识别位置
        self.move_release = self.ui.pushButton_place            #释放位置
        self.quit = self.ui.pushButton_quit                       #退出

        self.RGB_picture = self.ui.label_picture                  #图像检测框
        self.robot_pos = self.ui.textBrowser_coordinate       #苹果坐标显示
        self.picture_text = self.ui.textBrowser_info          #其他信息显示

        self.init_robot_slots()
        center(self.ui)
        self.ui.show()
    
    #动作控制
    def init_robot_slots(self):
        self.all_reset.clicked.connect(lambda:project_reset())       #复位
        self.auto_pickup.clicked.connect(self.auto_pick)    #自动采摘
        self.recognize.clicked.connect(self.start_camera_and_detect)  #打开相机
        self.open_claw.clicked.connect(self.claw_open) #打开手爪
        self.close_claw.clicked.connect(self.claw_close) #关闭手爪
        self.lock_claw.clicked.connect(self.lock)    #锁定手爪
        self.unlock_claw.clicked.connect(self.unlock)    #解锁手爪
        self.move_recognize.clicked.connect(lambda:project_detect())  #识别位置
        self.move_release.clicked.connect(lambda:move_realse())  #释放位置
        self.quit.clicked.connect(self.app_quit)  #退出
    
    def claw_open(self):
        # port_open()
        port_send_open()
        self.ui.textBrowser_info.setText("手爪已打开")


    
    def claw_close(self):
        # port_open()
        port_send_close()
        self.ui.textBrowser_info.setText("手爪已闭合")

        

    def lock(self):
        port_close()
        self.ui.textBrowser_info.setText("手爪已锁定")
    
    def unlock(self):
        port_open()
        self.ui.textBrowser_info.setText("手爪已解锁")

    def start_camera_and_detect(self):
        # 打开相机并开始识别
        self.pipeline, self.align = self._open_camera()
        if self.pipeline:
            # 初始化模型
            self.session, self.model_inputs, _, _ = init_detect_model('./picture/CCFM-ShuffleNetV2_150.onnx')
           # 启动一个线程来更新视频流并进行检测
            self.timer = QtCore.QTimer()
            self.timer.timeout.connect(self.detect_and_update_frames)
            self.timer.start(30)  # 每30ms更新一次
            self.ui.textBrowser_info.setText("相机已打开")

    def detect_and_update_frames(self):
    # 获取帧
        frames = self.pipeline.wait_for_frames()
        # 对齐深度图和彩色图像
        aligned_frames = self.align.process(frames)
        # 获取对齐后的深度帧和彩色帧
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return

        # 转换为 numpy 数组
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # 进行目标检测
        output_image, detections = detect_object(
            color_image, self.session, self.model_inputs,
            self.input_width, self.input_height, depth_frame
        )

        # 1. 过滤 + 排序（只做一次）
        apple = [d for d in detections if d['class_name'] == 'apple']
        apple = sorted(apple, key=lambda x: x['depth'])
        blocked = [d for d in detections if d['class_name'] == 'blocked']
        blocked = sorted(blocked, key=lambda x: x['depth'])

        # 2. 保存到成员变量（抓取用）

        self.apple_pixel_coords = apple
        self.blocked_pixel_coords = blocked
        # 3. 显示坐标（两类都展示）
        text = "识别结果：\n"
        for idx, a in enumerate(apple, 1):
            text += f"苹果 {idx}: ({a['cx']}, {a['cy']}, {a['depth']:.3f}m)\n"
        for idx, b in enumerate(blocked, 1):
            text += f"遮挡 {idx}: ({b['cx']}, {b['cy']}, {b['depth']:.3f}m)\n"
        self.ui.textBrowser_coordinate.setText(text if apple or blocked else "未检测到目标")

        # 4. 图像显示流程
        output_image = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        if output_image is None or not isinstance(output_image, np.ndarray):
            return
        if output_image.dtype != np.uint8:
            output_image = (output_image * 255).astype(np.uint8)

        output_image_rgb = cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB)
        color_qimage = self.convert_to_qimage(output_image_rgb)
        if color_qimage is None:
            return

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
        depth_qimage = self.convert_to_qimage(depth_colormap)

        # 显示在 QLabel 中
        self.ui.label_picture.setPixmap(QtGui.QPixmap.fromImage(color_qimage))



    def auto_pick(self):
        if not self.apple_pixel_coords and not self.blocked_pixel_coords:
            self.ui.textBrowser_info.setText("未检测到任何目标")
            return

        # 初始化机械臂控制
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

            for a in self.apple_pixel_coords:
                x,y,depth = a['cx'], a['cy'], a['depth']
                if 0.1< depth < 0.6:
                    port_open()          
                    port_send_open()
                    sleep(0.5)
                    # y=int(round(y+(45.95/depth)))
                    pick_up(robot, (x,y,depth)) #46.05是苹果实际大小×焦距fy/1000
                    print(x,y,depth)
                    sleep(1)

                    #移动4关节
                    # j = get_current_joints(robot)   # 6 个弧度  
                    # robot.move_joint((j[0], j[1], j[2], j[3]+radians(20), j[4], j[5]))   
                    
                    #抓取
                    port_send_close()
                    port_send_close()
                    port_send_close()

                    robot.move_to_target_in_cartesian((0.213,-0.12,0.4),(90,0,90))
                    # 放
                    sleep(1)
                    port_send_open()
                    sleep(0.5)
                    # j=get_current_joints(robot)
                    # robot.move_joint((j[0]+radians(45),j[1],j[2],j[3],j[4],j[5]))
                    # project_detect()
                    
                    
                    port_send_close()
                    port_close()
                    sleep(0.5)


            for b in self.blocked_pixel_coords:
                x, y, depth = b['cx'], b['cy'], b['depth']
                if 0.1< depth < 0.6:
                    port_open()          
                    port_send_open()
                    sleep(0.5)

                    # y=int(round(y+(45.95/depth)))
                    pick_up(robot,(x,y,depth))           #移动到目标点
                    sleep(1)

                    #移动4关节
                    # j = get_current_joints(robot)   # 6 个弧度  
                    # robot.move_joint((j[0], j[1], j[2], j[3]+radians(20), j[4], j[5]))   
                    
                    #抓取
                    port_send_close()
                    port_send_close()
                    port_send_close()

                    robot.move_to_target_in_cartesian((0.213,-0.12,0.4),(90,0,90))
                    # 放
                    sleep(1)
                    port_send_open()
                    sleep(0.5)
                    # j=get_current_joints(robot)
                    # robot.move_joint((j[0]+radians(45),j[1],j[2],j[3],j[4],j[5]))
                    # project_detect()
                    
                    
                    port_send_close()
                    port_close()
                    sleep(0.5)

        finally:
            # 断开连接
            if robot.connected:
                robot.disconnect()
            # 释放资源
            Auboi5Robot.uninitialize()



    def app_quit(self):
        # 退出应用程序前确保相机已关闭
        if self.pipeline:
            self.timer.stop()
            self._close_camera(self.pipeline)
        sys.exit()

    def _open_camera(self):
        # 创建管道和配置对象
        pipeline = rs.pipeline()
        config = rs.config()
        
        # 配置深度和颜色流，设置分辨率为 
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        
        # 开始流
        profile = pipeline.start(config)
        
        # 创建对齐对象，将深度图对齐到彩色图像
        align_to = rs.stream.color
        align = rs.align(align_to)
        
        return pipeline, align

    def _close_camera(self, pipeline):
        # 停止流
        pipeline.stop()

    def convert_to_qimage(self, image):
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        # 将 BGR 图像转换为 RGB 图像
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 转换图像为 QImage 格式
        height, width, channel = image_rgb.shape
        bytes_per_line = 3 * width
        qimage = QtGui.QImage(image_rgb.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        return qimage        # 转换图像为 QImage 格式


if __name__ == "__main__":
    app = QApplication(sys.argv)
    login_window = LoginWindow()
    login_window.ui.show()
    sys.exit(app.exec_())
