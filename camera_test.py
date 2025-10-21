import cv2
import onnxruntime as ort
from PIL import Image
import numpy as np
import time
import pyrealsense2 as rs


# 置信度
confidence_thres = 0.5
# iou阈值
iou_thres = 0.4
# 类别
classes = {0: 'apple', 1: 'blocked'}
# 随机颜色BRG
color_palette = np.array([[0, 0, 255], [0, 255, 0]])

# 判断是使用GPU或CPU
providers = [
    'CPUExecutionProvider',  # 也可以设置CPU作为备选
]

# 相机内参
fx, fy, cx, cy = 910.77051, 908.94745, 643.65656, 362.56644
camera_matrix = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
])


def calculate_iou(box, other_boxes):
    # 计算交集的左上角坐标
    x1 = np.maximum(box[0], np.array(other_boxes)[:, 0])
    y1 = np.maximum(box[1], np.array(other_boxes)[:, 1])
    # 计算交集的右下角坐标
    x2 = np.minimum(box[0] + box[2], np.array(other_boxes)[:, 0] + np.array(other_boxes)[:, 2])
    y2 = np.minimum(box[1] + box[3], np.array(other_boxes)[:, 1] + np.array(other_boxes)[:, 3])
    # 计算交集区域的面积
    intersection_area = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    # 计算给定边界框的面积
    box_area = box[2] * box[3]
    # 计算其他边界框的面积
    other_boxes_area = np.array(other_boxes)[:, 2] * np.array(other_boxes)[:, 3]
    # 计算IoU值
    iou = intersection_area / (box_area + other_boxes_area - intersection_area)
    return iou


def custom_NMSBoxes(boxes, scores, confidence_threshold, iou_threshold):
    # 如果没有边界框，则直接返回空列表
    if len(boxes) == 0:
        return []
    # 将得分和边界框转换为NumPy数组
    scores = np.array(scores)
    boxes = np.array(boxes)
    # 根据置信度阈值过滤边界框
    mask = scores > confidence_threshold
    filtered_boxes = boxes[mask]
    filtered_scores = scores[mask]
    # 如果过滤后没有边界框，则返回空列表
    if len(filtered_boxes) == 0:
        return []
    # 根据置信度得分对边界框进行排序
    sorted_indices = np.argsort(filtered_scores)[::-1]
    # 初始化一个空列表来存储选择的边界框索引
    indices = []
    # 当还有未处理的边界框时，循环继续
    while len(sorted_indices) > 0:
        # 选择得分最高的边界框索引
        current_index = sorted_indices[0]
        indices.append(current_index)
        # 如果只剩一个边界框，则结束循环
        if len(sorted_indices) == 1:
            break
        # 获取当前边界框和其他边界框
        current_box = filtered_boxes[current_index]
        other_boxes = filtered_boxes[sorted_indices[1:]]
        # 计算当前边界框与其他边界框的IoU
        iou = calculate_iou(current_box, other_boxes)
        # 找到IoU低于阈值的边界框，即与当前边界框不重叠的边界框
        non_overlapping_indices = np.where(iou <= iou_threshold)[0]
        # 更新sorted_indices以仅包含不重叠的边界框
        sorted_indices = sorted_indices[non_overlapping_indices + 1]
    # 返回选择的边界框索引
    return indices


def draw_detections(img, box, score, class_id):
    # 提取边界框的坐标
    x1, y1, w, h = box
    # 根据类别ID检索颜色
    color = color_palette[class_id].tolist()
    # 在图像上绘制边界框
    cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)
    # 创建标签文本，包括类名和得分
    label = f'{classes[class_id]}: {score:.1f}'
    # 计算标签文本的尺寸
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    # 计算标签文本的位置（放在框上方中间位置）
    label_x = x1 + (w - label_width) // 2
    label_y = y1 - 10 if y1 - 10 > label_height else y1
    # 绘制填充的矩形作为标签文本的背景
    cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color, cv2.FILLED)
    # 在图像上绘制标签文本
    cv2.putText(img, label, (label_x - 40, label_y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    # 计算坐标文本的尺寸
    coord_text = "Center:()"  # 先初始化一个占位文本，用于计算尺寸
    (coord_text_width, coord_text_height), _ = cv2.getTextSize(coord_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    # 计算坐标文本的位置（放在类别标签下方合适位置）
    coord_text_x = label_x
    coord_text_y = label_y + label_height
    return label_x, label_y, coord_text_x, coord_text_y, coord_text_width, coord_text_height


def preprocess(img, input_width, input_height):
    # 获取输入图像的高度和宽度
    img_height, img_width = img.shape[:2]
    # 将图像颜色空间从BGR转换为RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 根据相机输入图像大小和模型期望输入大小调整图像尺寸（这里假设模型期望输入也是1280*720，如果不是需调整）
    img = cv2.resize(img, (input_width, input_height))
    # 通过除以255.0来归一化图像数据
    image_data = np.array(img) / 255.0
    # 转置图像，使通道维度为第一维
    image_data = np.transpose(image_data, (2, 0, 1))  # 通道首
    # 扩展图像数据的维度以匹配预期的输入形状
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
    # 返回预处理后的图像数据
    return image_data, img_height, img_width


def postprocess(input_image, output, input_width, input_height, img_width, img_height, depth_frame):
    # 转置和压缩输出以匹配预期的形状
    outputs = np.transpose(np.squeeze(output[0]))
    # 获取输出数组的行数
    rows = outputs.shape[0]
    # 用于存储检测的边界框、得分和类别ID的列表
    boxes = []
    scores = []
    class_ids = []
    apple_pixel_coords = []  # 用于存储苹果的像素坐标和深度信息
    # 计算边界框坐标的缩放因子
    x_factor = img_width / input_width
    y_factor = img_height / input_height
    # 遍历输出数组的每一行
    for i in range(rows):
        # 从当前行提取类别得分
        classes_scores = outputs[i][4:]
        # 找到类别得分中的最大得分
        max_score = np.amax(classes_scores)
        # 如果最大得分高于置信度阈值
        if max_score >= confidence_thres:
            # 获取得分最高的类别ID
            class_id = np.argmax(classes_scores)
            # 从当前行提取边界框坐标
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
            # 计算边界框的缩放坐标
            left = int((x - w / 2) * x_factor)
            top = int((y - h / 2) * y_factor)
            width = int(w * x_factor)
            height = int(h * x_factor)
            # 将类别ID、得分和框坐标添加到各自的列表中
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])
    # 应用非最大抑制过滤重叠的边界框
    indices = custom_NMSBoxes(boxes, scores, confidence_thres, iou_thres)
    selected_boxes = [boxes[i] for i in indices] if indices else []
    selected_scores = [scores[i] for i in indices] if indices else []
    selected_class_ids = [class_ids[i] for i in indices] if indices else []
    # 遍历非最大抑制后的选定索引
    for i in indices:
        # 根据索引获取框、得分和类别ID
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]
        # 在输入图像上绘制检测结果
        label_x, label_y, coord_text_x, coord_text_y, coord_text_width, coord_text_height = draw_detections(input_image, box, score, class_id)
        # 计算物体中心坐标（x坐标是边界框左边加上宽度一半，y坐标是边界框上边加上高度一半）
        center_x = int(box[0] + box[2] / 2)
        center_y = int(box[1] + box[3] / 2)
        depth_value = depth_frame.get_distance(center_x, center_y)
        apple_pixel_coords.append((center_x, center_y, depth_value))  # 存储苹果的像素坐标和深度信息
        # 显示像素坐标和深度信息
        coord_text = f"({center_x:.0f}, {center_y:.0f}, {depth_value:.2f}m)"
        cv2.putText(input_image, coord_text, (coord_text_x - 40, coord_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
    # 返回修改后的输入图像和苹果的像素坐标
    return input_image, apple_pixel_coords


def detect_object(image, session, model_inputs, input_width, input_height, depth_frame):
    # 如果输入的图像是PIL图像对象，将其转换为NumPy数组
    if isinstance(image, Image.Image):
        result_image = np.array(image)
    else:
        # 否则，直接使用输入的图像（假定已经是NumPy数组）
        result_image = image
    # 预处理图像数据，调整图像大小并可能进行归一化等操作
    img_data, img_height, img_width = preprocess(result_image, input_width, input_height)
    # 使用预处理后的图像数据进行推理
    outputs = session.run(None, {model_inputs[0].name: img_data})
    # 对推理结果进行后处理，例如解码检测框，过滤低置信度的检测等
    output_image, apple_coords = postprocess(result_image, outputs, input_width, input_height, img_width, img_height, depth_frame)
    # 返回处理后的图像和苹果的像素坐标
    return output_image, apple_coords


def init_detect_model(model_path):
    # 使用ONNX模型文件创建一个推理会话，并指定执行提供者
    session = ort.InferenceSession(model_path, providers=providers)
    # 获取模型的输入信息
    model_inputs = session.get_inputs()
    # 获取输入的形状，用于后续使用
    input_shape = model_inputs[0].shape
    # 从输入形状中提取输入宽度，这里假设模型输入形状符合预期且能正确获取整数值（需根据实际情况确认）
    input_width = 640
    # 从输入形状中提取输入高度，这里假设模型输入形状符合预期且能正确获取整数值（需根据实际情况确认）
    input_height = 640
    # 返回会话、模型输入信息、输入宽度和输入_height
    return session, model_inputs, input_width, input_height


def main():
    # 初始化模型
    model_path = "./picture/lab_ultra.onnx"  # 替换为你的模型路径
    session, model_inputs, input_width, input_height = init_detect_model(model_path)

    # 初始化相机管道
    pipeline = rs.pipeline()
    config = rs.config()
    # 配置彩色流
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    # 配置深度流
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    # 开始流
    pipeline.start(config)

    # 初始化帧率计数器
    start_time = time.time()
    frame_count = 0

    try:
        while True:
            # 等待相机帧
            frames = pipeline.wait_for_frames()
            # 获取彩色和深度帧
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            # 检查是否获取到帧
            if not color_frame or not depth_frame:
                continue

            # 将彩色帧转换为numpy数组
            color_image = np.asanyarray(color_frame.get_data())

            # 检测苹果
            start_inference = time.time()
            output_image, _ = detect_object(color_image, session, model_inputs, input_width, input_height, depth_frame)
            end_inference = time.time()
            inference_time = end_inference - start_inference

            # 更新帧率计数器
            frame_count += 1
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time

            # 在视频流中显示帧率和推理时间
            cv2.putText(output_image, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(output_image, f"Inference Time: {inference_time * 1000:.2f}ms", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            # 显示结果图像
            cv2.namedWindow('Apple Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Apple Detection', 800, 600)
            cv2.moveWindow('Apple Detection', 50, 50)
            cv2.imshow('Apple Detection', output_image)

            # 退出循环的按键
            key = cv2.waitKey(1)
            if key == 27 or key == ord('q'):
                break

    finally:
        # 停止相机管道
        pipeline.stop()
        # 关闭所有OpenCV窗口
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()