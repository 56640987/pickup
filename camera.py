import cv2
import onnxruntime as ort
from PIL import Image
import numpy as np
import time
import pyrealsense2 as rs



# 置信度
confidence_thres = 0.6
# iou阈值
iou_thres = 0.6
# 类别
classes = {0: 'apple', 1: 'blocked'}
# classes = {0: 'tomato'}

# 随机颜色BRG

color_palette = np.array([[0, 0, 255] for _ in range(len(classes))])
# 判断是使用GPU或CPU
providers = [
    'CPUExecutionProvider',  # 也可以设置CPU作为备选
]

# 相机内参
fx, fy, cx, cy =  910.77051,908.94745,643.65656,362.56644
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
    x1, y1, w, h = box
    color = color_palette[class_id].tolist()

    # 1. 画框
    cv2.rectangle(img, (int(x1), int(y1)),
                  (int(x1 + w), int(y1 + h)), color, 2)

    # 2. 生成标签：类别:置信度
    label = f"{classes[class_id]}:{score:.1f}"

    # 3. 文字参数
    font, fs, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    (tw, th), _ = cv2.getTextSize(label, font, fs, thick)

    # 4. 计算位置：水平居中，紧贴框上方
    label_x = int(x1 + (w - tw) / 2)
    label_y = int(y1 - 25) if y1 - 25 > th else int(y1 + th + 5)

    # 5. 绘制文字
    cv2.putText(img, label, (label_x, label_y),
                font, fs, (0, 255, 0), thick, cv2.LINE_AA)

def letterbox(im, new_shape=640, color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    """
    将图像等比缩放并用灰色填充至 new_shape×new_shape
    返回：letterbox 后的图像、缩放比例 ratio、单边 padding(dw,dh)
    """
    shape = im.shape[:2]  # current shape [h, w]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r  # 宽、高同比例缩放
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)


def preprocess(img, input_width, input_height):
    """
    前置处理：BGR → RGB → letterbox → 归一化 → NCHW
    返回：网络输入张量、缩放比例 ratio、单边 padding(dw,dh)
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)          # BGR → RGB
    img_l, ratio, (dw, dh) = letterbox(img_rgb, new_shape=input_width)  # 等比缩放 + 灰边
    img_norm = img_l.astype(np.float32) / 255.0             # 0-1 归一化
    img_chw = np.transpose(img_norm, (2, 0, 1))             # HWC → CHW
    img_batch = np.expand_dims(img_chw, axis=0)             # 添加 batch 维度
    return img_batch, ratio, (dw, dh)


def postprocess(input_image,output,input_width,input_height,ratio,dw,dh,depth_frame):
    """
    后处理：解析 ONNX 输出 → NMS → 画框 → 返回结果
    参数：
        input_image     : 原始图像 (H,W,3)
        output          : ONNX 输出 list，output[0] 形状 (1,6,8400)
        input_width/height: 模型输入尺寸（640,640）
        ratio           : 缩放比例（来自 letterbox）
        dw, dh          : 单边 padding（来自 letterbox）
        depth_frame     : RealSense 深度帧
    返回：
        output_image    : 画好框的图像
        apple_pixel_coords: 苹果中心点列表 [(x,y,depth), ...]
    """
    # 将输出 reshape 为 (8400, 6)，方便遍历
    outputs = np.transpose(np.squeeze(output[0]))  # (8400, 6)
    rows = outputs.shape[0]

    boxes, scores, class_ids = [], [], []
    detections=[]

    for i in range(rows):
        cls_scores = outputs[i][4:]
        max_score = np.amax(cls_scores)
        if max_score >= confidence_thres:
            class_id = int(np.argmax(cls_scores))
            xc, yc, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]

            # 首先减去 padding，再除以缩放比例 → 还原到原图坐标
            x1 = (xc - w / 2 - dw) / ratio
            y1 = (yc - h / 2 - dh) / ratio
            w_norm = w / ratio
            h_norm = h / ratio

            # 转 int
            x1, y1, w_norm, h_norm = int(x1), int(y1), int(w_norm), int(h_norm)

            boxes.append([x1, y1, w_norm, h_norm])
            scores.append(float(max_score))
            class_ids.append(class_id)

    # NMS
    indices = custom_NMSBoxes(boxes, scores, confidence_thres, iou_thres)
    selected_boxes = [boxes[i] for i in indices] if indices else []
    selected_scores = [scores[i] for i in indices] if indices else []
    selected_class_ids = [class_ids[i] for i in indices] if indices else []
    

    # 画框
    for box, score, cls_id in zip(selected_boxes, selected_scores, selected_class_ids):
        draw_detections(input_image, box, score, cls_id)

        # 深度信息
        cx = box[0] + box[2] // 2
        cy = box[1] + box[3] // 2
        depth = depth_frame.get_distance(cx, cy)

        detections.append({
            'class_id': cls_id,
            'class_name': classes[cls_id],
            'cx': cx,
            'cy': cy,
            'depth': depth,
            'box': box
        })


        coord_text = f"({cx},{cy},{depth:.2f}m)"
        cv2.putText(input_image, coord_text,
                    (box[0], box[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    # return input_image, apple_pixel_coords
    return input_image, detections


def detect_object(image, session, model_inputs, input_width, input_height, depth_frame):
    if isinstance(image, Image.Image):
        result_image = np.array(image)
    else:
        result_image = image

    # 预处理：返回的是 img_batch, ratio, (dw, dh)
    img_data, ratio, (dw, dh) = preprocess(result_image, input_width, input_height)

    # 推理
    outputs = session.run(None, {model_inputs[0].name: img_data})

    # 后处理：传入 ratio 和 dw/dh，而不是 img_width/img_height
    output_image, apple_coords = postprocess(result_image, outputs, input_width, input_height, ratio, dw, dh, depth_frame)

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

