import cv2
import onnxruntime as ort
from PIL import Image
import numpy as np

# 置信度
confidence_thres = 0.35
# iou阈值
iou_thres = 0.5
# 类别
classes = {0: 'apple', 1: 'blocked'}
# classes = {0: 'tomato'}

# 随机颜色
color_palette = np.random.uniform(100, 255, size=(len(classes), 1))

# 判断是使用GPU或CPU，这里先只使用CPUExecutionProvider作为示例，如果后续GPU环境配置好可按需求调整
providers = [
    'CPUExecutionProvider',
]


def calculate_iou(box, other_boxes):
    """
    计算给定边界框与一组其他边界框之间的交并比（IoU）。

    参数：
    - box: 单个边界框，格式为 [x1, y1, width, height]。
    - other_boxes: 其他边界框的数组，每个边界框的格式也为 [x1, y1, width, height]。

    返回值：
    - iou: 一个数组包含给定边界框与每个其他边界框的IoU值。
    """

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
    """
    在输入图像上绘制检测到的对象的边界框和标签。

    参数:
            img: 要在其上绘制检测结果的输入图像。
            box: 检测到的边界框。
            score: 对应的检测得分。
            class_id: 检测到的对象的类别ID。

    返回:
            无
    """

    # 提取边界框的坐标
    x1, y1, w, h = box
    # 根据类别ID检索颜色
    color = color_palette[class_id]
    # 在图像上绘制边界框
    cv2.rectangle(img, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 1)
    # 创建标签文本，包括类名和得分w
    label = f'{classes[class_id]}: {score:.2f}'
    # 计算标签文本的尺寸
    (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,0, 1)
    # 计算标签文本的位置
    label_x = x1
    label_y = y1 - 10 if y1 - 10 > label_height else y1 + 10
    # 绘制填充的矩形作为标签文本的背景
    cv2.rectangle(img, (label_x, label_y - label_height), (label_x + label_width, label_y + label_height), color,
                  cv2.FILLED)
    # 在图像上绘制标签文本
    cv2.putText(img, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)


def preprocess(img, input_width, input_height):
    """
    在执行推理之前预处理输入图像。

    返回:
        image_data: 为推理准备好的预处理后的图像数据。
    """

    # 获取输入图像的高度和宽度
    img_height, img_width = img.shape[:2]
    # 将图像颜色空间从BGR转换为RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # 确保input_width和input_height为整数类型进行图像尺寸调整
    input_width = int(input_width)
    input_height = int(input_height)
    img = cv2.resize(img, (input_width, input_height))
    # 通过除以255.0来归一化图像数据
    image_data = np.array(img) / 255.0
    # 转置图像，使通道维度为第一维
    image_data = np.transpose(image_data, (2, 0, 1))  # 通道首
    # 扩展图像数据的维度以匹配预期的输入形状
    image_data = np.expand_dims(image_data, axis=0).astype(np.float32)
    # 返回预处理后的图像数据以及原始图像的尺寸信息
    return image_data, img_height, img_width


def postprocess(input_image, output, input_width, input_height, img_width, img_height):
    # 转置和压缩输出以匹配预期的形状
    outputs = np.transpose(np.squeeze(output[0]))
    # 获取输出数组的行数
    rows = outputs.shape[0]
    # 用于存储检测的边界框、得分和类别ID的列表
    boxes = []
    scores = []
    class_ids = []
    # 计算边界框坐标的缩放因子
    x_factor = img_width / input_width
    y_factor = img_height / input_height
    # 遍历输出数组的每一行
    # 假设模型输出 x_center,y_center,w,h
    for i in range(rows):
        classes_scores = outputs[i][4:]
        max_score = np.amax(classes_scores)
        if max_score >= confidence_thres:
            class_id = np.argmax(classes_scores)
            cx, cy, w, h = outputs[i][:4]

            # 缩放到原图
            left   = int((cx - w/2) * img_width  / input_width)
            top    = int((cy - h/2) * img_height / input_height)
            right  = int((cx + w/2) * img_width  / input_width)
            bottom = int((cy + h/2) * img_height / input_height)

            boxes.append([left, top, right - left, bottom - top])
            scores.append(max_score)
            class_ids.append(class_id)
    # 应用非最大抑制过滤重叠的边界框
    indices = custom_NMSBoxes(boxes, scores, confidence_thres, iou_thres)
    # 遍历非最大抑制后的选定索引
    for i in indices:
        # 根据索引获取框、得分和类别ID
        box = boxes[i]
        score = scores[i]
        class_id = class_ids[i]
        # 在输入图像上绘制检测结果
        draw_detections(input_image, box, score, class_id)

    # return input_image

    # 在画框等检测操作完成后，将图像调整为640 * 480大小
    output_image = cv2.resize(input_image, (1280, 720))
    # 返回调整尺寸后的图像
    return output_image

def init_detect_model(model_path):
    session = ort.InferenceSession(model_path, providers=providers)
    model_inputs = session.get_inputs()
    input_shape = model_inputs[0].shape        # 例如 [1,3,640,640]
    print("Real input_shape:", input_shape)

    # 防止出现 'width'/'height' 字符串
    if any(isinstance(dim, str) for dim in input_shape):
        # 如果 shape 里有字符串，手动指定常用尺寸
        input_height = 640
        input_width  = 640
    else:
        input_height = int(input_shape[2])
        input_width  = int(input_shape[3])

    print(f"Using {input_width}×{input_height}")
    return session, model_inputs, input_width, input_height



def detect_object(image, session, model_inputs, input_width, input_height):
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
    output_image = postprocess(result_image, outputs, input_width, input_height, img_width, img_height)
    return output_image

if __name__ == '__main__':
    # 模型文件的路径
    model_path = "./picture/lab_ultra.onnx"
    # model_path = "./picture/CCFM-ShuffleNetV2.onnx"
    # 初始化检测模型，加载模型并获取模型输入节点信息和输入图像的宽度、高度
    session, model_inputs, input_width, input_height = init_detect_model(model_path)

        # 读取图像文件
    image_path = "./blocked_Color.png"
    image_data = cv2.imread(image_path)

    # 使用检测模型对读入的图像进行对象检测
    result_image = detect_object(image_data, session, model_inputs, input_width, input_height)
    # 将检测后的图像保存到文件（保存尺寸为640 * 480）
    cv2.imwrite("lab_ultra.jpg", result_image)
    # cv2.imwrite("CCFM-ShuffleNetV2.jpg", result_image)
    # 等待用户按键，然后关闭显示窗口
    cv2.waitKey(0)





