import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
from pickletools import uint8
import random
import time
import argparse
import numpy as np
from models.gradcam import YOLOV3GradCAM, YOLOV3GradCAMPP
from models.yolo_detector import YOLOV3TorchObjectDetector
import cv2
from style_transfer import load_img

names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']  # class names
# yolov3网络中的三个detect层
target_layers = ['model_15_act', 'model_22_act', 'model_27_1_cv2_act']
# Arguments
parser = argparse.ArgumentParser()
# Gradio用这个
# parser.add_argument('--model-path', type=str, default="./yolov3/weights/yolov3.pt", help='Path to the model')
# parser.add_argument('--img-path', type=str, default='./yolov3/data/images', help='input image path')
# parser.add_argument('--output-dir', type=str, default='./yolov3/outputs/', help='output dir')

# Main用这个
parser.add_argument('--model-path', type=str, default="./weights/yolov3.pt", help='Path to the model')
# parser.add_argument('--img-path', type=str, default='./data/images', help='input image path')
# parser.add_argument('--output-dir', type=str, default='./outputs/', help='output dir')

# parser.add_argument('--output-dir', type=str, default='./outputs/', help='output dir')
parser.add_argument('--img-size', type=int, default=640, help="input image size")
parser.add_argument('--target-layer', type=str, default='model_15_act',
                    help='The layer hierarchical address to which gradcam will applied,'
                         ' the names should be separated by underline')
parser.add_argument('--method', type=str, default='gradcam', help='gradcam method')
parser.add_argument('--device', type=str, default='cpu', help='cuda or cpu')
parser.add_argument('--no_text_box', action='store_true',
                    help='do not show label and box on the heatmap')
args = parser.parse_args()


def get_res_img(bbox, mask, res_img):
    mask = mask.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy().astype(
        np.uint8)
    heatmap = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    n_heatmat = (heatmap / 255).astype(np.float32)
    res_img = res_img / 255
    res_img = cv2.add(res_img, n_heatmat)
    res_img = (res_img / res_img.max())
    return res_img, n_heatmat, mask


def plot_one_box(x, img, color=None, label=None, line_thickness=3):
    # this is a bug in cv2. It does not put box on a converted image from torch unless it's buffered and read again!
    cv2.imwrite('temp.jpg', (img * 255).astype(np.uint8))
    img = cv2.imread('temp.jpg')

    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    return img


# 检测单个图片
def main(img_path, model, SAVE_DIR=None, attack_range=220):
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    device = args.device
    input_size = (args.img_size, args.img_size)
    # 读入图片
    img = cv2.imread(img_path)  # 读取图像格式：BGR
    img_size = (img.shape[0], img.shape[1])
    # print('[INFO] Loading the model')

    # img[..., ::-1]: BGR --> RGB
    # (480, 640, 3) --> (1, 3, 480, 640)
    # torch_img = model.preprocessing(img[..., ::-1])

    torch_img = load_img(img_path, args.img_size)

    tic = time.time()
    mask_im = np.zeros(img_size)

    # 遍历三层检测层
    for target_layer in target_layers:
        # 获取grad-cam方法
        if args.method == 'gradcam':
            saliency_method = YOLOV3GradCAM(model=model, layer_name=target_layer, img_size=input_size)
        elif args.method == 'gradcampp':
            saliency_method = YOLOV3GradCAMPP(model=model, layer_name=target_layer, img_size=input_size)
        masks, logits, [boxes, _, class_names, conf] = saliency_method(torch_img)  # 得到预测结果
        result = torch_img.squeeze(0).mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).detach().cpu().numpy()
        result = result[..., ::-1]  # convert to bgr
        # 保存设置
        imgae_name = os.path.basename(img_path)  # 获取图片名
        if SAVE_DIR is None:
            save_path = f'{args.output_dir}{imgae_name[:-4]}/'
        else:
            save_path = f'{SAVE_DIR}{imgae_name[:-4]}/'
        # save_path = 'mask/'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        print(f'[INFO] Saving the final image at {save_path}')
        # 遍历每张图片中的每个目标
        for i, mask in enumerate(masks):
            # 遍历图片中的每个目标
            res_img = result.copy()
            # 获取目标的位置和类别信息
            bbox, cls_name = boxes[0][i], class_names[0][i]
            if (cls_name == 'stop sign'):  # 设置哪一类别的热力图
                label = f'{cls_name} {conf[0][i]}'  # 类别+置信分数
                # 获取目标的热力图
                res_img, heat_map, mask = get_res_img(bbox, mask, res_img)
                res_img = plot_one_box(bbox, res_img, label=label, color=colors[int(names.index(cls_name))],
                                       line_thickness=3)
                tmp = np.zeros((res_img.shape[0], res_img.shape[1], 1))
                tmp[bbox[1]:bbox[3], bbox[0]:bbox[2], ] = 1
                mask = mask * tmp
                # 缩放到原图片大小
                res_img = cv2.resize(res_img, dsize=(img.shape[:-1][::-1]))
                mask = cv2.resize(mask, dsize=(img.shape[:-1][::-1]))
                output_path = f'{save_path}/{target_layer[6:8]}_{i}.jpg'
                cv2.imwrite(output_path, res_img)
                print(f'{target_layer[6:8]}_{cls_name}.jpg done!!')
                mask_im += mask
    mask_im = np.where(mask_im < attack_range, 0, 255)  # 设置阈值
    # print(len(mask_im[mask_im==255]))
    # print(len(mask_im[mask_im==0]))
    num = len(mask_im[mask_im == 255])

    print(mask_im.shape)
    output_path = f'{save_path}/mask.jpg'
    # mask_path = f'./mask/{imgae_name}'
    cv2.imwrite(output_path, mask_im)
    # cv2.imwrite(mask_path, mask_im)

    print(f'mask_{cls_name}.jpg done!!')
    print(f'Total time : {round(time.time() - tic, 4)} s')
    return num, save_path


if __name__ == '__main__':
    IMAGES_PATH = '../gradio/images/'
    SAVE_DIR = '../gradio/gradcam_images/'
    device = args.device
    input_size = (args.img_size, args.img_size)
    model = YOLOV3TorchObjectDetector(args.model_path, device, img_size=input_size, names=names)

    images = [os.path.join(IMAGES_PATH, file) for file in os.listdir(IMAGES_PATH)]
    for image in images:
        main(image, model, SAVE_DIR=SAVE_DIR)
