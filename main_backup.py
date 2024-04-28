from __future__ import print_function

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import sys

sys.path.append('/PyTorchYOLOv3')
from torch.autograd import Variable
import argparse
from torchvision.utils import save_image

from style_transfer_4 import *
from PyTorchYOLOv3.detect import DetectorYolov3
from patch import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
         'hair drier', 'toothbrush']  # class names

parser = argparse.ArgumentParser()
parser.add_argument('--img-size', type=int, default=416, help="input image size")

parser.add_argument('--img-path', type=str, default='yolov3/data/images', help='input image path')
parser.add_argument('--output-dir', type=str, default='output_adv/', help='output dir')
parser.add_argument("--content_weight", dest='content_weight', nargs='?', type=float,
                    help="weight of content loss", default=5e0)
parser.add_argument("--style_weight", dest='style_weight', nargs='?', type=float,
                    help="weight of style loss", default=1e2)
parser.add_argument("--tv_weight", dest='tv_weight', nargs='?', type=float,
                    help="weight of total variational loss", default=1e-3)
parser.add_argument("--attack_weight", dest='attack_weight', nargs='?', type=float,
                    help="weight of attack loss", default=5e3)
parser.add_argument('--mode', type=str, default='disappeared', help='untargeted/targeted/disappeared')

args = parser.parse_args()


def main(item, detectorYolov3):
    style_img_path = './yolov3/tie2.jpg'
    content_img_path = f'yolov3/data/test/{item}'
    image_name = os.path.basename(content_img_path)[:-4]
    mask_path = f'yolov3/outputs/{image_name}/mask.jpg'  # 获取mask
    patch_path = f'{args.output_dir}/patch/{image_name}.jpg'  # 最终补丁存储路径
    save_path = f'{args.output_dir}/adv_img/{image_name}.jpg'  # 最终对抗样本存储路径
    save_detpath = f'{args.output_dir}/det_img/{image_name}.jpg'  # 对抗样本检测效果存储路径

    num_epoches = 1000

    detector = detectorYolov3
    # yoloModel = YOLOV3TorchObjectDetector(args.model_path, device, img_size=input_size, names=names)
    loss_func = torch.nn.CrossEntropyLoss()

    # 加载不规则mask以及未攻击的干净图片
    # mask_img = load_img(mask_path,(args.img_size, args.img_size)).to(device)
    clear_image = load_img(content_img_path, (args.img_size, args.img_size)).to(device)

    # 获取mask最小外接矩阵位置
    tmp, loc = draw_min_rect_rectangle(mask_path, args.img_size)
    mask = get_patch(mask_path, loc, 416).to(device)

    patch = get_patch(content_img_path, loc, args.img_size).to(device)

    # 风格图片和内容图片
    content_img = get_patch(content_img_path, loc, args.img_size).to(device)
    style_img = load_img(style_img_path, (loc[2], loc[3])).to(device)

    # 风格迁移模型
    vgg = models.vgg19(pretrained=True).features
    vgg = vgg.to(device)
    model, style_loss_list, content_loss_list = get_style_model_and_loss(style_img, content_img, vgg)

    patch = Variable(patch)
    patch.requires_grad = True
    optimizer = torch.optim.Adam([patch], lr=0.01)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=50)
    epoch = 0
    maxl = 0
    flag = 0
    exp = 0
    att_num = 11
    target_num = 11
    target = torch.tensor(np.eye(80)[target_num]).unsqueeze(0).to(device)
    att_cls = torch.tensor(np.eye(80)[att_num]).unsqueeze(0).to(device)
    patch_applier = PatchApplier().cuda()

    while epoch < num_epoches:
        patch.data.clamp_(0, 1)  # 更新图像的数据
        optimizer.zero_grad()

        adv_image = patch_applier(clear_image, patch, mask, loc, args.img_size)
        max_prob_obj_cls, overlap_score, det, logits = detector.detect(input_imgs=adv_image, cls_id_attacked=att_num,
                                                                       with_bbox=False)
        try:
            model(patch)
        except:
            exp = 1
            break
        style_score = 0
        content_score = 0

        # 计算总损失，并得到各个损失的梯度
        for sl in style_loss_list:
            style_score = style_score + sl.backward()
        for cl in content_loss_list:
            content_score = content_score + cl.backward()
            # tv_loss = total_variation(patch)
        if (args.mode == 'untargeted'):
            label = det[0][6]
            if (label.data != target_num) and (11 not in det[:, 6]):
                save_image(adv_image, save_path)
                save_image(patch, patch_path)
                print("success,label={}".format(label))
                break
            loss_det = -loss_func(logits, target)
        elif (args.mode == 'targeted'):
            label = det[0][6]
            if (label.data == target_num):
                save_image(adv_image, save_path)
                save_image(patch, patch_path)
                print("success,label={}".format(label))
                break
            loss_det = loss_func(logits, target)
        elif (args.mode == 'disappeared'):
            if det is None:
                save_image(adv_image, save_path)
                save_image(patch, patch_path)
                flag = 1
                break
            # if (11 not in det[:,6]) and len(det[:,6])<maxl:
            #     save_image(adv_image, save_path)
            #     save_image(patch, patch_path)
            #     flag = 1
            #     break

            loss_det = torch.mean(max_prob_obj_cls)

        obj_cls = det[:, 4]
        loss_fab = torch.mean(obj_cls)
        label = det[0][6]
        # loss_det.backward(retain_graph=True)
        if epoch == 0:
            maxl = len(det[:, 6])

        loss = loss_det * 10000 + content_score + style_score
        if (label != att_num):
            loss = loss_det * 10 + loss_fab * 10000 + content_score * 10 + style_score * 100
        loss.backward(retain_graph=True)

        # print("epoch={} loss={} label = {}".format(epoch,loss_det,label))

        # print("epoch={} loss_det={} loss_fab={} label = {} Style Loss: {:.4f} Content Loss: {:.4f}"
        #       .format(epoch, loss_det, loss_fab, label, style_score.data.item(), content_score.data.item()))

        epoch += 1

        save_image(adv_image, save_path)  # 存储对抗样本
        save_image(patch, patch_path)  # 存储补丁
        optimizer.step()
        # scheduler.step(loss)

    max_prob_obj_cls, overlap_score, det, logits, bboxes = detector.detect(input_imgs=adv_image, cls_id_attacked=11,
                                                                           clear_imgs=content_img)
    print(item, ":", det)  # 打印检测结果
    det_adv_image = detector.plot(adv_image, names, det, 0.5)  # 检测对抗样本
    save_image(det_adv_image, save_detpath)  # 存储检测结果


def detect(item, detectorYolov3):
    # content_img_path = f'output_adv/adv_img/{item}'
    content_img_path = item
    image_name = os.path.basename(content_img_path)[:-4]
    mask_path = f'yolov3/outputs/{image_name}/mask.jpg'  # 获取mask
    save_detpath = f'{args.output_dir}/detect/{image_name}.jpg'  # 对抗样本检测效果存储路径
    detector = detectorYolov3
    clear_image = load_img(content_img_path, (args.img_size, args.img_size)).to(device)
    tmp, loc = draw_min_rect_rectangle(mask_path, args.img_size)
    mask = get_patch(mask_path, loc, 416).to(device)

    patch = get_patch(content_img_path, loc, args.img_size).to(device)

    content_img = get_patch(content_img_path, loc, args.img_size).to(device)

    patch = Variable(patch)
    patch.requires_grad = True
    patch_applier = PatchApplier().cuda()

    adv_image = patch_applier(clear_image, patch, mask, loc, args.img_size)
    max_prob_obj_cls, overlap_score, det, logits, bboxes = detector.detect(input_imgs=adv_image, cls_id_attacked=11,
                                                                           clear_imgs=content_img)
    det_adv_image = detector.plot(adv_image, names, det, 0.5)  # 检测对抗样本
    save_image(det_adv_image, save_detpath)  # 存储检测结果
    return os.path.dirname(os.path.abspath(save_detpath))

if __name__ == "__main__":
    image1 = '000000026162.jpg'
    image2 = '000000015157.jpg'
    image3 = '000000025797.jpg'
    image4 = '000000097369.jpg'
    image5 = '000000440298.jpg'
    image6 = '000000529777.jpg'
    detectorYolov3 = DetectorYolov3(show_detail=False, tiny=True)
    images = [image1, image2, image3, image4, image5, image6]
    for image in images:
        detect(image, detectorYolov3)
        # main(image, detectorYolov3)
    # detectorYolov3.detect(image1, cls_id_attacked=11)
    # main(image2, detectorYolov3)
