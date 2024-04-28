import os
import sys

import torch

import gradio as gr

sys.path.append('/')
sys.path.append('/yolov3')
from yolov3.detectorForGradio import prepare_model, predict_image
from yolov3.models.yolo_detector import YOLOV3TorchObjectDetector
from yolov3.main_gradcam_for_gradio import args as gradcam_args
from yolov3.main_gradcam_for_gradio import names as gradcam_names
from yolov3.main_gradcam_for_gradio import main as gradcam_main

import subprocess
import shutil

css = """
.column2 {
    padding: 100px 10px 0px;
}
.column34{
    padding: 40px 10px 0px;
}
.column2_btn{
    padding: 112px 0px 0px;
}
"""

# 把原图和各种处理的图片单独创建一个gradio文件夹存起来
PREDICTED_IMAGES = 'gradio/predicted_images/'
GRADCAM_IMAGES = 'gradio/gradcam_images/'

STYLE_IMAGE = 'gradio/style_images/tie1.jpg'

MASK_IMAGE = 'gradio/gradcam_images/{}/mask.jpg'
PATCH_IMAGE = 'gradio/attack_images/patch/{}'
ADV_IMAGE = 'gradio/attack_images/adv_img/{}'
DET_IMAGE = 'gradio/attack_images/det_img/{}'


def delete_files(folder_path):
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


# 原图检测detect
def wrapped_predict_image(image):
    global predict_model
    global PREDICTED_IMAGES
    delete_files(PREDICTED_IMAGES)
    save_dir = predict_image(model=predict_model, source=image, SAVE_DIR=PREDICTED_IMAGES)
    return os.path.join(save_dir, os.path.basename(image))


# 生成攻击区域gradcam
def wrapped_gradcam_main(image, attack):
    global gradcam_model
    global GRADCAM_IMAGES
    delete_files(GRADCAM_IMAGES)
    _, save_dir = gradcam_main(img_path=image, model=gradcam_model, SAVE_DIR=GRADCAM_IMAGES, attack_range=attack)
    result_img = os.path.join(save_dir, "15_0.jpg")
    mask = os.path.join(save_dir, "mask.jpg")
    return result_img, mask


# 攻击结果4.py 复制了一份改名为attack_main.py
def wrapped_attack_main(image, style=None):
    global detectorYolov3
    global MASK_IMAGE
    global PATCH_IMAGE
    global ADV_IMAGE
    global DET_IMAGE

    basename = os.path.basename(image)

    delete_files('gradio/attack_images/adv_img/')
    delete_files('gradio/attack_images/det_img/')
    delete_files('gradio/attack_images/patch/')

    command = [sys.executable, "attack_main.py",
               "--img", image,
               "--style", "None" if style is None else style,
               "--mask", MASK_IMAGE.format(os.path.splitext(basename)[0]),
               "--patch", PATCH_IMAGE.format(basename),
               "--adv", ADV_IMAGE.format(basename),
               "--det", DET_IMAGE.format(basename)]
    try:
        results = subprocess.check_output(command).decode()
        print(results)
        print("输出结果: ", results)
        left_pos = results.find('[', results.index('return'))
        right_pos = results.find(']', results.index('return'))
        paths = eval(results[left_pos:right_pos + 1])
        return paths[1], paths[2]
    except subprocess.CalledProcessError as e:
        print(e.output.decode())


# 前端页面代码
with gr.Blocks(css=css) as demo:
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(label="上传图像", type='filepath')
            style = gr.Image(label="上传风格图像", type='filepath')
            attack_param = gr.Slider(minimum=0, maximum=250, step=1, label="攻击区域", value=220)
        with gr.Column(elem_classes=["column2"]):
            output_img = gr.Image(interactive=False, show_label=False, height="30vh", type='filepath')
            with gr.Row(elem_classes=["column2_btn"]):
                detect_btn = gr.Button(value="原图检测")
                detect_btn.click(fn=wrapped_predict_image, inputs=input_img, outputs=output_img)
        with gr.Column(elem_classes=["column34"]):
            output_img_1 = gr.Image(interactive=False, show_label=False, type="filepath")
            output_img_2 = gr.Image(interactive=False, show_label=False, type="filepath")
            attack_btn = gr.Button(value="生成攻击区域")
            attack_btn.click(fn=wrapped_gradcam_main, inputs=[input_img, attack_param],
                             outputs=[output_img_1, output_img_2])
        with gr.Column(elem_classes=["column34"]):
            result_img_1 = gr.Image(interactive=False, label="对抗样本检测结果", type="filepath")
            result_img_2 = gr.Image(interactive=False, label="生成的补丁", type="filepath")
            result_btn = gr.Button(value="攻击结果")
            result_btn.click(fn=wrapped_attack_main, inputs=[input_img, style], outputs=[result_img_1, result_img_2])


if __name__ == '__main__':

    # 加载原图检测模型
    predict_model_path = './yolov3/weights/yolov3.pt'
    predict_model = prepare_model(weights=predict_model_path)

    # 加载热力图模型
    input_size = (gradcam_args.img_size, gradcam_args.img_size)
    gradcam_model = YOLOV3TorchObjectDetector(
        predict_model_path,
        gradcam_args.device,
        img_size=input_size,
        names=gradcam_names
    )

    demo.launch()
