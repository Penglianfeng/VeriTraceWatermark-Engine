import numpy as np
import gradio as gr
from gzhu import init, infer
from gzhu_utils import load_image_from_path, closing_resize
import os
from tqdm import tqdm
import PIL
from PIL import Image, ImageOps


def reverse_mask(mask):
    r, g, b, a = mask.split()
    mask = PIL.Image.merge('RGB', (r, g, b))
    return ImageOps.invert(mask)


config = init()
target_image_path = os.path.join(os.getcwd(), 'MIST.png')


def process_image(image, eps, steps, input_size, rate, mode, block_mode, no_resize):
    print('Processing....')
    if mode == '语义模式':
        mode_value = 1
    elif mode == '内容模式':
        mode_value = 0
    elif mode == '融合模式':
        mode_value = 2
    if image is None:
        raise ValueError

    processed_mask = reverse_mask(image['mask'])

    image = image['image']

    print('tar_img loading fin')
    config['parameters']['epsilon'] = eps / 255.0 * (1 - (-1))
    config['parameters']['steps'] = steps

    config['parameters']["rate"] = 10 ** (rate + 3)

    config['parameters']['mode'] = mode_value
    block_num = len(block_mode) + 1
    resize = len(no_resize)
    bls = input_size // block_num
    if resize:
        img, target_size = closing_resize(image, input_size, block_num, True)
        bls_h = target_size[0]//block_num
        bls_w = target_size[1]//block_num
        tar_img = load_image_from_path(target_image_path, target_size[0],
                                       target_size[1])
    else:
        img = load_image_from_path(image, input_size, input_size, True)
        tar_img = load_image_from_path(target_image_path, input_size)
        bls_h = bls_w = bls
        target_size = [input_size, input_size]
    processed_mask = load_image_from_path(processed_mask, target_size[0], target_size[1], True)
    config['parameters']['input_size'] = bls
    print(config['parameters'])
    output_image = np.zeros([input_size, input_size, 3])
    for i in tqdm(range(block_num)):
        for j in tqdm(range(block_num)):
            if processed_mask is not None:
                input_mask = Image.fromarray(np.array(processed_mask)[bls_w*i: bls_w*i+bls_w, bls_h*j: bls_h*j + bls_h])
            else:
                input_mask = None
            img_block = Image.fromarray(np.array(img)[bls_w*i: bls_w*i+bls_w, bls_h*j: bls_h*j + bls_h])
            tar_block = Image.fromarray(np.array(tar_img)[bls_w*i: bls_w*i+bls_w, bls_h*j: bls_h*j + bls_h])

            output_image[bls_w*i: bls_w*i+bls_w, bls_h*j: bls_h*j + bls_h] = infer(img_block, config, tar_block, input_mask)
    output = Image.fromarray(output_image.astype(np.uint8))
    return output



if __name__ == "__main__":
    with gr.Blocks() as demo:
        with gr.Column():
            gr.Image("title.png", show_label=False)
            with gr.Row():
                with gr.Column():
                    image = gr.Image(type='pil')
                    eps = gr.Slider(0, 32, step=4, value=16, label='强度',
                                    info="强度越大，保护能力越好，但是噪声也越明显。")
                    steps = gr.Slider(0, 1000, step=1, value=100, label='步长',
                                      info="步长越大，保护能力越好，但是运行时间也越长。")
                    input_size = gr.Slider(256, 768, step=256, value=512, label='输出尺寸',
                                           info="输出图片的尺寸.")

                    mode = gr.Radio(["内容模式", "语义模式", "融合模式"], value="融合模式", label="模式",
                                    info="推荐用户使用融合模式。")

                    with gr.Accordion("融合模式的参数", open=False):
                        rate = gr.Slider(0, 5, step=1, value=1, label='融合权重',
                                         info="融合权重越大，语义模式占比越大。")

                    block_mode = gr.CheckboxGroup(["低VRAM容量可选项"],
                                                  info="如果设备的VRAM不足，请点击此选项。",
                                                  label='关于VRAM的可选项')
                    with gr.Accordion("非正方形输入的操作选项", open=False):
                        no_resize = gr.CheckboxGroup(["非正方形输入可选项"],
                                                  info="如果不想将图像调整为正方形，请点击此选项。此选项仍处于试验阶段，可能会降低智绘疫苗的强度。",
                                                  label='非正方形输入可选项')
                    inputs = [image, eps, steps, input_size, rate, mode, block_mode, no_resize]
                    image_button = gr.Button("运行")
                outputs = gr.Image(label='输出图片')
            image_button.click(process_image, inputs=inputs, outputs=outputs)

    demo.queue().launch(share=True)

# gr.HTML("""
#             <style>
#                 body {
#                     background-color: black;
#                     color: white;
#                 }
#                 .gradio-container {
#                     background-color: black;
#                     color: white;
#                 }
#                 .gradio-container .gradio-component {
#                     color: white;
#                 }
#                 .gradio-container .gradio-input {
#                     background-color: #333;
#                     color: white;
#                 }
#                 .gradio-container .gradio-output {
#                     background-color: #333;
#                     color: white;
#                 }
#                 .gradio-container .gradio-slider input[type='range'] {
#                     background-color: #333;
#                 }
#                 .gradio-container .gradio-slider input[type='range']::-webkit-slider-thumb {
#                     background-color: white;
#                 }
#                 .gradio-container .gradio-checkboxgroup,
#                 .gradio-container .gradio-accordion,
#                 .gradio-container .gradio-radio,
#                 .gradio-container .gradio-button {
#                     background-color: #333;
#                     color: white;
#                 }
#                 .gradio-container .gradio-button:hover {
#                     background-color: #555;
#                 }
#             </style>
#         """)