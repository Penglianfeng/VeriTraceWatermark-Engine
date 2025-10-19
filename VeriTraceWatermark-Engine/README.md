1.“溯真水印：AIGC 大模型训练图片的版权保护系统”使用总览
![image](https://github.com/Penglianfeng/VeriTraceWatermark-Engine/blob/main/VeriTraceWatermark-Engine/%E6%80%BB%E8%A7%88%E5%9B%BE.png)

首先将未受保护的原始图片输入到本系统中，经过噪声仿真网络与 AIGC 大模型获得对抗样本的损失 Loss，将 Loss与原始图片叠加生成受保护的对抗样本,系统生成的对抗样本不仅不会影响原始图片的视觉感受，而且在经过不同的社交平台压缩传输，信道加噪之后仍具有保护能力。本系统旨在利用对抗样本技术保护原创作品的版权，增强城市数据安全与隐私保护，促进生成式大模型与数字经济持续健康发展。

2.“溯真水印：AIGC 大模型训练图片的版权保护系统”界面
![image](https://github.com/Penglianfeng/VeriTraceWatermark-Engine/blob/main/VeriTraceWatermark-Engine/GUI.png)

在 Web 操作界面左上角输入原始图片，用户设置好保护强度，输出尺寸等参数后即可生成视觉不变的受保护的对抗样本。

3.“溯真水印：AIGC 大模型训练图片的版权保护系统”版权保护效果展示图
![image](https://github.com/13859/WatermarkVaccine_AIGC/blob/master/效果图.png)

若使用未受保护的图片训练文生图大模型，输入相关提示词后即可生成右侧与训练图片画风和内容相似的图片，侵犯了原始图片的版权。若使用受保护的图片训练文生图大模型，无法生成画风与内容相似的图片，有效保护了图片的版权。
