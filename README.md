# 基于OpenVINO的影音视频创作方案(开源版)  
**商业版不开源，后续将开放商业版体验链接**  
## 项目概述
基于OpenVINO框架实现的智能影音生成系统，支持通过自然语言指令生成视频内容。集成Stable Video Diffusion等AI模型，提供高性价比的的AI视频创作解决方案。

## 集成模型  
+ Stable Video Diffusion
+ Wan2.1  
+ OuteTTS  
+ Wav2Lip  
+ DeepSeek-R1-Distill-Qwen-7B  

## 开源模型权重  
**本项目基于OpenVINO构建，开源了以下几个模型权重，欢迎大家下载使用**    
1. OpenVINO中间格式的的Stable Video Diffusion权重  
[stable-video-diffusion-img2vid-xt-ov](https://huggingface.co/liuming9157/stable-video-diffusion-img2vid-xt-ov)
2. OpenVINO中间格式的Wan2.1权重  
[Wan2.1-T2V-1.3B-Diffusers-ov-fp16](https://huggingface.co/liuming9157/Wan2.1-T2V-1.3B-Diffusers-ov-fp16)
3. OpenVINO中间格式的Wan2.1权重-int4量化版  
[Wan2.1-T2V-1.3B-Diffusers-ov-int4](https://huggingface.co/liuming9157/Wan2.1-T2V-1.3B-Diffusers-ov-int4)  
4. OpenVINO中间格式的OuteTTS权重  
[OuteTTS-0.2-500M-ov](https://huggingface.co/liuming9157/OuteTTS-0.2-500M-ov)  

> 以上模型的转换脚本和推理脚本请进入ov_pipe文件夹查看

## 主要功能
✅ 文本到视频生成
✅ 图像到视频扩展
✅ 音频驱动的口型同步
✅ 多模态内容编辑 
✅ OpenVINO加速推理

## 安装说明
```bash
# 克隆仓库
git clone https://github.com/liuming9157/openvino_video.git
# or
git clone https://gitee.com/liuming9157/openvino_video.git

# 安装依赖
pip install -r requirements.txt

```
## 下载模型文件  
可到上文介绍的开源模型权重中直接下载，然后放置到`models`文件夹下对应目录，也可进入`/openvino`文件夹运行转换脚本

## 快速使用
```python
# 运行主程序
python app.py --prompt "夕阳下的奔跑场景"
```

## 贡献指南
欢迎通过Pull Request提交改进，请确保：
1. 通过pep8格式检查
2. 添加对应的单元测试
3. 更新文档说明

## 许可协议
Apache License 2.0