import torch
import os
import folder_paths
from PIL import Image
import comfy.utils
import numpy as np
import sys

current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_directory)  # 添加当前目录到Python路径
folder_paths.folder_names_and_paths["WANMODELS"] = ([os.path.join(folder_paths.models_dir, "WANMODELS")], [".pt", ".safetensors", ".pth"])

import wan
from wan.configs import WAN_CONFIGS


class WanT2V_ModelLoader_Zho:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": (folder_paths.get_filename_list("WANMODELS"), ),
            }
        }

    RETURN_TYPES = ("WANMODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "🎞️WanT2V"
  
    def load_model(self, ckpt_name):
        if not ckpt_name:
            raise ValueError("请提供检查点文件名")

        ckpt_path = folder_paths.get_full_path("WANMODELS", ckpt_name)
        directory_path = os.path.dirname(ckpt_path)
        print(f"Directory path: {directory_path}")
            
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"检查点文件 {ckpt_path} 不存在")

        cfg = WAN_CONFIGS['t2v-1.3B']
        model = wan.WanT2V(
            config=cfg,
            checkpoint_dir=directory_path,  # 直接使用检查点路径
            device_id=0,
            rank=0,
            t5_fsdp=False,
            dit_fsdp=False,
            use_usp=False,
        )
        return (model,)

class WanT2V_Generation_Zho:
    def __init__(self):
        pass

    @classmethod 
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("WANMODEL",),
                "prompt": ("STRING", {"multiline": True, "default": "一艘宇宙飞船在太空中飞行"}),
                "resolution": (["480*832", "832*480", "624*624", "704*544", "544*704"], {"default": "480*832"}),
                "sampling_steps": ("INT", {"default": 50, "min": 1, "max": 1000}),
                "guidance_scale": ("FLOAT", {"default": 6.0, "min": 0, "max": 20}),
                "shift_scale": ("FLOAT", {"default": 8.0, "min": 0, "max": 20}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "低质量，模糊"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("video",)
    FUNCTION = "generate_video"
    CATEGORY = "🎞️WanT2V"
    OUTPUT_NODE = True

    def generate_video(self, model, prompt, resolution, sampling_steps, guidance_scale, shift_scale, seed, negative_prompt):
        # 解析分辨率
        W, H = map(int, resolution.split('*'))
        
        # 设置随机种子
        if seed == -1:
            seed = torch.seed()
        generator = torch.Generator().manual_seed(seed)
        
        # 生成视频张量
        video_tensor = model.generate(
            prompt,
            size=(W, H),
            shift=shift_scale,
            sampling_steps=sampling_steps,
            guide_scale=guidance_scale,
            n_prompt=negative_prompt,
            seed=seed,
            offload_model=False,
        )
        
        # ===== 调试输出 =====
        print(f"原始输出形状: {video_tensor.shape} 数据类型: {video_tensor.dtype} 值范围: [{video_tensor.min().item():.2f}, {video_tensor.max().item():.2f}]")

        # ===== 维度修正 =====
        video_nhwc = video_tensor.permute(1, 2, 3, 0)  # [C, frames, H, W] -> [frames, H, W, C]
    
        # ===== 验证最终输出 =====
        print(f"最终输出形状: {video_nhwc.shape} 值范围: [{video_nhwc.min().item():.2f}, {video_nhwc.max().item():.2f}]")
    
        return (video_nhwc,)



NODE_CLASS_MAPPINGS = {
    "WanT2V_ModelLoader_Zho": WanT2V_ModelLoader_Zho,
    "WanT2V_Generation_Zho": WanT2V_Generation_Zho
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WANMODELS_ModelLoader_Zho": "🎞️WANMODELS Model Loader Zho",
    "WANMODELS_Generation_Zho": "🎞️WANMODELS Generation Zho"
}
