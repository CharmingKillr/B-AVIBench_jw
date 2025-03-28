import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info  # 需要确保该工具函数可用
from PIL import Image
import os

class TestQwen25:
    def __init__(self, device=None, device_map="auto", use_flash_attention=False, min_pixels = None, max_pixels=256*28*28):
        # 初始化配置
        self.model_name = "/data/jw/huggingfacemodel/Qwen2.5-VL-7B-Instruct"
        self.use_flash_attention = use_flash_attention
        self.target_size = 224
        # 设置设备
        self.device = device
        self.device_map = {"":self.device}
        # 加载模型
        self._load_model(min_pixels, max_pixels)

    def _load_model(self, min_pixels, max_pixels):
        """加载模型和处理器"""
        # 加载处理器
        processor_kwargs = {}
        if min_pixels is not None:
            processor_kwargs["min_pixels"] = min_pixels
        if max_pixels is not None:
            processor_kwargs["max_pixels"] = max_pixels
        processor_kwargs['size'] = {"height": self.target_size, "width": self.target_size}
        processor_kwargs['keep_ratio'] = False
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            **processor_kwargs
        )

        # 加载模型
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if self.use_flash_attention else "auto",
            "device_map": self.device_map
        }
        if self.use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        self.model = self.model.to(self.device)

    def _build_messages(self, image_paths, questions):
        """构建消息结构"""
        messages_list = []
        for img_path, question in zip(image_paths, questions):
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": img_path},
                    {"type": "text", "text": question}
                ]
            }]
            messages_list.append(messages)
        return messages_list

    def _process_inputs(self, messages_list):
        """处理输入数据"""
        # 生成模板文本
        texts = [self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        ) for messages in messages_list]

        # 处理视觉输入
        image_inputs, video_inputs = process_vision_info(messages_list)
        
        # 处理输入
        inputs = self.processor(
            text=texts,
            images=image_inputs if image_inputs else None,
            videos=video_inputs if video_inputs else None,
            padding=True,
            return_tensors="pt"
        ).to(self.device)
        
        return inputs

    @torch.no_grad()
    def generate(self, image_path, question, max_new_tokens=128,method=None, level=0):
        return self.batch_generate([image_path], [question], max_new_tokens,method=None, level=0)[0]

    @torch.no_grad()
    def batch_generate(self, image_paths, questions, max_new_tokens=128,method=None, level=0):
        images=[]
        for image in image_paths:
            if method is not None and level!=0:
                if level == 1:
                    DATA_PATA = '/data/jw/projects/B-AVIBench_jw/eval-data/corruption/lvlm_tiny_corruption_1'
                elif level == 3:
                    DATA_PATA = '/data/jw/projects/B-AVIBench_jw/eval-data/corruption/lvlm_tiny_corruption_3'
                elif level == 5:
                    DATA_PATA = '/data/jw/projects/B-AVIBench_jw/eval-data/corruption/lvlm_tiny_corruption_5'
                tmp=image.split('/')
                image=os.path.join(DATA_PATA,tmp[-2]+'_{}_{}'.format(method,level),tmp[-1])
            images.append(image)
        image_paths=images
        # 构建消息结构
        messages_list = self._build_messages(image_paths, questions)
        
        
        # # 处理输入
        # inputs = self._process_inputs(messages_list)
        
        # # 生成配置
        # generate_kwargs = {
        #     "max_new_tokens": max_new_tokens,
        #     "pad_token_id": self.processor.tokenizer.eos_token_id,
        #     'num_beams':1
        # }

        # # 执行生成
        # generated_ids = self.model.generate(**inputs, **generate_kwargs)
        # Debug 视觉输入
        print(f"Processing batch with {len(messages_list)} samples.")
        inputs = self._process_inputs(messages_list)

        # Debug: 确保输入在 GPU
        print(f"Input IDs Shape: {inputs.input_ids.shape}, Device: {inputs.input_ids.device}")

        # 修复可能的 batch_size 问题
        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "pad_token_id": self.processor.tokenizer.eos_token_id,
            "num_beams": 1,
            "use_cache": False  # 禁用 KV Cache 试试看
        }

        generated_ids = self.model.generate(**inputs, **generate_kwargs)
        print(f"Generated ID Shape: {generated_ids.shape}")  # Debug batch 生成的 ID
        
        # 后处理输出
        return self._post_process_outputs(inputs.input_ids, generated_ids)

    def _post_process_outputs(self, input_ids, generated_ids):
        """后处理生成结果"""
        # 修剪输入ID
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
        ]
        
        # 解码文本
        outputs = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )
        
        return [output.strip() for output in outputs]

    def process_corrupted_images(self, image_paths, method=None, level=0):
        """处理损坏图像路径（示例实现）"""
        processed_paths = []
        for img_path in image_paths:
            if method and level != 0:
                # 示例路径处理逻辑
                base_path = f"/path/to/corruption/lvlm_tiny_corruption_{level}"
                parts = img_path.split('/')
                new_path = os.path.join(
                    base_path,
                    f"{parts[-2]}_{method}_{level}",
                    parts[-1]
                )
                processed_paths.append(new_path)
            else:
                processed_paths.append(img_path)
        return processed_paths