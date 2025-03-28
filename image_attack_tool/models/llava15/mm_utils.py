from PIL import Image
from io import BytesIO
import base64

import torch
from transformers import StoppingCriteria
from .constants import IMAGE_TOKEN_INDEX
from torchvision.transforms.functional import to_pil_image

def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    metadata_list = []
    if image_aspect_ratio == 'pad':
        for i,image in enumerate(images):
            if isinstance(image, torch.Tensor):
                if image.ndimension() == 4:
                    image = image.squeeze(0)
                image = to_pil_image(image)
            original_w, original_h = image.size
            image_expand = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            padded_w, padded_h = image_expand.size
            # 计算填充边界
            if original_w > original_h:
                pad_top = (original_w - original_h) // 2
                pad_bottom = original_w - original_h - pad_top
                padding = (0, pad_top, 0, pad_bottom)  # (左, 上, 右, 下)
            else:
                pad_left = (original_h - original_w) // 2
                pad_right = original_h - original_w - pad_left
                padding = (pad_left, 0, pad_right, 0)
            scale = 336 / max(padded_w, padded_h)
            # 记录元数据
            metadata = {
                'original_size': (original_w, original_h),   
                'padded_size': (padded_w, padded_h),
                'padding' : padding,
                'scale_factor': scale,        
            }
            metadata_list.append(metadata)
            # show_image(image, image_expand, i)
            image_expand = image_processor.preprocess(image_expand, return_tensors='pt')['pixel_values'][0]
            # tensor_pil = tensor_to_pil(image_expand, image_processor)
            # tensor_pil.save(f"/data/jw/projects/B-AVIBench_jw/eval-data/debug_image/tensor_pil_{i}.png")
            
            # recover_image = recover_adv_image(image_expand, metadata, image_processor)
            # recover_image.save(f"/data/jw/projects/B-AVIBench_jw/eval-data/debug_image/recover_{i}.png")
            new_images.append(image_expand)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images
def show_image(PIL1, PIL2, i):
    import matplotlib.pyplot as plt
    # 显示原始图像
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(PIL1)
    plt.axis("off")

    # 显示填充后的图像
    plt.subplot(1, 2, 2)
    plt.title("Square Image")
    plt.imshow(PIL2)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"/data/jw/projects/B-AVIBench_jw/eval-data/debug_image/expand2square_{i}.png")
def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids
def tensor_to_pil(tensor, image_processor):
    import numpy as np
    # --- 逆归一化 ---
    # 获取标准化的均值和标准差
    mean = torch.tensor(image_processor.image_mean).view(3, 1, 1)  # 形状变为 (3,1,1)
    std = torch.tensor(image_processor.image_std).view(3, 1, 1)

    # 逆标准化
    image_tensor = tensor * std + mean  # 逆标准化
    image_tensor = torch.clamp(image_tensor, 0, 1)  # 限制范围在 [0,1]

    # 转换为 PIL 图像
    image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)  # 变换通道顺序并转换为 uint8
    image_pil = Image.fromarray(image_np)

    return image_pil
def recover_adv_image(adv_tensor, metadata, image_processor):
    """
    完整恢复流程：逆归一化 → 逆缩放 → 裁剪填充 → 恢复原图尺寸
    :param adv_tensor: 对抗样本张量 (3, 336, 336)
    :param metadata: 预处理元数据
    :param image_processor: CLIPImageProcessor实例
    :return: 恢复后的PIL图像 (原始尺寸)
    """
    from torchvision import transforms
    # --- 逆归一化 ---
    mean = torch.tensor(image_processor.image_mean).view(3, 1, 1)
    std = torch.tensor(image_processor.image_std).view(3, 1, 1)
    adv_denorm = (adv_tensor * std + mean) * 255
    adv_denorm = adv_denorm.clamp(0, 255).byte().cpu()
    
    # --- 逆缩放：精确缩放到填充后的原始尺寸 ---
    # 计算目标尺寸（保持宽高比）
    padded_w, padded_h = metadata['padded_size']
    scale = metadata['scale_factor']
    target_w = int(padded_w * scale)
    target_h = int(padded_h * scale)
    
    # 验证缩放一致性（必须等于336）
    assert target_w == 336 or target_h == 336, f"缩放尺寸错误: {target_w}x{target_h}"
    
    # 转换为PIL并精确逆缩放
    adv_pil = transforms.ToPILImage()(adv_denorm)
    adv_rescaled = adv_pil.resize((padded_w, padded_h), Image.Resampling.LANCZOS)
    
    # --- 精确裁剪填充区域 ---
    orig_w, orig_h = metadata['original_size']
    left, top, right, bottom = metadata['padding']
    
    # 计算裁剪框（关键修复）
    if orig_w > orig_h:
        # 原图宽 > 高：裁剪上下填充
        crop_box = (0, top, orig_w, orig_h + top)
    else:
        # 原图高 > 宽：裁剪左右填充
        crop_box = (left, 0, orig_w + left, orig_h)
    
    adv_cropped = adv_rescaled.crop(crop_box)
    
    # --- 最终调整到原始尺寸（避免缩放误差）---
    adv_final = adv_cropped.resize((orig_w, orig_h), Image.Resampling.LANCZOS)
    
    return adv_cropped
def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]




class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False