import torch
torch.autograd.set_detect_anomaly(True)
import torch_npu
from torch_npu.contrib import transfer_to_npu

import torch.nn as nn
from models.llava15.model.builder import load_pretrained_model
from models.llava15.mm_utils import get_model_name_from_path
from PIL import Image
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch NPU Model gardient-based attack')
    parser.add_argument('--model_name', type=str, default='LLaVA15', help='Path to the NPU model')
    parser.add_argument('--device', type=int, default=1, help='NPU device ID (default: 0)')
    parser.add_argument('--epsilon', type=float, default= 8/255, help='扰动强度')
    parser.add_argument('--alpha', type=float, default= 2/255, help='PGD步长')
    parser.add_argument('--num_iter', type=int, default=10, help='PGD迭代次数')
    args = parser.parse_args()
    return args

# 配置参数
model_path = "/data/jw/huggingfacemodel/llava-v1.5-7b"  # 可替换为13b版本
device = torch.device('npu') if torch.cuda.is_available() else "cpu"
torch_npu.npu.set_device(2)
epsilon = 8/255  # 扰动强度（像素值归一化到0-1时）
alpha = 2/255    # PGD步长
num_iter = 10    # PGD迭代次数
target_answer = "no"  # 有目标攻击时的指定错误答案

# 加载LLaVA模型和处理器
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=model_name,
    device_map='sequential',
    device=device
)

# 冻结模型参数（仅计算输入梯度）
for param in model.parameters():
    param.requires_grad = False
model.eval()
model.to(torch.float16)
# 准备原始输入
image_path = "/data/jw/projects/B-AVIBench_jw/eval-data/tiny_lvlm_datasets/AOKVQAOpen/COCO_val2014_000000263463.jpg"  # 示例图片
raw_image = Image.open(image_path)
question = "Is there a dog in the picture?"  # 假设正确答案应为"yes"

# 图像预处理（保留梯度计算）
def process_image(image):
    image_tensor = image_processor(image)["pixel_values"][0]
    image_tensor = torch.from_numpy(image_tensor).float()
    image_tensor = image_tensor.unsqueeze(0)
    
    return image_tensor.to(device).to(torch.float16).requires_grad_(True)

# 文本处理
def process_text(text):
    return tokenizer(text, return_tensors="pt").to(device)

# 生成对抗样本的核心函数
def generate_attack(model, image, question, true_answer, method="pgd"):
    # 初始化对抗图像
    adv_image = image.clone().detach().requires_grad_(True)

    # 处理文本输入
    text_input = tokenizer(question, return_tensors="pt").to(device)
    input_ids = text_input.input_ids
    attention_mask = text_input.attention_mask

    # 获取真实答案的token
    answer_input = process_text(true_answer)
    target_ids = answer_input.input_ids[:, 1:]  # 跳过起始token
    
    # 攻击循环
    for _ in range(num_iter if method == "pgd" else 1):
        adv_image = adv_image.detach().requires_grad_(True)
        # 前向传播
        with torch.enable_grad():
            inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "images": adv_image,
                "return_dict": True
            }
            outputs = model(**inputs)
        
        # 计算损失（交叉熵）
        logits = outputs.logits[:, -target_ids.shape[1]:, :]
        loss = nn.CrossEntropyLoss()(
            logits.reshape(-1, logits.shape[-1]),
            target_ids.reshape(-1)
        )
        model.zero_grad()
        # 梯度计算
        loss.backward()
        grad = torch.autograd.grad(loss, adv_image, create_graph=False)[0]
        
        # 更新对抗样本
        if method.lower() == "fgsm":
            perturbed_image = adv_image + epsilon * data_grad.sign()
        elif method.lower() == "pgd":
            perturbed_image = adv_image + alpha * data_grad.sign()
            # 投影到epsilon邻域内
            perturbed_image = torch.max(torch.min(perturbed_image, image + epsilon), image - epsilon)
        
        # 像素值裁剪到合法范围
        perturbed_image = torch.clamp(perturbed_image, 0, 1)
        adv_image = perturbed_image.detach().requires_grad_(True)
    
    return adv_image

# 执行攻击
clean_image = process_image(raw_image)
adv_image = generate_attack(model, clean_image, question, true_answer="yes", method="pgd")

# 测试攻击效果
def vqa_inference(model, image, question):
    inputs = {
        "images": image,
        "text": process_text(question)["input_ids"],
        "return_dict": True
    }
    outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# 原始预测
print("Clean prediction:", vqa_inference(model, clean_image, question))

# 对抗样本预测
print("Adversarial prediction:", vqa_inference(model, adv_image, question))