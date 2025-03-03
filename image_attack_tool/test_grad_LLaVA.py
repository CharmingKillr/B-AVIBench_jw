import torch
import torch.nn as nn
from models.llava15.model.builder import load_pretrained_model
from models.llava15.mm_utils import get_model_name_from_path
from PIL import Image
import requests
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
model_path = "/seu_nvme/home/230239304/huggingfacemodel/llava-v1.5-7b"  # 可替换为13b版本
device = torch.device('cuda') if torch.cuda.is_available() else "cpu"
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
    device=device
)

# ====================================================================
# 打印图像编码器（ViT）结构（基于CLIPVisionTransformer）
# ====================================================================
print("\n" + "="*50)
print("Image Encoder (CLIP ViT) Architecture")
print("="*50)

def print_vit_structure(encoder, prefix=""):
    for name, module in encoder.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(module, torch.nn.ModuleList):
            for i, block in enumerate(module):
                print(f"{full_name}[{i}]")
                print(block)
        elif list(module.named_children()):
            print(f"\n{full_name}")
            print_vit_structure(module, full_name)
        else:
            print(f"{full_name}: {type(module).__name__}")

# 根据结构修正访问路径
vit_encoder = model.model.vision_tower.vision_tower.vision_model  # 关键路径修正
print_vit_structure(vit_encoder)

# ====================================================================
# 打印文本编码器（Llama）结构
# ====================================================================
print("\n" + "="*50)
print("Text Encoder (Llama) Architecture")
print("="*50)

def print_llama_layers(layers):
    for i, layer in enumerate(layers):
        print(f"\nLayer {i}:")
        print("-"*30)
        for name, module in layer.named_children():
            if name == "self_attn":
                print(f"  {name}:")
                print(f"    q_proj: {module.q_proj}")
                print(f"    k_proj: {module.k_proj}")
                print(f"    v_proj: {module.v_proj}")
                print(f"    o_proj: {module.o_proj}")
            elif name == "mlp":
                print(f"  {name}:")
                print(f"    gate_proj: {module.gate_proj}")
                print(f"    up_proj: {module.up_proj}")
                print(f"    down_proj: {module.down_proj}")
            else:
                print(f"  {name}: {module}")

# 访问语言模型层
llama_layers = model.model.layers
print_llama_layers(llama_layers)

# ====================================================================
# 打印关键组件参数统计
# ====================================================================
print("\n" + "="*50)
print("Parameter Statistics")
print("="*50)

def print_param_stats(module, name):
    params = sum(p.numel() for p in module.parameters())
    print(f"{name}:")
    print(f"  Trainable: {any(p.requires_grad for p in module.parameters())}")
    print(f"  Parameters: {params/1e6:.1f}M")

# 图像编码器参数
print_param_stats(vit_encoder, "Vision Transformer")

# 文本编码器参数
print_param_stats(llama_layers, "Llama Layers")

# 跨模态投影器
print_param_stats(model.model.mm_projector, "MM Projector")