import os
import json

# 过滤后的输出目录
output_dir = "/seu_nvme/home/230239304/projects_jw/B-AVIBench_jw/result_process_re"

# 定义数据集分类
dataset_categories = {
    "CAP": ["NoCaps", "Flickr", "MSCOCO_caption_karpathy", "WHOOPSCaption"], 
    "CLS": ['CIFAR10','CIFAR100','Flowers102','ImageNet','OxfordIIITPet'],  
    "imagenetvc": ["ImageNetVC_color", "ImageNetVC_component", "ImageNetVC_material", "ImageNetVC_others", "ImageNetVC_shape"], 
    "KIE": ["FUNSD", "POIE", "SROIE"], 
    "MCI": ["MSCOCO_MCI", "VCR1_MCI", "MSCOCO_OC", "VCR1_OC"], 
    "OBJECT": ["MSCOCO_pope_random", "MSCOCO_pope_adversarial", "MSCOCO_pope_popular"], 
    "OCR": ["COCO-Text", "CTW", "CUTE80", "HOST", "IC13", "IC15", "IIIT5K", "SVTP", "SVT", "Total-Text", "WOST", "WordArt"], 
    "VQA": ['AOKVQAClose','AOKVQAOpen','DocVQA','GQA','OCRVQA','OKVQA','STVQA','TextVQA','VizWiz','WHOOPSVQA','WHOOPSWeird','Visdial'], 
    "VQACHOICE": ['ScienceQAIMG','IconQA','VSR'] 
}

def safe_load_json(file_path):
    """安全加载 JSON 文件，防止文件为空或损坏"""
    if not os.path.exists(file_path):
        return {}
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        print(f"⚠️ 警告: {file_path} 解析失败，可能为空或格式错误！")
        return {}

# 遍历 `output_dir` 里的所有模型
for model_dir in os.listdir(output_dir):
    model_path = os.path.join(output_dir, model_dir)
    if not os.path.isdir(model_path):
        continue  # 跳过非目录项

    # 统计 `output_dir` 里 `result.json` 的 key 总数
    total_filtered_keys = 0
    for category in dataset_categories.keys():
        category_path = os.path.join(model_path, category)
        if os.path.isdir(category_path):
            result_json_path = os.path.join(category_path, "result.json")
            filtered_data = safe_load_json(result_json_path)
            total_filtered_keys += len(filtered_data)

    # 输出统计结果
    print(f"模型: {model_dir}")
    print(f"  过滤后 key 总数: {total_filtered_keys}\n")
