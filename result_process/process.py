import os
import json
import shutil
import re

# 定义数据集分类
dataset_categories = {
    "CAP": ["NoCaps", "Flickr", "MSCOCO_caption_karpathy", "WHOOPSCaption"], # 4
    "CLS": ['CIFAR10','CIFAR100','Flowers102','ImageNet','OxfordIIITPet'],  # 5
    "imagenetvc": ["ImageNetVC_color", "ImageNetVC_component", "ImageNetVC_material", "ImageNetVC_others", "ImageNetVC_shape"], # 5
    "KIE": ["FUNSD", "POIE", "SROIE"], # 3
    "MCI": ["MSCOCO_MCI", "VCR1_MCI", "MSCOCO_OC", "VCR1_OC"], # 4
    "OBJECT": ["MSCOCO_pope_random", "MSCOCO_pope_adversarial", "MSCOCO_pope_popular"], # 3
    "OCR": ["COCO-Text", "CTW", "CUTE80", "HOST", "IC13", "IC15", "IIIT5K", "SVTP", "SVT", "Total-Text", "WOST", "WordArt"], # 12
    "VQA": ['AOKVQAClose','AOKVQAOpen','DocVQA','GQA','OCRVQA','OKVQA','STVQA','TextVQA','VizWiz','WHOOPSVQA','WHOOPSWeird','Visdial'], # 12
    "VQACHOICE": ['ScienceQAIMG','IconQA','VSR'] # 3
}

# 定义方法后缀
methods = ['Fog', 'Zoom_Blur', 'Glass_Blur', 'Gaussian_Noise', 'Shot_Noise', 'Impulse_Noise', 'Defocus_Blur', 'Motion_Blur', 'Snow',
           'Frost', 'Brightness', 'Contrast', 'Elastic', 'Pixelate', 'JPEG', 'Speckle_Noise', 'Gaussian_Blur', 'Spatter', 'Saturate']

# 输入和输出目录
input_dir = "/seu_nvme/home/230239304/projects_jw/B-AVIBench_jw/result_process"
output_dir = "/seu_nvme/home/230239304/projects_jw/B-AVIBench_jw/result_process_re"

# 遍历所有模型子目录
for model_dir in os.listdir(input_dir):
    model_path = os.path.join(input_dir, model_dir)
    if os.path.isdir(model_path):
        # 遍历每个子目录下的九个子目录
        for category in dataset_categories.keys():
            category_path = os.path.join(model_path, category)
            if os.path.isdir(category_path):
                result_json_path = os.path.join(category_path, "result.json")
                if os.path.exists(result_json_path):
                    # 读取result.json文件
                    with open(result_json_path, 'r') as f:
                        result_data = json.load(f)
                    
                    # 过滤出对应数据集的内容
                    filtered_data = {}
                    for key, value in result_data.items():
                        for dataset in dataset_categories[category]:
                            if re.match(rf'^{dataset}(_|$)', key):  # 确保 dataset 后面是 "_" 或者结束
                                filtered_data[key] = value
                    
                    # 创建输出目录
                    output_category_path = os.path.join(output_dir, model_dir, category)
                    os.makedirs(output_category_path, exist_ok=True)
                    
                    # 保存过滤后的数据到新的result.json文件
                    output_json_path = os.path.join(output_category_path, "result.json")
                    with open(output_json_path, 'w') as f:
                        json.dump(filtered_data, f, indent=4)
    print(f'{model_dir}已完成！')
print("分类保存完成！")