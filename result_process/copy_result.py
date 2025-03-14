import os
import json

# 定义数据集分类
dataset_categories = {
    "CAP": ["NoCaps", "Flickr", "MSCOCO_caption_karpathy", "WHOOPSCaption"],
    "CLS": ['CIFAR10', 'CIFAR100', 'Flowers102', 'ImageNet', 'OxfordIIITPet'],
    "imagenetvc": ["ImageNetVC_color", "ImageNetVC_component", "ImageNetVC_material", "ImageNetVC_others", "ImageNetVC_shape"],
    "KIE": ["FUNSD", "POIE", "SROIE"],
    "MCI": ["MSCOCO_MCI", "VCR1_MCI", "MSCOCO_OC", "VCR1_OC"],
    "OBJECT": ["MSCOCO_pope_random", "MSCOCO_pope_adversarial", "MSCOCO_pope_popular"],
    "OCR": ["COCO-Text", "CTW", "CUTE80", "HOST", "IC13", "IC15", "IIIT5K", "SVTP", "SVT", "Total-Text", "WOST", "WordArt"],
    "VQA": ['AOKVQAClose', 'AOKVQAOpen', 'DocVQA', 'GQA', 'OCRVQA', 'OKVQA', 'STVQA', 'TextVQA', 'VizWiz', 'WHOOPSVQA', 'WHOOPSWeird', 'Visdial'],
    "VQACHOICE": ['ScienceQAIMG', 'IconQA', 'VSR']
}

# 输入和输出目录
input_dir = "/seu_nvme/home/230239304/projects_jw/B-AVIBench_jw/result_process"
output_dir = "/seu_nvme/home/230239304/projects_jw/B-AVIBench_jw/image_corruption_attack_tool/tiny_answers"

# 遍历 `tiny_answers` 目录，获取所有模型
for model_dir_output in os.listdir(output_dir):
    model_path_output = os.path.join(output_dir, model_dir_output)
    if os.path.isdir(model_path_output):
        # 生成 `input_dir` 对应的模型目录（添加 `_ff`）
        model_dir_input = model_dir_output + "_ff"
        model_path_input = os.path.join(input_dir, model_dir_input)

        # 确保 `input_dir` 里有对应的模型目录
        if not os.path.isdir(model_path_input):
            print(f"⚠️ {model_dir_input} 在 {input_dir} 中不存在，跳过...")
            continue

        # 进入 `output_dir` 该模型的子目录（假设只有一个子目录）
        sub_dirs = [d for d in os.listdir(model_path_output) if os.path.isdir(os.path.join(model_path_output, d))]
        if not sub_dirs:
            print(f"⚠️ {model_dir_output} 目录下没有子目录，跳过...")
            continue

        sub_path = os.path.join(model_path_output, sub_dirs[0])
        result_json_path = os.path.join(sub_path, "result.json")

        # 确保 `result.json` 存在
        if not os.path.exists(result_json_path):
            print(f"⚠️ {model_dir_output} 的 {result_json_path} 不存在，跳过...")
            continue

        # 读取 `tiny_answers` 中该模型的 `result.json`
        with open(result_json_path, 'r') as f:
            result_data = json.load(f)

        # 遍历该模型 `result_process` 下的每个分类目录
        for category in dataset_categories.keys():
            category_path_result = os.path.join(model_path_input, category)
            if os.path.isdir(category_path_result):
                result_json_path_result = os.path.join(category_path_result, "result.json")

                # 仅当 `result.json` 存在时才写入
                if os.path.exists(result_json_path_result):
                    with open(result_json_path_result, 'w') as f:
                        json.dump(result_data, f, indent=4)

        print(f"✅ {model_dir_output} → {model_dir_input} 处理完成！")

print("🎉 所有模型处理完成！")
