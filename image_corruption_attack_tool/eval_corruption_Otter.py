import os
import json
import argparse
import datetime
#from wand.image import Image as WandImage
#from wand.api import library as wandlibrary
#import wand.color as WandColor
import torch
import numpy as np
import random
from models import get_model
from utils import dataset_task_dict
from tiny_datasets import dataset_class_dict, GeneralDataset
import socket
import deepspeed

method = ['Fog','Zoom_Blur','Glass_Blur','Gaussian_Noise','Shot_Noise','Impulse_Noise','Defocus_Blur','Motion_Blur','Snow',
'Frost','Brightness','Contrast','Elastic','Pixelate','JPEG','Speckle_Noise','Gaussian_Blur','Spatter','Saturate'] 
"""
model_name: BLIP2; MiniGPT-4; mPLUG-Owl; Otter; Otter-Image; InstructBLIP; VPGTrans; LLaVA; sharegpt4v; moellava; LLaVA15; LLaMA-Adapter-v2; internlm-xcomposer; PandaGPT; OFv2.

dataset_name: Visual perception--image_cls:ImageNetVC_color,ImageNetVC_component,ImageNetVC_material,ImageNetVC_others,ImageNetVC_shape; Visual perception--MCI and OC: MSCOCO_MCI,VCR1_MCI,MSCOCO_OC,VCR1_OC.

Visual knowledge acquisition--KIE: FUNSD,POIE,SROIE; Visual knowledge acquisition--OCR: COCO-Text,CTW,CUTE80,HOST,IC13,IC15,IIIT5K,SVTP,SVT,Total-Text,WOST,WordArt; Visual knowledge acquisition--Image Caption: NoCaps,Flickr,MSCOCO_caption_karpathy,WHOOPSCaption.

Visual reasoning--VQA: AOKVQAClose,AOKVQAOpen,DocVQA,GQA,OCRVQA,OKVQA,STVQA,TextVQA,WHOOPSVQA,WHOOPSWeird,Visdial,IconQA,VSR; Visual reasoning--KGID: ScienceQAIMG,VizWiz.

Visual commonsense: ImageNetVC_color,ImageNetVC_component,ImageNetVC_material,ImageNetVC_others,ImageNetVC_shape.

Object hallucination: MSCOCO_pope_random,MSCOCO_pope_adversarial,MSCOCO_pope_popular.
"""
def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    # models
    parser.add_argument("--model_name", type=str, default="Otter")
    parser.add_argument("--device", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)

    # datasets
    parser.add_argument("--dataset_name", type=str, default='ImageNetVC_component')
    parser.add_argument("--sample_num", type=int, default=500)
    parser.add_argument("--sample_seed", type=int, default=20230719)

    # result_path
    parser.add_argument("--answer_path", type=str, default="/seu_nvme/home/230239304/projects_jw/B-AVIBench_jw/image_corruption_attack_tool/tiny_answers")

    args = parser.parse_args()
    return args


def sample_dataset(dataset, max_sample_num=5000, seed=0):
    if max_sample_num == -1:
        return dataset

    if len(dataset) > max_sample_num:
        np.random.seed(seed)
        random_indices = np.random.choice(
            len(dataset), max_sample_num, replace=False
        )
        dataset = torch.utils.data.Subset(dataset, random_indices)
    return dataset

def initialize_distributed():
    os.environ['MASTER_IP'] = os.getenv('MASTER_ADDR', 'localhost')
    
    # 生成一个随机端口号
    port = random.randint(10000, 60000)
    
    # 检查随机端口号是否可用
    while True:
        try:
            # 创建一个临时的socket对象并尝试绑定到随机端口号
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('localhost', port))
            break
        except OSError:
            # 如果端口号已被占用，则选择一个新的随机端口号
            port = random.randint(10000, 60000)
    
    # 将随机端口号设置为环境变量
    os.environ['MASTER_PORT'] = str(port)
    
    # 初始化分布式训练
    deepspeed.init_distributed(dist_backend='nccl')

def main(args):
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)
    # os.environ['DS_INIT_PROCESS_PORT'] = str(10008)
    # 将随机端口号设置为环境变量    os.environ['MASTER_PORT'] = str(port)
    time = datetime.datetime.now().strftime("%Y_%m%d_%H_%M_%S")
    answer_path = f"{args.answer_path}/{args.model_name}"
    model = get_model(args.model_name, device=torch.device('cuda')) 
       

    #result = {}
    args.renew = True
    if args.renew :
        time_renew = '2025_0201_18_23_23'
        time = time_renew
        
    for dataset_name in args.dataset_name:
        eval_function, task_type = dataset_task_dict[dataset_name]      
        
        for method_name in method:     
            for k in [0,1,3,5]:#    
                final_path = os.path.join(answer_path,time,f'{dataset_name}_{method_name}_{k}.json')
                if args.renew and os.path.exists(final_path):
                    print(f"result file {final_path} exists, skip.")
                    continue

                dataset = GeneralDataset(dataset_name) 
                metrics = eval_function(model, dataset, args.model_name, dataset_name, task_type, time, args.batch_size, answer_path=answer_path, method=method_name, level=k)
                result_key = "{}_severity_{}_{}".format(dataset_name, method_name, k)

                result_path = os.path.join(os.path.join(answer_path, time), 'result.json')  

                if args.renew:
                    with open(result_path, 'r') as f:
                        try:
                            existing_results = json.load(f)
                        except:
                            existing_results={}
                else:
                    existing_results={}
                # 更新 existing_results
                existing_results[result_key] = metrics
                
                # 写回文件（确保所有数据不会丢失）
                with open(result_path, "w") as f:
                    json.dump(existing_results, f, indent=4)


if __name__ == "__main__":
    args = parse_args()
    args.dataset_name = ['ImageNetVC_color','ImageNetVC_component','ImageNetVC_material','ImageNetVC_others','ImageNetVC_shape','MSCOCO_MCI','VCR1_MCI','MSCOCO_OC','VCR1_OC','FUNSD','POIE','SROIE',
                         'COCO-Text','CTW','CUTE80','HOST','IC13','IC15','IIIT5K','SVTP','SVT','NoCaps','Flickr','MSCOCO_caption_karpathy','WHOOPSCaption','AOKVQAClose','AOKVQAOpen','DocVQA','GQA',
                         'OCRVQA','OKVQA','STVQA','TextVQA','WHOOPSVQA','WHOOPSWeird','Visdial','IconQA','VSR','ScienceQAIMG','VizWiz','MSCOCO_pope_random','MSCOCO_pope_adversarial','MSCOCO_pope_popular',
                         'CIFAR10','CIFAR100','Flowers102','ImageNet','OxfordIIITPet','Total-Text','WOST','WordArt']
    main(args)