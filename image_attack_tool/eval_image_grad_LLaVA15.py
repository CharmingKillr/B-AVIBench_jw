import torch
import argparse
from models import get_model
import datetime
from utils import dataset_task_dict
from tiny_datasets import dataset_class_dict, GeneralDataset
import os
import json
import numpy as np
"""
model_name: BLIP2; MiniGPT-4; mPLUG-Owl; Otter; Otter-Image; InstructBLIP; VPGTrans; LLaVA; sharegpt4v; moellava; LLaVA15; LLaMA-Adapter-v2; internlm-xcomposer; PandaGPT; OFv2.

dataset_name: Visual perception--image_cls:ImageNetVC_color,ImageNetVC_component,ImageNetVC_material,ImageNetVC_others,ImageNetVC_shape; Visual perception--MCI and OC: MSCOCO_MCI,VCR1_MCI,MSCOCO_OC,VCR1_OC.

Visual knowledge acquisition--KIE: FUNSD,POIE,SROIE; Visual knowledge acquisition--OCR: COCO-Text,CTW,CUTE80,HOST,IC13,IC15,IIIT5K,SVTP,SVT,Total-Text,WOST,WordArt; Visual knowledge acquisition--Image Caption: NoCaps,Flickr,MSCOCO_caption_karpathy,WHOOPSCaption.

Visual reasoning--VQA: AOKVQAClose,AOKVQAOpen,DocVQA,GQA,OCRVQA,OKVQA,STVQA,TextVQA,WHOOPSVQA,WHOOPSWeird,Visdial,IconQA,VSR; Visual reasoning--KGID: ScienceQAIMG,VizWiz.

Visual commonsense: ImageNetVC_color,ImageNetVC_component,ImageNetVC_material,ImageNetVC_others,ImageNetVC_shape.

Object hallucination: MSCOCO_pope_random,MSCOCO_pope_adversarial,MSCOCO_pope_popular.
"""
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch NPU Model gardient-based attack')
    parser.add_argument('--model_name', type=str, default='LLaVA15', help='Path to the CUDA model')
    parser.add_argument('--device', type=int, default=1, help='GPU device ID (default: 0)')
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--answer_path", type=str, default="/seu_nvme/home/230239304/projects_jw/B-AVIBench_jw/image_attack_tool/tiny_grad_answers")
    parser.add_argument("--grad_attacks", action='store_true', help='Enable gradient-based attacks')
    args = parser.parse_args()
    return args

def main(args):
    args.grad_attacks = True

    model = get_model(args.model_name, device=torch.device('cuda'))
    time = datetime.datetime.now().strftime("%Y_%m%d_%H_%M_%S")
    for dataset_name in args.dataset_names:
        result = {}
        print("ll",dataset_name)
        eval_function, task_type = dataset_task_dict[dataset_name]
        dataset = GeneralDataset(dataset_name)
        metrics = eval_function(model, dataset, args.model_name, dataset_name, task_type, time, args.batch_size, answer_path=args.answer_path, method=None, level=0, grad_attacks=args.grad_attacks)
        print("***---",metrics)
        result["{}_severity_{}".format(dataset_name,0)] = metrics

        result_path = os.path.join(os.path.join(args.answer_path, time), 'result_{}.json'.format(dataset_name))
        with open(result_path, "w") as f:
            f.write(json.dumps(result, indent=4))

if __name__ == '__main__':
    args = parse_args()
    args.dataset_names = ['VCR1_MCI','MSCOCO_OC','VCR1_OC']
    # args.dataset_names = ['ImageNetVC_color','ImageNetVC_component','ImageNetVC_material','ImageNetVC_others','ImageNetVC_shape','MSCOCO_MCI','VCR1_MCI','MSCOCO_OC','VCR1_OC','FUNSD','POIE','SROIE',
    #                      'COCO-Text','CTW','CUTE80','HOST','IC13','IC15','IIIT5K','SVTP','SVT','NoCaps','Flickr','MSCOCO_caption_karpathy','WHOOPSCaption','AOKVQAClose','AOKVQAOpen','DocVQA','GQA',
    #                      'OCRVQA','OKVQA','STVQA','TextVQA','WHOOPSVQA','WHOOPSWeird','Visdial','IconQA','VSR','ScienceQAIMG','VizWiz','MSCOCO_pope_random','MSCOCO_pope_adversarial','MSCOCO_pope_popular']
    main(args)