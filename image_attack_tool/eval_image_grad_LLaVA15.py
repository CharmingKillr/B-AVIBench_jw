import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
import argparse
from models import get_model
import datetime
from utils import dataset_task_dict
from tiny_datasets import dataset_class_dict, GeneralDataset
import os
import json
import numpy as np
import logging
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
    parser.add_argument('--model_name', type=str, default='LLaVA15', help='Path to the NPU model')
    parser.add_argument('--device', type=int, default=2, help='GPU device ID (default: 0)')
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--answer_path", type=str, default="/data/jw/projects/B-AVIBench_jw/image_attack_tool/tiny_grad_answers")
    parser.add_argument("--grad_attacks", action='store_true', help='Enable gradient-based attacks')
    parser.add_argument("--logging_path", type=str, default="/data/jw/projects/B-AVIBench_jw/eval-data/transfer_perturbed_images")
    args = parser.parse_args()
    return args

def main(args):
    args.grad_attacks = True
    torch_npu.npu.set_device(args.device)
    model = get_model(args.model_name, device=torch.device('npu'))
    time = datetime.datetime.now().strftime("%Y_%m%d_%H_%M_%S")
    for dataset_name in args.dataset_names:
        result = {}
        print("ll",dataset_name)
        eval_function, task_type = dataset_task_dict[dataset_name]
        dataset = GeneralDataset(dataset_name)

        log_dir = os.path.join(args.logging_path, dataset_name)
        log_file = os.path.join(log_dir, 'log_{}.log'.format(time))
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(filename=log_file,
                            level=logging.INFO,
                            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        logging.info(f"This log on {args.model_name} grad_attacks is {args.grad_attacks} ")

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