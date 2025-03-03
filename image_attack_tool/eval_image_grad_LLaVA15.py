import torch
import argparse
from models import get_model
import datetime
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
    parser.add_argument('--device', type=int, default=1, help='NPU device ID (default: 0)')
    args = parser.parse_args()
    return args
def main(args):
    model = get_model(args.model_name, device=torch.device('npu'))
    time = datetime.datetime.now().strftime("%Y_%m%d_%H_%M_%S")
    print(model)
    print(time)

if __name__ == '__main__':
    args = parse_args()
    main(args) 