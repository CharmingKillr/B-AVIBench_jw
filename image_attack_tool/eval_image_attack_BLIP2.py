import os
import json
import argparse
import datetime
#from wand.image import Image as WandImage
#from wand.api import library as wandlibrary
#import wand.color as WandColor
import torch
import numpy as np
from bat.attacks import SimBA
from models import get_model
from utils import dataset_task_dict
from tiny_datasets import dataset_class_dict, GeneralDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")

    # models
    parser.add_argument("--model_name", type=str, default="BLIP2")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)

    # datasets
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--sample_num", type=int, default=500)
    parser.add_argument("--sample_seed", type=int, default=20230719)

    # result_path
    parser.add_argument("--answer_path", type=str, default="/seu_nvme/home/230239304/projects_jw/B-AVIBench_jw/image_attack_tool/tiny_answers")
    
    # renew
    parser.add_argument("--renew", action="store_true", default=False)
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


def main(args):
    model = get_model(args.model_name, device=torch.device('cuda'))
    
    time = datetime.datetime.now().strftime("%Y_%m%d_%H_%M_%S")
    answer_path = f"{args.answer_path}/{args.model_name}"

    args.renew = True
    if args.renew :
        time_renew = '2025_0227_06_45_45'
        time = time_renew
    for dataset_name in args.dataset_names:
        final_path = os.path.join(os.path.join(answer_path, time), 'result_{}.json'.format(dataset_name))
        if args.renew and os.path.exists(final_path):
                    print(f"result file {final_path} exists, skip.")
                    continue

        result = {}
        print("ll",dataset_name)
        eval_function, task_type = dataset_task_dict[dataset_name]
        dataset = GeneralDataset(dataset_name)    
        
        metrics = eval_function(model, dataset, args.model_name, dataset_name, task_type, time, args.batch_size, answer_path=answer_path, method=None, level=0)
        print("***---",metrics)
        result["{}_severity_{}".format(dataset_name,0)] = metrics
                # result[dataset_name] = metrics

        result_path = os.path.join(os.path.join(answer_path, time), 'result_{}.json'.format(dataset_name))
        with open(result_path, "w") as f:
            f.write(json.dumps(result, indent=4))


if __name__ == "__main__":
    args = parse_args()
    args.dataset_names = ['ImageNetVC_color','ImageNetVC_component','ImageNetVC_material','ImageNetVC_others','ImageNetVC_shape','MSCOCO_MCI','VCR1_MCI','MSCOCO_OC','VCR1_OC','FUNSD','POIE','SROIE',
                         'COCO-Text','CTW','CUTE80','HOST','IC13','IC15','IIIT5K','SVTP','SVT','NoCaps','Flickr','MSCOCO_caption_karpathy','WHOOPSCaption','AOKVQAClose','AOKVQAOpen','DocVQA','GQA',
                         'OCRVQA','OKVQA','STVQA','TextVQA','WHOOPSVQA','WHOOPSWeird','Visdial','IconQA','VSR','ScienceQAIMG','VizWiz','MSCOCO_pope_random','MSCOCO_pope_adversarial','MSCOCO_pope_popular']
    main(args)