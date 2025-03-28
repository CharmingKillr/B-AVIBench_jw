import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import CLIPImageProcessor, CLIPVisionModel, StoppingCriteria

from . import get_image
import collections
import pdb
import os

d = collections.OrderedDict()

from PIL import Image



DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False


def get_model_name(model_path):
    # get model name
    if model_path.endswith("/"):
        model_path = model_path[:-1]
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        model_name = model_paths[-2] + "_" + model_paths[-1]
    else:
        model_name = model_paths[-1]
    
    return model_name



def load_model(model_path, model_name, dtype=torch.float16, device='npu'):
    
    tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
    
    
    model = AutoModelForCausalLM.from_pretrained(model_path,torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,trust_remote_code=True)
    


    context_len = 8192
    model.to(device=device)

    return tokenizer, model, context_len


class Testglm4v:
    def __init__(self, device=None):
        model_path="/data/jw/huggingfacemodel/glm-4v-9b"
        model_name = get_model_name(model_path)
        self.tokenizer, self.model, self.context_len = load_model(model_path, model_name)
        

        if device is not None:
            self.move_to_device(device)
        
    def move_to_device(self, device=None):
        if device is not None and 'npu' in device.type:
            self.dtype = torch.bfloat16
            self.device = device
        else:
            self.dtype = torch.float32
            self.device = 'cpu'
        
        self.model.to(device=self.device, dtype=self.dtype)
    
    @torch.no_grad()
    def batch_generate(self, image_list, question_list, max_new_tokens=256,method=None, level=0):
        images, prompts = [], []
        # print(image_list)
        for image, question in zip(image_list, question_list):
            # print("jfk",image)
            # print(image)
            if method is not None and level!=0:
                if level == 1:
                    DATA_PATA = '/data/jw/projects/B-AVIBench_jw/eval-data/corruption/lvlm_tiny_corruption_1'
                elif level == 3:
                    DATA_PATA = '/data/jw/projects/B-AVIBench_jw/eval-data/corruption/lvlm_tiny_corruption_3'
                elif level == 5:
                    DATA_PATA = '/data/jw/projects/B-AVIBench_jw/eval-data/corruption/lvlm_tiny_corruption_5'
                tmp=image.split('/')
                image=os.path.join(DATA_PATA,tmp[-2]+'_{}_{}'.format(method,level),tmp[-1])
            
            image = get_image(image)
            image = image.resize((224, 224), Image.BICUBIC)

            prompts.append(question)
            images.append(image)
        
        outputs = self.do_generate(prompts, images, dtype=self.dtype, max_new_tokens=max_new_tokens,method=method, level=level)

        return outputs

    @torch.no_grad()
    def do_generate(self, prompts, images, dtype=torch.float16, temperature=0, max_new_tokens=256,method=None, level=0):
        gen_kwargs = {"max_length": max_new_tokens, "do_sample": True, "top_k": 1}
        outputs=[]
        for prompt,image in zip(prompts,images):
            input = self.tokenizer.apply_chat_template([{"role": "user", "image": image, "content": prompt}],
                                       add_generation_prompt=True, tokenize=True, return_tensors="pt",
                                       return_dict=True).to(self.device)
            with torch.no_grad():
                output = self.model.generate(**input, **gen_kwargs)
                output = output[:, input['input_ids'].shape[1]:]
                output = self.tokenizer.decode(output[0])
                outputs.append(output)
        
        return outputs