#coding=utf-8
#基于patch的攻击方法，一个patch一个patch进行攻击

import numpy as np
import time
import copy
import torch
import os
####
from .distances import Distance
from .distances import MSE
from PIL import Image
from .tools import has_word, remove_special_chars
from .cider import CiderScorer
from .tools import VQAEval
from models import llama_adapter_v2 as llama
from collections import defaultdict
import torch.nn.functional as F
from functools import reduce
import torch.nn as nn
import math
import gc
from torchviz import make_dot
from torchvision import transforms
import logging
####
def has_nested_attr(obj, attr_path):
    """检查对象是否具有嵌套属性"""
    try:
        reduce(getattr, attr_path.split('.'), obj)
        return True
    except AttributeError:
        return False
def softmax(logits):
    """Transforms predictions into probability values.

    Parameters
    ----------
    logits : array_like
        The logits predicted by the model.

    Returns
    -------
    `numpy.ndarray`
        Probability values corresponding to the logits.
    """

    assert logits.ndim == 1

    # for numerical reasons we subtract the max logit
    # (mathematically it doesn't matter!)
    # otherwise exp(logits) might become too large or too small
    logits = logits - np.max(logits)
    e = np.exp(logits)
    return e / np.sum(e)
def denormalize(image_tensor, mean, std):
    """将标准化后的图像反转换到原始像素空间 [0,1]"""
    if image_tensor.dim() == 3:  # [C, H, W]
        return torch.clamp(image_tensor * std.reshape(-1,1,1) + mean.reshape(-1,1,1), 0.0, 1.0)
    elif image_tensor.dim() == 4:  # [B, C, H, W]
        return torch.clamp(image_tensor * std + mean, 0.0, 1.0)
    else:
        raise ValueError("Unexpected image tensor shape")
class StopAttack(Exception):
    """Exception thrown to request early stopping of an attack
    if a given (optional!) threshold is reached."""

    pass
class F1Scorer:
    def __init__(self):
        self.n_detected_words = 0
        self.n_gt_words = 0        
        self.n_match_words = 0

    def add_string(self, ref, pred):        
        pred_words = list(pred.split())
        ref_words = list(ref.split())
        self.n_gt_words += len(ref_words)
        self.n_detected_words += len(pred_words)
        for pred_w in pred_words:
            if pred_w in ref_words:
                self.n_match_words += 1
                ref_words.remove(pred_w)

    def score(self):
        if float(self.n_detected_words)!=0:
            prec = self.n_match_words / float(self.n_detected_words) * 100
            recall = self.n_match_words / float(self.n_gt_words) * 100
        else:
            prec = 0
            recall = self.n_match_words / float(self.n_gt_words) * 100
        
        if prec + recall==0:
            f1=0
        else:
            f1 = 2 * (prec * recall) / (prec + recall)
        return prec, recall, f1

    def result_string(self):
        prec, recall, f1 = self.score()
        return f"Precision: {prec:.3f} Recall: {recall:.3f} F1: {f1:.3f}"


def crossentropy(label, logits):
    """Calculates the cross-entropy.

    Parameters
    ----------
    logits : array_like
        The logits predicted by the model.
    label : int
        The label describing the target distribution.

    Returns
    -------
    float
        The cross-entropy between softmax(logits) and onehot(label).

    """

    assert logits.ndim == 1

    # for numerical reasons we subtract the max logit
    # (mathematically it doesn't matter!)
    # otherwise exp(logits) might become too large or too small
    logits = logits - np.max(logits)
    e = np.exp(logits)
    s = np.sum(e)
    ce = np.log(s) - logits[label]
    return ce


def to_cuda(x): #将numpy转换为tensor在显卡上计算
    return torch.from_numpy(x).cuda()


def l2_distance(a, b):
    # if type(b) != torch.Tensor:
    #     b = torch.ones_like(a).cuda() * b        

    # dist = (torch.sum((torch.round(a)/255.0 - torch.round(b)/255.0) ** 2))**0.5
    
    # 确保 a 和 b 都在同一个设备上
    device = a.device  # 获取 a 所在的设备
    if type(b) != torch.Tensor:
        b = torch.ones_like(a).to(device) * b  # 确保 b 在相同设备上

    # 将 a 和 b 转换为 float32 类型
    a = a.to(torch.float32)
    b = b.to(torch.float32)

    # 计算 L2 距离，确保使用 round 操作的张量是 float32 类型
    dist = torch.sqrt(torch.sum((torch.round(a) / 255.0 - torch.round(b) / 255.0) ** 2))
    return dist


def normalize_noise(direction, distance, original_image):
    norm_direction = direction/l2_distance(direction, 0)   #归一化

    clipped_direction = torch.clip(torch.round(norm_direction*distance + original_image), 0, 255) - original_image

    clipped_dist = l2_distance(clipped_direction, 0)

    return clipped_direction, clipped_dist



def scatter_draw(data):
    save_path = "/home/syc/adversarial_machine_learning/nips18-avc-attack-template__/"
    fig = plt.figure(figsize=(16,9))
    plt.scatter(data[1], data[0], s=1)
    plt.savefig(save_path+"data.png", bbox_inches='tight')



def clip(x, min_x=-1, max_x=1):
    x[x < min_x] = min_x
    x[x > max_x] = max_x
    return x

def value_mask_init(patch_num):    #初始化查询价值mask
    value_mask = torch.ones([patch_num, patch_num]).cuda()
    # value_mask[int(patch_num*0.25):int(patch_num*0.75) , int(patch_num*0.25):int(patch_num*0.75)] = 0.5

    return value_mask

def noise_mask_init(x, image, patch_num, patch_size):    #初始化噪声幅度mask
    noise = x - image
    noise_mask = torch.zeros([patch_num, patch_num]).cuda()
    for row_counter in range(patch_num):
        for col_counter in range(patch_num):
            noise_mask[row_counter][col_counter] = l2_distance(noise[(row_counter*patch_size):(row_counter*patch_size+patch_size) , (col_counter*patch_size):(col_counter*patch_size+patch_size) ], 0)

    return noise_mask


def translate(index, patch_num):  #将价值最高patch的行列输出出来
    best_row = index//patch_num
    best_col = index - patch_num*best_row

    return best_row, best_col




class Attacker:
    def __init__(self, model,task,label):
        self.model = model
        self.task=task
        self.__original_class=label

    def attack(self, inputs):
        return NotImplementedError

    def attack_target(self, inputs, targets):
        return NotImplementedError


class PatchAttack(Attacker):
    def __init__(self, model,task,label): 
        self.model = model
        self.task=task
        self.__original_class=label

    def predictions(self, inputs, question_list=None, chat_list=None, max_new_tokens=None,model_name=None,vis_proc=None):
        if model_name=="TestMiniGPT4"  or model_name=="vpgtrans":
            chat_list_new=chat_list.copy()
            outputs = self.model.batch_answer([Image.fromarray(np.uint8(inputs))], [question_list], [chat_list_new],max_new_tokens=max_new_tokens)
            del chat_list_new
        elif model_name=="blip2":
            imgs = vis_proc["eval"](Image.fromarray(np.uint8(inputs))).unsqueeze(0).to("cuda", dtype=torch.float32)
            prompts = f"Question: {question_list} Answer:" 
            outputs = self.model.generate({"image": imgs, "prompt": prompts}, max_length=max_new_tokens)
        elif model_name=="instruct_blip":
            imgs = vis_proc["eval"](Image.fromarray(np.uint8(inputs))).unsqueeze(0).to("cuda", dtype=torch.float32)
            prompts = question_list 
            outputs = self.model.generate({"image": imgs, "prompt": prompts}, max_length=max_new_tokens)
        elif model_name=="adv2":
            imgs = vis_proc(Image.fromarray(np.uint8(inputs))).unsqueeze(0).to("cuda", dtype=torch.float32)
            prompts =[ llama.format_prompt(question_list) ]
            outputs =[ self.model.generate(imgs, prompts, temperature=0,max_gen_len=max_new_tokens)[0].strip()]
        elif model_name=="panda":  
            Image.fromarray(np.uint8(inputs)).save("./panda2_{}.png".format(vis_proc[0]) )      
            image_list= "./panda2_{}.png".format(vis_proc[0])        
            outputs = [self.model(image_list, question_list, max_new_tokens)]
        elif model_name=="otter": 
            imgs = vis_proc([Image.fromarray(np.uint8(inputs))],return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0).to("cuda", dtype=torch.float16)
            prompts = [f"<image> User: {question_list} GPT: <answer>"]
            lang_x = self.model.text_tokenizer(prompts, return_tensors="pt", padding=True)
            generated_text = self.model.generate(
            vision_x=imgs,
            lang_x=lang_x["input_ids"].to("cuda"),
            attention_mask=lang_x["attention_mask"].to("cuda", dtype=torch.float16),
            max_new_tokens=max_new_tokens,
            num_beams=3,
            no_repeat_ngram_size=3,
            )
            output = self.model.text_tokenizer.decode(generated_text[0])
            output = [x for x in output.split(' ') if not x.startswith('<')]
            out_label = output.index('GPT:')
            outputs = [' '.join(output[out_label + 1:])]
        elif model_name=="owl":
            prompt_template = "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: <image>\nHuman: {}\nAI:"

            prompts = [prompt_template.format(question_list)]
            inputs = vis_proc[2](text=prompts, images=[Image.fromarray(np.uint8(inputs))], return_tensors='pt')
            inputs = {k: v.to("cuda", dtype=torch.float32) if v.dtype == torch.float else v for k, v in inputs.items()}
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            generate_kwargs = {
            'do_sample': False,
            'top_k': 5,
            'max_length': max_new_tokens
            }
            with torch.no_grad():
                res = self.model.generate(**inputs, **generate_kwargs)
            outputs = [vis_proc[1].decode(res.tolist()[0], skip_special_tokens=True)]  
        elif model_name=="ofv2":
            vision_x = vis_proc[0](Image.fromarray(np.uint8(inputs))).unsqueeze(0).unsqueeze(0).unsqueeze(0).to("cuda", dtype=torch.float16)
            prompts = [f"<image>Question: {question_list} Short answer:"]            
            lang_x = vis_proc[1](
            prompts,
            return_tensors="pt", padding=True,
            ).to("cuda")
            generated_text = self.model.generate(
            vision_x=vision_x,
            lang_x=lang_x["input_ids"].to("cuda"),
            attention_mask=lang_x["attention_mask"].to("cuda", dtype=torch.float16),
            max_new_tokens=max_new_tokens,
            num_beams=3,pad_token_id=vis_proc[1].eos_token_id
            )
            outputs = vis_proc[1].batch_decode(generated_text, skip_special_tokens=True)
            outputs = [y[len(x)-len('<image>'):].strip() for x, y in zip(prompts, outputs)]
        elif model_name == "internlm":
            Image.fromarray(np.uint8(inputs)).save("./internlm2{}.png".format(vis_proc[2]))   
            image_list= "./internlm2{}.png".format(vis_proc[2])  
            texts=f" <|User|>:<ImageHere> {question_list}" + vis_proc[1] + " <|Bot|>:"
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    outputs=[vis_proc[0](self.model,texts, image_list,max_new_tokens=max_new_tokens) ]  
            
        elif model_name=="llava" or model_name=="llava15" or model_name=="moellava" or model_name=="sharegpt4v":
            outputs = self.model([question_list],[Image.fromarray(np.uint8(inputs))],stop_str=vis_proc[0], dtype=torch.float16, max_new_tokens=max_new_tokens,method=vis_proc[1], level=vis_proc[2],image_listnew=None)
       
        if self.task=="cls" or self.task=="ocr":
            predict = remove_special_chars(outputs[0]).lower()            
        else:
            predict = outputs[0]
        temp_result=predict
        if 1:
            is_adversarial=False
            if self.task=="cls" or self.task=="ocr":        
                adv=True
                if len(self.__original_class)!=0 and not isinstance(self.__original_class, str):
                    for gt in self.__original_class:
                        is_adversarial_tmp=not (bool(has_word(temp_result, gt)) or bool(has_word(temp_result, gt+'s')) )
                        adv=adv and is_adversarial_tmp
                    is_adversarial= is_adversarial or adv
                else:
                    is_adversarial=not (bool(has_word(temp_result, self.__original_class)) or bool(has_word(temp_result, self.__original_class+'s')) )                    
            if self.task=="caption":                
                cider_scorer = CiderScorer(n=4, sigma=6.0)
                cider_scorer += (temp_result, self.__original_class)
                (score, scores) = cider_scorer.compute_score()                
                if scores==0:
                    is_adversarial=True
            elif self.task=="kie":
                f1_scorer = F1Scorer()
                if isinstance(self.__original_class, list) :
                    gt_answers =" ".join(self.__original_class)
                else:
                    gt_answers=self.__original_class
                f1_scorer.add_string(gt_answers, temp_result)
                prec, recall, f1 = f1_scorer.score()
                if f1==0:
                    is_adversarial=True                    
            elif self.task=="mrr":
                eval = VQAEval()
                mrr = eval.evaluate_MRR(temp_result, self.__original_class)
                if mrr==0:
                    is_adversarial=True  
            elif self.task=="vqa" or self.task=="vqachoice" or self.task=="imagenetvc":
                eval = VQAEval()
                mrr = eval.evaluate(temp_result, self.__original_class)
                if mrr==0:
                    is_adversarial=True     
        
        return predict,is_adversarial

    def distance(self, input1, input2, min_, max_):
        return np.mean((input1 - input2) ** 2) / ((max_ - min_) ** 2)

    def print_distance(self, distance):
        return np.sqrt(distance * 1*28*28)

    def log_step(self, step, distance, spherical_step, source_step, message=''):
        print('Step {}: {:.5f}, stepsizes = {:.1e}/{:.1e}: {}'.format(
            step,
            self.print_distance(distance),
            spherical_step,
            source_step,
            message))

    def patch_attack(
            self,
            original,    #原始图像
            label,       #原始标签
            starting_point,   #初始对抗样本
            iterations=1000,  #总的查询次数
            min_=0.0,         
            max_=255.0,
            mode='targeted', question_list=None, chat_list=None, max_new_tokens=None,model_name=None,vis_proc=None):

        from numpy.linalg import norm
        from scipy import interpolate
        import collections

        #全部转换为torch来计算
        original = to_cuda(original)
        starting_point = to_cuda(starting_point)
        step = 0

        patch_num = 4   #横纵几等分
        patch_size = int(original.shape[0] / patch_num)


        success_num = 0    #成功和失败的次数
        fail_num = 0

        value_mask = value_mask_init(patch_num)
        noise_mask = noise_mask_init(starting_point, original, patch_num, patch_size)

        best_noise = starting_point - original
        current_min_noise = l2_distance(starting_point, original)

        #FIXME
        evolutionary_doc = np.zeros(iterations)   #记录下当前最小噪声   这个不管了，先不记录了

        while step < iterations:

            if torch.sum(value_mask * noise_mask) == 0:  #当前平分方法下没有可以查询的了
                #FIXME
                # print("*************-----------------")
                # pdb.set_trace()
                print("patch num * 2", step)
                patch_num *= 2

                if patch_num == 64:  
                    print("only", step)
                    break

                patch_size = int(original.shape[0] / patch_num)

                value_mask = value_mask_init(patch_num)
                noise_mask = noise_mask_init(best_noise, original, patch_num, patch_size)
            total_mask = value_mask*noise_mask
            best_index = torch.argmax(total_mask)
            best_row, best_col = translate(best_index, patch_num)
            temp_noise = copy.deepcopy(best_noise)
            temp_noise[(best_row*patch_size):(best_row*patch_size+patch_size) , (best_col*patch_size):(best_col*patch_size+patch_size) ] = 0
            candidate = torch.clip(torch.round(original + temp_noise), 0, 255)   
            if l2_distance(candidate, original) >= current_min_noise:
                value_mask[best_row, best_col] = 0
                continue
            if chat_list is not None:
                chat_list_new=chat_list.copy()
                temp_result,is_adversarial = self.predictions((candidate).cpu().numpy(), question_list=question_list, chat_list=chat_list_new, max_new_tokens=max_new_tokens,model_name=model_name,vis_proc=vis_proc)
                del chat_list_new
            else:
                # 224,224,3  'Yes, there is a person in the image, and they are standing in front of a cabinet.'
                temp_result,is_adversarial = self.predictions((candidate).cpu().numpy(), question_list=question_list, chat_list=None, max_new_tokens=max_new_tokens,model_name=model_name,vis_proc=vis_proc)

            if is_adversarial:
                current_min_noise = l2_distance(candidate, original)
                success_num += 1
                best_noise = candidate - original
                noise_mask[best_row, best_col] = l2_distance(best_noise[(best_row*patch_size):(best_row*patch_size+patch_size) , (best_col*patch_size):(best_col*patch_size+patch_size) ], 0)
            else:
                fail_num += 1
                value_mask[best_row, best_col] = 0
            step += 1 
        final_best_adv_example = best_noise+original
        final_best_adv_example = final_best_adv_example.cpu().numpy().astype(np.float32)
        print("success_num", success_num, step)
        return final_best_adv_example, step


    def attack(
            self, 
            image,
            label,
            starting_point, 
            iterations=1000,
            val_samples = 1000,
            min_=0.0, 
            max_=255.0,
            mode = 'untargeted',
            strategy = 0, question_list=None, chat_list=None, max_new_tokens=None,model_name=None,vis_proc=None):

        if mode == 'untargeted':
            return self.patch_attack(image, label, starting_point, iterations, min_, max_, mode='untargeted', question_list=question_list, chat_list=chat_list, max_new_tokens=max_new_tokens,model_name=model_name,vis_proc=vis_proc)

class PatchGradAttack(Attacker):
    def __init__(self, model, task, label, tokenizer, stop_str, 
                image_layers=[12, 18, 23], 
                text_layers=[16, 24, 31],
                patch_size=14,
                topk_ratio=0.3):
        self.model = model
        self.task=task
        self.__original_class=label
        self.model_dtype = next(model.parameters()).dtype
        # 模型结构验证
        self._validate_model_structure()
        
        # 配置参数
        self.tokenizer = tokenizer
        self.image_layers = sorted(set(image_layers))
        self.text_layers = sorted(set(text_layers))
        self.stop_str = stop_str
        self.patch_size = patch_size
        self.topk_ratio = topk_ratio
        self.device = next(model.parameters()).device
        
        # 初始化特征缓存（带梯度保留）
        self.image_features = defaultdict(lambda: [])
        self.text_features = defaultdict(lambda: [])
        self._register_hooks()

        # 动态投影层初始化
        self._init_dynamic_projector()
        
        # 课程学习参数
        self.curriculum_phase = 0  # 0:粗粒度,1:中粒度,2:细粒度
        self.temperature = nn.Parameter(torch.tensor(0.07))
    
    def _validate_model_structure(self):
        """深度验证模型结构完整性"""
        required_components = [
        'model.vision_tower.vision_tower.vision_model.encoder.layers',
        'model.layers',
        'model.vision_tower.vision_tower.config.image_size',
        'model.vision_tower.vision_tower.config.patch_size'
        ]
    
        for comp in required_components:
            if not has_nested_attr(self.model, comp):
                raise AttributeError(f"模型缺少关键组件: {comp}")
    def _init_dynamic_projector(self):
        """动态多尺度投影网络"""
        model_dtype = next(self.model.parameters()).dtype
        
        # 图像特征投影 (核心修改)
        self.img_proj = nn.ModuleDict({
            'shallow': nn.Sequential(
                nn.Linear(1024, 4096),  # 图像1024维 -> 文本4096维
                nn.GELU(),
                nn.LayerNorm(4096)
            ),
            'deep': nn.Sequential(
                nn.Linear(1024, 4096),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1)
            )
        }).to(self.device, dtype=model_dtype)

        def _init_weights(module):
            if isinstance(module, nn.Linear):
                # He初始化适配LeakyReLU
                nn.init.kaiming_normal_(module.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)

        self.img_proj.apply(_init_weights)
        # 自适应融合参数
        self.alpha = nn.Parameter(torch.tensor([0.6, 0.4])).to(model_dtype)  # 浅层/深层权重
        
        # 初始化验证
        for param in self.img_proj.parameters():
            param.requires_grad_(False)  # 禁用所有参数

    def _register_hooks(self):
        """注册带梯度保留的钩子"""
        # 视觉编码器钩子
        vision_encoder = self.model.model.vision_tower.vision_tower.vision_model.encoder
        max_vision_layer = len(vision_encoder.layers) - 1
        for layer_idx in self.image_layers:
            if layer_idx > max_vision_layer:
                raise ValueError(f"视觉层索引{layer_idx}超出范围(最大{max_vision_layer})")
            layer = vision_encoder.layers[layer_idx]
            layer.register_forward_hook(self._create_vision_hook(layer_idx))
        
        # 文本解码器钩子
        text_decoder = self.model.model.layers
        max_text_layer = len(text_decoder) - 1
        for layer_idx in self.text_layers:
            if layer_idx > max_text_layer:
                raise ValueError(f"文本层索引{layer_idx}超出范围(最大{max_text_layer})")
            layer = text_decoder[layer_idx]
            layer.register_forward_hook(self._create_text_hook(layer_idx))

    def _create_vision_hook(self, layer_idx):
        """视觉特征处理（保留空间信息）"""
        def hook(module, inputs, outputs):
            feat = outputs[0] if isinstance(outputs, tuple) else outputs
            # 确保特征形状为 [B, seq_len, D]
            if feat.dim() == 4:  # 来自CNN的特征 [B, C, H, W]
                feat = feat.flatten(2).permute(0,2,1)  # [B, C, H, W] -> [B, H*W, C]
            elif feat.dim() == 3:  # ViT特征 [B, seq_len, D]
                pass
            else:
                raise ValueError(f"不支持的视觉特征维度: {feat.shape}")
            
            feat = feat.contiguous().reshape(feat.size(0), -1, feat.size(-1)) 
            self.image_features[layer_idx].append(feat)
        return hook

    def _create_text_hook(self, layer_idx):
        """文本特征处理（带时间步管理）"""
        def hook(module, inputs, outputs):
            if self.is_generating:  # 添加生成状态标记
                hidden_states = outputs[0][:, -1:, :]
                self.text_features[layer_idx].append({
                    'states': hidden_states,
                    'timesteps': None  # 延迟到后处理选择
                })
        return hook
    def _post_process_features(self):
        """生成完成后统一处理特征"""
        for layer in self.text_layers:
            if not self.text_features[layer]:
                continue
                
            # 获取完整生成序列的隐藏状态 [B, T_total, D]
            all_states = torch.cat([d['states'] for d in self.text_features[layer]], dim=1)
            
            # 全局选择重要时间步
            timesteps = self._get_important_timesteps(all_states, topk=3)
            
            # 更新存储数据
            self.text_features[layer][-1]['timesteps'] = timesteps
    def _get_important_timesteps(self, features, topk=3):
        """动态选择重要时间步"""
        # 确保特征维度正确
        if features.dim() == 2:
            features = features.unsqueeze(1)  # [B,D] -> [B,1,D]
        
        B, T, D = features.shape
        assert T >= 1, "时间步维度不能为0"
        # 动态调整k值
        k = min(topk, T)
        if k <= 0:
            k = 1
        
        # 计算每个token重要性 (无需平均)
        importance = torch.norm(features, dim=-1)  # [B, T]
        
        # 获取每个样本的前k个时间步索引
        _, indices = torch.topk(importance, k=k, dim=1)  # [B, k]
        assert indices.max() < T, "选择索引超出序列范围"
        return indices
    def _create_gaussian_target(self, H, W, sigma_scale=0.2):
        """生成高斯目标分布"""
        device = self.device
        sigma = max(H, W) * sigma_scale
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=device), 
            torch.arange(W, device=device), 
            indexing='ij'
        )
        center = (W//2, H//2)
        spatial_mask = torch.exp(-(
            (grid_x - center[0])**2 + 
            (grid_y - center[1])**2
        )/(2*sigma**2))
        
        # 添加CLS token位置（索引0）
        spatial_mask = F.pad(spatial_mask.flatten(), (1,0), value=0)  # [H*W+1]
        return spatial_mask / spatial_mask.sum()
    def _compute_hierarchical_loss(self):
        """层次化特征对齐损失"""
        total_loss = 0
        device = next(iter(self.image_features.values()))[0].device
        # 图像特征处理
        img_shallow = torch.cat(self.image_features[self.image_layers[0]], dim=1)  # [B, 577, 1024] 
        img_deep = torch.cat(self.image_features[self.image_layers[-1]], dim=1)  # [B, 577, 1024]
        
        # 投影到文本空间
        proj_img_shallow = self.img_proj['shallow'](img_shallow)  # [B, 577, 4096] -> [B, 577, 4096]
        proj_img_deep = self.img_proj['deep'](img_deep)         # [B, 577, 4096] -> [B, 577, 4096]
        
        # 文本特征对齐
        for layer_idx in self.text_layers:
            if not self.text_features[layer_idx]:
                continue
                
            text_data = self.text_features[layer_idx][-1]
            text_states = text_data['states']  # [B, 1, D]
            if torch.isnan(text_states).any():
                print("警告:文本编码器输出含NaN!")
            selected_steps = text_data['timesteps'].squeeze(0).cpu().numpy().tolist()
            selected_tensors = []
            for step in selected_steps:
                feat = self.text_features[layer_idx][step]
                selected_tensors.append(feat['states'])
            selected_features = torch.cat(selected_tensors, dim=1)  # [B, T_total, D]

            # 分层投影
            if layer_idx < 20:
                assert not torch.isnan(selected_features).any(), "文本特征含NaN!"
                assert not torch.isnan(proj_img_shallow).any(), "图像投影特征含NaN!"

                text_norm = F.normalize(selected_features, p=2, dim=-1).to(device)
                img_norm = F.normalize(proj_img_shallow, p=2, dim=-1).to(device)

                attn_scores = torch.einsum('btd,bpd->btp', text_norm, img_norm) / self.temperature
                
                # 动态生成高斯目标分布（示例：聚焦中心区域）
                seq_len = attn_scores.size(-1) - 1
                H = int(math.sqrt(seq_len))
                W = seq_len // H
                
                spatial_mask = self._create_gaussian_target(H, W).reshape(1, 1, -1)
                
                # KL散度损失
                loss = F.kl_div(
                    F.log_softmax(attn_scores, dim=-1),
                    spatial_mask.expand_as(attn_scores),
                    reduction='batchmean',
                    log_target=False
                )
                total_loss += self.alpha[0] * loss
            # 深层对齐（语义相似性）
            else:
                text_norm = F.normalize(selected_features, p=2, dim=-1).to(device)
                img_norm = F.normalize(proj_img_deep, p=2, dim=-1).to(device)

                sim_matrix = torch.einsum('btd,bpd->btp', text_norm, img_norm)
                max_sim = sim_matrix.max(dim=-1)[0].mean()
                
                total_loss += self.alpha[1] * (1 - max_sim)
        
        return total_loss.to(torch.float16)
    def _reset_features(self):
        """彻底清除特征缓存和计算图"""
        # 清空特征缓存
        self.image_features.clear()
        self.text_features.clear()
        
        # 释放钩子中的特征
        for layer_idx in self.image_layers:
            if layer_idx in self.image_features:
                for feat in self.image_features[layer_idx]:
                    del feat
                self.image_features[layer_idx].clear()
        
        for layer_idx in self.text_layers:
            if layer_idx in self.text_features:
                for feat in self.text_features[layer_idx]:
                    del feat
                self.text_features[layer_idx].clear()
        
        # 清除所有参数的梯度
        self.model.zero_grad(set_to_none=True)
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()
        
        # 强制回收内存
        torch.cuda.empty_cache()
    
    def attack(self, image, label, iterations=10, alpha=0.003, momentum=0.9, epsilon=8/255, log_file=None, **kwargs):
        """层次化对抗攻击"""
        mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1).to(self.device)
    
        save_dir = os.path.dirname(kwargs.get('save_dir'))

        # 初始化图像和扰动(只做一次)
        image_tensor = self._preprocess_image(image)  # 原始图像
        image_denorm = denormalize(image_tensor, mean, std).to(self.device)
        delta = torch.zeros_like(image_denorm, dtype=self.model_dtype, requires_grad=True).to(self.device)  # 初始化扰动

        # 保存resize后的原始图像
        tmp=kwargs['save_dir'].split('/')
        save_path = os.path.join(save_dir, 'clean', tmp[-1])
        os.makedirs(os.path.join(save_dir, 'clean'), exist_ok=True)
        test_img = image_denorm.permute(0,2,3,1).detach().cpu().squeeze(0).numpy()
        test_img = (test_img * 255).astype(np.uint8)
        Image.fromarray(test_img).save(save_path)

        best_adv = image.copy()
        min_dist = float('inf')

        logging.info(f"Attack started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        logging.info(f"Parameters: alpha={alpha}, epsilon={epsilon}\n\n")
        for step in range(iterations):
            self._reset_features()
            
            adv_image = torch.clamp(image_denorm + delta, 0.0, 1.0)
            adv_image_norm = (adv_image - mean) / std

            self._update_curriculum(step, iterations)  # 课程学习更新

            # 前向传播获取特征
            self.is_generating = True

            outputs_ids = self._trigger_forward(adv_image_norm, **kwargs)

            self.is_generating = False
            self._post_process_features()
            outputs = self._postprocess_outputs(kwargs['input_ids'],outputs_ids)
            if step == 0:
                logging.info(f"Original prediction: {outputs}\n\n")
            # 计算层次化损失
            loss = self._compute_hierarchical_loss()

            loss.backward(retain_graph=False)

            # Patch重要性mask
            with torch.no_grad():
                grad_magnitude = delta.grad.abs().sum(dim=1, keepdim=True)
                patch_mask = self._compute_patch_importance(grad_magnitude)
                
                # 带mask的PGD更新
                signed_grad = delta.grad.sign()
                delta.data += alpha * signed_grad * patch_mask
                
                # 扰动裁剪
                delta.data = torch.clamp(delta.data, -epsilon, epsilon)
                delta.data = torch.clamp(image_denorm + delta, 0, 1) - image_denorm
            
            # 评估对抗样本
            with torch.no_grad():
                current_adv = image_denorm + delta.detach()
                current_dist = F.mse_loss(current_adv, image_denorm)
            
                current_adv_norm = (current_adv - mean) / std

                predict, is_adv = self.predictions(current_adv_norm, max_new_tokens=kwargs['max_new_tokens'], model_name=kwargs['model_name'], vis_proc=kwargs['vis_proc'],
                                                stopping_criteria=kwargs['stopping_criteria'], input_ids=kwargs['input_ids'])

                # 记录日志
                log_msg = f"Step {step:03d} | Loss: {loss.item():.4f} | Distance: {current_dist:.4f} | Success: {is_adv} | Prediction: {predict}\n"
                logging.info(log_msg)
                print(log_msg, end='')

                if current_dist < min_dist:
                    best_adv = current_adv.detach()
                    min_dist = current_dist

        # 保存最终结果
        if 'save_dir' in kwargs:
            save_path = os.path.join(save_dir, 'adv', tmp[-1])
            os.makedirs(os.path.join(save_dir, 'adv'), exist_ok=True)
            best_adv_denorm = denormalize(best_adv, mean, std)
            adv_img = best_adv_denorm.permute(0,2,3,1).cpu().squeeze(0).numpy()
            adv_img = (adv_img * 255).astype(np.uint8)
            Image.fromarray(adv_img).save(save_path)

        return best_adv, min_dist, predict

    def _update_curriculum(self, step, total_steps):
        """动态课程学习策略"""
        phase = step / total_steps
        if phase < 0.3:  # 粗粒度阶段
            self.alpha.data = torch.tensor([0.7, 0.2, 0.1], device=self.device)
        elif phase < 0.7:  # 中粒度阶段
            self.alpha.data = torch.tensor([0.3, 0.5, 0.2], device=self.device)
        else:  # 细粒度阶段
            self.alpha.data = torch.tensor([0.1, 0.3, 0.6], device=self.device)

    def _preprocess_image(self, image):
        """带梯度保留的预处理"""
        tensor = torch.from_numpy(np.array(image)).unsqueeze(0)
        tensor = tensor.to(self.device, dtype=next(self.model.parameters()).dtype)
        return tensor.requires_grad_(True)

    def _trigger_forward(self, image_tensor, **kwargs):
        """带梯度上下文的前向传播"""
        model_dtype = next(self.model.parameters()).dtype
        if kwargs['input_ids'].dim() == 1:
            input_ids = kwargs['input_ids'].unsqueeze(0)
        image_tensor = image_tensor.clone().detach().requires_grad_(True)
        
        with torch.autograd.graph.saved_tensors_hooks(
            pack_hook=lambda x: x.detach().to(model_dtype),
            unpack_hook=lambda x: x.to(model_dtype).requires_grad_(True)):
            return self.model.generate(
                input_ids=input_ids,
                images=image_tensor,
                max_new_tokens=kwargs['max_new_tokens'],
                use_cache=True,
                stopping_criteria=kwargs['stopping_criteria']
            )
        
    def _inference_forward(self, image_tensor, **kwargs):
        """仅推理的前向传播"""
        if kwargs['input_ids'].dim() == 1:
                input_ids = kwargs['input_ids'].unsqueeze(0)
        with torch.inference_mode():
            # 执行完整生成流程
            outputs_ids = self.model.generate(
                    input_ids=input_ids,
                    images=image_tensor,
                    max_new_tokens=kwargs['max_new_tokens'],
                    use_cache=True,
                    stopping_criteria=kwargs['stopping_criteria']
                )
            outputs = self._postprocess_outputs(kwargs['input_ids'],outputs_ids)
        return outputs
    def _compute_patch_importance(self, grad_magnitude, topk=10):
        """基于梯度显著性+特征激活的复合评估"""
        # 梯度幅值计算
        grad_energy = grad_magnitude.pow(2).sum(dim=1, keepdim=True)  # [B,1,H,W]
        
        # 特征激活度计算（来自中间层）
        with torch.no_grad():
            activation = self.image_features[self.image_layers[-1]][0].mean(dim=-1)
            activation = activation.view_as(grad_energy)
        
        # 复合重要性评分
        importance = grad_energy * activation.sigmoid()
        
        # 稀疏化处理
        B, C, H, W = importance.shape
        patch_size = self.patch_size
        patches = F.unfold(importance, patch_size, stride=patch_size)  # [B, C*K, N]
        
        # 选择Top-K补丁
        topk_values, topk_indices = patches.topk(topk, dim=-1)
        
        # 生成稀疏mask
        mask = torch.zeros_like(patches)
        mask.scatter_(-1, topk_indices, 1.0)
        mask = F.fold(mask, (H,W), patch_size, stride=patch_size)
        
        return mask
    def _postprocess_outputs(self, input_ids, output_ids):
        if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        if self.stop_str is not None:
            for i in range(len(outputs)):
                tmp = outputs[i].strip()
                if tmp.endswith(self.stop_str):
                    tmp = tmp[:-len(self.stop_str)]
                outputs[i] = tmp.strip()
        
        return outputs
    def predictions(self, inputs, question_list=None, chat_list=None, max_new_tokens=None,model_name=None,vis_proc=None,**kwargs):
        if model_name=="TestMiniGPT4"  or model_name=="vpgtrans":
            chat_list_new=chat_list.copy()
            outputs = self.model.batch_answer([Image.fromarray(np.uint8(inputs))], [question_list], [chat_list_new],max_new_tokens=max_new_tokens)
            del chat_list_new
        elif model_name=="blip2":
            imgs = vis_proc["eval"](Image.fromarray(np.uint8(inputs))).unsqueeze(0).to("cuda", dtype=torch.float32)
            prompts = f"Question: {question_list} Answer:" 
            outputs = self.model.generate({"image": imgs, "prompt": prompts}, max_length=max_new_tokens)
        elif model_name=="instruct_blip":
            imgs = vis_proc["eval"](Image.fromarray(np.uint8(inputs))).unsqueeze(0).to("cuda", dtype=torch.float32)
            prompts = question_list 
            outputs = self.model.generate({"image": imgs, "prompt": prompts}, max_length=max_new_tokens)
        elif model_name=="adv2":
            imgs = vis_proc(Image.fromarray(np.uint8(inputs))).unsqueeze(0).to("cuda", dtype=torch.float32)
            prompts =[ llama.format_prompt(question_list) ]
            outputs =[ self.model.generate(imgs, prompts, temperature=0,max_gen_len=max_new_tokens)[0].strip()]
        elif model_name=="panda":  
            Image.fromarray(np.uint8(inputs)).save("./panda2_{}.png".format(vis_proc[0]) )      
            image_list= "./panda2_{}.png".format(vis_proc[0])        
            outputs = [self.model(image_list, question_list, max_new_tokens)]
        elif model_name=="otter": 
            imgs = vis_proc([Image.fromarray(np.uint8(inputs))],return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0).to("cuda", dtype=torch.float16)
            prompts = [f"<image> User: {question_list} GPT: <answer>"]
            lang_x = self.model.text_tokenizer(prompts, return_tensors="pt", padding=True)
            generated_text = self.model.generate(
            vision_x=imgs,
            lang_x=lang_x["input_ids"].to("cuda"),
            attention_mask=lang_x["attention_mask"].to("cuda", dtype=torch.float16),
            max_new_tokens=max_new_tokens,
            num_beams=3,
            no_repeat_ngram_size=3,
            )
            output = self.model.text_tokenizer.decode(generated_text[0])
            output = [x for x in output.split(' ') if not x.startswith('<')]
            out_label = output.index('GPT:')
            outputs = [' '.join(output[out_label + 1:])]
        elif model_name=="owl":
            prompt_template = "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: <image>\nHuman: {}\nAI:"

            prompts = [prompt_template.format(question_list)]
            inputs = vis_proc[2](text=prompts, images=[Image.fromarray(np.uint8(inputs))], return_tensors='pt')
            inputs = {k: v.to("cuda", dtype=torch.float32) if v.dtype == torch.float else v for k, v in inputs.items()}
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            generate_kwargs = {
            'do_sample': False,
            'top_k': 5,
            'max_length': max_new_tokens
            }
            with torch.no_grad():
                res = self.model.generate(**inputs, **generate_kwargs)
            outputs = [vis_proc[1].decode(res.tolist()[0], skip_special_tokens=True)]  
        elif model_name=="ofv2":
            vision_x = vis_proc[0](Image.fromarray(np.uint8(inputs))).unsqueeze(0).unsqueeze(0).unsqueeze(0).to("cuda", dtype=torch.float16)
            prompts = [f"<image>Question: {question_list} Short answer:"]            
            lang_x = vis_proc[1](
            prompts,
            return_tensors="pt", padding=True,
            ).to("cuda")
            generated_text = self.model.generate(
            vision_x=vision_x,
            lang_x=lang_x["input_ids"].to("cuda"),
            attention_mask=lang_x["attention_mask"].to("cuda", dtype=torch.float16),
            max_new_tokens=max_new_tokens,
            num_beams=3,pad_token_id=vis_proc[1].eos_token_id
            )
            outputs = vis_proc[1].batch_decode(generated_text, skip_special_tokens=True)
            outputs = [y[len(x)-len('<image>'):].strip() for x, y in zip(prompts, outputs)]
        elif model_name == "internlm":
            Image.fromarray(np.uint8(inputs)).save("./internlm2{}.png".format(vis_proc[2]))   
            image_list= "./internlm2{}.png".format(vis_proc[2])  
            texts=f" <|User|>:<ImageHere> {question_list}" + vis_proc[1] + " <|Bot|>:"
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    outputs=[vis_proc[0](self.model,texts, image_list,max_new_tokens=max_new_tokens) ]  
            
        elif model_name=="llava" or model_name=="llava15" or model_name=="moellava" or model_name=="sharegpt4v":
            outputs = self._inference_forward(inputs, max_new_tokens=max_new_tokens, **kwargs)
       
        if self.task=="cls" or self.task=="ocr":
            predict = remove_special_chars(outputs[0]).lower()            
        else:
            predict = outputs[0]
        temp_result=predict
        if 1:
            is_adversarial=False
            if self.task=="cls" or self.task=="ocr":        
                adv=True
                if len(self.__original_class)!=0 and not isinstance(self.__original_class, str):
                    for gt in self.__original_class:
                        is_adversarial_tmp=not (bool(has_word(temp_result, gt)) or bool(has_word(temp_result, gt+'s')) )
                        adv=adv and is_adversarial_tmp
                    is_adversarial= is_adversarial or adv
                else:
                    is_adversarial=not (bool(has_word(temp_result, self.__original_class)) or bool(has_word(temp_result, self.__original_class+'s')) )                    
            if self.task=="caption":                
                cider_scorer = CiderScorer(n=4, sigma=6.0)
                cider_scorer += (temp_result, self.__original_class)
                (score, scores) = cider_scorer.compute_score()                
                if scores==0:
                    is_adversarial=True
            elif self.task=="kie":
                f1_scorer = F1Scorer()
                if isinstance(self.__original_class, list) :
                    gt_answers =" ".join(self.__original_class)
                else:
                    gt_answers=self.__original_class
                f1_scorer.add_string(gt_answers, temp_result)
                prec, recall, f1 = f1_scorer.score()
                if f1==0:
                    is_adversarial=True                    
            elif self.task=="mrr":
                eval = VQAEval()
                mrr = eval.evaluate_MRR(temp_result, self.__original_class)
                if mrr==0:
                    is_adversarial=True  
            elif self.task=="vqa" or self.task=="vqachoice" or self.task=="imagenetvc":
                eval = VQAEval()
                mrr = eval.evaluate(temp_result, self.__original_class)
                if mrr==0:
                    is_adversarial=True     
        
        return predict,is_adversarial
            
class PatchGradAttacktest(Attacker):
    def __init__(self, model, task, label, tokenizer, stop_str, 
                image_layers=[12, 18, 23], 
                text_layers=[16, 24, 31],
                patch_size=14,
                topk_ratio=0.3):
        self.model = model
        self.task=task
        self.__original_class=label
        
        # 模型结构验证
        self._validate_model_structure()
        
        # 配置参数
        self.tokenizer = tokenizer
        self.image_layers = sorted(set(image_layers))
        self.text_layers = sorted(set(text_layers))
        self.stop_str = stop_str
        self.patch_size = patch_size
        self.topk_ratio = topk_ratio
        self.device = next(model.parameters()).device
        
        # 初始化特征缓存（带梯度保留）
        self.image_features = defaultdict(lambda: [])
        self.text_features = defaultdict(lambda: [])
        self._register_hooks()

        # 动态投影层初始化
        self._init_dynamic_projector()
        
        # 课程学习参数
        self.curriculum_phase = 0  # 0:粗粒度,1:中粒度,2:细粒度
    
    def _validate_model_structure(self):
        """深度验证模型结构完整性"""
        required_components = [
        'model.vision_tower.vision_tower.vision_model.encoder.layers',
        'model.layers',
        'model.vision_tower.vision_tower.config.image_size',
        'model.vision_tower.vision_tower.config.patch_size'
        ]
    
        for comp in required_components:
            if not has_nested_attr(self.model, comp):
                raise AttributeError(f"模型缺少关键组件: {comp}")
    def _init_dynamic_projector(self):
        """动态多尺度投影网络"""
        model_dtype = next(self.model.parameters()).dtype
        
        # 图像特征投影 (核心修改)
        self.img_proj = nn.ModuleDict({
            'shallow': nn.Sequential(
                nn.Linear(1024, 4096),  # 图像1024维 -> 文本4096维
                nn.GELU(),
                nn.LayerNorm(4096)
            ),
            'deep': nn.Sequential(
                nn.Linear(1024, 4096),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.1)
            )
        }).to(self.device, dtype=model_dtype)

        def _init_weights(module):
            if isinstance(module, nn.Linear):
                # He初始化适配LeakyReLU
                nn.init.kaiming_normal_(module.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)

        self.img_proj.apply(_init_weights)
        # 自适应融合参数
        self.alpha = nn.Parameter(torch.tensor([0.6, 0.4]))  # 浅层/深层权重
        
        # 初始化验证
        for param in self.img_proj.parameters():
            param.requires_grad_(False)  # 禁用所有参数

    def _register_hooks(self):
        """注册带梯度保留的钩子"""
        # 视觉编码器钩子
        vision_encoder = self.model.model.vision_tower.vision_tower.vision_model.encoder
        for layer_idx in self.image_layers:
            if layer_idx < len(vision_encoder.layers):
                layer = vision_encoder.layers[layer_idx]
                layer.register_forward_hook(self._create_vision_hook(layer_idx))
        
        # 文本解码器钩子（带梯度保留）
        text_decoder = self.model.model.layers
        for layer_idx in self.text_layers:
            if layer_idx < len(text_decoder):
                layer = text_decoder[layer_idx]
                layer.register_forward_hook(self._create_text_hook(layer_idx))

    def _create_vision_hook(self, layer_idx):
        """视觉特征处理（保留空间信息）"""
        def hook(module, inputs, outputs):
            feat = outputs[0] if isinstance(outputs, tuple) else outputs
            # 保留原始梯度流
            feat = feat.contiguous().view(feat.size(0), -1, feat.size(-1))  # [B, H*W, D]
            self.image_features[layer_idx].append(feat)
        return hook

    def _create_text_hook(self, layer_idx):
        """文本特征处理（带时间步管理）"""
        def hook(module, inputs, outputs):
            if self.is_generating:  # 添加生成状态标记
                hidden_states = outputs[0][:, -1:, :]
                self.text_features[layer_idx].append({
                    'states': hidden_states,
                    'timesteps': None  # 延迟到后处理选择
                })
        return hook
    def _post_process_features(self):
        """生成完成后统一处理特征"""
        for layer in self.text_layers:
            if not self.text_features[layer]:
                continue
                
            # 获取完整生成序列的隐藏状态 [B, T_total, D]
            all_states = torch.cat([d['states'] for d in self.text_features[layer]], dim=1)
            
            # 全局选择重要时间步
            timesteps = self._get_important_timesteps(all_states, topk=3)
            
            # 更新存储数据
            self.text_features[layer][-1]['timesteps'] = timesteps
    def _get_important_timesteps(self, features, topk=3):
        """动态选择重要时间步"""
        # 确保特征维度正确
        if features.dim() == 2:
            features = features.unsqueeze(1)  # [B,D] -> [B,1,D]
        
        B, T, D = features.shape
        assert T >= 1, "时间步维度不能为0"
        # 动态调整k值
        k = min(topk, T)
        if k <= 0:
            k = 1
        
        # 计算每个token重要性 (无需平均)
        importance = torch.norm(features, dim=-1)  # [B, T]
        
        # 获取每个样本的前k个时间步索引
        _, indices = torch.topk(importance, k=k, dim=1)  # [B, k]
        assert indices.max() < T, "选择索引超出序列范围"
        return indices

    def _compute_hierarchical_loss(self):
        """层次化特征对齐损失"""
        total_loss = 0
        B = next(iter(self.image_features.values()))[0].size(0)
        
        # 图像特征处理
        img_shallow = torch.cat(self.image_features[self.image_layers[0]], dim=1).to(torch.float32)  # [B, 577, 1024] 
        img_deep = torch.cat(self.image_features[self.image_layers[-1]], dim=1).to(torch.float32)   # [B, 577, 1024]
        
        # 投影到文本空间
        proj_img_shallow = self.img_proj['shallow'](img_shallow).to(torch.float32)  # [B, 577, 4096] -> [B, 577, 4096]
        proj_img_deep = self.img_proj['deep'](img_deep).to(torch.float32)          # [B, 577, 4096] -> [B, 577, 4096]
        
        # 文本特征对齐
        for layer_idx in self.text_layers:
            if not self.text_features[layer_idx]:
                continue
                
            text_data = self.text_features[layer_idx][-1]
            text_states = text_data['states']  # [B, 1, D]
            if torch.isnan(text_states).any():
                print("警告:文本编码器输出含NaN!")
            selected_steps = text_data['timesteps'].squeeze(0).cpu().numpy().tolist()
            
            selected_tensors = []
            for step in selected_steps:
                feat = self.text_features[layer_idx][step]
                selected_tensors.append(feat['states'])
            selected_features = torch.cat(selected_tensors, dim=1)  # [B, T_total, D]
            # 分层投影
            if layer_idx < 20:
                assert not torch.isnan(selected_features).any(), "文本特征含NaN!"
                assert not torch.isnan(proj_img_shallow).any(), "图像投影特征含NaN!"

                scale = 1.0 / math.sqrt(selected_features.size(-1))
                selected_norm = F.normalize(selected_features + 1e-8, p=2, dim=-1)
                proj_shallow_norm = F.normalize(proj_img_shallow + 1e-8, p=2, dim=-1)

                attn_scores = torch.einsum('bkd,bqd->bkq', selected_norm, proj_shallow_norm) * scale
                attn_scores = attn_scores - attn_scores.max(dim=-1, keepdim=True).values  # 数值平移
                
                # 动态生成高斯目标分布（示例：聚焦中心区域）
                H, W = 24, 24  # CLIP-ViT-L/14的特征图尺寸为24x24
                center_x, center_y = W//2, H//2  # 中心坐标
                grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
                spatial_mask = torch.exp(-((grid_x-center_x)**2 + (grid_y-center_y)**2)/(2*(H/6)**2)) + 1e-8
                spatial_mask = spatial_mask / spatial_mask.sum(dim=-1, keepdim=True)  # 重新归一化
                spatial_mask = spatial_mask.view(1, 1, H*W).to(attn_scores.device)  # [1,1,576]
                
                # 合并CLS位置（第0位设为0）
                target = torch.cat([torch.zeros(1,1,1, device=attn_scores.device), spatial_mask], dim=-1)
                target = target[:, :, :attn_scores.size(-1)]  # 适配实际长度577
                
                # KL散度损失（float32计算）
                loss = F.kl_div(
                    F.log_softmax(attn_scores.float(), dim=-1), 
                    target.float().expand_as(attn_scores),
                    reduction='batchmean',
                    log_target=False
                ).to(torch.float32)
                
                total_loss += self.alpha[0] * loss
            # 深层对齐（语义相似性）
            else:
                # 强制特征归一化
                selected_norm = F.normalize(selected_features, p=2, dim=-1)
                img_deep_norm = F.normalize(proj_img_deep, p=2, dim=-1)    

                # 维度对齐检查
                assert selected_norm.dim() == 3 and img_deep_norm.dim() == 3
                selected_exp = selected_norm.unsqueeze(2)  # [B,k,1,D]
                img_exp = img_deep_norm.unsqueeze(1)  # [B,1,N,D]

                # 余弦相似度计算（显式点积）
                sim_matrix = torch.sum(selected_exp * img_exp, dim=-1)  # [B,k,N]
                
                # 排除CLS token（仅使用空间特征）
                sim_matrix = sim_matrix[:, :, 1:]  # [1, k, 576]

                if torch.all(sim_matrix == 0):
                    print("警告：相似度全零！特征范数：", 
                        torch.norm(selected_norm), torch.norm(img_deep_norm))
                # 取每个文本token的最大相似度
                max_sim = sim_matrix.max(dim=-1)[0]  # [1, k]
                
                # 损失 = 1 - 平均最大相似度
                loss = 1 - torch.clamp(max_sim.mean(), min=-1, max=1)
                total_loss += self.alpha[1] * loss
        
        return total_loss.to(torch.float16)
    def _reset_features(self):
        """彻底清除特征缓存和计算图"""
        # 清空特征缓存
        self.image_features.clear()
        self.text_features.clear()
        
        # 释放钩子中的特征
        for layer_idx in self.image_layers:
            if layer_idx in self.image_features:
                for feat in self.image_features[layer_idx]:
                    del feat
                self.image_features[layer_idx].clear()
        
        for layer_idx in self.text_layers:
            if layer_idx in self.text_features:
                for feat in self.text_features[layer_idx]:
                    del feat
                self.text_features[layer_idx].clear()
        
        # 清除所有参数的梯度
        self.model.zero_grad(set_to_none=True)
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()
        
        # 强制回收内存
        torch.cuda.empty_cache()
    def attack(self, image, label, iterations=50, lr=0.01, momentum=0.9, epsilon=8/255, **kwargs):
        """层次化对抗攻击"""
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        # 创建日志记录
        log_dir = os.path.dirname(kwargs.get('save_dir', './logs'))
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f'attack_log_{time.strftime("%Y%m%d_%H%M%S")}.txt')

        # 初始化图像和扰动(只做一次)
        image_tensor = self._preprocess_image(image)  # 原始图像

        image_denorm = denormalize(image_tensor, mean, std).to(self.device)
        delta = torch.zeros_like(image_denorm).to(self.device)  # 初始化扰动
        delta.requires_grad_(True)
        # 保存resize后的原始图像
        tmp=kwargs['save_dir'].split('/')
        save_path = os.path.join(log_dir, 'clean', tmp[-1])
        os.makedirs(os.path.join(log_dir, 'clean'), exist_ok=True)
        test_img = image_denorm.permute(0,2,3,1).detach().cpu().squeeze(0).numpy()
        test_img = (test_img * 255).astype(np.uint8)
        Image.fromarray(test_img).save(save_path)

        best_adv = image.copy()
        min_dist = float('inf')
        optimizer = torch.optim.SGD([delta], lr=lr, momentum=momentum)
        with open(log_path, 'w') as f:
            f.write(f"Attack started at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Parameters: lr={lr}, epsilon={epsilon}\n\n")
            for step in range(iterations):
                # 清除上一轮的梯度和特征
                optimizer.zero_grad(set_to_none=True)
                self._reset_features()
            
                adv_image_denorm = image_denorm + delta
                adv_image_denorm = torch.clamp(adv_image_denorm, 0.0, 1.0)
                adv_image_norm = transforms.Normalize(mean, std)(adv_image_denorm)

                self._update_curriculum(step, iterations)  # 课程学习更新

                # 前向传播获取特征
                self.is_generating = True
                outputs_ids = self._trigger_forward(adv_image_norm, **kwargs)
                self.is_generating = False
                self._post_process_features()
                outputs = self._postprocess_outputs(kwargs['input_ids'],outputs_ids)
                
                # 计算层次化损失
                loss = self._compute_hierarchical_loss()
                loss.backward(retain_graph=True)

                # Patch重要性mask
                with torch.no_grad():
                    delta_grad = delta.grad.data
                    patch_mask = self._compute_patch_importance(delta_grad.abs(), k=int(0.2*self.patch_size**2))
                    # PGD更新（只在重要区域添加扰动）
                    delta.data = delta_grad.sign()

                    optimizer.step()

                    delta.data = torch.clamp(delta.data, -epsilon, epsilon)
                    delta.data = torch.clamp(image_denorm + delta, 0, 1) - image_denorm

                delta = delta.detach()
                delta.requires_grad_(True)
                
                # 生成对抗样本
                current_adv_denorm = image_denorm + delta
                current_adv_norm = transforms.Normalize(mean, std)(current_adv_denorm)



                predict, is_adv = self.predictions(current_adv_norm, max_new_tokens=kwargs['max_new_tokens'], model_name=kwargs['model_name'], vis_proc=kwargs['vis_proc'],
                                                stopping_criteria=kwargs['stopping_criteria'], input_ids=kwargs['input_ids'])
                
                current_dist = l2_distance(current_adv_norm, image_tensor)

                # 记录日志
                log_msg = f"Step {step:03d} | Loss: {loss.item():.4f} | Distance: {current_dist:.4f} | Success: {is_adv} | Prediction: {predict}\n"
                f.write(log_msg)
                f.flush()  # 实时写入
                print(log_msg, end='')

                # 更新最佳样本
                if current_dist < min_dist:
                    best_adv = current_adv_norm.detach().cpu()
                    min_dist = current_dist

            # 保存最终结果
            if 'save_dir' in kwargs:
                save_path = os.path.join(log_dir, 'adv', tmp[-1])
                os.makedirs(os.path.join(log_dir, 'adv'), exist_ok=True)
                best_adv_denorm = denormalize(best_adv, mean, std)
                adv_img = best_adv_denorm.permute(0,2,3,1).squeeze(0).numpy()
                adv_img = (adv_img * 255).astype(np.uint8)
                Image.fromarray(adv_img).save(save_path)

        return best_adv, min_dist, predict

    def _update_curriculum(self, step, total_steps):
        """动态课程学习策略"""
        phase = step / total_steps
        if phase < 0.3:  # 粗粒度阶段
            self.alpha.data = torch.tensor([0.7, 0.2, 0.1], device=self.device)
        elif phase < 0.7:  # 中粒度阶段
            self.alpha.data = torch.tensor([0.3, 0.5, 0.2], device=self.device)
        else:  # 细粒度阶段
            self.alpha.data = torch.tensor([0.1, 0.3, 0.6], device=self.device)

    def _preprocess_image(self, image):
        """带梯度保留的预处理"""
        tensor = torch.from_numpy(np.array(image)).unsqueeze(0)
        tensor = tensor.to(self.device, dtype=next(self.model.parameters()).dtype)
        return tensor.requires_grad_(True)

    def _trigger_forward(self, image_tensor, **kwargs):
        """带梯度上下文的前向传播"""

        if kwargs['input_ids'].dim() == 1:
                input_ids = kwargs['input_ids'].unsqueeze(0)
        #image_tensor.retain_grad()
        # 确保输入张量可导
        image_tensor = image_tensor.clone().requires_grad_(True)
        with torch.enable_grad():
            # 执行完整生成流程
            outputs_ids = self.model.generate(
                    input_ids=input_ids,
                    images=image_tensor,
                    max_new_tokens=kwargs['max_new_tokens'],
                    use_cache=True,
                    stopping_criteria=kwargs['stopping_criteria']
                )
        return outputs_ids
    def _inference_forward(self, image_tensor, **kwargs):
        """仅推理的前向传播"""
        if kwargs['input_ids'].dim() == 1:
                input_ids = kwargs['input_ids'].unsqueeze(0)
        with torch.inference_mode():
            # 执行完整生成流程
            outputs_ids = self.model.generate(
                    input_ids=input_ids,
                    images=image_tensor,
                    max_new_tokens=kwargs['max_new_tokens'],
                    use_cache=True,
                    stopping_criteria=kwargs['stopping_criteria']
                )
            outputs = self._postprocess_outputs(kwargs['input_ids'],outputs_ids)
        return outputs
    def _compute_patch_importance(self, gradients, k=5):
        """计算基于梯度显著性的Patch重要性"""
        # 输入梯度形状: [B, C, H, W]
        b, c, h, w = gradients.shape
        
        # 将梯度转换为Patch粒度
        patch_size = self.patch_size
        num_h = h // patch_size
        num_w = w // patch_size
        
        # 计算每个Patch的L2显著性
        patch_importance = []
        for i in range(num_h):
            for j in range(num_w):
                # 提取当前Patch区域
                y_start = i * patch_size
                y_end = (i+1) * patch_size
                x_start = j * patch_size
                x_end = (j+1) * patch_size
                
                # 计算该区域梯度范数
                patch_grad = gradients[:, :, y_start:y_end, x_start:x_end]
                importance = torch.norm(patch_grad, p=2)  # L2范数
                patch_importance.append(importance)
        
        # 生成重要性mask
        patch_importance = torch.stack(patch_importance, dim=0)  # [num_patches]
        
        # 选择topk重要patch
        _, topk_indices = torch.topk(patch_importance, k=k)
        mask = torch.zeros_like(patch_importance)
        mask[topk_indices] = 1.0
        
        # 将mask转换为空间维度 [H, W]
        spatial_mask = torch.zeros(h, w, device=gradients.device)
        for idx in topk_indices:
            i = idx // num_w
            j = idx % num_w
            y_start = i * patch_size
            y_end = (i+1) * patch_size
            x_start = j * patch_size
            x_end = (j+1) * patch_size
            spatial_mask[y_start:y_end, x_start:x_end] = 1.0
        
        return spatial_mask.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    def _postprocess_outputs(self, input_ids, output_ids):
        if input_ids.dim() == 1:
                input_ids = input_ids.unsqueeze(0)
        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
        if self.stop_str is not None:
            for i in range(len(outputs)):
                tmp = outputs[i].strip()
                if tmp.endswith(self.stop_str):
                    tmp = tmp[:-len(self.stop_str)]
                outputs[i] = tmp.strip()
        
        return outputs
    def predictions(self, inputs, question_list=None, chat_list=None, max_new_tokens=None,model_name=None,vis_proc=None,**kwargs):
        if model_name=="TestMiniGPT4"  or model_name=="vpgtrans":
            chat_list_new=chat_list.copy()
            outputs = self.model.batch_answer([Image.fromarray(np.uint8(inputs))], [question_list], [chat_list_new],max_new_tokens=max_new_tokens)
            del chat_list_new
        elif model_name=="blip2":
            imgs = vis_proc["eval"](Image.fromarray(np.uint8(inputs))).unsqueeze(0).to("cuda", dtype=torch.float32)
            prompts = f"Question: {question_list} Answer:" 
            outputs = self.model.generate({"image": imgs, "prompt": prompts}, max_length=max_new_tokens)
        elif model_name=="instruct_blip":
            imgs = vis_proc["eval"](Image.fromarray(np.uint8(inputs))).unsqueeze(0).to("cuda", dtype=torch.float32)
            prompts = question_list 
            outputs = self.model.generate({"image": imgs, "prompt": prompts}, max_length=max_new_tokens)
        elif model_name=="adv2":
            imgs = vis_proc(Image.fromarray(np.uint8(inputs))).unsqueeze(0).to("cuda", dtype=torch.float32)
            prompts =[ llama.format_prompt(question_list) ]
            outputs =[ self.model.generate(imgs, prompts, temperature=0,max_gen_len=max_new_tokens)[0].strip()]
        elif model_name=="panda":  
            Image.fromarray(np.uint8(inputs)).save("./panda2_{}.png".format(vis_proc[0]) )      
            image_list= "./panda2_{}.png".format(vis_proc[0])        
            outputs = [self.model(image_list, question_list, max_new_tokens)]
        elif model_name=="otter": 
            imgs = vis_proc([Image.fromarray(np.uint8(inputs))],return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0).to("cuda", dtype=torch.float16)
            prompts = [f"<image> User: {question_list} GPT: <answer>"]
            lang_x = self.model.text_tokenizer(prompts, return_tensors="pt", padding=True)
            generated_text = self.model.generate(
            vision_x=imgs,
            lang_x=lang_x["input_ids"].to("cuda"),
            attention_mask=lang_x["attention_mask"].to("cuda", dtype=torch.float16),
            max_new_tokens=max_new_tokens,
            num_beams=3,
            no_repeat_ngram_size=3,
            )
            output = self.model.text_tokenizer.decode(generated_text[0])
            output = [x for x in output.split(' ') if not x.startswith('<')]
            out_label = output.index('GPT:')
            outputs = [' '.join(output[out_label + 1:])]
        elif model_name=="owl":
            prompt_template = "The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.\nHuman: <image>\nHuman: {}\nAI:"

            prompts = [prompt_template.format(question_list)]
            inputs = vis_proc[2](text=prompts, images=[Image.fromarray(np.uint8(inputs))], return_tensors='pt')
            inputs = {k: v.to("cuda", dtype=torch.float32) if v.dtype == torch.float else v for k, v in inputs.items()}
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            generate_kwargs = {
            'do_sample': False,
            'top_k': 5,
            'max_length': max_new_tokens
            }
            with torch.no_grad():
                res = self.model.generate(**inputs, **generate_kwargs)
            outputs = [vis_proc[1].decode(res.tolist()[0], skip_special_tokens=True)]  
        elif model_name=="ofv2":
            vision_x = vis_proc[0](Image.fromarray(np.uint8(inputs))).unsqueeze(0).unsqueeze(0).unsqueeze(0).to("cuda", dtype=torch.float16)
            prompts = [f"<image>Question: {question_list} Short answer:"]            
            lang_x = vis_proc[1](
            prompts,
            return_tensors="pt", padding=True,
            ).to("cuda")
            generated_text = self.model.generate(
            vision_x=vision_x,
            lang_x=lang_x["input_ids"].to("cuda"),
            attention_mask=lang_x["attention_mask"].to("cuda", dtype=torch.float16),
            max_new_tokens=max_new_tokens,
            num_beams=3,pad_token_id=vis_proc[1].eos_token_id
            )
            outputs = vis_proc[1].batch_decode(generated_text, skip_special_tokens=True)
            outputs = [y[len(x)-len('<image>'):].strip() for x, y in zip(prompts, outputs)]
        elif model_name == "internlm":
            Image.fromarray(np.uint8(inputs)).save("./internlm2{}.png".format(vis_proc[2]))   
            image_list= "./internlm2{}.png".format(vis_proc[2])  
            texts=f" <|User|>:<ImageHere> {question_list}" + vis_proc[1] + " <|Bot|>:"
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    outputs=[vis_proc[0](self.model,texts, image_list,max_new_tokens=max_new_tokens) ]  
            
        elif model_name=="llava" or model_name=="llava15" or model_name=="moellava" or model_name=="sharegpt4v":
            outputs = self._inference_forward(inputs, max_new_tokens=max_new_tokens, **kwargs)
       
        if self.task=="cls" or self.task=="ocr":
            predict = remove_special_chars(outputs[0]).lower()            
        else:
            predict = outputs[0]
        temp_result=predict
        if 1:
            is_adversarial=False
            if self.task=="cls" or self.task=="ocr":        
                adv=True
                if len(self.__original_class)!=0 and not isinstance(self.__original_class, str):
                    for gt in self.__original_class:
                        is_adversarial_tmp=not (bool(has_word(temp_result, gt)) or bool(has_word(temp_result, gt+'s')) )
                        adv=adv and is_adversarial_tmp
                    is_adversarial= is_adversarial or adv
                else:
                    is_adversarial=not (bool(has_word(temp_result, self.__original_class)) or bool(has_word(temp_result, self.__original_class+'s')) )                    
            if self.task=="caption":                
                cider_scorer = CiderScorer(n=4, sigma=6.0)
                cider_scorer += (temp_result, self.__original_class)
                (score, scores) = cider_scorer.compute_score()                
                if scores==0:
                    is_adversarial=True
            elif self.task=="kie":
                f1_scorer = F1Scorer()
                if isinstance(self.__original_class, list) :
                    gt_answers =" ".join(self.__original_class)
                else:
                    gt_answers=self.__original_class
                f1_scorer.add_string(gt_answers, temp_result)
                prec, recall, f1 = f1_scorer.score()
                if f1==0:
                    is_adversarial=True                    
            elif self.task=="mrr":
                eval = VQAEval()
                mrr = eval.evaluate_MRR(temp_result, self.__original_class)
                if mrr==0:
                    is_adversarial=True  
            elif self.task=="vqa" or self.task=="vqachoice" or self.task=="imagenetvc":
                eval = VQAEval()
                mrr = eval.evaluate(temp_result, self.__original_class)
                if mrr==0:
                    is_adversarial=True     
        
        return predict,is_adversarial
