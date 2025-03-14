#coding=utf-8
#基于patch的攻击方法，一个patch一个patch进行攻击

import numpy as np
import time
import copy
import torch

####
from .distances import Distance
from .distances import MSE
from PIL import Image
from .tools import has_word, remove_special_chars
from .cider import CiderScorer
import pdb
from .tools import VQAEval
from models import llama_adapter_v2 as llama
import imageio
from collections import defaultdict
from functools import reduce
import torch.nn.functional as F
import torch.nn as nn
import math



def has_nested_attr(obj, attr_path):
    """检查对象是否具有嵌套属性"""
    try:
        reduce(getattr, attr_path.split('.'), obj)
        return True
    except AttributeError:
        return False
####
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

    def forward(self, x):
        return torch.nn.functional.gelu(x)

class PatchGradAttack(Attacker):
    def __init__(self, model, task, label, tokenizer, 
                image_layers=[12, 18, 23], 
                text_layers=[16, 24, 31],
                patch_size=16,
                topk_ratio=0.3):
        super().__init__(model, task, label)
        
        # 模型组件验证
        self._validate_model_structure()
        
        # 特征配置
        self.tokenizer = tokenizer
        self.image_layers = sorted(set(image_layers))
        self.text_layers = sorted(set(text_layers))
        self.patch_size = patch_size
        self.topk_ratio = topk_ratio
        self.device = next(model.parameters()).device
        
        # 特征存储（使用内存映射防止OOM）
        self.image_features = defaultdict(lambda: [])
        self.text_features = defaultdict(lambda: [])
        
        # 注册钩子
        self._register_hooks()

        self._init_projection_layer()

    def _init_projection_layer(self):
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                # 第一层使用He初始化
                if m.in_features == 1024:
                    nn.init.xavier_normal_(m.weight, gain=1.1)
                    # 保持初始输出幅度稳定
                    with torch.no_grad():
                        m.weight.data *= math.sqrt(2.0 / (1 + math.sqrt(2/math.pi)))  # GELU校正因子
                # 第二层使用Xavier初始化
                elif m.out_features == 4096:
                    nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('linear'))
                
                # 偏置初始化
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
                # 添加权重归一化约束
                m.weight.data = torch.nn.functional.normalize(m.weight, dim=1) * 0.1
    
        # 网络结构定义
        self.projection = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.GELU(),
            nn.Linear(2048, 4096),
            nn.Dropout(p=0.1)  # 添加正则化
        ).to(self.device).train()
        
        # 应用初始化
        self.projection.apply(_init_weights)
        
        # 残差连接初始化
        self.residual = nn.Linear(1024, 4096).to(self.device).train()
        nn.init.eye_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)
        
        # 自适应缩放因子
        self.alpha = nn.Parameter(torch.tensor(0.1))

    def forward_projection(self, x):
        # 主路径
        main_path = self.projection(x)
        # 残差路径
        residual = self.residual(x)
        # 自适应融合
        return self.alpha * main_path + (1 - self.alpha) * residual

    def _reset_features(self):
        """重置特征缓存（关键补充）"""
        # 清空图像特征缓存
        for key in self.image_features:
            self.image_features[key].clear()
        # 清空文本特征缓存
        for key in self.text_features:
            self.text_features[key].clear()
        # 可选：释放GPU缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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

    def _register_hooks(self):
        """优化钩子注册逻辑"""
        # 视觉编码器钩子
        vision_encoder = self.model.model.vision_tower.vision_tower.vision_model.encoder
        for layer_idx in self.image_layers:
            layer = vision_encoder.layers[layer_idx]
            layer.register_forward_hook(
                self._create_vision_hook(layer_idx)
            )

        # 文本解码器钩子
        text_decoder = self.model.model.layers
        for layer_idx in self.text_layers:
            layer = text_decoder[layer_idx]
            layer.register_forward_hook(
                self._create_text_hook(layer_idx)
            )

    def _create_vision_hook(self, layer_idx):
        """动态生成视觉特征处理闭包"""
        def hook(module, inputs, outputs):
            feat = outputs[0] if isinstance(outputs, tuple) else outputs
            # 标准化特征尺寸 [B, Seq, D] -> [B*Seq, D]
            self.image_features[layer_idx].append(
                feat.detach().view(-1, feat.size(-1)).cpu()
            )
        return hook

    def _create_text_hook(self, layer_idx):
        """动态生成文本特征处理闭包""" 
        def hook(module, inputs, outputs):
            hidden_states = outputs[0]
            # 标准化特征尺寸 [B, Seq, D] -> [B*Seq, D]
            self.text_features[layer_idx].append(
                hidden_states.detach().view(-1, hidden_states.size(-1)).cpu()
            )
        return hook

    def _compute_feature_loss(self):
        """改进的特征对齐策略"""
        dtype = torch.float32
        # 图像特征池化
        img_feats = torch.stack([torch.cat(feats).mean(dim=0) 
                                for feats in self.image_features.values()]).to(self.device, dtype=dtype)  # [L_img, D_img]
        
        # 文本特征池化
        txt_feats = torch.stack([torch.cat(feats).mean(dim=0) 
                                for feats in self.text_features.values()]).to(self.device, dtype=dtype)  # [L_txt, D_txt]
        
        # 动态维度对齐
        projected_feats = self.forward_projection(img_feats).to(dtype)  # [L_img, D_txt]
        
        # 跨模态注意力
        attn_weights = F.softmax(projected_feats @ txt_feats.T, dim=-1)  # [L_img, L_txt]
        
        attended_feats = attn_weights.to(dtype) @ txt_feats.to(dtype)  # [L_img, D_txt]
        
        # 对比损失计算
        sim_matrix = F.cosine_similarity(
            projected_feats.unsqueeze(1).to(dtype),  # [L_img, 1, D_txt]
            attended_feats.unsqueeze(0).to(dtype),  # [1, L_img, D_txt]
            dim=-1
        )  # [L_img, L_img]
        
        loss = -sim_matrix.mean()  # 计算平均相似度损失
        
        return loss  # 直接返回损失

    def _get_patch_importance(self, grad_map):
        """优化梯度显著性计算"""
        # 通道平均梯度 [B, C, H, W] -> [H, W]
        grad_map = grad_map.abs().mean(dim=1).squeeze(0)
        
        # 动态计算补丁网格
        h, w = grad_map.shape
        ph = h // self.patch_size
        pw = w // self.patch_size
        
        # 池化计算显著性
        importance = F.avg_pool2d(
            grad_map.unsqueeze(0).unsqueeze(0),
            kernel_size=self.patch_size,
            stride=self.patch_size
        ).view(ph, pw)
        
        return importance

    def attack(self, image, label, iterations=50, lr=0.01, momentum=0.9, 
              question_list=None, max_new_tokens=256, model_name=None, vis_proc=None):
        """优化攻击流程"""
        # 输入标准化
        image_tensor = self._preprocess_image(image)
        optimizer = torch.optim.SGD([image_tensor], lr=lr, momentum=momentum)
        
        best_adv = image.copy()
        min_dist = float('inf')
        
        for step in range(iterations):
            self._reset_features()
            
            # 前向传播
            with torch.no_grad():
                self._trigger_forward(image_tensor)
            
            # 损失计算
            loss = self._compute_feature_loss()
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度处理
            grad_map = image_tensor.grad.data.clone()
            importance_map = self._get_patch_importance(grad_map)
            
            # 动态topk选择
            k = max(1, int(self.topk_ratio * importance_map.numel()))
            _, topk_indices = torch.topk(importance_map.view(-1), k)
            
            # 生成掩码
            mask = torch.zeros_like(importance_map)
            mask.view(-1)[topk_indices] = 1
            
            # 参数更新
            with torch.no_grad():
                delta = lr * grad_map * mask.unsqueeze(0).unsqueeze(0)
                image_tensor.data = torch.clamp(image_tensor + delta, 0, 255)
            
            # 评估对抗样本
            current_adv = self._postprocess_image(image_tensor)
            _, is_adv = self.predictions(current_adv, question_list, None, 
                                       max_new_tokens, model_name, vis_proc)
            
            # 更新最佳样本
            current_dist = l2_distance(current_adv, image)
            if is_adv and current_dist < min_dist:
                best_adv = current_adv
                min_dist = current_dist
                
        return best_adv, min_dist

    def _preprocess_image(self, image):
        """图像预处理标准化"""
        # 转换为模型输入尺寸
        processed = Image.fromarray(image.astype(np.uint8)).resize(
            (self.model.model.vision_tower.config.image_size, 
             self.model.model.vision_tower.config.image_size),
            Image.BICUBIC
        )
        
        # 转换为张量 [H, W, C] -> [C, H, W]
        tensor = torch.from_numpy(np.array(processed)).permute(2,0,1).float()
        return tensor.unsqueeze(0).to(self.device).requires_grad_(True)

    def _postprocess_image(self, tensor):
        """图像后处理标准化"""
        return tensor.squeeze().permute(1,2,0).detach().cpu().numpy().astype(np.uint8)

    def _trigger_forward(self, image_tensor):
        """重构前向传播（保持梯度流）"""
        # 禁用模型参数的梯度计算（只保留图像梯度）
        with torch.no_grad():
            # 视觉编码
            vision_outputs = self.model.model.vision_tower(image_tensor)
            last_hidden_state = vision_outputs
            
            # 文本解码（使用虚拟输入）
            dummy_input = torch.tensor([[self.tokenizer.bos_token_id]], 
                                     device=self.device)
            self.model(dummy_input, images=last_hidden_state)

class MMProjectorAttack(Attacker):
    def __init__(self, model, task, label, tokenizer, 
                projector_layers=['mm_projector.0', 'mm_projector.2'],  # 典型的两层结构
                device='cuda'):
        super().__init__(model, task, label)
        
        # 模型结构验证
        self._validate_projector_structure(projector_layers)
        
        # 配置参数
        self.tokenizer = tokenizer
        self.device = device
        self.projector = self._get_projector(projector_layers)
        
        # 特征缓存
        self.visual_feats = None
        self.proj_feats = None
        self.text_feats = None
        
        # 注册钩子
        self.hook_handles = []
        self._register_projector_hooks()

    def _validate_projector_structure(self, layers):
        """验证mm_projector结构完整性"""
        for layer in layers:
            if not has_nested_attr(self.model, layer):
                raise AttributeError(f"Missing projector layer: {layer}")
                
        # 验证典型的两层结构（Linear +激活函数+ Linear）
        try:
            layer0 = reduce(getattr, layers[0].split('.'), self.model)
            layer1 = reduce(getattr, layers[1].split('.'), self.model)
            if not (isinstance(layer0, torch.nn.Linear) and isinstance(layer1, torch.nn.Linear)):
                raise ValueError("Projector layers should be Linear layers")
        except Exception as e:
            raise RuntimeError(f"Projector structure validation failed: {str(e)}")

    def _get_projector(self, layers):
        """获取投影层引用"""
        return [reduce(getattr, layer.split('.'), self.model) for layer in layers]

    def _register_projector_hooks(self):
        """注册投影层特征捕获钩子"""
        # 第一层输入（原始视觉特征）
        handle = self.projector[0].register_forward_pre_hook(
            lambda module, input: self._capture_visual_feats(input[0])
        )
        self.hook_handles.append(handle)
        
        # 最后一层输出（投影后特征）
        handle = self.projector[-1].register_forward_hook(
            lambda module, input, output: self._capture_proj_feats(output)
        )
        self.hook_handles.append(handle)

    def _capture_visual_feats(self, features):
        """捕获原始视觉特征"""
        self.visual_feats = features.detach().clone()

    def _capture_proj_feats(self, features):
        """捕获投影后特征"""
        self.proj_feats = features.detach().clone()

    def _get_text_features(self, text):
        """获取目标文本特征"""
        with torch.no_grad():
            inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
            outputs = self.model.model.language_model(**inputs, output_hidden_states=True)
            return outputs.hidden_states[-1][:, -1]  # 取最后层[CLS]特征

    def _similarity_loss(self, proj_feats, text_feats):
        """计算跨模态相似度损失"""
        # 特征归一化
        proj_feats = F.normalize(proj_feats, dim=-1)
        text_feats = F.normalize(text_feats, dim=-1)
        
        # 对比损失计算
        logits = proj_feats @ text_feats.T
        return -logits.mean()  # 最大化相似度

    def attack(self, image, target_text, iterations=100, lr=0.1, momentum=0.9):
        """核心攻击方法"""
        # 初始化对抗样本
        adv_image = torch.tensor(image).permute(2,0,1).unsqueeze(0).to(self.device).float().requires_grad_(True)
        optimizer = torch.optim.SGD([adv_image], lr=lr, momentum=momentum)
        
        # 预计算文本特征
        text_feats = self._get_text_features(target_text)
        
        best_adv = None
        min_loss = float('inf')
        
        for step in range(iterations):
            # 清空特征缓存
            self.visual_feats = None
            self.proj_feats = None
            
            # 前向传播
            self.model(adv_image)
            
            # 计算损失
            loss = self._similarity_loss(self.proj_feats, text_feats)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度更新
            optimizer.step()
            
            # 数值截断
            adv_image.data = torch.clamp(adv_image, 0, 255)
            
            # 记录最佳样本
            if loss < min_loss:
                best_adv = adv_image.detach().clone()
                min_loss = loss.item()

        return best_adv.squeeze().permute(1,2,0).cpu().numpy().astype(np.uint8) 