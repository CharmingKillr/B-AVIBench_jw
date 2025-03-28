# gradient_attack.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Union
from PIL import Image
import numpy as np
import os
import json
from .distances import MSE
from .cider import CiderScorer
from .tools import has_word, remove_special_chars, VQAEval
class F1Scorer:
        """用于KIE任务的F1评估"""
        def __init__(self):
            self.n_detected = 0
            self.n_gt = 0
            self.n_match = 0
            
        def add(self, ref: str, pred: str):
            pred_words = pred.split()
            ref_words = ref.split()
            self.n_gt += len(ref_words)
            self.n_detected += len(pred_words)
            for w in pred_words:
                if w in ref_words:
                    self.n_match += 1
                    ref_words.remove(w)
        
        def score(self) -> float:
            prec = self.n_match / self.n_detected if self.n_detected else 0
            recall = self.n_match / self.n_gt if self.n_gt else 0
            return 2 * (prec * recall) / (prec + recall) if (prec + recall) else 0

# gradient_attack.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Union
from PIL import Image
import numpy as np
import os
import json

class MultiModalAdversarialAttack(nn.Module):
    def __init__(self, 
                 model: nn.Module,
                 task_type: str,
                 vision_layers: List[int] = [20, 23],  # CLIP ViT最后两层
                 text_layers: List[int] = [24, 31],    # LLaMA深层
                 cross_attn_layers: List[int] = [3,7],  # 跨模态交互层
                 eps: float = 0.05,
                 alpha: float = 0.01,
                 iters: int = 50,
                 temp: float = 0.07,
                 topk: float = 0.3):
        super().__init__()
        self.model = model
        self.task_type = task_type
        self.eps = eps
        self.alpha = alpha
        self.iters = iters
        self.temp = temp
        self.topk = topk
        
        # 特征注册配置
        self.feature_hooks = {}
        self._register_hooks(vision_layers, text_layers, cross_attn_layers)
        
        # 损失组件
        self.contrast_loss = nn.CrossEntropyLoss()
        
        # 任务特定参数
        self._init_task_params()
        
    def _init_task_params(self):
        """初始化任务相关参数"""
        self.success_threshold = {
            "cls": lambda p,t: not any(has_word(p.lower(), w) for w in (t if isinstance(t,list) else [t])),
            "vqa": lambda p,t: VQAEval().evaluate(p, t) == 0,
            "caption": self._cider_score_check,
            "kie": lambda p,t: self._f1_score_check(p,t) == 0
        }
        
    def _register_hooks(self, v_layers, t_layers, c_layers):
        """注册多模态特征钩子"""
        # 视觉编码器（CLIP ViT）
        vision_tower = self.model.get_vision_tower()
        for layer in v_layers:
            vision_tower.vision_model.encoder.layers[layer].register_forward_hook(
                self._create_hook(f"vision_{layer}")
            )
        
        # 文本解码器（LLaMA）
        text_model = self.model.get_text_model()
        for layer in t_layers:
            text_model.model.layers[layer].register_forward_hook(
                self._create_hook(f"text_{layer}")
            )
        
        # 跨模态投影层
        mm_projector = self.model.get_mm_projector()
        for layer in c_layers:
            mm_projector.mlp.layers[layer].register_forward_hook(
                self._create_hook(f"cross_{layer}")
            )
    
    def _create_hook(self, name):
        """创建特征保存钩子"""
        def hook(module, input, output):
            self.feature_hooks[name] = output[0] if isinstance(output, tuple) else output
        return hook
    
    def _compute_contrastive_loss(self):
        """多模态对比损失计算"""
        # 提取多级特征
        vision_feats = [v for k,v in self.feature_hooks.items() if "vision" in k]
        text_feats = [v for k,v in self.feature_hooks.items() if "text" in k]
        cross_feats = [v for k,v in self.feature_hooks.items() if "cross" in k]
        
        # 特征对齐损失
        v_loss = sum(F.cosine_similarity(v.mean(1), t.mean(1)) for v in vision_feats for t in text_feats)
        
        # 跨模态一致性损失
        c_loss = sum(F.mse_loss(c, torch.zeros_like(c)) for c in cross_feats)
        
        return 0.7 * v_loss + 0.3 * c_loss
    
    def _dynamic_patch_mask(self, grad):
        """生成动态显著性mask"""
        # 通道平均显著性
        saliency = F.avg_pool2d(grad.abs().mean(dim=1, keepdim=True), kernel_size=14)
        
        # Top-k区域选择
        k = int(saliency.numel() * self.topk)
        _, topk_idx = torch.topk(saliency.view(-1), k)
        mask = torch.zeros_like(saliency)
        mask.view(-1)[topk_idx] = 1.0
        
        return mask
    
    def attack(self, clean_images, questions, targets):
        """
        执行对抗攻击
        :param clean_images: 原始图像张量 [B,C,H,W]
        :param questions: 问题列表 [B]
        :param targets: 目标答案（根据任务类型不同）
        :return: 对抗样本，是否成功，迭代次数
        """
        delta = torch.zeros_like(clean_images, requires_grad=True)
        images = clean_images.clone().detach().to(self.model.device)
        batch_size = images.size(0)
        
        # 文本预处理
        text_inputs = self.model.tokenizer(
            questions, 
            padding=True, 
            return_tensors="pt"
        ).to(self.model.device)
        
        for step in range(self.iters):
            self.feature_hooks.clear()
            
            # 生成对抗样本
            adv_images = torch.clamp(images + delta, 0, 1)
            
            # 前向传播
            outputs = self.model(
                input_ids=text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask,
                images=adv_images
            )
            
            # 计算对比损失
            loss = self._compute_contrastive_loss()
            
            # 反向传播
            loss.backward()
            grad = delta.grad.data
            
            # 动态更新
            with torch.no_grad():
                mask = self._dynamic_patch_mask(grad)
                delta.data = delta - self.alpha * grad.sign() * mask
                delta.data = torch.clamp(delta, -self.eps, self.eps)
                delta.data = torch.clamp(images + delta, 0, 1) - images
                delta.grad.zero_()
            
            # 检查攻击成功率
            if self._check_success(adv_images, questions, targets):
                print(f"Batch attack succeeded at step {step+1}")
                break
        
        return torch.clamp(images + delta, 0, 1).detach()
    
    def _check_success(self, adv_images, questions, targets):
        """批量攻击成功判断"""
        outputs = self.model.generate(
            images=adv_images,
            texts=questions,
            max_new_tokens=50
        )
        
        batch_success = []
        for pred, target in zip(outputs, targets):
            checker = self.success_threshold[self.task_type]
            batch_success.append(checker(pred, target))
            
        return all(batch_success)
    
    # region 任务特定评估方法
    def _cider_score_check(self, pred, target):
        scorer = CiderScorer(n=4, sigma=6.0)
        scorer += (pred, [target])
        _, score = scorer.compute_score()
        return score == 0
    
    def _f1_score_check(self, pred, target):
        scorer = F1Scorer()
        gt = " ".join(target) if isinstance(target, list) else target
        scorer.add_string(gt, pred)
        _, _, f1 = scorer.score()
        return f1 == 0
    # endregion