import torch
import numpy as np
from PIL import Image
import os
DATA_DIR = ''

def skip(*args, **kwargs):
    pass
torch.nn.init.kaiming_uniform_ = skip
torch.nn.init.uniform_ = skip
torch.nn.init.normal_ = skip


def get_image(image):
    tmp=image.split('/')
    if 'corruption' in tmp:
        image=os.path.join('/seu_nvme/home/230239304/projects_jw/B-AVIBench_jw/eval-data/',tmp[-4],tmp[-3],tmp[-2],tmp[-1])#,tmp[-1]
    else:
        image=os.path.join('/seu_nvme/home/230239304/projects_jw/B-AVIBench_jw/eval-data/',tmp[-3],tmp[-2],tmp[-1])#,tmp[-1]
                    
    # print("******************",image)
    if type(image) is str:
        try:
            return Image.open(image).convert("RGB")
        except Exception as e:
            print(f"Fail to read image: {image}")
            exit(-1)
    elif type(image) is Image.Image:
        return image
    else:
        raise NotImplementedError(f"Invalid type of Image: {type(image)}")


def get_BGR_image(image):
    image = get_image(image)
    image = np.array(image)[:, :, ::-1]
    image = Image.fromarray(np.uint8(image))
    return image


def get_model(model_name, device=None):
    if model_name == 'BLIP2':#01
        from .test_blip2 import TestBlip2
        return TestBlip2(device)
    elif model_name == 'MiniGPT-4':##02
        from .test_minigpt4 import TestMiniGPT4
        return TestMiniGPT4(device)
    elif model_name == 'mPLUG-Owl':#03
        from .test_mplug_owl import TestMplugOwl
        return TestMplugOwl(device)
    elif model_name == 'Otter':#04
        from .test_otter import TestOtter
        return TestOtter(device)
    elif model_name == 'Otter-Image':
        from .test_otter_image import TestOtterImage
        return TestOtterImage(device)
    elif model_name == 'InstructBLIP':#05
        from .test_instructblip import TestInstructBLIP
        return TestInstructBLIP(device)
    elif model_name == 'VPGTrans':#06
        from .test_vpgtrans import TestVPGTrans
        return TestVPGTrans(device)
    elif model_name == 'LLaVA':#07
        from .test_llava import TestLLaVA
        return TestLLaVA(device)
    elif model_name == 'sharegpt4v':
        from .test_sharegpt4v import Testsharegpt4v
        return Testsharegpt4v(device)
    elif model_name == 'moellava':
        from .test_moellava import Testmoellava
        return Testmoellava(device)
    elif model_name == 'LLaVA15':
        from .test_llava15 import TestLLaVA15
        return TestLLaVA15(device)
    elif model_name == 'LLaMA-Adapter-v2':
        from .test_llama_adapter_v2 import TestLLamaAdapterV2, TestLLamaAdapterV2_web
        return TestLLamaAdapterV2(device)
    elif model_name == 'internlm-xcomposer':
        from .test_InternLM import TestInternLM
        return TestInternLM(device) 
    elif 'PandaGPT' in model_name:
        from .test_pandagpt import TestPandaGPT
        return TestPandaGPT(device)
    elif 'OFv2' in model_name:
        version = '4BI'
        from .test_OFv2 import OFv2
        return OFv2(version, device)   
    elif model_name == 'Qwen25':
        from .test_Qwen25 import TestQwen25
        return TestQwen25(device)
    elif model_name == 'glm4v':
        from .test_glm4v import Testglm4v
        return Testglm4v(device)
    else:
        raise ValueError(f"Invalid model_name: {model_name}")
