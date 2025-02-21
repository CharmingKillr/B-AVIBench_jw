from models.test_llava15 import TestLLaVA15
import torch_npu
from torch_npu.contrib import transfer_to_npu

if __name__ == '__main__':
    model = TestLLaVA15(device='npu:1')  # 如果有GPU，可以指定使用GPU
    image_path = '/data/jw/projects/B-AVIBench_jw/eval-data/tiny_lvlm_datasets/NoCaps/0f17235949709331.jpg'
    question = '这张图是什么？'
    output = model.generate(image_path, question)
    print(output)
    ouput_attack = model.batch_generate(image_path, question)
    print(output_attack)