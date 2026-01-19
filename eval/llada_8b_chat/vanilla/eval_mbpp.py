from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner, SizePartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import LLaDACausalLM

with read_base():
    from opencompass.configs.datasets.mbpp.sanitized_mbpp_mdblock_gen_a447ff import sanitized_mbpp_datasets as mbpp_datasets


datasets = []
datasets += mbpp_datasets

max_seq_len = 2048
max_out_len = 512

num_gpus = {
    'llada_8b_chat': 1
}

path_dict = {   
    'llada_8b_chat': 'GSAI-ML/LLaDA-8B-Instruct',  # path to your LLaDA-8B-Instruct
} 

models = [
    ('llada_8b_chat-b64_s512-bf16', {}, {'steps': 512, 'block_length': 64, }, None),
]

models = [
    dict(
        type=LLaDACausalLM, abbr=abbr, path=path_dict[abbr.split('-')[0]], local_window_size = local_window_size,
        scaling_config=scaling_config, diffusion_config=diffusion_config, seed=2025, model_type=abbr.split('_')[0],
        model_kwargs={'flash_attention': True}, max_seq_len = max_seq_len, max_out_len=max_out_len, batch_size=1, 
        run_cfg=dict(num_gpus=num_gpus[abbr.split('-')[0]], num_procs=num_gpus[abbr.split('-')[0]]),
    ) for abbr, scaling_config, diffusion_config, local_window_size in models
]

work_dir = './outputs/llada_8b_chat-b64_s512/'

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=40, gen_task_coef=4),
    runner=dict(
        type=LocalRunner,
        # max_num_workers=2, retry=2,  
        task=dict(type=OpenICLInferTask),
    ),
)

eval = dict(
    partitioner=dict(type=NaivePartitioner),
    runner=dict(
        type=LocalRunner,
        max_num_workers=16, retry=2, 
        task=dict(type=OpenICLEvalTask, dump_details=True),
    ),
)


