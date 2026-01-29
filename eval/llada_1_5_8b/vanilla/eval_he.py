from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, NumWorkerPartitioner, SizePartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import LLaDACausalLM

with read_base():
    from opencompass.configs.datasets.humaneval.humaneval_gen_v04 import humaneval_datasets 


datasets = []
datasets += humaneval_datasets 

max_seq_len = 2048
max_out_len = 512

num_gpus = {
    'llada_1_5_8b': 1
}

path_dict = {   
    'llada_1_5_8b': 'GSAI-ML/LLaDA-1.5',  # path to your LLaDA-1.5
} 

models = [
    ('llada_1_5_8b-b64_s512-bf16', {}, {'steps': 512, 'block_length': 64, }, None),
]

models = [
    dict(
        type=LLaDACausalLM, abbr=abbr, path=path_dict[abbr.split('-')[0]], local_window_size = local_window_size,
        scaling_config=scaling_config, diffusion_config=diffusion_config, seed=2025, model_type=abbr.split('_')[0],
        model_kwargs={'flash_attention': True}, max_seq_len = max_seq_len, max_out_len=max_out_len, batch_size=1, 
        run_cfg=dict(num_gpus=num_gpus[abbr.split('-')[0]], num_procs=num_gpus[abbr.split('-')[0]]),
    ) for abbr, scaling_config, diffusion_config, local_window_size in models
]

work_dir = './outputs/llada_1_5-b64_s512/'

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


