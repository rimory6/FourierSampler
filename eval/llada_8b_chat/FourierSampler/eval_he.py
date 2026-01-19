from mmengine.config import read_base
from opencompass.runners import LocalRunner
from opencompass.partitioners import NaivePartitioner, SizePartitioner
from opencompass.tasks import OpenICLInferTask, OpenICLEvalTask
from opencompass.models import LLaDACausalLM_FourierSampler

with read_base():
    from opencompass.configs.datasets.humaneval.humaneval_gen import humaneval_datasets 


datasets = []
datasets += humaneval_datasets 

max_seq_len = 2048
max_out_len = 512

num_gpus = {
    'llada_8b_chat': 1
}

path_dict = {   
    'llada_8b_chat': 'GSAI-ML/LLaDA-8B-Instruct',  # path to your LLaDA-8B-Instruct
} 


param_grid = [
    (0.4, 0.4, 0.6),
]

DIFFUSION_CFG = {'steps': 512, 'block_length': 64}
LOCAL_WINDOW_SIZE = None  


models = []
for wr, bb_min, bb_max in param_grid:
    abbr = f"llada_8b_chat_fouriersampler"
    fft_cfg = dict(
        enable=True,
        window_ratio=wr,
        boost_beta_min=bb_min, 
        boost_beta_max=bb_max,
        variance_scale=1.0,
        enforce_lowfreq_gate=False,
        gate_keep_ratio=1.0,
    )

    models.append(
        dict(
            type=LLaDACausalLM_FourierSampler,
            abbr=abbr,
            path=path_dict['llada_8b_chat'],
            local_window_size=LOCAL_WINDOW_SIZE,
            scaling_config={},  
            diffusion_config=DIFFUSION_CFG,
            seed=2025,
            model_type='llada', 
            model_kwargs={'flash_attention': True},
            max_seq_len=max_seq_len,
            max_out_len=max_out_len,
            batch_size=1,
            run_cfg=dict(
                num_gpus=num_gpus['llada_8b_chat'],
                num_procs=num_gpus['llada_8b_chat'],
            ),
            remasking='low_confidence',
            fft_cfg=fft_cfg,
        )
    )


work_dir = './outputs/llada_8b_chat_fouriersampler-b64_s512/'

infer = dict(
    partitioner=dict(type=SizePartitioner, max_task_size=400, gen_task_coef=4),
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
