import os
from functools import partial
from time import time

import psutil
import torch
import torch.nn as nn
from commons.utils import get_data, get_profile_context, get_tflops, get_time_stamp
from packaging import version
from torch.nn.parallel import DistributedDataParallel as DDP
from tokenization_glm import GLMChineseTokenizer

from modeling_glm import GLMForConditionalGeneration
# from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader

import colossalai
from colossalai.logging import disable_existing_loggers, get_dist_logger
from colossalai.nn.optimizer import HybridAdam
# from colossalai.nn.optimizer import GeminiAdamOptimizer
from colossalai.nn.parallel import zero_model_wrapper, zero_optim_wrapper
from colossalai.tensor import ColoParameter, ComputePattern, ComputeSpec, ProcessGroup, ReplicaSpec, ShardSpec
from colossalai.utils import get_current_device
from colossalai.utils.model.colo_init_context import ColoInitContext
import pdb
CAI_VERSION = colossalai.__version__


def parse_args():
    parser = colossalai.get_default_parser()
    parser.add_argument(
        "--distplan",
        type=str,
        default='CAI_Gemini',
        help="The distributed plan [colossalai, zero1, zero2, torch_ddp, torch_zero].",
    )
    parser.add_argument(
        "--tp_degree",
        type=int,
        default=1,
        help="Tensor Parallelism Degree. Valid when using colossalai as dist plan.",
    )
    parser.add_argument(
        "--placement",
        type=str,
        default='cpu',
        help="Placement Policy for Gemini. Valid when using colossalai as dist plan.",
    )
    parser.add_argument(
        "--shardinit",
        action='store_true',
        help=
        "Shard the tensors when init the model to shrink peak memory size on the assigned device. Valid when using colossalai as dist plan.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="batch size per DP group of training.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="gpt2_medium",
        help="model model scale",
    )
    parser.add_argument(
        "--train_step",
        type=int,
        default=10,
        help="training iterations for test",
    )

    args = parser.parse_args()
    return args


# Parameter Sharding Strategies for Tensor Parallelism
def split_param_single_dim_tp1d(dim: int, param: ColoParameter, pg: ProcessGroup):
    spec = (ShardSpec([dim], [pg.tp_world_size()]), ComputeSpec(ComputePattern.TP1D))
    param.set_tensor_spec(*spec)


def split_param_row_tp1d(param: ColoParameter, pg: ProcessGroup):
    split_param_single_dim_tp1d(0, param, pg)


def split_param_col_tp1d(param: ColoParameter, pg: ProcessGroup):
    split_param_single_dim_tp1d(-1, param, pg)


class GPTLMLoss(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        return self.loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


def get_cpu_mem():
    return psutil.Process().memory_info().rss / 1024**2


def get_gpu_mem():
    return torch.cuda.memory_allocated() / 1024**2


def get_mem_info(prefix=''):
    return f'{prefix}GPU memory usage: {get_gpu_mem():.2f} MB, CPU memory usage: {get_cpu_mem():.2f} MB'


def get_model_size(model: nn.Module):
    total_numel = 0
    for module in model.modules():
        for p in module.parameters(recurse=False):
            total_numel += p.numel()
    return total_numel


def model_size_formatter(numel: int) -> str:
    GB_SIZE = 10**9
    MB_SIZE = 10**6
    KB_SIZE = 10**3
    if numel >= GB_SIZE:
        return f'{numel / GB_SIZE:.1f}B'
    elif numel >= MB_SIZE:
        return f'{numel / MB_SIZE:.1f}M'
    elif numel >= KB_SIZE:
        return f'{numel / KB_SIZE:.1f}K'
    else:
        return str(numel)


def set_cpu_maximum_parallelism():
    conf_str = torch.__config__.parallel_info()
    inter_str = conf_str.split("hardware_concurrency() : ")[1]
    max_concurrency = '20'# inter_str.split('\n')[0]
    os.environ["OMP_NUM_THREADS"] = '20'#max_concurrency
    print(f"environmental variable OMP_NUM_THREADS is set to {max_concurrency}.")


# Tensor Parallel
def tensor_parallelize(model: torch.nn.Module, pg: ProcessGroup):
    """tensor_parallelize
    Sharding the Model Parameters.
    Args:
        model (torch.nn.Module): a torch module to be sharded
    """
    for mn, module in model.named_modules():
        if mn=='':
            continue
        for pn, param in module.named_parameters(recurse=False):
            # NOTE() a param maybe shared by two modules
            if hasattr(param, 'visited'):
                continue
            # print('*'*50, mn+'--'+pn, '*'*50)
            # if shard init, then convert param to replica and use the dp-only ProcessGroup
            param: ColoParameter = param
            param.set_dist_spec(ReplicaSpec())
            param.set_process_group(pg)

            # shard it w.r.t tp pattern
            if 'mlp.dense_4h_to_h' in mn: # 在gpt2里面c_fc用的是Conv1D所以weight矩阵是右乘的，即dim=768*3072，但glm是linear，左乘的，即16384*4096
                                 # 所以gpt2的mlp.c_fc需要做列并行，而在glm中是mlp.dense_4h_to_h做列并行
                if 'weight' in pn or 'bias' in pn:
                    split_param_col_tp1d(param, pg)    # colmn slice
                    # keep the shape of the output from c_fc
                    param.compute_spec.set_output_replicate(False)
                else:
                    param.set_dist_spec(ReplicaSpec())
            elif 'mlp.dense_h_to_4h' in mn:
                if 'weight' in pn:
                    split_param_row_tp1d(param, pg)    # row slice
                else:
                    param.set_dist_spec(ReplicaSpec())
            elif 'word_embeddings' in mn: # 在glm中因为没有采用传统的word embedding，所以这里需要对embedding weight做row并行
                if 'weight' in pn:
                    split_param_col_tp1d(param, pg)    # colmn slice
            elif 'position_embeddings' in mn:
                if 'weight' in pn:
                    split_param_col_tp1d(param, pg)    # colmn slice
            elif 'query_key_value':
                split_param_row_tp1d(param, pg)    # colmn slice
            else:
                param.set_dist_spec(ReplicaSpec())
            param.visited = True


def main():
    # version check
    # this example is supposed to work for versions greater than 0.2.0
    assert version.parse(CAI_VERSION) >= version.parse("0.2.0")

    set_cpu_maximum_parallelism()
    args = parse_args()

    # if args.distplan not in ["colossalai", "torch_ddp", "torch_zero", "zero1", "zero2"]:
    if args.distplan not in ["CAI_ZeRO1", "CAI_ZeRO2", "CAI_Gemini", "Pytorch_DDP", "Pytorch_ZeRO"]:
        raise TypeError(f"{args.distplan} is error")

    # batch size per DP degree
    BATCH_SIZE = args.batch_size
    
    NUM_STEPS = args.train_step

    WARMUP_STEPS = 1
    assert WARMUP_STEPS < NUM_STEPS, "warmup steps should smaller than the total steps"
    assert (NUM_STEPS - WARMUP_STEPS) % 2 == 1, "the number of valid steps should be odd to take the median"
    PROF_FLAG = False    # The flag of profiling, False by default

    disable_existing_loggers()
    colossalai.launch_from_torch(config={})

    logger = get_dist_logger()
    logger.info(f"{args.model_type}, {args.distplan}, batch size {BATCH_SIZE}", ranks=[0])

    # build criterion
    # criterion = GPTLMLoss()

    torch.manual_seed(123)
    if args.distplan.startswith("CAI"):
        # all param must use the same process group.
        world_size = torch.distributed.get_world_size()
        shard_pg = ProcessGroup(tp_degree=world_size) if args.shardinit else None
        default_dist_spec = ShardSpec([-1], [world_size]) if args.shardinit else None

        if args.shardinit and args.distplan != "CAI_Gemini":
            raise RuntimeError("You can only use shardinit with CAI_Gemini")

        # build GPT model
        with ColoInitContext(device=get_current_device(),
                             dtype=torch.half,
                             default_dist_spec=default_dist_spec,
                             default_pg=shard_pg):
            # model = model_builder(args.model_type)(checkpoint=True)
            model = GLMForConditionalGeneration.from_pretrained('BAAI/glm-10b-chinese', trust_remote_code=True)
            print('*'*50, 'load model finish', '*'*50)
            
        print('*'*50, 'start ProcessGroup', '*'*50)
        tp_pg = ProcessGroup(tp_degree=args.tp_degree)
        # Tensor Parallelism (TP)
        # You should notice that v0.1.10 is not compatible with TP degree > 1
        if args.tp_degree > 1:
            print('*'*50, 'start tensor parallel', '*'*50)
            tensor_parallelize(model, tp_pg)

        # asign running configurations
        gemini_config = None
        if args.distplan.startswith("CAI_ZeRO"):
            optim_config = dict(reduce_bucket_size=12 * 1024 * 1024, overlap_communication=True, verbose=True)
        elif args.distplan == "CAI_Gemini":
            gemini_config = dict(strict_ddp_mode=args.tp_degree == 1,
                                 device=get_current_device(),
                                 placement_policy=args.placement,
                                 pin_memory=True,
                                #  hidden_dim=model.config.n_embd,
                                 search_range_mb=128)
            optim_config = dict(gpu_margin_mem_ratio=0.)
        else:
            raise RuntimeError

        # build a highly optimized gpu/cpu optimizer
        #pdb.set_trace()
        #print('*'*50, 'build zero adam', '*'*50)
        #optimizer = GeminiAdamOptimizer(model.glm, lr=1e-3, initial_scale=1)
        print('*'*50, 'build adam', '*'*50)
        
        optimizer = HybridAdam(model.parameters(), lr=1e-3)

        if args.distplan == "CAI_ZeRO1":
            zero_stage = 1
        elif args.distplan == "CAI_ZeRO2":
            zero_stage = 2
        elif args.distplan == "CAI_Gemini":
            zero_stage = 3
        else:
            raise RuntimeError

        # wrap your model and optimizer
        print('*'*50, 'zero model process', '*'*50)
        model = zero_model_wrapper(model, zero_stage, gemini_config)
        print('*'*50, 'zero optim process', '*'*50)
        optimizer = zero_optim_wrapper(model, optimizer, optim_config=optim_config)
        print('*'*50, 'load pack parallel', '*'*50)
        logger.info(get_mem_info(prefix='After init optim, '), ranks=[0])
    elif args.distplan.startswith("Pytorch"):
        assert args.tp_degree == 1, "The degree of TP should be 1 for DDP examples."
        model = model_builder(args.model_type)(checkpoint=True).cuda()
        model = DDP(model)
        if args.distplan.endswith("DDP"):
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        elif args.distplan.endswith("ZeRO"):
            from torch.distributed.optim import ZeroRedundancyOptimizer
            optimizer = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=torch.optim.Adam, lr=1e-3)
    else:
        raise RuntimeError

    # model is shared after TP
    numel = get_model_size(model)
    # print('numel', numel)
    logger.info(f"the size of testing model size is {model_size_formatter(numel)}.")
    logger.info(get_mem_info(prefix='After init model, '), ranks=[0])

    # Tflops_per_GPU = global_batch * global_numel * seq_len * 8 / #gpu
    # = (batch_per_DP_group * dp_degree) * (numel * tp_degree) * seq_len * 8 / (tp_degree * dp_degree)
    # = batch_per_DP_group * numel * seq_len * 8
    get_tflops_func = partial(get_tflops, numel, BATCH_SIZE, 200)

    torch.cuda.synchronize()
    model.train()
    tflops_list = []

    def train_step(inputs):
        # we just use randomly generated data here

        optimizer.zero_grad()

        start = time()
        # outputs = model(input_ids, attn_mask)
        # loss = criterion(outputs, input_ids)
        # pdb.set_trace()

        for key in inputs:
            inputs[key] = inputs[key].to(model.module.glm.word_embeddings.weight.device)
        outputs = model(**inputs)
        loss = outputs.loss
        torch.cuda.synchronize()
        fwd_end = time()
        fwd_time = fwd_end - start
        logger.info(get_mem_info(prefix=f'[{n + 1}/{NUM_STEPS}] Forward '), ranks=[0])

        if args.distplan.startswith("CAI"):
            optimizer.backward(loss)
        elif args.distplan.startswith("Pytorch"):
            loss.backward()
        else:
            raise RuntimeError

        torch.cuda.synchronize()
        bwd_end = time()
        bwd_time = bwd_end - fwd_end
        logger.info(get_mem_info(prefix=f'[{n + 1}/{NUM_STEPS}] Backward '), ranks=[0])

        optimizer.step()
        torch.cuda.synchronize()
        optim_time = time() - bwd_end
        step_time = time() - start
        logger.info(get_mem_info(prefix=f'[{n + 1}/{NUM_STEPS}] Optimizer step '), ranks=[0])

        step_tflops = get_tflops_func(step_time)
        logger.info(
            f"[{n + 1}/{NUM_STEPS}] Loss:{loss.item():.3f}, Step time: {step_time:.3f}s, TFLOPS: {get_tflops_func(step_time):.3f}, FWD time: {fwd_time:.3f}s, BWD time: {bwd_time:.3f}s, OPTIM time: {optim_time:.3f}s",
            ranks=[0],
        )
        if n >= WARMUP_STEPS:
            tflops_list.append(step_tflops)

    demo_profiler = get_profile_context(PROF_FLAG,
                                        WARMUP_STEPS,
                                        NUM_STEPS - WARMUP_STEPS,
                                        save_dir=f"profile/{get_time_stamp()}-demo")

    print('load tokenizer')
    
    # pdb.set_trace()
    with demo_profiler as prof:
        tokenizer = GLMChineseTokenizer.from_pretrained('BAAI/glm-10b-chinese', trust_remote_code=True)
        dataset = persona2sentence_dataset('')
        collate = collate_fn(tokenizer)
        dataloader = DataLoader(dataset, BATCH_SIZE, shuffle=False, collate_fn=collate)
        # pdb.set_trace()
        for n, batch in enumerate(dataloader):
            train_step(batch)
            prof.step()
            

    tflops_list.sort()
    median_index = ((NUM_STEPS - WARMUP_STEPS) >> 1) + WARMUP_STEPS
    logger.info(f"Median TFLOPS is {tflops_list[median_index]:.3f}")
    torch.cuda.synchronize()


class persona2sentence_dataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        
        # use dummy data here
        self.data = []
        self.seq_len = []
        for i in range(200):
            dialog = 'dddddddddddddddddd'
            tmp = '根据对话例子，续写对话，以下给出信息：' + '人物的姓名：' + 'xxxx' + \
                '角色简介：' + 'xxxxxxxxxxxx' + '以下给出对话并续写。'+'xxxxxx' + 'xxxxxxx' + '：[MASK]'
            self.data.append([tmp, dialog])
        
    def __getitem__(self, index):
        
        return self.data[index]

    def __len__(self):
        return len(self.data)
    

class collate_fn():
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer


    def __call__(self, batch):
        inputs = [i[0] for i in batch]
        encoded_inputs = self.tokenizer(inputs, return_tensors="pt", padding=True)
        targets = [i[1] for i in batch]
        encoded_batch = self.tokenizer.build_inputs_for_generation(encoded_inputs, targets=targets, max_gen_length=70, padding=False)
        return encoded_batch

if __name__ == '__main__':
    main()
    # tokenizer = GLMChineseTokenizer.from_pretrained('BAAI/glm-10b-chinese')
    # dataset = persona2sentence_dataset('persona2sentence.txt', )
    # collate = collate_fn(tokenizer)
    # dataloader = DataLoader(dataset, 3, shuffle=False, collate_fn=collate)
    # import pdb
    # model = GLMForConditionalGeneration.from_pretrained('BAAI/glm-10b-chinese')
    # model.eval()
    # with torch.no_grad():
    #     for batch in dataloader:
    #         pdb.set_trace()