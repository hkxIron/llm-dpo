import torch
torch.backends.cuda.matmul.allow_tf32 = True
import torch.nn.functional as F
import torch.nn as nn
import transformers
from omegaconf import DictConfig

import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    BackwardPrefetch,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.api import FullStateDictConfig, FullOptimStateDictConfig
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoTokenizer, PreTrainedModel
import tensor_parallel as tp

from preference_datasets import get_batch_iterator
from utils import (
    slice_and_move_batch_for_device,
    formatted_dict,
    all_gather_if_needed,
    pad_to_length,
    get_block_class_from_model,
    rank0_print,
    get_local_dir, set_random_seed,
)
import numpy as np
import wandb
import tqdm
import random
import os
from collections import defaultdict
import time
import json
import functools
from typing import Optional, Dict, List, Union, Tuple

"""
该代码质量非常高，推荐学习
"""
def get_dpo_loss(policy_chosen_log_probs: torch.FloatTensor,
                 policy_rejected_log_probs: torch.FloatTensor,
                 reference_chosen_log_probs: torch.FloatTensor,
                 reference_rejected_log_probs: torch.FloatTensor,
                 beta: float,
                 reference_free: bool = False) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
    """Compute the DPO loss for a batch of policy and reference model log probabilities.
    
    Args:
        policy_chosen_log_probs: Log probabilities of the policy model for the chosen responses. Shape: (batch_size,)
        policy_rejected_log_probs: Log probabilities of the policy model for the rejected responses. Shape: (batch_size,)
        reference_chosen_log_probs: Log probabilities of the reference model for the chosen responses. Shape: (batch_size,)
        reference_rejected_log_probs: Log probabilities of the reference model for the rejected responses. Shape: (batch_size,)
        beta: Temperature parameter for the DPO loss, typically something in the range of 0.1 to 0.5. We ignore the reference model as beta -> 0.
        reference_free: If True, we ignore the _provided_ reference model and implicitly use a reference model that assigns equal probability to all responses.

    Returns:
        A tuple of three tensors: (losses, chosen_rewards, rejected_rewards).
        The losses tensor contains the DPO loss for each example in the batch.
        The chosen_rewards and rejected_rewards tensors contain the rewards for the chosen and rejected responses, respectively.
    """
    # policy_chosen_logps:[batch,]
    # policy_rejected_logps:[batch,]
    # pi_logratios:[batch,], 注意是针对整个response维度而非某个token维度
    pi_logratios = policy_chosen_log_probs - policy_rejected_log_probs # 均在log空间内
    ref_logratios = reference_chosen_log_probs - reference_rejected_log_probs

    if reference_free:
        ref_logratios = 0

    logits = pi_logratios - ref_logratios # 与paper中的公式(7)中的(policy_chosen_logps - reference_chosen_logps) - (policy_rejected_logps-reference_rejected_logps)等价
    # losses:[batch,]
    losses = -F.logsigmoid(beta * logits) # paper中的公式(7)

    # chosen_rewards:[batch,]
    chosen_rewards = beta * (policy_chosen_log_probs - reference_chosen_log_probs).detach()
    rejected_rewards = beta * (policy_rejected_log_probs - reference_rejected_log_probs).detach()
    return losses, chosen_rewards, rejected_rewards


def _get_batch_log_probs(logits: torch.FloatTensor, labels: torch.LongTensor, average_log_prob: bool = False) -> torch.FloatTensor:
    """Compute the log probabilities of the given labels under the given logits.

    Args:
        logits: Logits of the model (unnormalized). Shape: (batch_size, sequence_length, vocab_size)
        labels: Labels for which to compute the log probabilities. Label tokens with a value of -100 are ignored. Shape: (batch_size, sequence_length)
        average_log_prob: If True, return the average log probability per (non-masked) token. Otherwise, return the sum of the log probabilities of the (non-masked) tokens.

    Returns:
        A tensor of shape (batch_size,) containing the average/sum log probabilities of the given labels under the given logits.
    """
    assert logits.shape[:-1] == labels.shape

    # logits: [batch_size, sequence_length, vocab_size]
    # labels: (batch_size, sequence_length)
    logits = logits[:, :-1, :] # 取时间0~T-1
    labels = labels[:, 1:].clone() # 取时间1~T, labels取logits向后移一位的结果,注意是clone,国灰labels本身不需要计算梯度
    loss_mask = (labels != -100) # 对于labels为-100的token不计算loss

    # dummy token; we'll ignore the losses on these tokens later
    labels[labels == -100] = 0 # labels=-100在作为index时调用下面torch.gather会越界，所以统一先设为0,后面有loss_mask会将相应的loss去掉
    # log_softmax数学上等价于log(softmax(x)), 但做这两个单独操作速度较慢，数值上也不稳定。这个函数使用另一种公式来正确计算输出和梯度。
    log_probs = logits.log_softmax(dim=-1)
    per_token_log_probs = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(2)

    log_prob_sum = (per_token_log_probs * loss_mask).sum(-1)
    if average_log_prob:
        return log_prob_sum / loss_mask.sum(-1)
    else:
        return log_prob_sum


def concat_chosen_reject_inputs(batch: Dict[str, Union[List[str], torch.LongTensor]]) -> Dict[str, torch.LongTensor]:
    """Concatenate the chosen and rejected inputs into a single tensor.
    将正样本与负样本放在一个batch的tensor中,在batch维度，前面是正样本，后面是负样本

    Args:
        batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids',
        which are tensors of shape (batch_size, sequence_length).
        
    Returns:
        A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
    """
    max_length = max(batch['chosen_input_ids'].shape[1], batch['rejected_input_ids'].shape[1]) # max_length_in_batch, 注意：此处是因为计算整个batch内的最大长度
    concatenated_batch = {}
    for k in batch:
        if k.startswith('chosen') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0 # 对于input_id,就是pad=0, 对于labels就是-100
            concatenated_key = k.replace('chosen', 'concatenated')
            # chosen值只有一个，不需要concat
            concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value) # 对同一个batch内的token id进行padding,在timestep的尾部进行padding
    for k in batch:
        if k.startswith('rejected') and isinstance(batch[k], torch.Tensor):
            pad_value = -100 if 'labels' in k else 0 # attention_mask也是0
            concatenated_key = k.replace('rejected', 'concatenated')
            pad_rejected = pad_to_length(batch[k], max_length, pad_value=pad_value)
            # 因为rejected的值有多个，因此需要concat
            concatenated_batch[concatenated_key] = torch.cat((concatenated_batch[concatenated_key], pad_rejected), dim=0) # 只有第一个是正样本，其它的均为负枰本，在batch维度上concat
    return concatenated_batch


class BasicTrainer(object):
    def __init__(self, policy_model: PreTrainedModel, config: DictConfig, seed: int, run_dir: str, reference_model: Optional[PreTrainedModel] = None, rank: int = 0, world_size: int = 1):
        """A trainer for a language model, supporting either SFT or DPO training.
           
           If multiple GPUs are present, naively splits the model across them, effectively
           offering N times available memory, but without any parallel computation.
        """
        self.seed:int = seed
        self.rank:int = rank
        self.world_size:int = world_size
        self.config:DictConfig = config
        self.run_dir:str = run_dir

        tokenizer_name_or_path = config.model.tokenizer_name_or_path or config.model.name_or_path
        rank0_print(f'Loading tokenizer {tokenizer_name_or_path}')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, cache_dir=get_local_dir(config.local_dirs))
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        data_iterator_kwargs = dict(
            names=config.datasets,
            tokenizer=self.tokenizer,
            shuffle=True,
            max_length=config.max_length,
            max_prompt_length=config.max_prompt_length,
            sft_mode=config.loss.name == 'sft',
        )

        self.policy_model:PreTrainedModel = policy_model
        self.reference_model:PreTrainedModel = reference_model

        self.train_iterator = get_batch_iterator(**data_iterator_kwargs, split='train', n_epochs=config.n_epochs, n_examples=config.n_examples, batch_size=config.batch_size, silent=rank != 0, cache_dir=get_local_dir(config.local_dirs))
        rank0_print(f'Loaded train data iterator')
        self.eval_iterator = get_batch_iterator(**data_iterator_kwargs, split='test', n_examples=config.n_eval_examples, batch_size=config.eval_batch_size, silent=rank != 0, cache_dir=get_local_dir(config.local_dirs))
        self.eval_batches = list(self.eval_iterator) # list让迭代器取出第一批数据
        rank0_print(f'Loaded {len(self.eval_batches)} eval batches of size {config.eval_batch_size}')

    def get_batch_samples_text(self, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        """Generate samples from the policy (and reference model, if doing DPO training) for the given batch of inputs.
            将batch内的数据使用模型进行推理，然后用tokenizer进行解码
        """

        policy_output = self.policy_model.generate(
            batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'],
            max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)
        if self.config.loss.name == 'dpo':
            reference_output = self.reference_model.generate(
                batch['prompt_input_ids'], attention_mask=batch['prompt_attention_mask'],
                max_length=self.config.max_length, do_sample=True, pad_token_id=self.tokenizer.pad_token_id)

        policy_output = pad_to_length(policy_output, self.config.max_length, self.tokenizer.pad_token_id)
        policy_output = all_gather_if_needed(policy_output, self.rank, self.world_size)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        if self.config.loss.name == 'dpo':
            reference_output = pad_to_length(reference_output, self.config.max_length, self.tokenizer.pad_token_id)
            reference_output = all_gather_if_needed(reference_output, self.rank, self.world_size)
            reference_output_decoded = self.tokenizer.batch_decode(reference_output, skip_special_tokens=True)
        else:
            reference_output_decoded = []

        return policy_output_decoded, reference_output_decoded
    
    def concat_chosen_rejected_forward(self, model: PreTrainedModel, batch: Dict[str, Union[List[str], torch.LongTensor]]) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Run the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
        
           We do this to avoid doing two forward passes, because it's faster for FSDP.
        """
        chosen_reject_batch:Dict[str, torch.LongTensor] = concat_chosen_reject_inputs(batch)
        all_logits = model.forward(chosen_reject_batch['concatenated_input_ids'],
                                   attention_mask=chosen_reject_batch['concatenated_attention_mask']).logits.to(torch.float32)
        all_log_probs = _get_batch_log_probs(logits=all_logits, labels=chosen_reject_batch['concatenated_labels'], average_log_prob=False)
        chosen_num = batch['chosen_input_ids'].shape[0]
        chosen_log_probs = all_log_probs[:chosen_num] # batch前面的都是正样本
        rejected_log_probs = all_log_probs[chosen_num:] # batch后面的都是负样本
        return chosen_log_probs, rejected_log_probs

    def get_batch_forward_loss_and_metrics(self, batch: Dict[str, Union[List, torch.LongTensor]], loss_config: DictConfig, train=True):
        """
        Compute the SFT or DPO loss and other metrics for the given batch of inputs.
        """

        metrics = {}
        train_test = 'train' if train else 'eval'

        if loss_config.name == 'dpo':
            # 作者的训练过程loss见wandb: https://wandb.ai/eric_anthony_mitchell/dpo-demos?nw=nwusereric_anthony_mitchell
            policy_chosen_log_probs, policy_rejected_log_probs = self.concat_chosen_rejected_forward(self.policy_model, batch)

            with torch.no_grad(): # reference_model是freeze的，且不计算梯度
                reference_chosen_log_probs, reference_rejected_log_probs = self.concat_chosen_rejected_forward(self.reference_model, batch)

            # chosen_rewards, rejected_rewards只用来记录,只有loss有用
            losses, chosen_rewards, rejected_rewards = get_dpo_loss(policy_chosen_log_probs, policy_rejected_log_probs,
                                                                    reference_chosen_log_probs, reference_rejected_log_probs,
                                                                    beta=loss_config.beta, reference_free=loss_config.reference_free)
            # 如果chosen_rewards> rejected_rewards,则认为reward是正确的
            reward_accuracies = (chosen_rewards > rejected_rewards).float()
            chosen_rewards = all_gather_if_needed(chosen_rewards, self.rank, self.world_size)
            rejected_rewards = all_gather_if_needed(rejected_rewards, self.rank, self.world_size)
            reward_accuracies = all_gather_if_needed(reward_accuracies, self.rank, self.world_size)

            metrics[f'rewards_{train_test}/chosen'] = chosen_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/rejected'] = rejected_rewards.cpu().numpy().tolist()
            metrics[f'rewards_{train_test}/accuracies'] = reward_accuracies.cpu().numpy().tolist()
            # 正负样本之间的差距
            metrics[f'rewards_{train_test}/margins'] = (chosen_rewards - rejected_rewards).cpu().numpy().tolist()

            policy_rejected_log_probs = all_gather_if_needed(policy_rejected_log_probs.detach(), self.rank, self.world_size)
            # token被拒绝的概率
            metrics[f'logps_{train_test}/rejected'] = policy_rejected_log_probs.cpu().numpy().tolist()

        elif loss_config.name == 'sft': # sft时，只看choosen的loss
            policy_chosen_logits = self.policy_model(batch['chosen_input_ids'], attention_mask=batch['chosen_attention_mask']).logits.to(torch.float32)
            policy_chosen_log_probs = _get_batch_log_probs(policy_chosen_logits, batch['chosen_labels'], average_log_prob=False)
            losses = -policy_chosen_log_probs

        # token被选择的概率
        policy_chosen_log_probs = all_gather_if_needed(policy_chosen_log_probs.detach(), self.rank, self.world_size)
        metrics[f'logps_{train_test}/chosen'] = policy_chosen_log_probs.cpu().numpy().tolist()

        # 记录loss
        all_devices_losses = all_gather_if_needed(losses.detach(), self.rank, self.world_size)
        metrics[f'loss/{train_test}'] = all_devices_losses.cpu().numpy().tolist()

        return losses.mean(), metrics

    def _eval_if_need(self):
        #### BEGIN EVALUATION ####
        if self.example_counter % self.config.eval_every == 0 and (self.example_counter > 0 or self.config.do_first_eval):
            rank0_print(f'Running evaluation after {self.example_counter} train examples')
            self.policy_model.eval()

            all_eval_metrics = defaultdict(list)
            if self.config.sample_during_eval:
                all_policy_samples, all_reference_samples = [], []
                policy_text_table = wandb.Table(columns=["step", "prompt", "sample"])
                if self.config.loss.name == 'dpo':
                    reference_text_table = wandb.Table(columns=["step", "prompt", "sample"])

            for eval_batch in (tqdm.tqdm(self.eval_batches) if self.rank == 0 else self.eval_batches):
                # 只在rank=0上运行eval
                local_eval_batch = slice_and_move_batch_for_device(eval_batch, self.rank, self.world_size, self.rank)
                with torch.no_grad():
                    _, eval_metrics = self.get_batch_forward_loss_and_metrics(local_eval_batch, self.config.loss, train=False)

                for k, v in eval_metrics.items():
                    all_eval_metrics[k].extend(v)

                if self.config.sample_during_eval:
                    if 'FSDP' in self.config.trainer:
                        with FSDP.summon_full_params(self.policy_model, writeback=False, recurse=False):
                            policy_samples, reference_samples = self.get_batch_samples_text(local_eval_batch)
                    else:
                        policy_samples, reference_samples = self.get_batch_samples_text(local_eval_batch)

                    all_policy_samples.extend(policy_samples)
                    all_reference_samples.extend(reference_samples)

                    for prompt, sample in zip(eval_batch['prompt'], policy_samples):
                        policy_text_table.add_data(self.example_counter, prompt, sample)
                    if self.config.loss.name == 'dpo':
                        for prompt, sample in zip(eval_batch['prompt'], reference_samples):
                            reference_text_table.add_data(self.example_counter, prompt, sample)

            mean_eval_metrics = {k: sum(v) / len(v) for k, v in all_eval_metrics.items()}
            rank0_print(f'eval after {self.example_counter}: {formatted_dict(mean_eval_metrics)}')
            if self.config.sample_during_eval:
                rank0_print(json.dumps(all_policy_samples[:10], indent=2))
                if self.config.loss.name == 'dpo':
                    rank0_print(json.dumps(all_reference_samples[:10], indent=2))

            if self.config.wandb.enabled and self.rank == 0:
                wandb.log(mean_eval_metrics, step=self.example_counter)

                if self.config.sample_during_eval:
                    wandb.log({"policy_samples": policy_text_table}, step=self.example_counter)
                    if self.config.loss.name == 'dpo':
                        wandb.log({"reference_samples": reference_text_table}, step=self.example_counter)

            if self.example_counter > 0:
                if self.config.debug:
                    rank0_print('skipping save in debug mode')
                else:
                    output_dir = os.path.join(self.run_dir, f'step-{self.example_counter}')
                    rank0_print(f'creating checkpoint to write to {output_dir}...')
                    self.save(output_dir, mean_eval_metrics)
        #### END EVALUATION ####

    def train(self):
        """Begin either SFT or DPO training, with periodic evaluation."""

        rank0_print(f'Using {self.config.optimizer} optimizer')
        self.optimizer = getattr(torch.optim, self.config.optimizer)(self.policy_model.parameters(), lr=self.config.lr) # 这个反射用得出神入化，但会降低代码可读性,  torch.optim.RMSprop()
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / (self.config.warmup_steps + 1)))

        set_random_seed(self.seed)

        if self.config.loss.name == 'dpo':
            self.reference_model.eval()

        self.example_counter = 0
        self.batch_counter = 0
        last_log = None

        for batch in self.train_iterator:
            #### BEGIN EVALUATION ####
            self._eval_if_need()
            #### END EVALUATION ####

            #### BEGIN TRAINING ####
            self.policy_model.train()

            start_time = time.time()
            batch_metrics = defaultdict(list)
            for microbatch_idx in range(self.config.gradient_accumulation_steps):
                # 将batch内的数据分给不同的microbatch_idx,得到global_microbatch
                global_microbatch = slice_and_move_batch_for_device(batch, microbatch_idx, self.config.gradient_accumulation_steps, self.rank)
                # 将batch global_microbatch的数据分给不同的gpu rank
                local_microbatch = slice_and_move_batch_for_device(global_microbatch, self.rank, self.world_size, self.rank)
                loss, metrics = self.get_batch_forward_loss_and_metrics(local_microbatch, self.config.loss, train=True)
                """
                实际上这⾥就是做了⼀次求mean的操作。原因是直接累加的 accum_iter 次梯度值作为⼀次参数更新的梯度，
                是将梯度值放⼤了 accum_iter 倍，⽽Pytorch的参数更新是写在optimizer.step() ⽅法内部，
                ⽆法⼿动控制，因此只能根据链式法则 ，在loss处进⾏缩放，来达到缩放梯度的⽬的。
                """
                # 梯度累加后平均再求loss
                avg_loss = loss / self.config.gradient_accumulation_steps # 梯度累加：对gradient_accumulation_steps次的梯度进行平均
                avg_loss.backward() # 这里进行了gradient_accumulation_steps次backward，计算每个tensor的梯度，多次计算的梯度进行累加，注意并未进行参数更新

                for k, v in metrics.items():
                    batch_metrics[k].extend(v)

            grad_norm = self.clip_gradient()
            self.optimizer.step() # 对参数进行梯度, 注意：只有此时才进行了参数的更新
            self.scheduler.step() # scheduler计数+1
            self.optimizer.zero_grad() # 清空参数的梯度

            step_time = time.time() - start_time
            examples_per_second = self.config.batch_size / step_time
            batch_metrics['examples_per_second'].append(examples_per_second)
            batch_metrics['grad_norm'].append(grad_norm) # 记录梯度grad_norm,所有参数组成的向量的模长

            self.batch_counter += 1
            self.example_counter += self.config.batch_size

            # 计算所有batch_metrics的平均
            if last_log is None or time.time() - last_log > self.config.minimum_log_interval_secs:
                mean_train_metrics = {k: sum(v) / len(v) for k, v in batch_metrics.items()}
                mean_train_metrics['counters/examples'] = self.example_counter
                mean_train_metrics['counters/updates'] = self.batch_counter
                rank0_print(f'train stats after {self.example_counter} examples: {formatted_dict(mean_train_metrics)}')

                if self.config.wandb.enabled and self.rank == 0:
                    wandb.log(mean_train_metrics, step=self.example_counter)

                last_log = time.time()
            else:
                rank0_print(f'skipping logging after {self.example_counter} examples to avoid logging too frequently')
            #### END TRAINING ####


    def clip_gradient(self):
        """Clip the gradient norm of the parameters of a non-FSDP policy.
            将所有参数看成一个向量，默认计算2范数,即向量模长
        """
        return torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), self.config.max_grad_norm).item()

    def write_state_dict(self, step: int, state: Dict[str, torch.Tensor], metrics: Dict, filename: str, dir_name: Optional[str] = None):
        """Write a checkpoint to disk."""
        if dir_name is None:
            dir_name = os.path.join(self.run_dir, f'LATEST')

        os.makedirs(dir_name, exist_ok=True)
        output_path = os.path.join(dir_name, filename)
        rank0_print(f'writing checkpoint to {output_path}...')
        torch.save({
            'step_idx': step,
            'state': state,
            'metrics': metrics if metrics is not None else {},
        }, output_path)
    
    def save(self, output_dir: Optional[str] = None, metrics: Optional[Dict] = None):
        """Save policy, optimizer, and scheduler state to disk."""

        policy_state_dict = self.policy_model.state_dict()
        self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict

        optimizer_state_dict = self.optimizer.state_dict()
        self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
        del optimizer_state_dict

        scheduler_state_dict = self.scheduler.state_dict()
        self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)


class FSDPTrainer(BasicTrainer):
    def __init__(self, policy_model: nn.Module, config: DictConfig, seed: int, run_dir: str, reference_model: Optional[nn.Module] = None, rank: int = 0, world_size: int = 1):
        """A trainer subclass that uses PyTorch FSDP to shard the model across multiple GPUs.
        
           This trainer will shard both the policy and reference model across all available GPUs.
           Models are sharded at the block level, where the block class name is provided in the config.
        """

        super().__init__(policy_model, config, seed, run_dir, reference_model, rank, world_size)
        assert config.model.block_name is not None, 'must specify model.block_name (e.g., GPT2Block or GPTNeoXLayer) for FSDP'

        wrap_class = get_block_class_from_model(policy_model, config.model.block_name)
        model_auto_wrap_policy = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={wrap_class},)

        # 没有使用transformer的fsdp,而直接手动使用原始的fsdp
        shared_fsdp_kwargs = dict(
            auto_wrap_policy=model_auto_wrap_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            cpu_offload=CPUOffload(offload_params=False),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=rank,
            ignored_modules=None,
            limit_all_gathers=False,
            use_orig_params=False,
            sync_module_states=False
        )

        rank0_print('Sharding policy...')
        # the mixed precision dtype if using FSD
        mix_precision_dtype = getattr(torch, config.model.fsdp_policy_mp) if config.model.fsdp_policy_mp is not None else None
        policy_mix_precision_policy = MixedPrecision(param_dtype=mix_precision_dtype,
                                                     reduce_dtype=mix_precision_dtype,
                                                     buffer_dtype=mix_precision_dtype)
        self.policy = FSDP(policy_model, **shared_fsdp_kwargs, mixed_precision=policy_mix_precision_policy)

        if config.activation_checkpointing:
            rank0_print('Attempting to enable activation checkpointing...')
            try:
                # use activation checkpointing, according to:
                # https://pytorch.org/blog/scaling-multimodal-foundation-models-in-torchmultimodal-with-pytorch-distributed/
                #
                # first, verify we have FSDP activation support ready by importing:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    checkpoint_wrapper,
                    apply_activation_checkpointing,
                )
            except Exception as e:
                rank0_print('FSDP activation checkpointing not available:', e)
            else:
                check_fn = lambda submodule: isinstance(submodule, wrap_class)
                rank0_print('Applying activation checkpointing wrapper to policy...')
                apply_activation_checkpointing(self.policy, checkpoint_wrapper_fn=checkpoint_wrapper, check_fn=check_fn)
                rank0_print('FSDP activation checkpointing enabled!')

        if config.loss.name == 'dpo':
            rank0_print('Sharding reference model...')
            self.reference_model = FSDP(reference_model, **shared_fsdp_kwargs)
        
        print('Loaded model on rank', rank)
        dist.barrier() #     Synchronize all processes.

    def clip_gradient(self):
        """Clip the gradient norm of the parameters of an FSDP policy, gathering the gradients across all GPUs."""
        return self.policy.clip_grad_norm_(self.config.max_grad_norm).item()
    
    def save(self, output_dir=None, metrics=None):
        """Save policy, optimizer, and scheduler state to disk, gathering from all processes and saving only on the rank 0 process."""
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT, state_dict_config=save_policy):
            policy_model_state_dict = self.policy.state_dict()

        if self.rank == 0:
            self.write_state_dict(self.example_counter, policy_model_state_dict, metrics, 'policy.pt', output_dir) # model
        del policy_model_state_dict
        dist.barrier() # 同步

        save_policy = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.policy, StateDictType.FULL_STATE_DICT, optim_state_dict_config=save_policy):
            optimizer_state_dict = FSDP.optim_state_dict(self.policy, self.optimizer)

        if self.rank == 0:
            self.write_state_dict(self.example_counter, optimizer_state_dict, metrics, 'optimizer.pt', output_dir)
        del optimizer_state_dict
        dist.barrier()

        if self.rank == 0:
            scheduler_state_dict = self.scheduler.state_dict()
            self.write_state_dict(self.example_counter, scheduler_state_dict, metrics, 'scheduler.pt', output_dir)
        dist.barrier()
        

class TensorParallelTrainer(BasicTrainer):
    def __init__(self, policy_model:PreTrainedModel, config, seed, run_dir, reference_model=None, rank=0, world_size=1):
        """A trainer subclass that uses TensorParallel to shard the model across multiple GPUs.

           Based on https://github.com/BlackSamorez/tensor_parallel. Note sampling is extremely slow,
              see https://github.com/BlackSamorez/tensor_parallel/issues/66.
        """
        super().__init__(policy_model, config, seed, run_dir, reference_model, rank, world_size)
        
        rank0_print('Sharding policy...')
        self.policy:PreTrainedModel = tp.tensor_parallel(policy_model, sharded=True)
        if config.loss.name == 'dpo':
            rank0_print('Sharding reference model...')
            self.reference_model = tp.tensor_parallel(reference_model, sharded=False)

    def save(self, output_dir=None, metrics=None):
        """Save (unsharded) policy state to disk."""
        with tp.save_tensor_parallel(self.policy):
            policy_state_dict = self.policy.state_dict()
    
        self.write_state_dict(self.example_counter, policy_state_dict, metrics, 'policy.pt', output_dir)
        del policy_state_dict
        