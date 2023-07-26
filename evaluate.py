#!/usr/bin/env python3 -u
# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.


import os
import sys
import logging
import torch
import numpy as np

from fairseq import distributed_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.logging import progress_bar
from fairseq.utils import reset_logging
from omegaconf import DictConfig
from utils import checkpoint_utils
from utils.eval_utils import eval_step, merge_results

# for clip
from clip import clip
from clip.visualizer import visualize_attention


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("ofa.evaluate")


def apply_half(t):
    if t.dtype is torch.float32:
        return t.to(dtype=torch.half)
    return t


def ofa_score_mapping(ofa_scr, eps = 1.0):
    threshold = (1./8192)*10
    ofa_scr[ofa_scr<=threshold] = 0.0   # set to 0 if the porb<1/codebooksize
    ofa_scr[ofa_scr>threshold] = np.log(ofa_scr[ofa_scr>threshold]) - np.log(threshold)  # set to log if the prob>1/codebooksize
    # eps is the minimum socre and set to 1
    ofa_scr += eps

    return ofa_scr


def calculate_score(ofa_scr, overrides, sample_id, use_indicator, clip_scr=None):
    
    if use_indicator:
        # calculate the final score
        ofa_scr = ofa_score_mapping(ofa_scr)

    if clip_scr is not None:
        fin_scr = np.multiply(clip_scr, ofa_scr)
    else:
        fin_scr = ofa_scr

    # save for visualizing
    np.save(os.path.join(overrides["output_path"], "{}_woall.npy".format(sample_id)), fin_scr)

    naive_scr = fin_scr.mean()
    log_scr = fin_scr.mean()
    har_scr = fin_scr.mean()

    print("sample_id: ", sample_id, "| naive_scr: ", naive_scr,
          "| log_scr: ", log_scr,
          "| har_scr: ", har_scr)

    with open(overrides["score_filepath"], "a") as f:
        f.write("{}\t{}\t{}\t{}\n".format(sample_id, naive_scr, log_scr, har_scr))


def main_eval(cfg: DictConfig, **kwargs):
    utils.import_user_module(cfg.common)

    reset_logging()
    logger.info(cfg)

    assert (
            cfg.dataset.max_tokens is not None or cfg.dataset.batch_size is not None
    ), "Must specify batch size either with --max-tokens or --batch-size"

    # Fix seed for stochastic decoding
    if cfg.common.seed is not None and not cfg.generation.no_seed_provided:
        np.random.seed(cfg.common.seed)
        utils.set_torch_seed(cfg.common.seed)

    use_fp16 = cfg.common.fp16
    use_cuda = torch.cuda.is_available() and not cfg.common.cpu

    if use_cuda:
        torch.cuda.set_device(cfg.distributed_training.device_id)

    # Load ensemble
    overrides = eval(cfg.common_eval.model_overrides)
    # Deal with beam-search / all-candidate VQA eval
    if cfg.task._name == "vqa_gen":
        overrides['val_inference_type'] = "beamsearch" if kwargs['beam_search_vqa_eval'] else "allcand"

    logger.info("loading model(s) from {}".format(cfg.common_eval.path))
    if kwargs["zero_shot"]:
        task = tasks.setup_task(cfg.task)
        models, saved_cfg = checkpoint_utils.load_model_ensemble(
            utils.split_paths(cfg.common_eval.path),
            arg_overrides=overrides,
            task=task,
            suffix=cfg.checkpoint.checkpoint_suffix,
            strict=(cfg.checkpoint.checkpoint_shard_count == 1),
            num_shards=cfg.checkpoint.checkpoint_shard_count,
        )
    else:
        models, saved_cfg, task = checkpoint_utils.load_model_ensemble_and_task(
            utils.split_paths(cfg.common_eval.path),
            arg_overrides=overrides,
            suffix=cfg.checkpoint.checkpoint_suffix,
            strict=(cfg.checkpoint.checkpoint_shard_count == 1),
            num_shards=cfg.checkpoint.checkpoint_shard_count,
        )

    # loading the dataset should happen after the checkpoint has been loaded so we can give it the saved task config
    task.load_dataset(cfg.dataset.gen_subset, task_cfg=saved_cfg.task)

    # Move models to GPU
    for model, ckpt_path in zip(models, utils.split_paths(cfg.common_eval.path)):
        if kwargs['ema_eval']:
            logger.info("loading EMA weights from {}".format(ckpt_path))
            model.load_state_dict(checkpoint_utils.load_ema_from_checkpoint(ckpt_path)['model'])
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(cfg.dataset.gen_subset),
        max_tokens=cfg.dataset.max_tokens,
        max_sentences=cfg.dataset.batch_size,
        max_positions=utils.resolve_max_positions(
            task.max_positions(), *[m.max_positions() for m in models]
        ),
        ignore_invalid_inputs=cfg.dataset.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=cfg.dataset.required_batch_size_multiple,
        seed=cfg.common.seed,
        num_shards=cfg.distributed_training.distributed_world_size,
        shard_id=cfg.distributed_training.distributed_rank,
        num_workers=cfg.dataset.num_workers,
        data_buffer_size=cfg.dataset.data_buffer_size,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=cfg.common.log_format,
        log_interval=cfg.common.log_interval,
        default_log_format=("tqdm" if not cfg.common.no_progress_bar else "simple"),
    )

    # Initialize generator
    generator = task.build_generator(models, cfg.generation)

    results = []
    score_sum = torch.FloatTensor([0]).cuda()
    score_cnt = torch.FloatTensor([0]).cuda()
    clip_tool = clip.CLIPTool(clip_mode=overrides["clip_mode"])
    for sample in progress:
        # ofa
        if "net_input" not in sample:
            continue
        sample = utils.move_to_cuda(sample) if use_cuda else sample
        sample = utils.apply_to_sample(apply_half, sample) if cfg.common.fp16 else sample
        with torch.no_grad():
            result, scores, probs_list = eval_step(task, generator, models, sample, cfg, **kwargs)
        results += result
        score_sum += sum(scores) if scores is not None else 0
        score_cnt += len(scores) if scores is not None else 0
        progress.log({"sentences": sample["nsentences"]})

        # collect probs dict
        ofa_scr = np.asarray(probs_list)

        # clip
        image_path = os.path.join(overrides["image_path"], "{:012d}.jpg".format(int(sample["code_images"])))

        if kwargs['use_credit'] or kwargs['use_image_credit']:
            clip_scr = clip_tool.clip_score(image_path=image_path, caption=sample["text"][0], output_path=overrides["output_path"], mask_size=32, use_credit = kwargs['use_credit'], use_image_credit=kwargs['use_image_credit'], use_smooth_exp=kwargs['use_smooth_exp'])
        else:
            clip_scr = None

        # calculate and save the scores
        calculate_score(ofa_scr=ofa_scr, overrides=overrides, sample_id=sample["id"][0], clip_scr=clip_scr, use_indicator=kwargs['use_indicator'])


def cli_main():
    parser = options.get_generation_parser()
    parser.add_argument("--ema-eval", action='store_true', help="Use EMA weights to make evaluation.")
    parser.add_argument("--beam-search-vqa-eval", action='store_true', help="Use beam search for vqa evaluation (faster inference speed but sub-optimal result), if not specified, we compute scores for each answer in the candidate set, which is slower but can obtain best result.")
    parser.add_argument("--zero-shot", action='store_true')
    args = options.parse_args_and_arch(parser)
    cfg = convert_namespace_to_omegaconf(args)
    distributed_utils.call_main(
        cfg, main_eval, ema_eval=args.ema_eval, beam_search_vqa_eval=args.beam_search_vqa_eval, zero_shot=args.zero_shot
    )


if __name__ == "__main__":
    cli_main()
