import os
import shutil
import argparse
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm
import importlib
import copy
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
import librosa
from pathlib import Path
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import wandb

from diffusers.optimization import get_scheduler
from omegaconf import OmegaConf

from emage_evaltools.mertic import FGD, BC, L1div, LVDFace, MSEFace
from emage_utils.motion_io import beat_format_load, beat_format_save, MASK_DICT, recover_from_mask
import emage_utils.rotation_conversions as rc
from emage_utils import fast_render
from emage_utils.motion_rep_transfer import get_motion_rep_numpy
from models.emage_audio import CausalEmageAudioTokenModel, StreamableVQModel, shift_tokens_with_bos


# ---------------------------------  train,val,test fn here --------------------------------- #
def inference_fn(cfg, model, device, test_path, save_path, **kwargs):
    motion_vq = kwargs["motion_vq"]
    actual_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    actual_model.eval()
    test_list = []
    for data_meta_path in test_path:
        test_list.extend(json.load(open(data_meta_path, "r")))
    test_list = [item for item in test_list if item.get("mode") == "test"]
    seen_ids = set()
    test_list = [item for item in test_list if not (item["video_id"] in seen_ids or seen_ids.add(item["video_id"]))]

    save_list = []
    start_time = time.time()
    total_length = 0
    for test_file in tqdm(test_list, desc="Testing"):
        audio, _ = librosa.load(test_file["audio_path"], sr=cfg.audio_sr)
        audio = torch.from_numpy(audio).to(device).unsqueeze(0)
        speaker_id = torch.zeros(1,1).to(device).long()

        # The predictor conditions on its own token history (starting from BOS);
        # no seed motion or motion feedback is used.
        token_logits = actual_model.inference(audio, speaker_id)
        motion_all = motion_vq.decode(
            face_index=token_logits["cls_face"].argmax(dim=-1),
            upper_index=token_logits["cls_upper"].argmax(dim=-1),
            hands_index=token_logits["cls_hands"].argmax(dim=-1),
            lower_index=token_logits["cls_lower"].argmax(dim=-1),
        )
       
        motion_pred = motion_all["motion_axis_angle"]
        t = motion_pred.shape[1]
        motion_pred = motion_pred.cpu().numpy().reshape(t, -1)
        expression_pred = motion_all["expression"].cpu().numpy().reshape(t, -1)
        trans_pred = motion_all["trans"].cpu().numpy().reshape(t, -1)
        # print(motion_pred.shape, expression_pred.shape, trans_pred.shape)
        beat_format_save(os.path.join(save_path, f"{test_file['video_id']}_output.npz"), motion_pred, upsample=30//cfg.pose_fps, expressions=expression_pred, trans=trans_pred)
        save_list.append(
            {
                "audio_path": test_file["audio_path"],
                "motion_path": os.path.join(save_path, f"{test_file['video_id']}_output.npz"),
                "video_id": test_file["video_id"],
            }
        )
        total_length+=t
    time_cost = time.time() - start_time
    print(f"\n cost {time_cost:.2f} seconds to generate {total_length / cfg.pose_fps:.2f} seconds of motion")
    return test_list, save_list

def get_cls_loss(motion_pred, motion_gt, cu, cl, ch, cf, ClsFn):
    ClsFn = ClsFn.to(motion_pred["cls_upper"].device)
    pred_upper = F.log_softmax(motion_pred["cls_upper"], dim=2)
    pred_lower = F.log_softmax(motion_pred["cls_lower"], dim=2)
    pred_hands = F.log_softmax(motion_pred["cls_hands"], dim=2)
    pred_face = F.log_softmax(motion_pred["cls_face"], dim=2)
    pred_upper = pred_upper.permute(0, 2, 1)  
    pred_lower = pred_lower.permute(0, 2, 1)
    pred_hands = pred_hands.permute(0, 2, 1)
    pred_face = pred_face.permute(0, 2, 1)
    cls_loss_upper = cu * ClsFn(pred_upper, motion_gt["upper"])
    cls_loss_lower = cl * ClsFn(pred_lower, motion_gt["lower"])
    cls_loss_hands = ch * ClsFn(pred_hands, motion_gt["hands"])
    cls_loss_face = cf * ClsFn(pred_face, motion_gt["face"])
    return cls_loss_upper+cls_loss_lower+cls_loss_hands+cls_loss_face

def train_val_fn(cfg, batch, model, device, mode="train", **kwargs):
    if mode == "train":
        model.train()
        kwargs["optimizer"].zero_grad()
    else:
        model.eval()

    motion_vq = kwargs["motion_vq"]
    motion_gt = batch["motion"].to(device)
    audio = batch["audio"].to(device)
    expressions_gt = batch["expressions"].to(device)
    bs, t, jc = motion_gt.shape
    j = jc // 3
    speaker_id = torch.zeros(bs,1).to(device).long()
    motion_gt = rc.axis_angle_to_rotation_6d(motion_gt.reshape(bs,t,j,3)).reshape(bs, t, j*6)
   
    # Stage-2: predict the shifted future sequence, conditioned on past tokens.
    # target = the 4-frame-shifted future (B1..B15) encoded as one VQ stream;
    # the model input is the right-shifted target tokens with a BOS prepended
    # (teacher forcing).
    factor = cfg.model.token_downsample_factor
    future_motion = motion_gt[:, factor:]
    future_expr = expressions_gt[:, factor:]
    # future must line up to whole VQ streams (each token = `factor` frames).
    assert future_motion.shape[1] % factor == 0, f"future_motion length {future_motion.shape[1]} not divisible by {factor}"
    latent_index_dict = motion_vq.map2index(future_motion, future_expr)
    full_inputs = shift_tokens_with_bos(latent_index_dict, cfg.model.vae_codebook_size)

    # Train on a random contiguous sub-window of the full shifted sequence so
    # training covers the same window configurations inference sees: windows
    # that still contain the BOS (start=0), slid windows without BOS (start>0),
    # and varying history lengths. Middle windows keep their real predecessor
    # tokens; no new BOS is inserted. Audio is sliced to the same positions
    # inside the model via audio_token_start.
    n_tokens = full_inputs["face"].shape[1]
    start = random.randint(0, n_tokens - 1) if mode == "train" else 0
    past_tokens = {part: tokens[:, start:] for part, tokens in full_inputs.items()}
    window_targets = {part: tokens[:, start:] for part, tokens in latent_index_dict.items()}
    motion_pred = model(audio, speaker_id, past_tokens, audio_token_start=start)
    loss_dict = {
        "cls": get_cls_loss(motion_pred, window_targets, cfg.model.cu, cfg.model.cl, cfg.model.ch, cfg.model.cf, kwargs["ClsFn"]),
    }
    
    all_loss = sum(loss_dict.values())
    loss_dict["all"] = all_loss
  
    if mode == "train":
        all_loss.backward()
        if cfg.solver.max_grad_norm > 0:
          torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.solver.max_grad_norm)
        kwargs["optimizer"].step()
        kwargs["lr_scheduler"].step()

    if mode == "val":
        _, cls_face =  torch.max(F.log_softmax(motion_pred["cls_face"], dim=2), dim=2)
        _, cls_upper =  torch.max(F.log_softmax(motion_pred["cls_upper"], dim=2), dim=2)
        _, cls_hands =  torch.max(F.log_softmax(motion_pred["cls_hands"], dim=2), dim=2)
        _, cls_lower =  torch.max(F.log_softmax(motion_pred["cls_lower"], dim=2), dim=2)
        decode_dict = motion_vq.decode(
            face_index=cls_face,
            upper_index=cls_upper,
            hands_index=cls_hands,
            lower_index=cls_lower,
        )
        motion_pred_rot6d = decode_dict["motion_rot6d"]
        # compare against the same shifted future that is being predicted
        kwargs["fgd_evaluator"].update(motion_pred_rot6d, future_motion)
    return loss_dict


# ---------------------------------  main train loop here --------------------------------- #
def main(cfg):
    seed_everything(cfg.seed)
    os.environ["WANDB_API_KEY"] = cfg.wandb_key
    local_rank = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else 0
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.distributed.init_process_group(backend="nccl")
    log_dir = os.path.join(cfg.output_dir, cfg.exp_name)
    experiment_ckpt_dir = os.path.join(log_dir, "checkpoints")
    os.makedirs(experiment_ckpt_dir, exist_ok=True)

    if local_rank == 0 and cfg.validation.wandb:  
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        wandb.init(
            project=cfg.wandb_project,
            name=f"{cfg.exp_name}_{run_time}",
            entity=cfg.wandb_entity,
            dir=log_dir,
            config=OmegaConf.to_container(cfg)
        )

    motion_vq = StreamableVQModel.from_config(cfg.model).to(device)
    for param in motion_vq.parameters():
        param.requires_grad = False
    motion_vq.eval()
    # The predictor uses one shared vae_codebook_size for all four heads, which
    # is only valid if the four VQ codebooks are actually the same size.
    for part in ("face", "upper", "hands", "lower"):
        n_tokens = getattr(motion_vq, f"vq_model_{part}").quantizer.num_tokens
        if n_tokens != cfg.model.vae_codebook_size:
            raise ValueError(
                f"vq_model_{part} codebook size {n_tokens} != "
                f"model.vae_codebook_size {cfg.model.vae_codebook_size}"
            )
    
    if cfg.test:
        model = CausalEmageAudioTokenModel.from_pretrained(cfg.test_model_path).to(device)
    else:
        model = init_hf_class(cfg.model.name_pyfile, cfg.model.class_name, cfg.model).to(device)
  
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    for name, param in model.named_parameters():
        param.requires_grad = True  
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True, broadcast_buffers=False)

    # optimizer
    optimizer_cls = torch.optim.Adam
    optimizer = optimizer_cls(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.solver.learning_rate,
        betas=(cfg.solver.adam_beta1, cfg.solver.adam_beta2),
        weight_decay=cfg.solver.adam_weight_decay,
        eps=cfg.solver.adam_epsilon
    )
    lr_scheduler = get_scheduler(
        cfg.solver.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=cfg.solver.lr_warmup_steps * cfg.solver.gradient_accumulation_steps,
        num_training_steps=cfg.solver.max_train_steps * cfg.solver.gradient_accumulation_steps
    )

    # loss
    ClsFn = nn.NLLLoss()

    # dataset
    train_dataset = init_class(cfg.data.name_pyfile, cfg.data.class_name, cfg, split='train')
    test_dataset = init_class(cfg.data.name_pyfile, cfg.data.class_name, cfg, split='test')
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    train_loader = DataLoader(train_dataset, batch_size=cfg.data.train_bs, sampler=train_sampler, drop_last=True, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=cfg.data.train_bs, sampler=test_sampler, drop_last=False, num_workers=8)

    # resume
    if cfg.resume_from_checkpoint:
        checkpoint = torch.load(cfg.resume_from_checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler_state_dict"])
        iteration = checkpoint["iteration"]
    else:  
        iteration = 0
    if cfg.test:
        iteration = 0

    max_epochs = (cfg.solver.max_train_steps // len(train_loader)) + (1 if cfg.solver.max_train_steps % len(train_loader) != 0 else 0)
    start_epoch = iteration // len(train_loader)
    start_step_in_epoch = iteration % len(train_loader)
    fgd_evaluator = FGD(download_path="./emage_evaltools/")
    bc_evaluator = BC(download_path="./emage_evaltools/", sigma=0.3, order=7)
    l1div_evaluator= L1div()
    lvd_evaluator = LVDFace()
    mse_evaluator = MSEFace()
    loss_meters = {}
    loss_meters_val = {}
    best_fgd_val = np.inf
    best_fgd_iteration_val= 0
    best_fgd_test = np.inf
    best_fgd_iteration_test = 0

    # train loop
    epoch = start_epoch
    while iteration < cfg.solver.max_train_steps:
        train_sampler.set_epoch(epoch)
        data_start = time.time()
        pbar = tqdm(train_loader, leave=True)
        for i, batch in enumerate(pbar):
            # for correct resume, if the dataset is very large. since we fixed the seed, we can skip the data
            if i < start_step_in_epoch: 
              iteration += 1
              continue
           
            # test
            if iteration % cfg.validation.test_steps == 0 and local_rank == 0:
                test_save_path = os.path.join(log_dir, f"test_{iteration}")
                os.makedirs(test_save_path, exist_ok=True)
                with torch.no_grad():
                    test_list, save_list = inference_fn(cfg.model, model, device, cfg.data.test_meta_paths, test_save_path, motion_vq=motion_vq)
                if cfg.validation.evaluation:
                    metrics = evaluation_fn([True]*55, test_list, save_list, fgd_evaluator, bc_evaluator, l1div_evaluator, device, lvd_evaluator, mse_evaluator, factor=cfg.model.token_downsample_factor)
                if cfg.validation.visualization: visualization_fn(save_list, test_save_path, test_list, only_check_one=True, factor=cfg.model.token_downsample_factor)
                if cfg.validation.evaluation: best_fgd_test, best_fgd_iteration_test =  log_test(model, metrics, iteration, best_fgd_test, best_fgd_iteration_test, cfg, local_rank, experiment_ckpt_dir, test_save_path)
                if cfg.test: return 0

            # validation
            if iteration % cfg.validation.validation_steps == 0:
                loss_meters = {}
                loss_meters_val = {}
                fgd_evaluator.reset()
                pbar_val = tqdm(test_loader, leave=True)

                data_start_val = time.time()  
                for j, batch in enumerate(pbar_val):
                    data_time_val = time.time() - data_start_val
                    with torch.no_grad():
                        val_loss_dict = train_val_fn(cfg, batch, model, device, mode="val", fgd_evaluator=fgd_evaluator, motion_vq=motion_vq, ClsFn=ClsFn, iteration=iteration)
                    net_time_val = time.time() - data_start_val
                    val_loss_dict["fgd"] = fgd_evaluator.compute() if j == len(test_loader) - 1 else 0
                    log_train_val(cfg, val_loss_dict, local_rank, loss_meters_val, pbar_val, epoch, max_epochs, iteration, net_time_val, data_time_val, optimizer, "Val  ")
                    data_start_val = time.time()
                    if cfg.debug and j > 1: break

                if local_rank == 0:
                    best_fgd_val, best_fgd_iteration_val = save_last_and_best_ckpt(
                        model, optimizer, lr_scheduler, iteration, experiment_ckpt_dir, best_fgd_val, best_fgd_iteration_val, val_loss_dict["fgd"], lower_is_better=True, mertic_name="fgd")

            # train
            data_time = time.time() - data_start
            loss_dict = train_val_fn(cfg, batch, model, device, mode="train", motion_vq=motion_vq, optimizer=optimizer, lr_scheduler=lr_scheduler, ClsFn=ClsFn, iteration=iteration)
            net_time = time.time() - data_start - data_time
            log_train_val(cfg, loss_dict, local_rank, loss_meters, pbar, epoch, max_epochs, iteration, net_time, data_time, optimizer, "Train")
            data_start = time.time()

            iteration += 1
   
        start_step_in_epoch = 0
        epoch += 1

    if local_rank == 0 and cfg.validation.wandb:
        wandb.finish()
    torch.distributed.destroy_process_group()


# ---------------------------------  utils fn here --------------------------------- #
def evaluation_fn(joint_mask, gt_list, pred_list, fgd_evaluator, bc_evaluator, l1_evaluator, device, lvd_evaluator, mse_evaluator, factor=4):
    # The predictor's first generated block is B1 (frame `factor` of the GT
    # timeline), so shift GT and audio by `factor` frames to align with the
    # saved predictions.
    fgd_evaluator.reset()
    bc_evaluator.reset()
    l1_evaluator.reset()
    lvd_evaluator.reset()
    mse_evaluator.reset()

    for test_file in tqdm(gt_list, desc="Evaluation"):
        # only load selective joints
        pred_file = [item for item in pred_list if item["video_id"] == test_file["video_id"]][0]
        if not pred_file:
            print(f"Missing prediction for {test_file['video_id']}")
            continue
        # print(test_file["motion_path"], pred_file["motion_path"])
        gt_dict = beat_format_load(test_file["motion_path"], joint_mask)
        pred_dict = beat_format_load(pred_file["motion_path"], joint_mask)

        motion_gt = gt_dict["poses"][factor:]
        motion_pred = pred_dict["poses"]
        expressions_gt = gt_dict["expressions"][factor:]
        expressions_pred = pred_dict["expressions"]
        betas = gt_dict["betas"]
        # motion_gt = recover_from_mask(motion_gt, joint_mask) # t1*165
        # motion_pred = recover_from_mask(motion_pred, joint_mask) # t2*165
    
        t = min(motion_gt.shape[0], motion_pred.shape[0])
        motion_gt = motion_gt[:t]
        motion_pred = motion_pred[:t]
        expressions_gt = expressions_gt[:t]
        expressions_pred = expressions_pred[:t]
       
        # bc and l1 require position representation
        motion_position_pred = get_motion_rep_numpy(motion_pred, device=device, betas=betas)["position"] # t*55*3
        motion_position_pred = motion_position_pred.reshape(t, -1)
        # ignore the start and end 2s, this may for beat dataset only
        audio_beat = bc_evaluator.load_audio(test_file["audio_path"], t_start=int((60 + factor) / 30 * 16000), t_end=int((t - 60 + factor) / 30 * 16000))
        motion_beat = bc_evaluator.load_motion(motion_position_pred, t_start=60, t_end=t-60, pose_fps=30, without_file=True)
        bc_evaluator.compute(audio_beat, motion_beat, length=t-120, pose_fps=30)
        # audio_beat = bc_evaluator.load_audio(test_file["audio_path"], t_start=0 * 16000, t_end=int((t-0)/30*16000))
        # motion_beat = bc_evaluator.load_motion(motion_position_pred, t_start=0, t_end=t-0, pose_fps=30, without_file=True)
        # bc_evaluator.compute(audio_beat, motion_beat, length=t-0, pose_fps=30)

        l1_evaluator.compute(motion_position_pred)
       
        face_position_pred = get_motion_rep_numpy(motion_pred, device=device, expressions=expressions_pred, expression_only=True, betas=betas)["vertices"] # t -1
        face_position_gt = get_motion_rep_numpy(motion_gt, device=device, expressions=expressions_gt, expression_only=True, betas=betas)["vertices"]
        lvd_evaluator.compute(face_position_pred, face_position_gt)
        mse_evaluator.compute(face_position_pred, face_position_gt)
       
        # fgd requires rotation 6d representaiton
        motion_gt = torch.from_numpy(motion_gt).to(device).unsqueeze(0)
        motion_pred = torch.from_numpy(motion_pred).to(device).unsqueeze(0)
        motion_gt = rc.axis_angle_to_rotation_6d(motion_gt.reshape(1, t, 55, 3)).reshape(1, t, 55*6)
        motion_pred = rc.axis_angle_to_rotation_6d(motion_pred.reshape(1, t, 55, 3)).reshape(1, t, 55*6)
        fgd_evaluator.update(motion_pred.float(), motion_gt.float())
       
    metrics = {}
    metrics["fgd"] = fgd_evaluator.compute()
    metrics["bc"] = bc_evaluator.avg()
    metrics["l1"] = l1_evaluator.avg()
    metrics["lvd"] = lvd_evaluator.avg()
    metrics["mse"] = mse_evaluator.avg()
    return metrics

def visualization_fn(pred_list, save_path, gt_list=None, only_check_one=True, factor=4):
    if gt_list is None: # single visualization
        for i in range(len(pred_list)):
            fast_render.render_one_sequence(
                pred_list[i]["motion_path"],
                save_path,
                pred_list[i]["audio_path"],
                model_folder="./evaluation/smplx_models/",
            )
            if only_check_one: break
    else: # paired visualization, pad the translation
        # The prediction starts at GT frame `factor` (first generated block is
        # B1), so trim the GT by `factor` frames to align the two timelines.
        for i in range(len(pred_list)):
            npz_pred = np.load(pred_list[i]["motion_path"], allow_pickle=True)
            gt_file = [item for item in gt_list if item["video_id"] == pred_list[i]["video_id"]][0]
            if not gt_file:
                print(f"Missing prediction for {pred_list[i]['video_id']}")
                continue
            npz_gt = np.load(gt_file["motion_path"], allow_pickle=True)
            t = min(npz_pred["poses"].shape[0], npz_gt["poses"].shape[0] - factor)
            np.savez(
                os.path.join(save_path, f"{pred_list[i]['video_id']}_transpad.npz"),
                betas=npz_pred['betas'][:t],
                poses=npz_pred['poses'][:t],
                expressions=npz_pred['expressions'][:t],
                trans=npz_pred["trans"][:t],
                model='smplx2020',
                gender='neutral',
                mocap_frame_rate=30,
            )
            gt_trimmed_path = os.path.join(save_path, f"{pred_list[i]['video_id']}_gt_trimmed.npz")
            np.savez(
                gt_trimmed_path,
                betas=npz_gt['betas'],
                poses=npz_gt['poses'][factor:factor + t],
                expressions=npz_gt['expressions'][factor:factor + t],
                trans=npz_gt["trans"][factor:factor + t],
                model='smplx2020',
                gender='neutral',
                mocap_frame_rate=30,
            )
            fast_render.render_one_sequence(
                os.path.join(save_path, f"{pred_list[i]['video_id']}_transpad.npz"),
                gt_trimmed_path,
                save_path,
                pred_list[i]["audio_path"],
                model_folder="./evaluation/smplx_models/",
            )
            if only_check_one: break
     
def log_test(model, metrics, iteration, best_mertics, best_iteration, cfg, local_rank, experiment_ckpt_dir, video_save_path=None):
    if local_rank == 0:
        print(f"\n Test Results at iteration {iteration}:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.10f}")
        if cfg.validation.wandb:
            for key, value in metrics.items():
                wandb.log({f"test/{key}": value}, step=iteration)
        if cfg.validation.wandb and cfg.validation.visualization:
            videos_to_log = []
            for filename in os.listdir(video_save_path):
                if filename.endswith(".mp4"):
                    videos_to_log.append(wandb.Video(os.path.join(video_save_path, filename)))
            if videos_to_log:
                wandb.log({"test/videos": videos_to_log}, step=iteration)
        if metrics["fgd"] < best_mertics:
            best_mertics = metrics["fgd"]
            best_iteration = iteration
            model.module.save_pretrained(os.path.join(experiment_ckpt_dir, "test_best"))
        # print(metrics, best_mertics, best_iteration)
        message = f"Current Test FGD: {metrics['fgd']:.4f} (Best: {best_mertics:.4f} at iteration {best_iteration})"
        log_metric_with_box(message)
    return best_mertics, best_iteration

def log_metric_with_box(message):
    box_width = len(message) + 2
    border = "-" * box_width
    print(f"\n{border}")
    print(f"|{message}|")
    print(f"{border}\n")

def log_train_val(cfg, loss_dict, local_rank, loss_meters, pbar, epoch, max_epochs, iteration, net_time, data_time, optimizer, ptype="Train"):
    new_loss_dict = {}
    for k, v in loss_dict.items():
        if "fgd" in k: continue
        v_cpu = torch.as_tensor(v).float().cpu().item()
        if k not in loss_meters:
            loss_meters[k] = {"sum":0,"count":0}
        loss_meters[k]["sum"] += v_cpu
        loss_meters[k]["count"] += 1
        new_loss_dict[k] = v_cpu
    mem_used = torch.cuda.memory_reserved() / 1E9
    lr = optimizer.param_groups[0]["lr"]
    loss_str = " ".join([f"{k}: {new_loss_dict[k]:.4f}({loss_meters[k]['sum']/loss_meters[k]['count']:.4f})" for k in new_loss_dict])
    desc = f"{ptype}: Epoch[{epoch}/{max_epochs}] Iter[{iteration}] {loss_str} lr: {lr:.2E} data_time: {data_time:.3f} net_time: {net_time:.3f} mem: {mem_used:.2f}GB"
    pbar.set_description(desc)
    pbar.bar_format = "{desc} {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    if cfg.validation.wandb and local_rank == 0:
        for k, v in new_loss_dict.items():
            wandb.log({f"loss/{ptype}/{k}": v}, step=iteration)

def save_last_and_best_ckpt(model, optimizer, lr_scheduler, iteration, save_dir, previous_best, best_iteration, current, lower_is_better=True, mertic_name="fgd"):
    checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict(),
            "iteration": iteration,
        }
    torch.save(checkpoint, os.path.join(save_dir, "last.bin"))
    model.module.save_pretrained(os.path.join(save_dir, "last"))
    if (lower_is_better and current < previous_best) or (not lower_is_better and current > previous_best):
        previous_best = current
        best_iteration = iteration
        shutil.copy(os.path.join(save_dir, "last.bin"), os.path.join(save_dir, "best.bin"))
        model.module.save_pretrained(os.path.join(save_dir, "best"))
    message = f"Current interation {iteration} {mertic_name}: {current:.4f} (Best: {previous_best:.4f} at iteration {best_iteration})"
    log_metric_with_box(message)
    return previous_best, best_iteration

def init_hf_class(module_name, class_name, config, **kwargs):
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    config_class = model_class.config_class
    config = config_class(config_obj=config)
    instance = model_class(config, **kwargs)
    return instance

def init_class(module_name, class_name, config, **kwargs):
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    instance = model_class(config, **kwargs)
    return instance

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

def init_env():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/train/stage2.yaml")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--visualization", action="store_true")
    parser.add_argument("--evaluation", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument('overrides', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    config.exp_name = os.path.splitext(os.path.basename(args.config))[0]

    if args.overrides: config = OmegaConf.merge(config, OmegaConf.from_dotlist(args.overrides))
    if args.debug:
        config.wandb_project = "debug"
        config.exp_name = "debug"
        config.solver.max_train_steps = 4
    else:
        run_time = datetime.now().strftime("%Y%m%d-%H%M")
        config.exp_name = config.exp_name + "_" + run_time
    if args.wandb:
        config.validation.wandb = True
    if args.visualization:
        config.validation.visualization = True
    if args.evaluation:
        config.validation.evaluation = True
    if args.test:
        config.test = True
    save_dir = os.path.join(config.output_dir, config.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    sanity_check_dir = os.path.join(save_dir, 'sanity_check')
    os.makedirs(sanity_check_dir, exist_ok=True)
    with open(os.path.join(sanity_check_dir, f'{config.exp_name}.yaml'), 'w') as f:
        OmegaConf.save(config, f)
    current_dir = Path.cwd()
    output_dir = Path(config.output_dir).resolve()
    for root, dirs, files in os.walk(current_dir):
        root_path = Path(root)
        dirs[:] = [name for name in dirs if (root_path / name).resolve() != output_dir]
        for name in files:
            if name.endswith('.py'):
                py_file = root_path / name
                dest_path = Path(sanity_check_dir) / py_file.relative_to(current_dir)
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(py_file, dest_path)
    return config

if __name__ == "__main__":
    config = init_env()
    main(config)
