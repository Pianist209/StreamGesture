import os
import argparse
import torch
from torchvision.io import write_video

import librosa
import time
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from emage_utils.motion_io import beat_format_save
from emage_utils import fast_render
from models.emage_audio import CausalEmageAudioTokenModel, StreamableVQModel


def inference(model, motion_vq, audio_path, device, save_folder, sr, pose_fps):
    audio, _ = librosa.load(audio_path, sr=sr)
    audio = torch.from_numpy(audio).to(device).unsqueeze(0)
    speaker_id = torch.zeros(1,1).long().to(device)
    with torch.no_grad():
        all_pred = model.generate_motion(audio, speaker_id, motion_vq)
        
    motion_pred = all_pred["motion_axis_angle"]
    t = motion_pred.shape[1]
    motion_pred = motion_pred.cpu().numpy().reshape(t, -1)
    face_pred = all_pred["expression"].cpu().numpy().reshape(t, -1)
    trans_pred = all_pred["trans"].cpu().numpy().reshape(t, -1)
    beat_format_save(os.path.join(save_folder, f"{os.path.splitext(os.path.basename(audio_path))[0]}_output.npz"),
                     motion_pred, upsample=30//pose_fps, expressions=face_pred, trans=trans_pred,
                     start_time_seconds=model.cfg.token_downsample_factor / pose_fps)
    return t

def visualize_one(save_folder, audio_path, nopytorch3d=False, gt_npz=None, extra_npz=None):
    npz_path = os.path.join(save_folder, f"{os.path.splitext(os.path.basename(audio_path))[0]}_output.npz")
    motion_dict = np.load(npz_path, allow_pickle=True)
    audio_start = float(motion_dict["start_time_seconds"])
    # Reference files are on the original audio timeline. Crop each reference
    # to the saved forecast's interval before side-by-side rendering.
    def align_reference(path, label):
        if path is None:
            return None
        with np.load(path, allow_pickle=True) as source:
            data = dict(source)
        source_start = float(data.get("start_time_seconds", 0.0))
        first = round((audio_start - source_start) * 30)
        if first < 0:
            raise ValueError("Reference starts after the prediction")
        for key in ("poses", "expressions", "trans"):
            data[key] = data[key][first:first + motion_dict["poses"].shape[0]]
        data["start_time_seconds"] = audio_start
        aligned = npz_path.replace(".npz", f"_{label}_aligned.npz")
        np.savez(aligned, **data)
        return aligned
    gt_npz = align_reference(gt_npz, "gt")
    extra_npz = align_reference(extra_npz, "extra")
    if not nopytorch3d:
        from emage_utils.npz2pose import render2d
        v2d_face = render2d(motion_dict, (512, 512), face_only=True, remove_global=True)
        write_video(npz_path.replace(".npz", "_2dface.mp4"), v2d_face.permute(0, 2, 3, 1), fps=30)
        fast_render.add_audio_to_video(npz_path.replace(".npz", "_2dface.mp4"), audio_path, npz_path.replace(".npz", "_2dface_audio.mp4"), audio_start_seconds=audio_start)
        v2d_body = render2d(motion_dict, (720, 480), face_only=False, remove_global=True)
        write_video(npz_path.replace(".npz", "_2dbody.mp4"), v2d_body.permute(0, 2, 3, 1), fps=30)
        fast_render.add_audio_to_video(npz_path.replace(".npz", "_2dbody.mp4"), audio_path, npz_path.replace(".npz", "_2dbody_audio.mp4"), audio_start_seconds=audio_start)
    if gt_npz is None:
        fast_render.render_one_sequence_no_gt(
            npz_path, os.path.dirname(npz_path), audio_path,
            model_folder="./emage_evaltools/smplx_models/",
            audio_start_seconds=audio_start,
        )
    else:
        fast_render.render_one_sequence(
            npz_path, gt_npz, os.path.dirname(npz_path), audio_path,
            model_folder="./emage_evaltools/smplx_models/",
            extra_npz_path=extra_npz,
            audio_start_seconds=audio_start,
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_folder", type=str, default="./examples/audio")
    parser.add_argument("--save_folder", type=str, default="./examples/motion")
    parser.add_argument("--config", type=str, default="./configs/emage_streamable_audio.yaml")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--visualization", action="store_true")
    parser.add_argument("--nopytorch3d", action="store_true")
    parser.add_argument(
        "--gt_npz", type=str, default=None,
        help="GT SMPL-X npz for side-by-side rendering of a single audio file",
    )
    parser.add_argument(
        "--extra_npz", type=str, default=None,
        help="Optional third SMPL-X npz for three-way rendering with --gt_npz",
    )
    args = parser.parse_args()

    os.makedirs(args.save_folder, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    config = OmegaConf.load(args.config)
    motion_vq = StreamableVQModel.from_config(config.model).to(device)
    motion_vq.eval()

    model = CausalEmageAudioTokenModel.from_pretrained(args.model_path).to(device).eval()

    # Accept either a folder of .wav files or a single .wav file.
    if os.path.isfile(args.audio_folder):
        audio_files = [args.audio_folder]
    else:
        audio_files = [os.path.join(args.audio_folder, f) for f in os.listdir(args.audio_folder) if f.endswith(".wav")]
    sr, pose_fps = model.cfg.audio_sr, model.cfg.pose_fps

    all_t = 0
    start_time = time.time()

    for audio_path in tqdm(audio_files, desc="Inference"):
        all_t += inference(model, motion_vq, audio_path, device, args.save_folder, sr, pose_fps)
        if args.visualization:
            visualize_one(args.save_folder, audio_path, args.nopytorch3d, args.gt_npz, args.extra_npz)
    print(f"generate total {all_t/pose_fps:.2f} seconds motion in {time.time()-start_time:.2f} seconds")
if __name__ == "__main__":
    main()
