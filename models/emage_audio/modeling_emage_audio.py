import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy
from transformers import PreTrainedModel
from .configuration_emage_audio import EmageAudioConfig, EmageVQVAEConvConfig, EmageVAEConvConfig
from .processing_emage_audio import Quantizer, VQEncoderV5, VQDecoderV5, WavEncoder, CausalWavEncoder, CausalMelAudioEncoder, MLP, PeriodicPositionalEncoding, VQEncoderV6, recover_from_mask_ts, rotation_6d_to_axis_angle, velocity2position, axis_angle_to_rotation_6d, rotation_6d_to_matrix, matrix_to_axis_angle, axis_angle_to_matrix, matrix_to_rotation_6d


def inverse_selection_tensor(filtered_t, selection_array, n):
    selection_array = torch.from_numpy(selection_array).cuda()
    original_shape_t = torch.zeros((n, 165)).cuda()
    selected_indices = torch.where(selection_array == 1)[0]
    for i in range(n):
        original_shape_t[i, selected_indices] = filtered_t[i]
    return original_shape_t

class EmageVAEConv(PreTrainedModel):
    config_class = EmageVAEConvConfig
    base_model_prefix = "emage_vaeconv"
    def __init__(self, config):
        super().__init__(config)
        self.encoder = VQEncoderV5(config)
        self.decoder = VQDecoderV5(config)
        
    def forward(self, inputs):
        pre_latent = self.encoder(inputs)
        rec_pose = self.decoder(pre_latent)
        return {
            "rec_pose": rec_pose
            }

class EmageVQVAEConv(PreTrainedModel):
    config_class = EmageVQVAEConvConfig
    base_model_prefix = "emage_vqvaeconv"
    def __init__(self, config):
        super().__init__(config)
        self.encoder = VQEncoderV5(config)
        self.quantizer = Quantizer(config.vae_codebook_size, config.vae_length, config.vae_quantizer_lambda)
        self.decoder = VQDecoderV5(config)
    def forward(self, inputs):
        pre_latent = self.encoder(inputs)
        embedding_loss, vq_latent, _, perplexity = self.quantizer(pre_latent)
        rec_pose = self.decoder(vq_latent)
        return {"poses_feat":vq_latent,"embedding_loss":embedding_loss,"perplexity":perplexity,"rec_pose": rec_pose}
    def map2index(self, inputs):
        pre_latent = self.encoder(inputs)
        index = self.quantizer.map2index(pre_latent)
        return index
    def map2latent(self, inputs):
        pre_latent = self.encoder(inputs)
        index = self.quantizer.map2index(pre_latent)
        z_q = self.quantizer.get_codebook_entry(index)
        return z_q
    def decode(self, index):
        z_q = self.quantizer.get_codebook_entry(index)
        rec_pose = self.decoder(z_q)
        return rec_pose
    def decode_from_latent(self, latent):
        # print(latent.shape)
        z_flattened = latent.contiguous().view(-1, self.quantizer.e_dim)

        d = torch.sum(z_flattened**2, dim=1, keepdim=True) + torch.sum(self.quantizer.embedding.weight**2, dim=1) - 2*torch.matmul(z_flattened, self.quantizer.embedding.weight.t())
        min_encoding_indices = torch.argmin(d, dim=1)
        # print(min_encoding_indices.shape)
        indices = min_encoding_indices.view(latent.shape[0], latent.shape[1])
        z_q = self.quantizer.get_codebook_entry(indices)
        rec_pose = self.decoder(z_q)
        return rec_pose

class EmageVQModel(nn.Module):
    def __init__(self, face_model, upper_model, hands_model, lower_model, global_model):
        super().__init__()
        self.joint_mask_upper = [
          False, False, False, True, False, False, True, False, False, True,
          False, False, True, True, True, True, True, True, True, True,
          True, True, False, False, False, False, False, False, False, False,
          False, False, False, False, False, False, False, False, False, False,
          False, False, False, False, False, False, False, False, False, False,
          False, False, False, False, False
        ]
        self.joint_mask_lower = [
          True, True, True, False, True, True, False, True, True, False,
          True, True, False, False, False, False, False, False, False, False,
          False, False, False, False, False, False, False, False, False, False,
          False, False, False, False, False, False, False, False, False, False,
          False, False, False, False, False, False, False, False, False, False,
          False, False, False, False, False
        ]
        self.vq_model_face = face_model
        self.vq_model_upper = upper_model
        self.vq_model_hands = hands_model
        self.vq_model_lower = lower_model
        self.global_motion = global_model

    def spilt_inputs(self, smplx_body_rot6d, expression, tar_contact=None, tar_trans=None):
        bs, t, j6 = smplx_body_rot6d.shape
        smplx_body_rot6d = smplx_body_rot6d.reshape(bs, t, j6//6, 6)
        jaw_rot6d = smplx_body_rot6d[:, :, 22:23, :].reshape(bs, t, 6)
        face = torch.cat([jaw_rot6d, expression], dim=2)
        upper_rot6d = smplx_body_rot6d[:, :,self.joint_mask_upper, :].reshape(bs, t, 78)
        hands_rot6d = smplx_body_rot6d[:, :,25:55, :].reshape(bs, t, 180)
        lower_rot6d = smplx_body_rot6d[:, :,self.joint_mask_lower, :].reshape(bs, t, 54)
        tar_contact = torch.zeros(bs, t, 4, device=smplx_body_rot6d.device) if tar_contact is None else tar_contact
        tar_trans = torch.zeros(bs, t, 3, device=smplx_body_rot6d.device) if tar_trans is None else tar_trans
        lower = torch.cat([lower_rot6d, tar_trans, tar_contact], dim=2)
        return dict(face=face, upper=upper_rot6d, hands=hands_rot6d, lower=lower)
    
    def map2index(self, smplx_body_rot6d, expression, tar_contact=None, tar_trans=None):
        inputs = self.spilt_inputs(smplx_body_rot6d, expression, tar_contact=tar_contact, tar_trans=tar_trans)
        face_index = self.vq_model_face.map2index(inputs["face"])
        upper_index = self.vq_model_upper.map2index(inputs["upper"])
        hands_index = self.vq_model_hands.map2index(inputs["hands"])
        lower_index = self.vq_model_lower.map2index(inputs["lower"])
        return dict(face=face_index, upper=upper_index, hands=hands_index, lower=lower_index)
    
    def map2latent(self, smplx_body_rot6d, expression, tar_contact=None, tar_trans=None):
        inputs = self.spilt_inputs(smplx_body_rot6d, expression,tar_contact=tar_contact, tar_trans=tar_trans)
        face_latent = self.vq_model_face.map2latent(inputs["face"])
        upper_latent = self.vq_model_upper.map2latent(inputs["upper"])
        hands_latent = self.vq_model_hands.map2latent(inputs["hands"])
        lower_latent = self.vq_model_lower.map2latent(inputs["lower"])
        return dict(face=face_latent, upper=upper_latent, hands=hands_latent, lower=lower_latent)
    
    def decode(self, face_index=None, upper_index=None, hands_index=None, lower_index=None, 
               face_latent=None, upper_latent=None, hands_latent=None, lower_latent=None, 
            get_global_motion=False, ref_trans=None):
        
        for input_tensor in [face_index, upper_index, hands_index, lower_index, face_latent, upper_latent, hands_latent, lower_latent]:
            if input_tensor is not None:
                bs, t = input_tensor.shape[:2]
                break
  
        if face_index is not None:
            face_mix = self.vq_model_face.decode(face_index) # bs, t, 106
            face_jaw_6d, expression = face_mix[:, :, :6], face_mix[:, :, 6:]
            face_jaw = rotation_6d_to_axis_angle(face_jaw_6d)
        elif face_latent is not None:
            face_mix = self.vq_model_face.decode_from_latent(face_latent)
            face_jaw_6d, expression = face_mix[:, :, :6], face_mix[:, :, 6:]
            face_jaw = rotation_6d_to_axis_angle(face_jaw_6d)
        else:
            face_jaw = torch.zeros(bs, t, 3, device=self.vq_model_face.device)
            expression = torch.zeros(bs, t, 100, device=self.vq_model_face.device)

        if upper_index is not None:
            # print(upper_index)
            upper_6d = self.vq_model_upper.decode(upper_index) # bs, t, 78
            upper = rotation_6d_to_axis_angle(upper_6d.reshape(bs, t, -1, 6)).reshape(bs, t, -1)
        elif upper_latent is not None:
            upper_6d = self.vq_model_upper.decode_from_latent(upper_latent)
            upper = rotation_6d_to_axis_angle(upper_6d.reshape(bs, t, -1, 6)).reshape(bs, t, -1)
        else:
            upper = torch.zeros(bs, t, 39, device=self.vq_model_upper.device)

        if hands_index is not None:
            hands_6d = self.vq_model_hands.decode(hands_index)
            hands = rotation_6d_to_axis_angle(hands_6d.reshape(bs, t, -1, 6)).reshape(bs, t, -1)
        elif hands_latent is not None:
            hands_6d = self.vq_model_hands.decode_from_latent(hands_latent)
            hands = rotation_6d_to_axis_angle(hands_6d.reshape(bs, t, -1, 6)).reshape(bs, t, -1)
        else:
            hands = torch.zeros(bs, t, 90, device=self.vq_model_hands.device)
        
        if lower_index is not None:
            lower_mix = self.vq_model_lower.decode(lower_index)
            lower_6d, transfoot = lower_mix[:, :, :-7], lower_mix[:, :, -7:]
            lower = rotation_6d_to_axis_angle(lower_6d.reshape(bs, t, -1, 6)).reshape(bs, t, -1)
        elif lower_latent is not None:
            lower_mix = self.vq_model_lower.decode_from_latent(lower_latent)
            lower_6d, transfoot = lower_mix[:, :, :-7], lower_mix[:, :, -7:]
            lower = rotation_6d_to_axis_angle(lower_6d.reshape(bs, t, -1, 6)).reshape(bs, t, -1)
        else:
            lower = torch.zeros(bs, t, 27, device=self.vq_model_lower.device)
            transfoot = torch.zeros(bs, t, 7, device=self.vq_model_lower.device)
            lower_6d = axis_angle_to_rotation_6d(lower.reshape(bs, t, -1, 3)).reshape(bs, t, -1)
            lower_mix = torch.cat([lower_6d, transfoot], dim=-1)

        upper2all = recover_from_mask_ts(upper, self.joint_mask_upper)
        hands2all = recover_from_mask_ts(hands, [False]*25+[True]*30)
        lower2all = recover_from_mask_ts(lower, self.joint_mask_lower)
        
        all_motion_axis_angle = upper2all + hands2all + lower2all
        all_motion_axis_angle[:, :, 22*3:22*3+3] = face_jaw
        all_motion_rot6d = axis_angle_to_rotation_6d(all_motion_axis_angle.reshape(bs, t, 55, 3)).reshape(bs, t, 55*6)

        all_motion4inference = torch.cat([all_motion_rot6d, transfoot], dim=2) # 330 + 3 + 4
        
        global_motion = None
        if get_global_motion:
            global_motion = self.get_global_motion(lower_mix, ref_trans)
        return dict(expression=expression, all_motion4inference=all_motion4inference, motion_axis_angle=all_motion_axis_angle, trans=global_motion)
    
    def get_global_motion(self, lower_body, ref_trans):
        global_motion = self.global_motion(lower_body)
        rec_trans_v_s = global_motion["rec_pose"][:, :, 54:57]
        if len(ref_trans.shape) == 2:
            ref_trans = ref_trans.unsqueeze(0).repeat(rec_trans_v_s.shape[0], 1, 1)
        
        rec_x_trans = velocity2position(rec_trans_v_s[:, :, 0:1], 1/30, ref_trans[:, 0, 0:1])
        rec_z_trans = velocity2position(rec_trans_v_s[:, :, 2:3], 1/30, ref_trans[:, 0, 2:3])
        rec_y_trans = rec_trans_v_s[:,:,1:2]
        global_motion = torch.cat([rec_x_trans, rec_y_trans, rec_z_trans], dim=-1)
        return global_motion


class EmageAudioModel(PreTrainedModel):
    config_class = EmageAudioConfig
    base_model_prefix = "emage_audio"
    def __init__(self, config: EmageAudioConfig):
        super().__init__(config)
        self.cfg = config
        # audio encoder
        self.audio_encoder_face = WavEncoder(self.cfg.audio_f)
        self.audio_encoder_body = WavEncoder(self.cfg.audio_f)
        #speaker id
        self.speaker_embedding_body = nn.Embedding(self.cfg.speaker_dims, self.cfg.hidden_size)
        self.speaker_embedding_face = nn.Embedding(self.cfg.speaker_dims, self.cfg.hidden_size)
        # mask embedding
        self.mask_embedding = nn.Parameter(torch.zeros(1,1,self.cfg.pose_dims+3+4))
        nn.init.normal_(self.mask_embedding, 0, self.cfg.hidden_size**-0.5)
        # nn.init.normal_(self.speaker_embedding_body.weight, 0, self.cfg.hidden_size/2**-0.5)
        # nn.init.normal_(self.speaker_embedding_face.weight, 0, self.cfg.hidden_size*2**-0.5)

        # motion pre encoder
        args_top = copy.deepcopy(self.cfg)
        args_top.vae_layer = 3
        args_top.vae_length = self.cfg.motion_f
        args_top.vae_test_dim = self.cfg.pose_dims+3+4
        self.motion_encoder = VQEncoderV6(args_top)
        self.bodyhints_face = MLP(self.cfg.motion_f, self.cfg.hidden_size, self.cfg.motion_f)
        self.bodyhints_body = MLP(self.cfg.motion_f, self.cfg.hidden_size, self.cfg.motion_f)
        # motion encoder
        self.audio_body_motion_proj = nn.Linear(self.cfg.audio_f, self.cfg.hidden_size)
        self.moton_proj = nn.Linear(self.cfg.motion_f, self.cfg.hidden_size)
        self.position_embeddings = PeriodicPositionalEncoding(self.cfg.hidden_size, period=self.cfg.pose_length, max_seq_len=self.cfg.pose_length)
        self.transformer_en_layer = nn.TransformerEncoderLayer(d_model=self.cfg.hidden_size,nhead=4,dim_feedforward=self.cfg.hidden_size*2)
        self.motion_self_encoder = nn.TransformerEncoder(self.transformer_en_layer, num_layers=1)
        # coss attn
        self.audio_motion_cross_attn_layer = nn.TransformerDecoderLayer(d_model=self.cfg.hidden_size,nhead=4,dim_feedforward=self.cfg.hidden_size*2)
        self.audio_motion_cross_attn = nn.TransformerDecoder(self.audio_motion_cross_attn_layer, num_layers=8)
        # feed forward
        self.motion2latent_upper = MLP(self.cfg.hidden_size, self.cfg.hidden_size, self.cfg.hidden_size)
        self.motion2latent_hands = MLP(self.cfg.hidden_size, self.cfg.hidden_size, self.cfg.hidden_size)
        self.motion2latent_lower = MLP(self.cfg.hidden_size, self.cfg.hidden_size, self.cfg.hidden_size)
        # refine
        self.body_motion_decoder_upper = nn.TransformerDecoder(self.audio_motion_cross_attn_layer, num_layers=1)
        self.body_motion_decoder_hands = nn.TransformerDecoder(self.audio_motion_cross_attn_layer, num_layers=1)
        self.body_motion_decoder_lower = nn.TransformerDecoder(self.audio_motion_cross_attn_layer, num_layers=1)
        # deocder
        self.motion_out_proj_upper = nn.Linear(self.cfg.hidden_size, self.cfg.vae_codebook_size)
        self.motion_out_proj_hands = nn.Linear(self.cfg.hidden_size, self.cfg.vae_codebook_size)
        self.motion_out_proj_lower = nn.Linear(self.cfg.hidden_size, self.cfg.vae_codebook_size)
        self.motion_cls_upper = MLP(self.cfg.vae_codebook_size, self.cfg.hidden_size, self.cfg.vae_codebook_size)
        self.motion_cls_hands = MLP(self.cfg.vae_codebook_size, self.cfg.hidden_size, self.cfg.vae_codebook_size)
        self.motion_cls_lower = MLP(self.cfg.vae_codebook_size, self.cfg.hidden_size, self.cfg.vae_codebook_size)

        # face decoder
        self.audio_face_motion_proj = nn.Linear(self.cfg.audio_f+self.cfg.motion_f, self.cfg.hidden_size)
        self.face_motion_decoder = nn.TransformerDecoder(self.audio_motion_cross_attn_layer, num_layers=4)
        self.face_out_proj = nn.Linear(self.cfg.hidden_size, self.cfg.vae_codebook_size)
        self.face_cls = MLP(self.cfg.vae_codebook_size, self.cfg.hidden_size, self.cfg.vae_codebook_size)
    
    def forward(self, audio, speaker_id, masked_motion, mask, use_audio=True):
        # mask motion
        masked_embeddings = self.mask_embedding.expand_as(masked_motion)
        masked_motion = torch.where(mask==1, masked_embeddings, masked_motion)

        # motion token (spatial hints)
        body_hint = self.motion_encoder(masked_motion)
        body_hint_body = self.bodyhints_body(body_hint)
        body_hint_face = self.bodyhints_face(body_hint)
        
        audio2face_fea = self.audio_encoder_face(audio)
        audio2body_fea = self.audio_encoder_body(audio)

        if audio2face_fea.shape[1] > body_hint_face.shape[1]:
            audio2face_fea = audio2face_fea[:, :body_hint_face.shape[1]]
        if audio2body_fea.shape[1] > body_hint_face.shape[1]:
            audio2face_fea = audio2face_fea[:, :body_hint_face.shape[1]]

        bs, t, _ = audio2face_fea.shape

        speaker_motion_fea_proj = self.speaker_embedding_body(speaker_id).repeat(1, t, 1)
        speaker_face_fea_proj = self.speaker_embedding_face(speaker_id).repeat(1, t, 1)
        
        audio2face_fea_proj = self.audio_face_motion_proj(torch.cat([audio2face_fea, body_hint_face], dim=2))
        # audio2face_fea_proj = self.position_embeddings(audio2face_fea_proj)
        # audio2face_fea_proj = speaker_face_fea_proj + audio2face_fea_proj
        face_proj = self.position_embeddings(speaker_face_fea_proj)
        decode_face = self.face_motion_decoder(tgt=face_proj.permute(1,0,2), memory=audio2face_fea_proj.permute(1,0,2)).permute(1,0,2)
        face_latent = self.face_out_proj(decode_face)
        classify_face = self.face_cls(face_latent)

        # motion self attn (temporal)
        masked_motion_proj = self.moton_proj(body_hint_body)
        masked_motion_proj = self.position_embeddings(masked_motion_proj)
        masked_motion_proj = speaker_motion_fea_proj + masked_motion_proj
        motion_fea = self.motion_self_encoder(masked_motion_proj.permute(1,0,2)).permute(1,0,2)

        # audio_cross_attn
        
        audio2body_fea_proj = self.audio_body_motion_proj(audio2body_fea)
        # audio2body_fea_proj = self.position_embeddings(audio2body_fea_proj)
        # audio2body_fea_proj = speaker_motion_fea_proj + audio2body_fea_proj
        motion_fea = motion_fea + speaker_motion_fea_proj
        motion_fea = self.position_embeddings(motion_fea)
        audio2body_fea_cross = self.audio_motion_cross_attn(tgt=motion_fea.permute(1,0,2), memory=audio2body_fea_proj.permute(1,0,2)).permute(1,0,2)
        if not use_audio:
          audio2body_fea_cross = audio2body_fea_cross * 0.
        motion_fea = motion_fea + audio2body_fea_cross 

        # mlp
        upper_latent = self.motion2latent_upper(motion_fea)
        hands_latent = self.motion2latent_hands(motion_fea)
        lower_latent = self.motion2latent_lower(motion_fea)

        # refine
        motion_upper_refine = self.body_motion_decoder_upper(tgt=upper_latent.permute(1,0,2)+speaker_motion_fea_proj.permute(1,0,2), memory=(hands_latent+lower_latent).permute(1,0,2)).permute(1,0,2)
        motion_hands_refine = self.body_motion_decoder_hands(tgt=hands_latent.permute(1,0,2)+speaker_motion_fea_proj.permute(1,0,2), memory=(upper_latent+lower_latent).permute(1,0,2)).permute(1,0,2)
        motion_lower_refine = self.body_motion_decoder_lower(tgt=lower_latent.permute(1,0,2)+speaker_motion_fea_proj.permute(1,0,2), memory=(upper_latent+hands_latent).permute(1,0,2)).permute(1,0,2)
        upper_latent = self.motion_out_proj_upper(upper_latent + motion_upper_refine)
        hands_latent = self.motion_out_proj_hands(hands_latent + motion_hands_refine)
        lower_latent = self.motion_out_proj_lower(lower_latent + motion_lower_refine)

        # decode body
        classify_upper = self.motion_cls_upper(upper_latent)
        classify_hands = self.motion_cls_hands(hands_latent)
        classify_lower = self.motion_cls_lower(lower_latent)

        return {
            "rec_face": face_latent,
            "rec_upper": upper_latent,
            "rec_hands": hands_latent,
            "rec_lower": lower_latent,
            "cls_face": classify_face,
            "cls_upper": classify_upper,
            "cls_hands": classify_hands,
            "cls_lower": classify_lower,
        }
    
    def inference(self, audio, speaker_id, vq_model, masked_motion=None, mask=None):
        # generate default mask and masked motion if not provided
        length = audio.shape[1] * 30 // 16000
        bs = audio.shape[0]

        fake_axis_angle = torch.zeros(bs, length, 55, 3).to(audio.device)
        fake_motion = axis_angle_to_rotation_6d(fake_axis_angle).reshape(bs, length, -1)
        fake_foot_and_trans = torch.zeros(bs, length, 7).to(audio.device)
        fake_motion = torch.cat([fake_motion, fake_foot_and_trans], dim=-1) 
        if masked_motion is not None:
            fake_motion[:, :masked_motion.shape[1]] = masked_motion 
        masked_motion = fake_motion

        fake_mask = torch.ones_like(masked_motion)
        if mask is not None:
            fake_mask[:, :mask.shape[1]] = mask 
        mask = fake_mask

        # print(length, masked_motion.shape, mask.shape)
        
        # Autoregressive inference
        bs, total_len, c = masked_motion.shape
        window = self.cfg.pose_length
        pre_frames = self.cfg.seed_frames
        rounds = (total_len - pre_frames) // (window - pre_frames)
        remain = (total_len - pre_frames) % (window - pre_frames)
        
        rec_all_face = []
        rec_all_lower = []
        rec_all_upper = []
        rec_all_hands = []
        cls_all_face = []
        cls_all_lower = []
        cls_all_upper = []
        cls_all_hands = []
        
        last_motion = masked_motion[:, :pre_frames, :]
        for i in range(rounds):
            start_idx = i*(window - pre_frames)
            end_idx = start_idx + window

            window_mask = mask[:, start_idx:end_idx, :].clone()
            window_motion = masked_motion[:, start_idx:end_idx, :].clone()
            window_motion[:, :pre_frames, :] = torch.where(
                (window_mask[:, :pre_frames, :] == 0),
                masked_motion[:, start_idx:start_idx+pre_frames, :],
                last_motion,
            )
            window_mask[:, :pre_frames, :] = 0

            audio_slice_len = (end_idx - start_idx)*(16000//30)
            audio_slice = audio[:, start_idx*(16000//30) : start_idx*(16000//30)+audio_slice_len]
            # print(i, audio_slice.shape, speaker_id.shape, window_motion.shape, window_mask.shape)
            net_out_val = self.forward(audio_slice, speaker_id, masked_motion=window_motion, mask=window_mask, use_audio=True)
       
            _, cls_face =  torch.max(F.log_softmax(net_out_val["cls_face"], dim=2), dim=2)
            _, cls_upper =  torch.max(F.log_softmax(net_out_val["cls_upper"], dim=2), dim=2)
            _, cls_hands =  torch.max(F.log_softmax(net_out_val["cls_hands"], dim=2), dim=2)
            _, cls_lower =  torch.max(F.log_softmax(net_out_val["cls_lower"], dim=2), dim=2)

            face_latent = net_out_val["rec_face"] if self.cfg.lf > 0 and self.cfg.cf == 0 else None
            upper_latent = net_out_val["rec_upper"] if self.cfg.lu > 0 and self.cfg.cu == 0 else None
            hands_latent = net_out_val["rec_hands"] if self.cfg.lh > 0 and self.cfg.ch == 0 else None
            lower_latent = net_out_val["rec_lower"] if self.cfg.ll > 0 and self.cfg.cl == 0 else None
            face_index = cls_face if self.cfg.cf > 0 else None
            upper_index = cls_upper if self.cfg.cu > 0 else None
            hands_index = cls_hands if self.cfg.ch > 0 else None
            lower_index = cls_lower if self.cfg.cl > 0 else None

            decode_dict = vq_model.decode(
            face_latent=face_latent, upper_latent=upper_latent, lower_latent=lower_latent, hands_latent=hands_latent,
            face_index=face_index, upper_index=upper_index, lower_index=lower_index, hands_index=hands_index,)
            
            # decode_dict = vq_model.decode(face_latent=net_out_val["rec_face"], upper_index=net_out_val["cls_upper"], hands_index=net_out_val["cls_hands"], lower_index=net_out_val["cls_lower"])
            
            last_motion = decode_dict["all_motion4inference"][:, -pre_frames:, :]
            rec_all_face.append(net_out_val["rec_face"][:, :-pre_frames, :])
            rec_all_upper.append(net_out_val["rec_upper"][:, :-pre_frames, :])
            rec_all_hands.append(net_out_val["rec_hands"][:, :-pre_frames, :])
            rec_all_lower.append(net_out_val["rec_lower"][:, :-pre_frames, :])
            cls_all_face.append(net_out_val["cls_face"][:, :-pre_frames])
            cls_all_upper.append(net_out_val["cls_upper"][:, :-pre_frames])
            cls_all_hands.append(net_out_val["cls_hands"][:, :-pre_frames])
            cls_all_lower.append(net_out_val["cls_lower"][:, :-pre_frames])

        if remain > pre_frames:
            final_start = rounds*(window - pre_frames)
            final_end = final_start + pre_frames + remain

            final_mask = mask[:, final_start:final_end, :].clone()
            final_motion = masked_motion[:, final_start:final_end, :].clone()
            final_motion[:, :pre_frames, :] = torch.where(
                (final_mask[:, :pre_frames, :] == 0),
                masked_motion[:, final_start:final_start+pre_frames, :],
                last_motion,
            )
            final_mask[:, :pre_frames, :] = 0

            audio_slice_len = (final_end - final_start)*(16000//30)
            audio_slice = audio[:, final_start*(16000//30) : final_start*(16000//30)+audio_slice_len]
            net_out_val = self.forward(audio_slice, speaker_id, masked_motion=final_motion, mask=final_mask, use_audio=True)

            _, cls_face =  torch.max(F.log_softmax(net_out_val["cls_face"], dim=2), dim=2)
            _, cls_upper =  torch.max(F.log_softmax(net_out_val["cls_upper"], dim=2), dim=2)
            _, cls_hands =  torch.max(F.log_softmax(net_out_val["cls_hands"], dim=2), dim=2)
            _, cls_lower =  torch.max(F.log_softmax(net_out_val["cls_lower"], dim=2), dim=2)

            face_latent = net_out_val["rec_face"] if self.cfg.lf > 0 and self.cfg.cf == 0 else None
            upper_latent = net_out_val["rec_upper"] if self.cfg.lu > 0 and self.cfg.cu == 0 else None
            hands_latent = net_out_val["rec_hands"] if self.cfg.lh > 0 and self.cfg.ch == 0 else None
            lower_latent = net_out_val["rec_lower"] if self.cfg.ll > 0 and self.cfg.cl == 0 else None
            face_index = cls_face if self.cfg.cf > 0 else None
            upper_index = cls_upper if self.cfg.cu > 0 else None
            hands_index = cls_hands if self.cfg.ch > 0 else None
            lower_index = cls_lower if self.cfg.cl > 0 else None

            decode_dict = vq_model.decode(
            face_latent=face_latent, upper_latent=upper_latent, lower_latent=lower_latent, hands_latent=hands_latent,
            face_index=face_index, upper_index=upper_index, lower_index=lower_index, hands_index=hands_index,)

            rec_all_face.append(net_out_val["rec_face"])
            rec_all_upper.append(net_out_val["rec_upper"])
            rec_all_hands.append(net_out_val["rec_hands"])
            rec_all_lower.append(net_out_val["rec_lower"])
            cls_all_face.append(net_out_val["cls_face"])
            cls_all_upper.append(net_out_val["cls_upper"])
            cls_all_hands.append(net_out_val["cls_hands"])
            cls_all_lower.append(net_out_val["cls_lower"])

        rec_all_face = torch.cat(rec_all_face, dim=1) 
        rec_all_upper = torch.cat(rec_all_upper, dim=1) 
        rec_all_hands = torch.cat(rec_all_hands, dim=1) 
        rec_all_lower = torch.cat(rec_all_lower, dim=1) 
        cls_all_face = torch.cat(cls_all_face, dim=1)
        cls_all_upper = torch.cat(cls_all_upper, dim=1) 
        cls_all_hands = torch.cat(cls_all_hands, dim=1) 
        cls_all_lower = torch.cat(cls_all_lower, dim=1) 

        return {
            "rec_face": rec_all_face,
            "rec_upper": rec_all_upper,
            "rec_hands": rec_all_hands,
            "rec_lower": rec_all_lower,
            "cls_face": cls_all_face,
            "cls_upper": cls_all_upper,
            "cls_hands": cls_all_hands,
            "cls_lower": cls_all_lower,
        }


class StreamableAudioTokenModel(PreTrainedModel):
    """Audio-to-token predictor for the 4x-downsampled streamable VQ-VAEs.

    It predicts one code for every four 30 FPS motion frames from causal audio
    features and causal token attention.
    """

    config_class = EmageAudioConfig
    base_model_prefix = "streamable_audio_token"

    def __init__(self, config: EmageAudioConfig):
        super().__init__(config)
        self.cfg = config
        self.audio_encoder = CausalWavEncoder(self.cfg.audio_f)
        self.token_downsample = nn.Conv1d(
            self.cfg.audio_f,
            self.cfg.hidden_size,
            kernel_size=self.cfg.token_downsample_factor,
            stride=self.cfg.token_downsample_factor,
        )
        self.speaker_embedding = nn.Embedding(self.cfg.speaker_dims, self.cfg.hidden_size)
        token_length = self.cfg.pose_length // self.cfg.token_downsample_factor
        self.position_embeddings = PeriodicPositionalEncoding(
            self.cfg.hidden_size,
            period=token_length,
            max_seq_len=8192,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.hidden_size,
            nhead=4,
            dim_feedforward=self.cfg.hidden_size * 2,
        )
        self.token_encoder = nn.TransformerEncoder(layer, num_layers=self.cfg.token_transformer_layers)
        self.out_face = nn.Linear(self.cfg.hidden_size, self.cfg.vae_codebook_size)
        self.out_upper = nn.Linear(self.cfg.hidden_size, self.cfg.vae_codebook_size)
        self.out_hands = nn.Linear(self.cfg.hidden_size, self.cfg.vae_codebook_size)
        self.out_lower = nn.Linear(self.cfg.hidden_size, self.cfg.vae_codebook_size)

    def forward(self, audio, speaker_id):
        audio_features = self.audio_encoder(audio)
        token_features = self.token_downsample(audio_features.transpose(1, 2)).transpose(1, 2)
        token_features = self.position_embeddings(token_features)
        speaker_features = self.speaker_embedding(speaker_id).expand(-1, token_features.shape[1], -1)
        token_length = token_features.shape[1]
        causal_mask = torch.ones(token_length, token_length, dtype=torch.bool, device=audio.device).triu(1)
        token_features = self.token_encoder(
            (token_features + speaker_features).permute(1, 0, 2),
            mask=causal_mask,
        ).permute(1, 0, 2)
        return {
            "cls_face": self.out_face(token_features),
            "cls_upper": self.out_upper(token_features),
            "cls_hands": self.out_hands(token_features),
            "cls_lower": self.out_lower(token_features),
        }

    def inference(self, audio, speaker_id):
        return self(audio, speaker_id)


class RotaryPositionEmbedding(nn.Module):
    """RoPE applied to (bs, heads, T, head_dim) query/key tensors."""

    def __init__(self, head_dim, base=10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, offset=0):
        freqs = torch.outer(
            torch.arange(offset, offset + x.shape[-2], device=x.device, dtype=self.inv_freq.dtype),
            self.inv_freq,
        )  # (T, head_dim/2)
        cos = freqs.cos()[None, None]
        sin = freqs.sin()[None, None]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x1 * cos - x2 * sin
        out[..., 1::2] = x1 * sin + x2 * cos
        return out


class SelfAttention(nn.Module):
    """Multi-head self-attention with optional RoPE and optional causal mask."""

    def __init__(self, dim, nhead, rope=None, causal=True):
        super().__init__()
        self.nhead = nhead
        self.rope = rope
        self.causal = causal
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)

    def forward(self, x):
        bs, t, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(bs, t, self.nhead, -1).transpose(1, 2)
        k = k.view(bs, t, self.nhead, -1).transpose(1, 2)
        v = v.view(bs, t, self.nhead, -1).transpose(1, 2)
        if self.rope is not None:
            q, k = self.rope(q), self.rope(k)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal)
        return self.out(out.transpose(1, 2).reshape(bs, t, dim))

    def forward_step(self, x, cache=None):
        """One new position attends to all cached positions and itself."""
        bs, _, dim = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [a.view(bs, 1, self.nhead, -1).transpose(1, 2) for a in (q, k, v)]
        offset = 0 if cache is None else cache[0].shape[-2]
        if self.rope is not None:
            q, k = self.rope(q, offset), self.rope(k, offset)
        if cache is not None:
            k, v = torch.cat((cache[0], k), dim=-2), torch.cat((cache[1], v), dim=-2)
        # No future keys exist. is_causal=True with a single query would apply
        # an upper-left mask and incorrectly hide most of the cached prefix.
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.out(out.transpose(1, 2).reshape(bs, 1, dim)), (k, v)


class CausalAudioCrossAttention(nn.Module):
    """Cross-attention from token positions to the 1:1-aligned causal audio tokens."""

    def __init__(self, dim, nhead, rope):
        super().__init__()
        self.nhead = nhead
        self.rope = rope
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out = nn.Linear(dim, dim)

    def forward(self, x, audio):
        bs, t, dim = x.shape
        s = audio.shape[1]
        q = self.q_proj(x).view(bs, t, self.nhead, -1).transpose(1, 2)
        k, v = self.kv_proj(audio).chunk(2, dim=-1)
        k = k.view(bs, s, self.nhead, -1).transpose(1, 2)
        v = v.view(bs, s, self.nhead, -1).transpose(1, 2)
        q, k = self.rope(q), self.rope(k)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(out.transpose(1, 2).reshape(bs, t, dim))

    def forward_step(self, x, audio, cache=None):
        bs, _, dim = x.shape
        q = self.q_proj(x).view(bs, 1, self.nhead, -1).transpose(1, 2)
        k, v = self.kv_proj(audio).chunk(2, dim=-1)
        k, v = [a.view(bs, 1, self.nhead, -1).transpose(1, 2) for a in (k, v)]
        offset = 0 if cache is None else cache[0].shape[-2]
        q, k = self.rope(q, offset), self.rope(k, offset)
        if cache is not None:
            k, v = torch.cat((cache[0], k), dim=-2), torch.cat((cache[1], v), dim=-2)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        return self.out(out.transpose(1, 2).reshape(bs, 1, dim)), (k, v)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        return self.net(x)


class RegionBlock(nn.Module):
    """One region-expert block: causal self-attn + causal audio cross-attn + FFN."""

    def __init__(self, dim, nhead, rope):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = SelfAttention(dim, nhead, rope=rope, causal=True)
        self.norm2 = nn.LayerNorm(dim)
        self.audio_attn = CausalAudioCrossAttention(dim, nhead, rope)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dim * 2)

    def forward(self, x, audio):
        x = x + self.self_attn(self.norm1(x))
        x = x + self.audio_attn(self.norm2(x), audio)
        return x + self.ffn(self.norm3(x))

    def forward_step(self, x, audio, cache=None):
        self_cache, audio_cache = (None, None) if cache is None else cache
        y, self_cache = self.self_attn.forward_step(self.norm1(x), self_cache)
        x = x + y
        y, audio_cache = self.audio_attn.forward_step(self.norm2(x), audio, audio_cache)
        x = x + y
        return x + self.ffn(self.norm3(x)), (self_cache, audio_cache)


class FusionBlock(nn.Module):
    """Coordinates the four region streams: same-time region attention, causal
    temporal attention, audio cross-attention, FFN."""

    def __init__(self, dim, nhead, rope):
        super().__init__()
        self.norm_region = nn.LayerNorm(dim)
        self.region_attn = SelfAttention(dim, nhead, causal=False)
        self.norm_time = nn.LayerNorm(dim)
        self.time_attn = SelfAttention(dim, nhead, rope=rope, causal=True)
        self.norm_audio = nn.LayerNorm(dim)
        self.audio_attn = CausalAudioCrossAttention(dim, nhead, rope)
        self.norm_ffn = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dim * 2)

    def forward(self, h, audio):
        # h: (bs, N, R, D), audio: (bs, N, D)
        bs, n, r, d = h.shape
        x = h.reshape(bs * n, r, d)
        x = x + self.region_attn(self.norm_region(x))
        x = x.reshape(bs, n, r, d).permute(0, 2, 1, 3).reshape(bs * r, n, d)
        x = x + self.time_attn(self.norm_time(x))
        audio_rep = audio.unsqueeze(1).expand(bs, r, n, d).reshape(bs * r, n, d)
        x = x + self.audio_attn(self.norm_audio(x), audio_rep)
        x = x + self.ffn(self.norm_ffn(x))
        return x.reshape(bs, r, n, d).permute(0, 2, 1, 3)

    def forward_step(self, h, audio, cache=None):
        bs, _, r, d = h.shape
        time_cache, audio_cache = (None, None) if cache is None else cache
        x = h[:, 0]
        x = x + self.region_attn(self.norm_region(x))
        x = x.reshape(bs * r, 1, d)
        y, time_cache = self.time_attn.forward_step(self.norm_time(x), time_cache)
        x = x + y
        audio_rep = audio.unsqueeze(1).expand(bs, r, 1, d).reshape(bs * r, 1, d)
        y, audio_cache = self.audio_attn.forward_step(self.norm_audio(x), audio_rep, audio_cache)
        x = x + y
        x = x + self.ffn(self.norm_ffn(x))
        return x.reshape(bs, 1, r, d), (time_cache, audio_cache)


def shift_tokens_with_bos(targets, bos_id):
    """Right-shift per-region target tokens and prepend BOS as history input."""
    return {
        part: torch.cat(
            (torch.full_like(tokens[:, :1], bos_id), tokens[:, :-1]), dim=1
        )
        for part, tokens in targets.items()
    }


class CausalEmageAudioTokenModel(PreTrainedModel):
    """LiveGesture-style audio-to-token predictor over the frozen sVQ-VAE codes.

    Four lightweight region branches first model each part's token history
    (all sharing one causal audio encoder), then a fusion module coordinates
    the regions; four classification heads predict the next VQ token per part.
    Predicted tokens feed back as history; decoded motion is only an output.
    """

    PARTS = ("face", "upper", "hands", "lower")

    config_class = EmageAudioConfig
    base_model_prefix = "causal_emage_audio_token"

    def __init__(self, config: EmageAudioConfig):
        super().__init__(config)
        self.cfg = config
        dim = self.cfg.hidden_size
        nhead = getattr(self.cfg, "num_heads", 4)
        if dim % nhead != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if (dim // nhead) % 2 != 0:
            raise ValueError("RoPE requires an even head dimension")
        codebook = self.cfg.vae_codebook_size
        self.bos_id = codebook  # embedding row used for the BOS token

        self.audio_encoder = CausalMelAudioEncoder(
            dim,
            sr=self.cfg.audio_sr,
            hidden_f=self.cfg.audio_f,
            token_downsample_factor=self.cfg.token_downsample_factor,
            pose_fps=self.cfg.pose_fps,
        )
        self.speaker_embedding = nn.Embedding(self.cfg.speaker_dims, dim)
        # +1 row for the BOS token used to right-shift the teacher-forced history
        self.token_embedding = nn.ModuleDict(
            {part: nn.Embedding(codebook + 1, dim) for part in self.PARTS}
        )
        rope = RotaryPositionEmbedding(dim // nhead)
        self.region_branches = nn.ModuleDict(
            {
                part: nn.ModuleList(
                    RegionBlock(dim, nhead, rope)
                    for _ in range(getattr(self.cfg, "region_transformer_layers", 2))
                )
                for part in self.PARTS
            }
        )
        self.fusion = nn.ModuleList(
            FusionBlock(dim, nhead, rope)
            for _ in range(getattr(self.cfg, "fusion_layers", 2))
        )
        self.heads = nn.ModuleDict(
            {part: nn.Linear(dim, codebook) for part in self.PARTS}
        )

    def forward(self, audio, speaker_id, past_tokens):
        """Parallel teacher forcing over the complete BOS-prefixed sequence."""
        token_length = past_tokens[self.PARTS[0]].shape[1]
        audio_feat = self.audio_encoder(audio, target_tokens=token_length)
        if audio_feat.shape[1] != token_length:
            raise ValueError("Audio does not cover the complete token prefix")
        speaker = self.speaker_embedding(speaker_id).expand(-1, token_length, -1)
        region_features = []
        for part in self.PARTS:
            x = self.token_embedding[part](past_tokens[part]) + speaker
            for block in self.region_branches[part]:
                x = block(x, audio_feat)
            region_features.append(x)
        h = torch.stack(region_features, dim=2)
        for block in self.fusion:
            h = block(h, audio_feat)
        return {
            f"cls_{part}": self.heads[part](h[:, :, region_id])
            for region_id, part in enumerate(self.PARTS)
        }

    def _token_step(self, audio_feat, speaker_id, past_tokens, cache=None):
        """Predict one time position, retaining the complete prefix in KV caches."""
        if cache is None:
            cache = {
                "regions": {part: [None] * len(self.region_branches[part]) for part in self.PARTS},
                "fusion": [None] * len(self.fusion),
            }
        speaker = self.speaker_embedding(speaker_id)
        regions = {}
        features = []
        for part in self.PARTS:
            x = self.token_embedding[part](past_tokens[part]) + speaker
            regions[part] = []
            for block, block_cache in zip(self.region_branches[part], cache["regions"][part]):
                x, block_cache = block.forward_step(x, audio_feat, block_cache)
                regions[part].append(block_cache)
            features.append(x)
        h = torch.stack(features, dim=2)
        fusion = []
        for block, block_cache in zip(self.fusion, cache["fusion"]):
            h, block_cache = block.forward_step(h, audio_feat, block_cache)
            fusion.append(block_cache)
        logits = {
            f"cls_{part}": self.heads[part](h[:, :, region_id])
            for region_id, part in enumerate(self.PARTS)
        }
        return logits, {"regions": regions, "fusion": fusion}

    @torch.no_grad()
    def stream_step(self, audio_chunk, speaker_id, state=None, motion_vq=None):
        """Consume new mono samples (B, samples); return ready blocks and state.

        Each result contains one token per region and, when motion_vq is supplied,
        exactly token_downsample_factor decoded frames. Pass the returned state
        into the next call. A new utterance starts with state=None.

        Block B1 starts at frame 4 (for factor=4), after receiving audio B0.
        start_frame/end_frame are on the original audio timeline. In eval mode,
        cached inference matches a full causal-prefix forward pass.
        """
        if self.training:
            raise RuntimeError("Call model.eval() before streaming inference")
        if state is None:
            state = {
                "audio": None, "attention": None, "decoder": None, "step": 0,
                "tokens": {
                    part: torch.full((audio_chunk.shape[0], 1), self.bos_id,
                                     dtype=torch.long, device=audio_chunk.device)
                    for part in self.PARTS
                },
            }
        audio_tokens, audio_state = self.audio_encoder.forward_stream(audio_chunk, state["audio"])
        state = dict(state, audio=audio_state)
        results = []
        for i in range(audio_tokens.shape[1]):
            logits, attention = self._token_step(
                audio_tokens[:, i:i + 1], speaker_id, state["tokens"], state["attention"]
            )
            indices = {part: logits[f"cls_{part}"].argmax(dim=-1) for part in self.PARTS}
            start_frame = (state["step"] + 1) * self.cfg.token_downsample_factor
            result = {
                "logits": logits, "indices": indices, "start_frame": start_frame,
                "end_frame": start_frame + self.cfg.token_downsample_factor,
            }
            decoder = state["decoder"]
            if motion_vq is not None:
                result["motion"], decoder = motion_vq.decode_stream(indices, decoder)
            results.append(result)
            state = dict(state, attention=attention, decoder=decoder, tokens=indices,
                         step=state["step"] + 1)
        return results, state

    def inference_stream(self, audio, speaker_id, motion_vq=None, num_tokens=None):
        """Yield each forecast immediately; offline adapter for stream_step.

        A recorded clip returns B1..B(N-1), excluding the initial ungenerated
        block and forecasts beyond the clip. Live callers use stream_step.
        """
        block = self.cfg.token_downsample_factor
        if num_tokens is None:
            total_blocks = audio.shape[1] * self.cfg.pose_fps // (block * self.cfg.audio_sr)
            num_tokens = total_blocks - 1
        if num_tokens < 1:
            raise ValueError("Audio must contain at least two motion blocks")
        required = (num_tokens * block * self.cfg.audio_sr + self.cfg.pose_fps - 1) // self.cfg.pose_fps
        if audio.shape[1] < required:
            raise ValueError("Audio does not cover the requested token count")
        state = None
        start = 0
        for step in range(num_tokens):
            # Ceil the rational boundary: never emit before the interval ends.
            end = ((step + 1) * block * self.cfg.audio_sr + self.cfg.pose_fps - 1) // self.cfg.pose_fps
            results, state = self.stream_step(audio[:, start:end], speaker_id, state, motion_vq)
            yield from results
            start = end

    @torch.no_grad()
    def inference(self, audio, speaker_id, num_tokens=None):
        outputs = list(self.inference_stream(audio, speaker_id, num_tokens=num_tokens))
        return {
            f"cls_{part}": torch.cat([out["logits"][f"cls_{part}"] for out in outputs], dim=1)
            for part in self.PARTS
        }

    @torch.no_grad()
    def generate_motion(self, audio, speaker_id, motion_vq, num_tokens=None):
        """Collect streamed decoder outputs for saving or offline evaluation."""
        chunks = [out["motion"] for out in self.inference_stream(
            audio, speaker_id, motion_vq=motion_vq, num_tokens=num_tokens
        )]
        return {key: torch.cat([chunk[key] for chunk in chunks], dim=1) for key in chunks[0]}


