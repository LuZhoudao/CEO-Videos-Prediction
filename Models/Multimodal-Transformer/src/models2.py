import numpy as np
import torch
from timm.models.layers import trunc_normal_
from torch import nn as nn
import torch.nn.functional as F
from src.model.utils.utils import normalize_embeddings
from src.model.utils.layers import get_projection
from src.model.utils.fusion_transformer import FusionTransformer
from src.model.utils.davenet import load_DAVEnet


class MULTModel(nn.Module):
    def __init__(self,
                 video_embed_dim=768,
                 text_embed_dim=768,
                 audio_embed_dim=128,
                 #fusion_params,
                 video_max_tokens=None,
                 text_max_tokens=None,
                 audio_max_tokens=None,
                 #audio_max_num_STFT_frames=None,
                 projection_dim=4096,
                 token_projection='gated',
                 projection='gated',
                 two_modal=True,
                 three_modal=False,
                 strategy_audio_pooling='none',
                 # davenet_v2=True,
                 individual_projections=True,
                 use_positional_emb=False,
                 ):
        super().__init__()

        #self.fusion = FusionTransformer(**fusion_params)
        self.fusion = FusionTransformer(embed_dim=4096, 
                                        use_cls_token=False, 
                                        depth=1,
                                        num_heads=64,
                                        mlp_ratio=1)

        self.individual_projections = individual_projections
        self.use_positional_emb = use_positional_emb
        self.strategy_audio_pooling = strategy_audio_pooling
        self.two_modal = two_modal
        self.three_modal = three_modal

        embed_dim = 4096 #fusion_params['embed_dim']
        #audio_embed_dim = 128

        self.video_norm_layer = nn.LayerNorm(embed_dim, eps=1e-6)
        self.text_norm_layer = nn.LayerNorm(embed_dim, eps=1e-6)
        self.audio_norm_layer = nn.LayerNorm(embed_dim, eps=1e-6)

        # Projection layers
        output_dim = 1
        self.proj1 = nn.Linear(embed_dim, embed_dim)
        self.proj2 = nn.Linear(embed_dim, embed_dim)
        self.out_layer = nn.Linear(embed_dim, output_dim)

        # audio token preprocess
        #self.davenet = load_DAVEnet(v2=davenet_v2)

        # if audio_max_num_STFT_frames is not None:
        #     if davenet_v2:
        #         audio_max_tokens = int(audio_max_num_STFT_frames / 64)
        #     else:
        #         audio_max_tokens = int(audio_max_num_STFT_frames / 16)
        #     self.audio_max_tokens = audio_max_tokens
        # else:
        #     self.audio_max_tokens = None

        if self.use_positional_emb:
            assert video_max_tokens is not None
            assert text_max_tokens is not None
            #assert audio_max_num_STFT_frames is not None
            self.video_pos_embed = nn.Parameter(torch.zeros(1, video_max_tokens, embed_dim))
            self.text_pos_embed = nn.Parameter(torch.zeros(1, text_max_tokens, embed_dim))
            self.audio_pos_embed = nn.Parameter(torch.zeros(1, audio_max_tokens, embed_dim))
        else:
            self.video_pos_embed = None
            self.text_pos_embed = None
            self.audio_pos_embed = None

        #audio_embed_dim = 4096 if davenet_v2 else 1024
        self.video_token_proj = get_projection(video_embed_dim, embed_dim, token_projection)
        self.text_token_proj = get_projection(text_embed_dim, embed_dim, token_projection)
        self.audio_token_proj = get_projection(audio_embed_dim, embed_dim, token_projection)

        if not self.individual_projections:
            self.proj = get_projection(embed_dim, projection_dim, projection)
        else:
            self.video_proj = get_projection(embed_dim, projection_dim, projection)
            self.text_proj = get_projection(embed_dim, projection_dim, projection)
            self.audio_proj = get_projection(embed_dim, projection_dim, projection)

        self.inp_size = 128
        self.hidden_size = 128
        self.mid_layers = 1
        self.out_dropout = 0.0
       
        self.lstm = nn.LSTM(input_size =self.inp_size, hidden_size=self.hidden_size, num_layers=1, batch_first=True)



        self.init_weights()

    def init_weights(self):
        for weights in [self.video_pos_embed, self.audio_pos_embed, self.text_pos_embed]:
            if weights is not None:
                trunc_normal_(weights, std=.02)

    def _check_and_fix_if_input_empty(self, x, attention_mask):
        nonempty_input_mask = attention_mask.sum(-1) != 0

        # if all tokens of modality is empty, add one masking token
        empty_input_mask = nonempty_input_mask == 0
        # n_masking_tokens = 1
        # x[empty_input_mask, :n_masking_tokens] = self.fusion.masking_token.type(x.dtype)
        # attention_mask[empty_input_mask, :n_masking_tokens] = 1
        return x, attention_mask, nonempty_input_mask

    def extract_video_tokens(self, video, attention_mask):
        x = self.video_token_proj(video)
        x = self.video_norm_layer(x)

        x, attention_mask, nonempty_input_mask = self._check_and_fix_if_input_empty(x, attention_mask)
        special_token_mask = attention_mask == 0

        return {'all_tokens': x, 'attention_mask': attention_mask, 'special_token_mask': special_token_mask,
                'nonempty_input_mask': nonempty_input_mask}

    def extract_audio_tokens(self, audio, attention_mask):
        # new audio LSTM
        audio = self.audio_token_proj(audio)
        audio = self.audio_norm_layer(audio)
 
        audio, attention_mask, nonempty_input_mask = self._check_and_fix_if_input_empty(audio, attention_mask)
        special_token_mask = attention_mask == 0
        return {'all_tokens': audio, 'attention_mask': attention_mask, 'special_token_mask': special_token_mask,
                'nonempty_input_mask': nonempty_input_mask}

    def extract_text_tokens(self, text, attention_mask):
        # print("text", text.shape)
        x = self.text_token_proj(text)
        x = self.text_norm_layer(x)
        x, attention_mask, nonempty_input_mask = self._check_and_fix_if_input_empty(x, attention_mask)
        special_token_mask = attention_mask == 0
        return {'all_tokens': x, 'attention_mask': attention_mask, 'special_token_mask': special_token_mask,
                'nonempty_input_mask': nonempty_input_mask}

    def forward(self, text, audio, video, text_mask, audio_mask, video_mask, force_cross_modal, modal_lst):
        batch_size, seq_len = audio.shape[0], audio.shape[1]
        output, _ = self.lstm(audio.view(batch_size*seq_len, audio.shape[2], audio.shape[-1]))
        audio = output[:, -1, :].view(batch_size, seq_len, -1)
        # print("mask", audio_mask.shape)
        middle = {}

        text_raw_embed = self.extract_text_tokens(text, text_mask)
        video_raw_embed = self.extract_video_tokens(video, video_mask)
        audio_raw_embed = self.extract_audio_tokens(audio, audio_mask)
        middle['text_nonempty_input_mask'] = text_raw_embed['nonempty_input_mask']
        middle['video_nonempty_input_mask'] = video_raw_embed['nonempty_input_mask']
        middle['audio_nonempty_input_mask'] = audio_raw_embed['nonempty_input_mask']

        # add positional embedding after masking
        if self.use_positional_emb:
            text_raw_embed['all_tokens'] = text_raw_embed['all_tokens'] + self.text_pos_embed
            video_raw_embed['all_tokens'] = video_raw_embed['all_tokens'] + self.video_pos_embed
            audio_raw_embed['all_tokens'] = audio_raw_embed['all_tokens'] + self.audio_pos_embed

        if self.individual_projections:
            text_proj, video_proj, audio_proj = self.text_proj, self.video_proj, self.audio_proj
        else:
            text_proj, video_proj, audio_proj = self.proj, self.proj, self.proj

        if len(modal_lst) == 1:
            if "T" in modal_lst:
                text = self.fusion(text=text_raw_embed)['text']
                middle["output"] = text_proj(text['embed'])
            elif "V" in modal_lst:
                video = self.fusion(video=video_raw_embed)['video']
                middle["output"] = video_proj(video['embed'])
            elif "A" in modal_lst:
                audio = self.fusion(audio=audio_raw_embed)['audio']
                middle["output"] = audio_proj(audio['embed'])


        elif len(modal_lst) == 2:
            if "T" in modal_lst and "V" in modal_lst:
                tv = self.fusion(text=text_raw_embed,
                                 video=video_raw_embed)
                middle["output"] = (normalize_embeddings(text_proj(tv['text']['embed'])) +
                                      normalize_embeddings(video_proj(tv['video']['embed']))) / 2
            elif "T" in modal_lst and "A" in modal_lst:
                ta = self.fusion(text=text_raw_embed,
                                 audio=audio_raw_embed)
                middle["output"] = (normalize_embeddings(text_proj(ta['text']['embed'])) +
                                      normalize_embeddings(audio_proj(ta['audio']['embed']))) / 2
            elif "A" in modal_lst and "V" in modal_lst:
                va = self.fusion(video=video_raw_embed,
                                 audio=audio_raw_embed)
                middle["output"] = (normalize_embeddings(video_proj(va['video']['embed'])) +
                                      normalize_embeddings(audio_proj(va['audio']['embed']))) / 2

        elif len(modal_lst) == 3:
            vat =  self.fusion(text=text_raw_embed,
                               video=video_raw_embed,
                               audio=audio_raw_embed)
            middle["output"] = (normalize_embeddings(video_proj(vat['video']['embed'])) +
                                normalize_embeddings(audio_proj(vat['audio']['embed']))+
                                normalize_embeddings(text_proj(vat['text']['embed']))) / 3



        # if force_cross_modal:
        #     #  needed for ablation
        #     middle["t+v_embed"] = (normalize_embeddings(middle["text_embed"]) +
        #                            normalize_embeddings(middle["video_embed"])) / 2
        #     middle["t+a_embed"] = (normalize_embeddings(middle["text_embed"]) +
        #                            normalize_embeddings(middle["audio_embed"])) / 2
        #     middle["v+a_embed"] = (normalize_embeddings(middle["video_embed"]) +
        #                            normalize_embeddings(middle["audio_embed"])) / 2

 
        last_hs_proj = self.proj2(F.dropout(F.relu(self.proj1(middle["output"])), p=self.out_dropout, training=self.training))
        last_hs_proj += middle["output"]

        output = self.out_layer(last_hs_proj)


        return output, middle["output"]


# class EverythingAtOnceModel_TV_Only(EverythingAtOnceModel):
#     def forward(self, data, force_cross_modal=False):
#         output = {}
#
#         text_raw_embed = self.extract_text_tokens(data['text'], data['text_mask'])
#         video_raw_embed = self.extract_video_tokens(data['video'], data['video_mask'])
#         audio_raw_embed = self.extract_audio_tokens(data['audio'], data['audio_mask'], data['audio_STFT_nframes'])
#         output['text_nonempty_input_mask'] = text_raw_embed['nonempty_input_mask']
#         output['video_nonempty_input_mask'] = video_raw_embed['nonempty_input_mask']
#         output['audio_nonempty_input_mask'] = audio_raw_embed['nonempty_input_mask']
#
#         # add positional embedding after masking
#         if self.use_positional_emb:
#             text_raw_embed['all_tokens'] = text_raw_embed['all_tokens'] + self.text_pos_embed
#             video_raw_embed['all_tokens'] = video_raw_embed['all_tokens'] + self.video_pos_embed
#
#         text = self.fusion(text=text_raw_embed)['text']
#         video = self.fusion(video=video_raw_embed)['video']
#
#         if not self.individual_projections:
#             output["text_embed"] = self.proj(text['embed'])
#             output["video_embed"] = self.proj(video['embed'])
#         else:
#             output["text_embed"] = self.text_proj(text['embed'])
#             output["video_embed"] = self.video_proj(video['embed'])
#         return output
#
#
# class TransformerPerModalityModel(EverythingAtOnceModel):
#     def __init__(self,
#                  video_embed_dim,
#                  text_embed_dim,
#                  fusion_params,
#                  video_max_tokens=None,
#                  text_max_tokens=None,
#                  audio_max_num_STFT_frames=None,
#                  projection_dim=6144,
#                  token_projection='gated',
#                  projection='gated',
#                  strategy_audio_pooling='none',
#                  davenet_v2=True,
#                  use_positional_emb=False,
#                  ):
#         super().__init__(video_embed_dim,
#                          text_embed_dim,
#                          fusion_params,
#                          video_max_tokens=video_max_tokens,
#                          text_max_tokens=text_max_tokens,
#                          audio_max_num_STFT_frames=audio_max_num_STFT_frames,
#                          projection_dim=projection_dim,
#                          token_projection=token_projection,
#                          projection=projection,
#                          cross_modal=False,
#                          strategy_audio_pooling=strategy_audio_pooling,
#                          davenet_v2=davenet_v2,
#                          individual_projections=True,
#                          use_positional_emb=use_positional_emb,
#                          )
#
#         self.fusion_text = self.fusion
#         self.fusion_video = FusionTransformer(**fusion_params)
#         self.fusion_audio = FusionTransformer(**fusion_params)
#
#     def forward(self, data, force_cross_modal=False):
#         output = {}
#
#         text_raw_embed = self.extract_text_tokens(data['text'], data['text_mask'])
#         video_raw_embed = self.extract_video_tokens(data['video'], data['video_mask'])
#         audio_raw_embed = self.extract_audio_tokens(data['audio'], data['audio_mask'], data['audio_STFT_nframes'])
#
#         output['text_nonempty_input_mask'] = text_raw_embed['nonempty_input_mask']
#         output['video_nonempty_input_mask'] = video_raw_embed['nonempty_input_mask']
#         output['audio_nonempty_input_mask'] = audio_raw_embed['nonempty_input_mask']
#
#         text = self.fusion_text(text=text_raw_embed)['text']
#         output["text_embed"] = self.text_proj(text['embed'])
#
#         video = self.fusion_video(video=video_raw_embed)['video']
#         output["video_embed"] = self.video_proj(video['embed'])
#
#         audio = self.fusion_audio(audio=audio_raw_embed)['audio']
#         output["audio_embed"] = self.audio_proj(audio['embed'])
#
#         if force_cross_modal:
#             #  needed for ablation
#             output["t+v_embed"] = (normalize_embeddings(output["text_embed"]) +
#                                    normalize_embeddings(output["video_embed"])) / 2
#             output["t+a_embed"] = (normalize_embeddings(output["text_embed"]) +
#                                    normalize_embeddings(output["audio_embed"])) / 2
#             output["v+a_embed"] = (normalize_embeddings(output["video_embed"]) +
#                                    normalize_embeddings(output["audio_embed"])) / 2
#
#         return output
#
#
# def create_audio_tokens(audio, audio_mask, audio_STFT_nframes, n_tokens, strategy='avg_pool'):
#     if torch.is_tensor(audio_STFT_nframes):
#         audio_STFT_nframes = int(audio_STFT_nframes.cpu().item())
#     if strategy == 'clip':
#         return audio[:n_tokens], audio_mask[:n_tokens]
#     elif strategy == 'nearest':
#         if audio_STFT_nframes <= n_tokens:
#             return create_audio_tokens(audio, audio_mask, audio_STFT_nframes, n_tokens, strategy='clip')
#         audio = audio[:audio_STFT_nframes]
#         audio = torch.nn.functional.interpolate(
#             audio.permute(1, 0).unsqueeze(0),
#             size=n_tokens,
#             mode='nearest').squeeze(0).permute(1, 0)
#         return audio, audio_mask[:n_tokens]
#     elif strategy == 'max_pool':
#         if audio_STFT_nframes <= n_tokens:
#             return create_audio_tokens(audio, audio_mask, audio_STFT_nframes, n_tokens, strategy='clip')
#         audio = audio[:audio_STFT_nframes]
#         audio = torch.nn.functional.adaptive_max_pool1d(
#             audio.permute(1, 0).unsqueeze(0),
#             output_size=n_tokens).squeeze(0).permute(1, 0)
#         return audio, audio_mask[:n_tokens]
#     elif strategy == 'avg_pool':
#         if audio_STFT_nframes <= n_tokens:
#             return create_audio_tokens(audio, audio_mask, audio_STFT_nframes, n_tokens, strategy='clip')
#         audio = audio[:audio_STFT_nframes]
#         audio = torch.nn.functional.adaptive_avg_pool1d(
#             audio.permute(1, 0).unsqueeze(0),
#             output_size=n_tokens).squeeze(0).permute(1, 0)
#         return audio, audio_mask[:n_tokens]
#     elif strategy == 'none':
#         return audio, audio_mask
#     else:
#         raise NotImplementedError