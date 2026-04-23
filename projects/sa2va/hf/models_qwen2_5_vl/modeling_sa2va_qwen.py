import torch
from torch import nn
from transformers import (AutoModel, GenerationConfig, Qwen2_5_VLForConditionalGeneration,
                          Qwen2ForCausalLM)
from transformers.modeling_utils import PreTrainedModel

from .configuration_sa2va_chat import Sa2VAChatConfigQwen

from .sam2 import SAM2

import numpy as np
from torchvision.transforms.functional import to_pil_image

import torch.nn.functional as F

from qwen_vl_utils import process_vision_info



class DirectResize:
    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        img = to_pil_image(image, mode='RGB')
        return np.array(img.resize((self.target_length, self.target_length)))

class Sa2VAChatModelQwen(PreTrainedModel):
    config_class = Sa2VAChatConfigQwen
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _no_split_modules = ['Qwen2_5_VisionTransformerPretrainedModel', 'Qwen2_5_VLDecoderLayer', 'SAM2']
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True



    def __init__(self, config: Sa2VAChatConfigQwen, model=None, use_flash_attn=True):
        super().__init__(config)
        self.extra_image_processor = DirectResize(target_length=1024, )

        self.min_pixels = 512 * 28 * 28
        self.max_pixels = 2048 * 28 * 28

        self.torch_dtype = torch.bfloat16

        if model is not None:
            self.model=model
        else:
            self.model = Qwen2_5_VLForConditionalGeneration(config)

        llm_hidden_size = config.text_config.hidden_size

        self.grounding_encoder = SAM2()
        out_dim = self.grounding_encoder.hidden_dim
        in_dim = llm_hidden_size
        self.text_hidden_fcs = nn.Sequential(
            nn.Linear(in_dim, in_dim), nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim), nn.Dropout(0.0)
        )

    @property
    def lm_head(self):
        return self.model.lm_head

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.model.get_output_embeddings()

    @staticmethod
    def _normalize_mask_prompts(mask_prompts):
        normalized = []
        for item in mask_prompts:
            tensor = torch.as_tensor(item, dtype=torch.float32)
            if tensor.ndim == 2:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 3:
                raise ValueError(f"mask_prompts must have shape (n_regions, h, w), got {tuple(tensor.shape)}")
            normalized.append(tensor)
        return normalized

    def _resize_mask_prompts(self, mask_prompts, image_grid_thw):
        normalized = self._normalize_mask_prompts(mask_prompts)
        if len(normalized) != int(image_grid_thw.shape[0]):
            raise ValueError(
                f"mask_prompts/image_grid_thw count mismatch: {len(normalized)} vs {int(image_grid_thw.shape[0])}"
            )

        spatial_merge_size = int(self.model.visual.spatial_merge_size)
        resized_masks = []
        region_token_counts = []
        for image_masks, grid_thw in zip(normalized, image_grid_thw):
            temporal = int(grid_thw[0].item())
            if temporal != 1:
                raise ValueError("mask_prompts currently only support single-image inference for Qwen Sa2VA.")
            target_h = int(grid_thw[1].item() // spatial_merge_size)
            target_w = int(grid_thw[2].item() // spatial_merge_size)
            resized = F.interpolate(
                image_masks.unsqueeze(0),
                size=(target_h, target_w),
                mode='nearest',
            ).squeeze(0).bool()
            resized_masks.append(resized)
            region_token_counts.extend(int(mask.sum().item()) for mask in resized)
        return resized_masks, region_token_counts

    @staticmethod
    def _build_visual_prompt_text(region_token_counts):
        if not region_token_counts:
            return ''

        vp_token_str = '\nThere are {} part regions in the picture: '.format(len(region_token_counts))
        for idx, pixels in enumerate(region_token_counts):
            vp_token_str += f"region{idx + 1}" + ('<vp>' * pixels) + '</vp>'
            vp_token_str += '.\n' if idx == len(region_token_counts) - 1 else ', '
        return vp_token_str

    def _inject_visual_prompt_embeds(self, mm_inputs, resized_mask_prompts):
        vp_token_id = self.processor.tokenizer.convert_tokens_to_ids('<vp>')
        if vp_token_id is None or vp_token_id < 0:
            raise ValueError("Tokenizer does not provide a valid <vp> token id.")

        input_ids = mm_inputs['input_ids']
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        image_embeds = self.model.get_image_features(mm_inputs['pixel_values'], mm_inputs['image_grid_thw'])

        vp_embeds = []
        for image_embed, image_masks in zip(image_embeds, resized_mask_prompts):
            flat_image_embed = image_embed.reshape(-1, image_embed.shape[-1])
            flat_masks = image_masks.to(device=flat_image_embed.device, dtype=torch.bool).reshape(image_masks.shape[0], -1)
            if flat_image_embed.shape[0] != flat_masks.shape[-1]:
                raise ValueError(
                    f"Resized mask/image feature mismatch: {flat_masks.shape[-1]} vs {flat_image_embed.shape[0]}"
                )
            for region_mask in flat_masks:
                vp_embeds.append(flat_image_embed[region_mask])

        expected_vp_tokens = int((input_ids == vp_token_id).sum().item())
        if vp_embeds:
            vp_embeds = torch.cat(vp_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        else:
            vp_embeds = inputs_embeds.new_empty((0, inputs_embeds.shape[-1]))

        if vp_embeds.shape[0] != expected_vp_tokens:
            raise ValueError(
                f"Visual prompt embedding count mismatch: expected {expected_vp_tokens}, got {vp_embeds.shape[0]}"
            )

        if expected_vp_tokens == 0:
            return inputs_embeds

        vp_mask = (input_ids == vp_token_id).unsqueeze(-1).expand_as(inputs_embeds)
        return inputs_embeds.masked_scatter(vp_mask, vp_embeds)

    def predict_forward(
            self,
            image=None,
            video=None,
            text=None,
            past_text='',
            mask_prompts=None,
            tokenizer=None,
            processor=None,
    ):
        assert processor is not None
        self.processor = processor
        inputs_embeds = None

        self.seg_token_idx = self.processor.tokenizer.convert_tokens_to_ids('[SEG]')

        text = text.replace('<image>', "")

        if image is None and video is None and '<image>' not in past_text:
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": past_text + text},
                    ],
                }
            ]

            # Preparation for inference
            processsed_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            mm_inputs = self.processor(
                text=[processsed_text],
                images=None,
                videos=None,
                padding=True,
                return_tensors="pt",
            )
            mm_inputs = mm_inputs.to(self.device)

            ret_masks = []
        else:
            input_dict = {}
            if video is not None:
                pixel_values = []
                extra_pixel_values = []
                images = []
                content = []
                ori_image_size = video[0].size
                for frame_idx, frame_image in enumerate(video):
                    # assert ori_image_size == frame_image.size
                    g_image = np.array(frame_image)  # for grounding
                    g_image = self.extra_image_processor.apply_image(g_image)
                    g_image = torch.from_numpy(g_image).permute(2, 0, 1).contiguous()
                    extra_pixel_values.append(g_image)
                    if frame_idx < 5:
                        content.append({"type": "image", "image": frame_image},)


                content.append({"type": "text", "text": text})
                messages = [
                    {
                        "role": "user",
                        "content": content,
                    }
                ]

                # Preparation for inference
                processsed_text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                image_inputs, video_inputs = process_vision_info(messages)
                mm_inputs = self.processor(
                    text=[processsed_text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels
                )
                mm_inputs = mm_inputs.to(self.device)

                g_pixel_values = torch.stack([
                    self.grounding_encoder.preprocess_image(pixel) for pixel in extra_pixel_values
                ]).to(self.torch_dtype)

                num_frames = min(5, len(video))

            else:
                ori_image_size = image.size

                # prepare grounding images
                g_image = np.array(image)  # for grounding
                g_image = self.extra_image_processor.apply_image(g_image)
                g_pixel_values = torch.from_numpy(g_image).permute(2, 0, 1).contiguous().to(self.torch_dtype)
                extra_pixel_values = [g_pixel_values]
                g_pixel_values = torch.stack([
                    self.grounding_encoder.preprocess_image(pixel) for pixel in extra_pixel_values
                ]).to(self.torch_dtype)

                user_text = text
                if mask_prompts is not None:
                    probe_messages = [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "image": image,
                                },
                                {"type": "text", "text": user_text},
                            ],
                        }
                    ]
                    probe_text = self.processor.apply_chat_template(
                        probe_messages, tokenize=False, add_generation_prompt=True
                    )
                    probe_image_inputs, probe_video_inputs = process_vision_info(probe_messages)
                    probe_mm_inputs = self.processor(
                        text=[probe_text],
                        images=probe_image_inputs,
                        videos=probe_video_inputs,
                        padding=True,
                        return_tensors="pt",
                        min_pixels=self.min_pixels,
                        max_pixels=self.max_pixels
                    )
                    _, region_token_counts = self._resize_mask_prompts(
                        mask_prompts,
                        probe_mm_inputs['image_grid_thw'],
                    )
                    user_text = self._build_visual_prompt_text(region_token_counts) + user_text

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": image,
                            },
                            {"type": "text", "text": user_text},
                        ],
                    }
                ]

                # Preparation for inference
                processsed_text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                image_inputs, video_inputs = process_vision_info(messages)
                mm_inputs = self.processor(
                    text=[processsed_text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                    min_pixels=self.min_pixels,
                    max_pixels=self.max_pixels
                )
                mm_inputs = mm_inputs.to(self.device)

                if mask_prompts is not None:
                    resized_mask_prompts, _ = self._resize_mask_prompts(
                        mask_prompts,
                        mm_inputs['image_grid_thw'],
                    )
                    inputs_embeds = self._inject_visual_prompt_embeds(
                        mm_inputs,
                        resized_mask_prompts,
                    )

                num_frames = 1
            
            input_dict['g_pixel_values'] = g_pixel_values
            ret_masks = []

        generate_output = self.model.generate(
            **mm_inputs,
            inputs_embeds=inputs_embeds,
            max_new_tokens=2048,
            do_sample=False,
            output_hidden_states=True,
            return_dict_in_generate=True
        )

        generate_output_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(mm_inputs.input_ids, generate_output.sequences)
        ]

        predict = self.processor.batch_decode(generate_output_trimmed, skip_special_tokens=False)[0].strip()

        if image is None and video is None and '<image>' not in past_text:
            return {'prediction': predict, 'prediction_masks': ret_masks, }

        # if have seg result, find the seg hidden states
        hidden_states = generate_output.hidden_states
        last_hidden_states = [item[-1][0] for item in hidden_states]
        last_hidden_states = torch.cat(last_hidden_states, dim=0)
        seg_hidden_states = get_seg_hidden_states(
            last_hidden_states, generate_output.sequences[0][:-1],
            seg_id=self.seg_token_idx
        )
        all_seg_hidden_states = self.text_hidden_fcs(seg_hidden_states)

        for seg_hidden_states in all_seg_hidden_states:
            seg_hidden_states = seg_hidden_states.unsqueeze(0)
            g_pixel_values = input_dict['g_pixel_values']
            sam_states = self.grounding_encoder.get_sam2_embeddings(g_pixel_values)
            pred_masks = self.grounding_encoder.language_embd_inference(sam_states, [seg_hidden_states] * num_frames)
            w, h = ori_image_size
            masks = F.interpolate(pred_masks, size=(h, w), mode='bilinear', align_corners=False)
            masks = masks[:, 0]
            masks = masks.sigmoid() > 0.5
            masks = masks.cpu().numpy()
            ret_masks.append(masks)

        return {'prediction': predict, 'prediction_masks': ret_masks,}

def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    if n_out == 0:
        return hidden_states[0:0]
    return hidden_states[-n_out:][seg_mask]