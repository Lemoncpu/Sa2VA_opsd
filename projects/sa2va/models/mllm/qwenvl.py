from collections import OrderedDict
from importlib.resources import path
from typing import Dict, Optional, Union, List


import torch
import torch.nn.functional as F
from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2_5_VLProcessor
from peft import PeftModelForCausalLM, get_peft_model, prepare_model_for_kbit_training


from xtuner.registry import BUILDER
from xtuner.model.utils import get_peft_model_state_dict
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from mmengine import print_log

class Qwen2_5_VL(BaseModel):
    r"""
    Qwen2.5-VL: Adapter for the Qwen2.5-VL model.
    Goal: Enable the training within the xtuner framework.
    """
    def __init__(
            self,
            model_path: str,
            freeze_llm: bool = False,
            freeze_visual_encoder: bool = False,
            llm_lora: Optional[dict] = None,
            pretrained_pth: Optional[str] = None
        ):
        super().__init__()

        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = freeze_visual_encoder
        self.use_llm_lora = llm_lora is not None


        # Note:
        # force to use flash_attention_2 and bfloat16 for training Qwen2.5-VL
        # for better acceleration and memory saving.
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            trust_remote_code=True
        )

        # self.model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        # "<|image_pad|>" is used to pad the image tokens to a fixed length.
        # This is consistent in the qwen2.5-vl model.
        img_context_token_id = tokenizer.convert_tokens_to_ids('<|image_pad|>')
        self.img_context_token_id = img_context_token_id

        if self.freeze_llm:
            self.model.language_model.requires_grad_(False)
        if self.freeze_visual_encoder:
            self.model.visual.requires_grad_(False)


        if self.use_llm_lora:
            self.llm_lora_config = llm_lora
            print_log(f'Qwen2_5_VL: Using Lora for the LLM with config {self.llm_lora_config} (delay the lora please call manual)', logger='current')

        self.tokenizer = None
        self.processor = None

    def add_special_tokens(self, tokenizer, special_tokens: List[str]) -> None:
        """Add special tokens to the tokenizer and resize embeddings if needed."""
        print_log(f'{self.__class__.__name__}:add_special_tokens [Before] The total number of tokens is now {len(tokenizer)}', logger='current')
        num_new_tokens = tokenizer.add_tokens(special_tokens, special_tokens=True)
        if num_new_tokens > 0:
            self.model.resize_token_embeddings(len(tokenizer))
            print_log(f'{self.__class__.__name__}:add_special_tokens Added {num_new_tokens} special tokens', logger='current')
            print_log(f'{self.__class__.__name__}:add_special_tokens [After] The total number of tokens is now {len(tokenizer)}', logger='current')
        self.tokenizer = tokenizer

    def _init_processor(self, image_processor, video_processor):
        self.processor = Qwen2_5_VLProcessor(
            image_processor=image_processor,
            tokenizer=self.tokenizer,
            video_processor=video_processor
        )
        
    def _parse_lora_config(self, lora_config):
        if isinstance(lora_config, dict) or isinstance(
                lora_config, Config) or isinstance(lora_config, ConfigDict):
            lora_config = BUILDER.build(lora_config)
        return lora_config

    def _prepare_llm_for_lora(self,
                              lora_config,
                              use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        self.model = prepare_model_for_kbit_training(self.model, use_activation_checkpointing)
        if lora_config.target_modules is None:
            _target_modules = []
            for name, module in self.model.language_model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    _target_modules.append('language_model.' + name)
            lora_config.target_modules = _target_modules
        self.model = get_peft_model(self.model, lora_config)

        self.model.print_trainable_parameters()


    def manual_prepare_llm_for_lora(self):
        if self.use_llm_lora:
          self._prepare_llm_for_lora(self.llm_lora_config)


    def get_embedding_size(self):
        return self.model.config.text_config.hidden_size

    @staticmethod
    def _normalize_prompt_masks(prompt_masks) -> torch.Tensor:
        if isinstance(prompt_masks, torch.Tensor):
            tensor = prompt_masks.to(dtype=torch.float32)
        else:
            tensor = torch.as_tensor(prompt_masks, dtype=torch.float32)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 3:
            raise ValueError(f"prompt_masks must have shape (n_regions, h, w), got {tuple(tensor.shape)}")
        return tensor

    def _resolve_prompt_masks_per_image(
        self,
        prompt_masks,
        vp_overall_mask,
        image_grid_thw_per_sample,
    ):
        total_images = sum(int(item.shape[0]) for item in image_grid_thw_per_sample)
        prompt_masks_per_image = [None] * total_images
        if prompt_masks is None:
            return prompt_masks_per_image

        if vp_overall_mask is not None:
            vp_overall_mask = vp_overall_mask.to(dtype=torch.bool).view(-1)
            if int(vp_overall_mask.numel()) == total_images:
                prompt_mask_idx = 0
                image_offset = 0
                for sample_grid_thw in image_grid_thw_per_sample:
                    sample_image_count = int(sample_grid_thw.shape[0])
                    sample_flags = vp_overall_mask[image_offset:image_offset + sample_image_count]
                    flagged_indices = torch.nonzero(sample_flags, as_tuple=False).flatten().tolist()
                    if flagged_indices:
                        if len(flagged_indices) != 1:
                            raise ValueError(
                                f"Qwen visual prompt training currently supports one prompted image per sample, got {flagged_indices}."
                            )
                        if prompt_mask_idx >= len(prompt_masks):
                            raise ValueError("vp_overall_mask expects more prompt_masks entries than provided.")
                        prompt_masks_per_image[image_offset + flagged_indices[0]] = self._normalize_prompt_masks(
                            prompt_masks[prompt_mask_idx]
                        )
                        prompt_mask_idx += 1
                    image_offset += sample_image_count

                if prompt_mask_idx != len(prompt_masks):
                    raise ValueError(
                        f"Unused prompt_masks entries remain after vp_overall_mask mapping: "
                        f"used {prompt_mask_idx}, total {len(prompt_masks)}."
                    )
                return prompt_masks_per_image

        if len(prompt_masks) == len(image_grid_thw_per_sample):
            image_offset = 0
            for sample_idx, sample_grid_thw in enumerate(image_grid_thw_per_sample):
                sample_image_count = int(sample_grid_thw.shape[0])
                if sample_image_count != 1:
                    raise ValueError(
                        "Without vp_overall_mask mapping, Qwen visual prompt training expects one image per sample."
                    )
                prompt_masks_per_image[image_offset] = self._normalize_prompt_masks(prompt_masks[sample_idx])
                image_offset += sample_image_count
            return prompt_masks_per_image

        if len(prompt_masks) == total_images:
            for image_idx, item in enumerate(prompt_masks):
                prompt_masks_per_image[image_idx] = self._normalize_prompt_masks(item)
            return prompt_masks_per_image

        raise ValueError(
            f"Cannot align prompt_masks with image_grid_thw: prompt_masks={len(prompt_masks)}, total_images={total_images}."
        )

    def _resize_prompt_masks_for_grid(self, prompt_masks: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        spatial_merge_size = int(self.model.visual.spatial_merge_size)
        target_h = int(grid_thw[1].item() // spatial_merge_size)
        target_w = int(grid_thw[2].item() // spatial_merge_size)
        if prompt_masks.shape[-2:] == (target_h, target_w):
            return prompt_masks.bool()
        resized = F.interpolate(
            prompt_masks.unsqueeze(0),
            size=(target_h, target_w),
            mode='nearest',
        ).squeeze(0)
        return resized.bool()

    def _build_inputs_embeds_with_visual_prompts(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        image_grid_thw: torch.Tensor,
        image_grid_thw_per_sample,
        prompt_masks=None,
        vp_overall_mask=None,
    ) -> torch.Tensor:
        if self.tokenizer is None:
            raise ValueError("Tokenizer must be initialized before using visual prompt training.")

        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
        flat_image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.model.get_placeholder_mask(
            input_ids,
            inputs_embeds=inputs_embeds,
            image_features=flat_image_embeds,
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, flat_image_embeds)

        prompt_masks_per_image = self._resolve_prompt_masks_per_image(
            prompt_masks,
            vp_overall_mask,
            image_grid_thw_per_sample,
        )
        vp_token_id = self.tokenizer.convert_tokens_to_ids('<vp>')
        if vp_token_id is None or vp_token_id < 0:
            raise ValueError("Tokenizer does not provide a valid <vp> token id.")

        vp_embeds = []
        for image_embed, grid_thw, prompt_masks_for_image in zip(image_embeds, image_grid_thw, prompt_masks_per_image):
            if prompt_masks_for_image is None:
                continue
            resized_masks = self._resize_prompt_masks_for_grid(prompt_masks_for_image, grid_thw)
            flat_image_embed = image_embed.reshape(-1, image_embed.shape[-1])
            flat_masks = resized_masks.to(device=flat_image_embed.device, dtype=torch.bool).reshape(
                resized_masks.shape[0], -1
            )
            if flat_masks.shape[-1] != flat_image_embed.shape[0]:
                raise ValueError(
                    f"Prompt mask/image feature mismatch: {flat_masks.shape[-1]} vs {flat_image_embed.shape[0]}"
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


    def forward(self,
                data: Dict[str, torch.Tensor],
                data_samples: Optional[list] = None,
                mode: str = 'loss') -> Union[Dict[str, torch.Tensor], list]:
        assert mode == 'loss', f'Only support loss mode in {self.__class__.__name__}, but got {mode}'
        pixel_values: List[torch.Tensor] = data['pixel_values']
        pixel_values = torch.cat(pixel_values, dim=0)
        image_grid_thw_per_sample = data['image_grid_thw']
        image_grid_thw = torch.cat(image_grid_thw_per_sample, dim=0)
        prompt_masks = data.get('prompt_masks')
        vp_overall_mask = data.get('vp_overall_mask')

        model_inputs = dict(
            input_ids=data['input_ids'],
            attention_mask=data['attention_mask'],
            labels=data['labels'],
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
        )

        if prompt_masks is not None:
            inputs_embeds = self._build_inputs_embeds_with_visual_prompts(
                input_ids=data['input_ids'],
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                image_grid_thw_per_sample=image_grid_thw_per_sample,
                prompt_masks=prompt_masks,
                vp_overall_mask=vp_overall_mask,
            )
            output = self.model(
                **model_inputs,
                pixel_values=None,
                inputs_embeds=inputs_embeds,
            )
        else:
            # DO NOT ENTER POSITION EMBEDDING HERE; Qwen2.5-VL will handle it inside (M-ROPE)
            output = self.model(
                **model_inputs,
                pixel_values=pixel_values,
            )
        return output


    def state_dict(self, *args, **kwargs):
        # filter out the untrainable parameters
        state_dict = super().state_dict(*args, **kwargs)
        to_return = OrderedDict()
        if isinstance(self.model, PeftModelForCausalLM):
            to_return.update(get_peft_model_state_dict(self.model, state_dict=state_dict))
        else:
            to_return.update(state_dict)
        return to_return

    def init_weights(self):
        # Always load from pretrained weights
        pass

if __name__ == "__main__":
    from peft import LoraConfig
    import copy
    model = Qwen2_5_VL(
        model_path="pretrained/qwen2_5vl/Qwen2.5-VL-7B-Instruct/",
        freeze_llm=False,
        freeze_visual_encoder=True,
        llm_lora=dict(
            type=LoraConfig,
            r=256,
            lora_alpha=512,
            lora_dropout=0.05,
            bias='none',
            task_type='CAUSAL_LM',
            modules_to_save=['lm_head', 'embed_tokens'],
            target_modules=None,
        ),
    )


    tokenizer = AutoTokenizer.from_pretrained("pretrained/qwen2_5vl/Qwen2.5-VL-7B-Instruct/", trust_remote_code=True)
    model.add_special_tokens(tokenizer, special_tokens=['[SEG]'])

    model.manual_prepare_llm_for_lora()
    model = model.to('cuda')

    processor = AutoProcessor.from_pretrained("pretrained/qwen2_5vl/Qwen2.5-VL-7B-Instruct/", trust_remote_code=True)
    from qwen_vl_utils import process_vision_info
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    # Preparation for inference
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)

    video_inputs = [copy.deepcopy(image_inputs[0]) for _ in range(10)]  # mock video with 10 frames
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")


    mock_data_dict = {
        'input_ids': inputs['input_ids'], # (1, 3602) torch.int64
        'attention_mask': inputs['attention_mask'], #  (1, 3602) torch.int64, all 1
        'labels': inputs['input_ids'], # (1, 3602) torch.int64
        'pixel_values': [inputs['pixel_values']], # torch.Size([14308, 1176]) torch.float32 torch.Size([11008, 1536]) for qwen3
        'image_grid_thw': [inputs['image_grid_thw']] # value: 1, 98, 146 (not shape) torch.int64; 1, 86. 128 for qwen3
    }


    output = model(mock_data_dict, mode='loss')
    print(output)