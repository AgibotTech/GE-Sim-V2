from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Union

import torch
from einops import rearrange

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.image_processor import PipelineImageInput
from diffusers.utils import replace_example_docstring, is_torch_xla_available
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from gesim.models.gesim_v2.pipelines.pipeline_output import CosmosPipelineOutput
from gesim.models.gesim_v2.pipelines.pipeline_cosmos2_video2world import Cosmos2VideoToWorldPipeline, \
    EXAMPLE_DOC_STRING, retrieve_timesteps, retrieve_latents
from gesim.models.gesim_v2.utils.statistics import StatisticInfo
import copy

@dataclass
class CosmosPosePipelineOutput(CosmosPipelineOutput):
    poses: Optional[torch.Tensor] = None
    pose_masks: Optional[torch.Tensor] = None
    pose_traces: Optional[torch.Tensor] = None


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False


class ACWMCosmos2PosePipeline(Cosmos2VideoToWorldPipeline):

    def __init__(
        self,
        text_encoder,
        tokenizer,
        transformer,
        vae,
        scheduler,
        safety_checker=None,
    ):
        super().__init__(
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            safety_checker=safety_checker,
        )
        self.scheduler_pose = copy.deepcopy(scheduler)
        self.pose_dataset_name: str = "g01"
        self.pose_max_stride: int = 1

    def _denormalize_pose(self, pose: torch.Tensor) -> torch.Tensor:
        """Detach-free in-place denormalization for pose tensors."""
        if pose is None:
            return pose

        dataset_key = getattr(self, "pose_dataset_name", "g01") or "g01"
        stats = StatisticInfo.get(dataset_key) or StatisticInfo.get("g01")
        if stats is None:
            return pose

        mean = torch.as_tensor(stats["mean"], device=pose.device, dtype=pose.dtype)
        std = torch.as_tensor(stats["std"], device=pose.device, dtype=pose.dtype)
        max_stride = max(1, int(getattr(self, "pose_max_stride", 1)))
        scale = max_stride * std
        offset = max_stride * mean

        pose[..., :6] = pose[..., :6] * scale[:6] + offset[:6]
        pose[..., 7:13] = pose[..., 7:13] * scale[7:13] + offset[7:13]
        pose[..., 6] = pose[..., 6] * 85.0 + 35.0
        pose[..., 13] = pose[..., 13] * 85.0 + 35.0
        return pose

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        image: PipelineImageInput = None,
        video: List[PipelineImageInput] = None,  # input video is (b v) t c h w (t is ahead of c)
        target_video: Optional[PipelineImageInput] = None,
        cond_to_concat: List[PipelineImageInput] = None,  # (b v) c t h w, could have different t than video t
        prompt: Union[str, List[str]] = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        height: int = 704,
        width: int = 1280,
        num_frames: int = 93, # equals chunk size
        num_inference_steps: int = 20,
        guidance_scale: float = 7.0,
        fps: int = 16,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        pose_states: Optional[torch.Tensor] = None,
        pose_mask: Optional[torch.Tensor] = None,
        pose_timesteps: Optional[torch.Tensor] = None,
        pose_history_token_num: Optional[int] = None,
        pose_initial_noise: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        pose_stats_dataset: Optional[str] = None,
        pose_stats_stride: Optional[int] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        return_pose: bool = False,
        callback_on_step_end: Optional[
            Union[Callable[[int, int, Dict], None], PipelineCallback, MultiPipelineCallbacks]
        ] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 512,
        sigma_conditioning: float = 0.0001,
        pose_buffer_store_step: Optional[int] = None,
        n_view: int = 3,
        n_prev: int = 4,
        merge_view_into_width: bool = True,
        postprocess_video: bool = True,
        denormalize_pose: bool = True,
        show_progress: bool = True,
        inference_sigmas: Optional[List[float]] = None,
        sparge_config: Optional[Dict] = None,
        liger_norm: bool = True,
        liger_layernorm: bool = False,
        triton_rope: bool = False,
        distilled_sampling: bool = False,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image`, `np.ndarray`, `torch.Tensor`, *optional*):
                The image to be used as a conditioning input for the video generation.
            video (`List[PIL.Image.Image]`, `np.ndarray`, `torch.Tensor`, *optional*):
                The video to be used as a conditioning input for the video generation.
            target_video (`List[PIL.Image.Image]`, `np.ndarray`, `torch.Tensor`, *optional*):
                Ground-truth future frames aligned with `num_frames`. When provided, the pipeline will log per-step
                noise prediction MSE against the implied diffusion noise of these frames.
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            height (`int`, defaults to `704`):
                The height in pixels of the generated image.
            width (`int`, defaults to `1280`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `93`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `35`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `7.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`.
            fps (`int`, defaults to `16`):
                The frames per second of the generated video.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                Random number generator for latent sampling. When None, the global RNG is used.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            pose_states (`torch.Tensor`, *optional*):
                Full pose token sequence (history actions, future actions, history poses, and future poses), shape
                should be `(batch, seq_len, pose_dim)`.
            pose_mask (`torch.Tensor`, *optional*):
                Mask aligned with `pose_states`; 0 indicates the token is ignored during inference.
            pose_timesteps (`torch.Tensor`, *optional*):
                Timestep information for history tokens; if provided, it will remain unchanged throughout the pose
                branch iterations.
            pose_history_token_num (`int`, *optional*):
                Number of history tokens. Tokens starting from this index are treated as future poses that need to be
                progressively denoised.
            pose_initial_noise (`torch.Tensor`, *optional*):
                Initial noise for future poses, used to reproduce a fixed inference trajectory. Shape must match
                `pose_states[:, pose_history_token_num:, :]`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. For PixArt-Sigma this negative prompt should be "". If not
                provided, negative_prompt_embeds will be generated from `negative_prompt` input argument.
            pose_stats_dataset (`str`, *optional*):
                Dataset key used to retrieve pose normalization statistics. Defaults to ``"g01"``.
            pose_stats_stride (`int`, *optional*):
                Maximum stride factor that was applied during normalization. Defaults to ``1``.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`CosmosPipelineOutput`] instead of a plain tuple.
            return_pose (`bool`, defaults to `False`):
                Whether to run the pose branch and include the full pose sequence in the output.
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `512`):
                The maximum number of tokens in the prompt. If the prompt exceeds this length, it will be truncated. If
                the prompt is shorter than this length, it will be padded.
            sigma_conditioning (`float`, defaults to `0.0001`):
                The sigma value used for scaling conditioning latents. Ideally, it should not be changed or should be
                set to a small value close to zero.
            pose_buffer_store_step (`int`, *optional*):
                Which denoising step to capture the video hidden states for the pose expert. Defaults to the final
                step. Negative values count from the end (e.g., ``-1`` is the last step).

        Examples:

        Returns:
            [`~CosmosPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`CosmosPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """

        # if self.safety_checker is None:
        #     raise ValueError(
        #         f"You have disabled the safety checker for {self.__class__}. This is in violation of the "
        #         "[NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license). "
        #         f"Please ensure that you are compliant with the license agreement."
        #     )

        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt, height, width, prompt_embeds, callback_on_step_end_tensor_inputs)

        self._guidance_scale = guidance_scale
        self._current_timestep = None
        self._interrupt = False

        if pose_stats_dataset is not None:
            self.pose_dataset_name = pose_stats_dataset
        if pose_stats_stride is not None:
            self.pose_max_stride = max(1, int(pose_stats_stride))

        device = self._execution_device

        # if self.safety_checker is not None:
        #     self.safety_checker.to(device)
        #     if prompt is not None:
        #         prompt_list = [prompt] if isinstance(prompt, str) else prompt
        #         for p in prompt_list:
        #             if not self.safety_checker.check_text_safety(p):
        #                 raise ValueError(
        #                     f"Cosmos Guardrail detected unsafe text in the prompt: {p}. Please ensure that the "
        #                     f"prompt abides by the NVIDIA Open Model License Agreement."
        #                 )
        #     self.safety_checker.to("cpu")

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        assert num_videos_per_prompt == 1  # TODO: only support num_videos_per_prompt=1 for now
        (
            prompt_embeds,
            negative_prompt_embeds,
        ) = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=self.do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            device=device,
            max_sequence_length=max_sequence_length,
        )

        # 4. Prepare timesteps
        sigmas_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64
        if distilled_sampling:
            if inference_sigmas is None:
                step_sigmas = torch.linspace(
                    1.0, 0.0, num_inference_steps + 1, dtype=torch.float32, device=device
                )
            else:
                step_sigmas = torch.tensor(inference_sigmas, dtype=torch.float32, device=device)
                if step_sigmas.ndim != 1 or step_sigmas.numel() == 0:
                    raise ValueError("`inference_sigmas` must be a non-empty 1D schedule for distilled sampling.")
                if step_sigmas[-1].abs() > 1e-8:
                    step_sigmas = torch.cat([step_sigmas, step_sigmas.new_zeros(1)])
            if step_sigmas.numel() < 2:
                raise ValueError("Distilled sampling needs at least one transition plus terminal sigma 0.")
            if not torch.all(step_sigmas[:-1] >= step_sigmas[1:]):
                raise ValueError(f"Distilled sigma schedule must be descending, got {step_sigmas.tolist()}.")
            timesteps = step_sigmas[:-1]
            num_inference_steps = int(step_sigmas.numel() - 1)
            for scheduler in (self.scheduler, self.scheduler_pose):
                scheduler.num_inference_steps = num_inference_steps
                scheduler.timesteps = timesteps
                scheduler.sigmas = step_sigmas
                scheduler._step_index = None
                scheduler._begin_index = None
        elif inference_sigmas is not None:
            sigmas = torch.tensor(inference_sigmas, dtype=sigmas_dtype)
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, device=device, sigmas=sigmas)
            _, _ = retrieve_timesteps(self.scheduler_pose, device=device, sigmas=sigmas)

            if self.scheduler.config.final_sigmas_type == "sigma_min":
                # Replace the last sigma (which is zero) with the minimum sigma value
                self.scheduler.sigmas[-1] = self.scheduler.sigmas[-2]

            if self.scheduler_pose.config.final_sigmas_type == "sigma_min":
                # Replace the last sigma (which is zero) with the minimum sigma value
                self.scheduler_pose.sigmas[-1] = self.scheduler_pose.sigmas[-2]
        else:
            sigmas = torch.linspace(0, 1, num_inference_steps, dtype=sigmas_dtype)
            timesteps, num_inference_steps = retrieve_timesteps(self.scheduler, device=device, sigmas=sigmas)
            _, _ = retrieve_timesteps(self.scheduler_pose, device=device, sigmas=sigmas)

            if self.scheduler.config.final_sigmas_type == "sigma_min":
                # Replace the last sigma (which is zero) with the minimum sigma value
                self.scheduler.sigmas[-1] = self.scheduler.sigmas[-2]

            if self.scheduler_pose.config.final_sigmas_type == "sigma_min":
                # Replace the last sigma (which is zero) with the minimum sigma value
                self.scheduler_pose.sigmas[-1] = self.scheduler_pose.sigmas[-2]
        # 5. Prepare latent variables
        vae_dtype = self.vae.dtype
        transformer_dtype = self.transformer.dtype

        if image is not None:
            video = self.video_processor.preprocess(image, height, width).unsqueeze(2)
        else:
            # input video is (b v) t c h w, output is (b v) c t h w
            video = self.video_processor.preprocess_video(video, height, width)
        video = video.to(device=device, dtype=vae_dtype)

        # num_channels_latents = self.transformer.config.in_channels - 1
        num_channels_latents = self.vae.z_dim
        if video.shape[2] > n_prev:  # pyright: ignore
            video = video[:, :, :n_prev]  # pyright: ignore
        latents, conditioning_latents, cond_indicator, uncond_indicator, cond_mask, uncond_mask = self.prepare_latents(
            video,  # memory only! (b v) c t h w
            batch_size * n_view,
            num_channels_latents,
            height,
            width,
            num_frames,
            self.do_classifier_free_guidance,
            torch.float32,
            device,
            generator,
            latents,
        )  # indicator and mask here are all about memory video, action condition not included
           # latents here only contains future
        if distilled_sampling:
            latents = latents / self.scheduler.config.sigma_max
            init_noise_future = latents.clone()
        unconditioning_latents = None

        noise_metrics: List[Dict[str, float]] = []
        target_latents_fp32: Optional[torch.Tensor] = None
        if target_video is not None:
            target_tensor = self.video_processor.preprocess_video(target_video, height, width)
            target_tensor = target_tensor.to(device=device, dtype=vae_dtype)
            target_latents = retrieve_latents(self.vae.encode(target_tensor), generator=None, sample_mode="argmax")
            target_latents = target_latents.to(device=device, dtype=transformer_dtype)

            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(device=device, dtype=transformer_dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(device=device, dtype=transformer_dtype)
            )
            target_latents = (target_latents - latents_mean) / latents_std * self.scheduler.config.sigma_data

            if target_latents.shape == latents.shape:
                target_latents_fp32 = target_latents.detach().to(dtype=torch.float32)
            else:
                target_latents_fp32 = None

        cond_mask = cond_mask.to(transformer_dtype)
        if self.do_classifier_free_guidance:
            uncond_mask = uncond_mask.to(transformer_dtype)
            unconditioning_latents = conditioning_latents

        padding_mask = latents.new_zeros(1, 1, height, width, dtype=transformer_dtype)  # ??? looks like it's always 0
        sigma_conditioning = torch.tensor(sigma_conditioning, dtype=torch.float32, device=device)
        t_conditioning = sigma_conditioning / (sigma_conditioning + 1)

        if liger_norm:
            from gesim.models.gesim_v2.networks.transformers.liger_norms import enable_liger_norms
            enable_liger_norms(self.transformer, fuse_layernorm=liger_layernorm)

        if triton_rope:
            from gesim.models.gesim_v2.networks.transformers.triton_rope import enable_triton_rope
            enable_triton_rope()

        _sparge_active = (sparge_config is not None
                          and sparge_config.get("enabled", False))
        if _sparge_active:
            from gesim.models.gesim_v2.networks.transformers.sparge_attention import (
                enable_sparge_attention, SpargeAttentionState,
            )
            enable_sparge_attention(self.transformer, sparge_config)

        # 6. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        if not show_progress:
            self.set_progress_bar_config(disable=True)
        pose_prediction = None
        pose_future_noisy = None
        pose_future_denoised = None
        video_pose_buffer_last: Optional[List[torch.Tensor]] = None
        cond_latent_last: Optional[torch.Tensor] = None
        cond_timestep_clean_last: Optional[torch.Tensor] = None
        pose_history_states: Optional[torch.Tensor] = None
        pose_history_feature_mask: Optional[torch.Tensor] = None
        pose_future_feature_mask: Optional[torch.Tensor] = None
        pose_timesteps_history: Optional[torch.Tensor] = None
        future_token_num = 0
        if return_pose:
            pose_states = pose_states.to(device=device, dtype=transformer_dtype)
            pose_mask = pose_mask.to(device=device, dtype=transformer_dtype)
            pose_timesteps_history = (
                pose_timesteps.to(device=device, dtype=transformer_dtype)[:, :pose_history_token_num]
                if pose_timesteps is not None and pose_history_token_num > 0
                else None
            )

            total_tokens = pose_states.shape[1]

            pose_history_states = pose_states[:, :pose_history_token_num] if pose_history_token_num > 0 else None
            pose_future_noisy = pose_states[:, pose_history_token_num:].clone()

            pose_history_feature_mask = (
                pose_mask[:, :pose_history_token_num] if pose_history_token_num > 0 else None
            )
            pose_future_feature_mask = pose_mask[:, pose_history_token_num:]
            init_sigma = self.scheduler_pose.sigmas[0]
            future_token_num = pose_future_noisy.shape[1]
            if pose_initial_noise is not None:
                pose_initial_noise = pose_initial_noise.to(device=device, dtype=transformer_dtype)
                pose_future_noisy = pose_initial_noise.clone()
            else:
                pose_initial_noise = randn_tensor(
                    (pose_states.shape[0], future_token_num, pose_states.shape[2]),
                    device=device,
                    dtype=transformer_dtype,
                )
            pose_future_noisy = pose_initial_noise*init_sigma
            
        else:
            pose_states = None
            pose_mask = None

        store_buffer_index = len(timesteps) - 1
        if pose_buffer_store_step is not None:
            store_buffer_index = int(pose_buffer_store_step)

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t
                current_sigma = self.scheduler.sigmas[i]

                if _sparge_active:
                    SpargeAttentionState.current_step = i

                if distilled_sampling:
                    current_sigma = current_sigma.to(device=latents.device, dtype=latents.dtype)
                    next_sigma = self.scheduler.sigmas[i + 1].to(device=latents.device, dtype=latents.dtype)
                    current_t = current_sigma
                    timestep = current_t.view(1, 1, 1, 1, 1).expand(
                        latents.size(0), -1, latents.size(2) + conditioning_latents.size(2), -1, -1
                    )
                    cond_latent = torch.cat([conditioning_latents, latents], dim=2)
                    if cond_to_concat is not None:
                        cond_latent = torch.cat([cond_latent, cond_to_concat], dim=1)
                    cond_latent = cond_latent.to(transformer_dtype)
                    cond_timestep = cond_indicator * t_conditioning + (1 - cond_indicator) * timestep
                    cond_timestep = cond_timestep.to(transformer_dtype)

                    store_buffer_flag = (i == store_buffer_index)
                    transformer_output = self.transformer(
                        hidden_states=cond_latent,
                        timestep=cond_timestep,
                        encoder_hidden_states=prompt_embeds,
                        fps=fps,
                        condition_mask=cond_mask,
                        padding_mask=padding_mask,
                        return_dict=False,
                        n_view=n_view,
                        return_pose=False,
                        store_buffer=store_buffer_flag,
                        compute_video_trunk=True,
                        is_inference=True,
                    )[0]

                    pred = transformer_output["video"]
                    if store_buffer_flag:
                        video_pose_buffer_last = transformer_output.get("video_pose_buffer", None)
                        cond_latent_last = cond_latent
                        cond_timestep_clean_last = cond_timestep

                    mem_latent_len = conditioning_latents.size(2)
                    pred = pred[:, :, mem_latent_len:].float()

                    if self.do_classifier_free_guidance:
                        uncond_latent = torch.cat([unconditioning_latents, latents], dim=2)
                        if cond_to_concat is not None:
                            uncond_latent = torch.cat([uncond_latent, cond_to_concat], dim=1)
                        uncond_latent = uncond_latent.to(transformer_dtype)
                        uncond_timestep = uncond_indicator * t_conditioning + (1 - uncond_indicator) * timestep
                        uncond_timestep = uncond_timestep.to(transformer_dtype)

                        pred_uncond = self.transformer(
                            hidden_states=uncond_latent,
                            timestep=uncond_timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            fps=fps,
                            condition_mask=uncond_mask,
                            padding_mask=padding_mask,
                            return_dict=False,
                            n_view=n_view,
                            return_pose=False,
                            compute_video_trunk=True,
                        )[0]["video"]
                        pred_uncond = pred_uncond[:, :, mem_latent_len:].float()
                        pred = pred + self.guidance_scale * (pred - pred_uncond)

                    if target_latents_fp32 is not None:
                        sigma_val = float(current_sigma.item())
                        if abs(sigma_val) > 1e-8:
                            latents_fp32 = latents.to(dtype=torch.float32)
                            pred_fp32 = pred.to(dtype=torch.float32)
                            noise_gt = (latents_fp32 - (1.0 - sigma_val) * target_latents_fp32) / sigma_val
                            mse_value = torch.mean((pred_fp32 - noise_gt) ** 2).item()
                            noise_metrics.append({
                                "step": int(i),
                                "sigma": sigma_val,
                                "mse": mse_value,
                            })

                    x0_pred = latents - current_sigma * pred
                    if next_sigma > 0:
                        latents = (1 - next_sigma) * x0_pred + next_sigma * init_noise_future
                    else:
                        latents = x0_pred
                else:
                    current_t = current_sigma / (current_sigma + 1)
                    c_in = 1 - current_t
                    c_skip = 1 - current_t
                    c_out = -current_t
                    # smallest current=0.002
                    # timestep = current_t.view(1, 1, 1, 1, 1).expand(
                    #     latents.size(0), -1, latents.size(2), -1, -1
                    # )  # [B, 1, T, 1, 1]
                    timestep = current_t.view(1, 1, 1, 1, 1).expand(
                        latents.size(0), -1, latents.size(2)+conditioning_latents.size(2), -1, -1
                    )  # [B, 1, T, 1, 1]
                    # timestep = timestep * 1000  # LTX, timestep ranges from 1 to 1000

                    cond_latent = latents * c_in  # all noise with a scale factor
                    # replace :n_prev frames with clean video latents
                    # cond_latent = cond_indicator * conditioning_latents + (1 - cond_indicator) * cond_latent
                    cond_latent = torch.cat([conditioning_latents, cond_latent], dim=2)  # frame
                    cond_latent = torch.cat([cond_latent, cond_to_concat], dim=1)  # channel
                    cond_latent = cond_latent.to(transformer_dtype)  # (b v) c t h w
                    cond_timestep = cond_indicator * t_conditioning + (1 - cond_indicator) * timestep
                    cond_timestep = cond_timestep.to(transformer_dtype)
                    # keep scheduler-driven per-step timestep for video branch (as before)

                    # Do not run pose branch in this pass; only collect buffer on the last step
                    store_buffer_flag = (i == store_buffer_index)
                    transformer_output = self.transformer(
                        hidden_states=cond_latent,
                        timestep=cond_timestep,
                        encoder_hidden_states=prompt_embeds,
                        fps=fps,
                        condition_mask=cond_mask,
                        padding_mask=padding_mask,
                        return_dict=False,
                        n_view=n_view,
                        return_pose=False,
                        store_buffer=store_buffer_flag,
                        # store_buffer=True,
                        compute_video_trunk=True,
                        is_inference=True,
                    )[0]

                    noise_pred = transformer_output["video"]
                    if store_buffer_flag:
                        video_pose_buffer_last = transformer_output.get("video_pose_buffer", None)
                        cond_latent_last = cond_latent
                        cond_timestep_clean_last = cond_timestep

                    # Remove memory portion based on actual encoded memory length
                    mem_latent_len = conditioning_latents.size(2)
                    noise_pred = noise_pred[:, :, mem_latent_len:]
                    noise_pred = (c_skip * latents + c_out * noise_pred.float()).to(transformer_dtype)
                    # noise_pred = cond_indicator * conditioning_latents + (1 - cond_indicator) * noise_pred

                    if self.do_classifier_free_guidance:
                        uncond_latent = latents * c_in
                        # replace :n_prev frames with clean video latents
                        # uncond_latent = uncond_indicator * unconditioning_latents + (1 - uncond_indicator) * uncond_latent
                        uncond_latent = torch.cat([conditioning_latents, uncond_latent], dim=2)  # frame
                        uncond_latent = torch.cat([uncond_latent, cond_to_concat], dim=1)  # channel
                        uncond_latent = uncond_latent.to(transformer_dtype)
                        uncond_timestep = uncond_indicator * t_conditioning + (1 - uncond_indicator) * timestep
                        uncond_timestep = uncond_timestep.to(transformer_dtype)

                        noise_pred_uncond = self.transformer(
                            hidden_states=uncond_latent,
                            timestep=uncond_timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            fps=fps,
                            condition_mask=uncond_mask,
                            padding_mask=padding_mask,
                            return_dict=False,
                            n_view=n_view,
                            return_pose=False,
                            compute_video_trunk=True,
                        )[0]
                        noise_pred_uncond = noise_pred_uncond[:, :, mem_latent_len:]  # remove memory
                        noise_pred_uncond = (c_skip * latents + c_out * noise_pred_uncond.float()).to(transformer_dtype)
                        # noise_pred_uncond = (
                        #     uncond_indicator * unconditioning_latents + (1 - uncond_indicator) * noise_pred_uncond
                        # )
                        noise_pred = noise_pred + self.guidance_scale * (noise_pred - noise_pred_uncond)

                    noise_pred = (latents - noise_pred) / current_sigma

                    if target_latents_fp32 is not None:
                        sigma_val = (
                            float(current_sigma.item())
                            if isinstance(current_sigma, torch.Tensor)
                            else float(current_sigma)
                        )
                        if abs(sigma_val) > 1e-8:
                            latents_fp32 = latents.to(dtype=torch.float32)
                            noise_pred_fp32 = noise_pred.to(dtype=torch.float32)
                            noise_gt = (latents_fp32 - (1.0 - sigma_val) * target_latents_fp32) / sigma_val
                            mse_value = torch.mean((noise_pred_fp32 - noise_gt) ** 2).item()
                            noise_metrics.append({
                                "step": int(i),
                                "sigma": sigma_val,
                                "mse": mse_value,
                            })
                    latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                # skip SE update inside loop

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                if XLA_AVAILABLE:
                    xm.mark_step()

        self._current_timestep = None

        # Second pass: run pose-only iterative denoising using the final video buffer
        if return_pose and video_pose_buffer_last is not None and pose_future_noisy is not None:

            # Iterate over scheduler_pose timesteps to gradually denoise pose tokens
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    if self.interrupt:
                        continue

                    current_sigma_pose = self.scheduler_pose.sigmas[i]
                    if distilled_sampling:
                        current_sigma_pose = current_sigma_pose.to(device=pose_future_noisy.device, dtype=pose_future_noisy.dtype)
                        next_sigma_pose = self.scheduler_pose.sigmas[i + 1].to(
                            device=pose_future_noisy.device,
                            dtype=pose_future_noisy.dtype,
                        )
                        current_t_pose = current_sigma_pose
                    else:
                        current_t_pose = current_sigma_pose / (current_sigma_pose + 1)
                        c_in_pose = 1 - current_t_pose
                        c_skip_pose = 1 - current_t_pose
                        c_out_pose = -current_t_pose

                    # Prepare pose inputs at this step
                    if distilled_sampling:
                        pose_future_scaled = pose_future_noisy
                    else:
                        pose_future_scaled = pose_future_noisy * c_in_pose

                    pose_states_combined = torch.cat([pose_history_states, pose_future_scaled], dim=1)
                    pose_mask_combined = torch.cat([pose_history_feature_mask,pose_future_feature_mask], dim=1)

                    future_timestep_step = torch.full(
                        (pose_future_scaled.shape[0], pose_future_scaled.shape[1]),
                        current_t_pose.to(dtype=transformer_dtype),
                        device=device,
                        dtype=transformer_dtype,
                    )
                    pose_timestep_combined = (torch.cat([pose_timesteps_history, future_timestep_step], dim=1))
                    # Forward only pose branch with reused video buffer (no video trunk recomputation)
                    transformer_output2 = self.transformer(
                        hidden_states=cond_latent_last,
                        timestep=cond_timestep_clean_last,
                        encoder_hidden_states=prompt_embeds,
                        fps=fps,
                        condition_mask=cond_mask,
                        padding_mask=padding_mask,
                        return_dict=False,
                        n_view=n_view,
                        return_pose=True,
                        store_buffer=False,
                        video_pose_buffer=video_pose_buffer_last,
                        compute_video_trunk=False,
                        pose_states=pose_states_combined,
                        pose_timestep=pose_timestep_combined,
                        pose_mask=pose_mask_combined,
                    )[0]

                    pose_step_output = transformer_output2.get("pose")
                    if self.do_classifier_free_guidance:
                        pose_uncond, pose_text = pose_step_output.chunk(2)
                        pose_step_output = pose_uncond + self.guidance_scale * (pose_text - pose_uncond)

                    pose_future_pred = pose_step_output[:, pose_history_token_num:]
                    if distilled_sampling:
                        pose_x0_pred = pose_future_noisy - current_sigma_pose * pose_future_pred
                        if next_sigma_pose > 0:
                            pose_future_noisy = (1 - next_sigma_pose) * pose_x0_pred + next_sigma_pose * pose_initial_noise
                        else:
                            pose_future_noisy = pose_x0_pred
                    else:
                        pose_noise_pred = c_skip_pose * pose_future_noisy + c_out_pose * pose_future_pred
                        pose_noise_pred = (pose_future_noisy - pose_noise_pred) / current_sigma_pose

                        # Euler update for pose branch
                        pose_future_noisy = self.scheduler_pose.step(
                            pose_noise_pred.to(transformer_dtype), t, pose_future_noisy, return_dict=False
                        )[0]
                    pose_future_noisy = pose_future_noisy * pose_future_feature_mask

                    # progress
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler_pose.order == 0):
                        progress_bar.update()

                # finalize
                pose_future_denoised = pose_future_noisy
                pose_prediction = torch.cat([pose_history_states, pose_future_denoised], dim=1)
                if denormalize_pose:
                    pose_prediction = self._denormalize_pose(pose_prediction)

        if not output_type == "latent":
            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents_std = (
                torch.tensor(self.vae.config.latents_std)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )
            latents = latents * latents_std / self.scheduler.config.sigma_data + latents_mean  # config.sigma_data=1.0

            video = self.vae.decode(latents.to(self.vae.dtype), return_dict=False)[0]
            if merge_view_into_width:
                video = rearrange(video, '(b v) c t h w -> b c t h (v w)', v=n_view)  # should be vw not wv !!!

            # if self.safety_checker is not None:
            #     self.safety_checker.to(device)
            #     video = self.video_processor.postprocess_video(video, output_type="np")
            #     video = (video * 255).astype(np.uint8)
            #     video_batch = []
            #     for vid in video:
            #         vid = self.safety_checker.check_video_safety(vid)
            #         video_batch.append(vid)
            #     video = np.stack(video_batch).astype(np.float32) / 255.0 * 2 - 1
            #     video = torch.from_numpy(video).permute(0, 4, 1, 2, 3)
            #     video = self.video_processor.postprocess_video(video, output_type=output_type)
            #     self.safety_checker.to("cpu")
            # else:
            if postprocess_video:
                video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()


        if not return_dict:
            if return_pose:
                return (video, pose_prediction)
            return (video,)

        if return_pose:
            return CosmosPosePipelineOutput(
                frames=video,
                noise_metrics=noise_metrics if noise_metrics else None,
                poses=pose_prediction,
                pose_masks=pose_mask,
            )
        return CosmosPipelineOutput(
            frames=video,
            noise_metrics=noise_metrics if noise_metrics else None,
        )

    def prepare_latents(
        self,
        video: torch.Tensor,  # (b v) c t h w
        batch_size: int,  # (b v)
        num_channels_latents: int = 16,
        height: int = 704,
        width: int = 1280,
        num_frames: int = 93,  # memory not included
        do_classifier_free_guidance: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        num_cond_frames = video.size(2)
        num_cond_latent_frames = num_cond_frames  # encode memory separately
        init_latents = [retrieve_latents(self.vae.encode(video[:, :, it].unsqueeze(2)), generator) for it in range(video.size(2))]

        init_latents = torch.cat(init_latents, dim=2).to(dtype)

        latents_mean = (
            torch.tensor(self.vae.config.latents_mean).view(1, self.vae.config.z_dim, 1, 1, 1).to(device, dtype)
        )
        latents_std = (
            torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(device, dtype)
        )
        init_latents = (init_latents - latents_mean) / latents_std * self.scheduler.config.sigma_data  # sigma_data=1.0

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        latent_height = height // self.vae_scale_factor_spatial
        latent_width = width // self.vae_scale_factor_spatial
        shape = (batch_size, num_channels_latents, num_latent_frames, latent_height, latent_width)

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device=device, dtype=dtype)

        latents = latents * self.scheduler.config.sigma_max  # sigma_max = 80.0

        padding_shape = (batch_size, 1, num_cond_latent_frames+num_latent_frames, latent_height, latent_width)
        ones_padding = latents.new_ones(padding_shape)
        zeros_padding = latents.new_zeros(padding_shape)

        cond_indicator = latents.new_zeros(1, 1, num_cond_latent_frames+num_latent_frames, 1, 1)
        cond_indicator[:, :, :num_cond_latent_frames] = 1.0
        cond_mask = cond_indicator * ones_padding + (1 - cond_indicator) * zeros_padding

        uncond_indicator = uncond_mask = None  # equals cond_indicator and cond_mask
        if do_classifier_free_guidance:
            uncond_indicator = latents.new_zeros(1, 1, num_cond_latent_frames+num_latent_frames, 1, 1)
            uncond_indicator[:, :, :num_cond_latent_frames] = 1.0
            uncond_mask = uncond_indicator * ones_padding + (1 - uncond_indicator) * zeros_padding

        return latents, init_latents, cond_indicator, uncond_indicator, cond_mask, uncond_mask
