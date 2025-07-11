from .openvino import OVWanPipeline, OVOutettsPipeline, OVStableVideoDiffusionPipeline
from ov_stable_video_diffusion_helper import OVStableVideoDiffusionPipeline
from transformers import CLIPImageProcessor
from pathlib import Path
from diffusers.utils import load_image, export_to_video
from lcm_scheduler import AnimateLCMSVDStochasticIterativeScheduler
import openvino as ov
import torch


class OVPipeline:
    def __init__(self, device: str = "GPU") -> None:
        self.device = device

        self.llm = ""
        self.tts = OVOutettsPipeline
        self.load_i2v()

    def load_llm(self, model_id: str):
        self.llm = OVOutettsPipeline(model_id, self.device)

    def load_tts(self, model_id: str):
        self.tts = OVOutettsPipeline(model_id, self.device)

    def load_t2v(self, model_id: str):
        self.t2v = OVWanPipeline(model_id, self.device)

    def load_i2v(self):
        core = ov.Core()
        model_dir = Path("./models/stable_video_diffusion_ov")
        vae_encoder = core.compile_model(model_dir / "vae_encoder.xml", self.device)
        image_encoder = core.compile_model(model_dir / "image_encoder.xml", self.device)
        unet = core.compile_model(model_dir / "unet.xml", self.device)
        vae_decoder = core.compile_model(model_dir / "vae_decoder.xml", self.device)
        scheduler = AnimateLCMSVDStochasticIterativeScheduler.from_pretrained(
            model_dir / "scheduler"
        )
        feature_extractor = CLIPImageProcessor.from_pretrained(
            model_dir / "feature_extractor"
        )
        self.i2v = OVStableVideoDiffusionPipeline(
            vae_encoder, image_encoder, unet, vae_decoder, scheduler, feature_extractor
        )

    def __call__(
        self,
        prompt: Union[str, list[str]] = None,
        negative_prompt: Union[str, list[str]] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, list[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
        prompt_embeds: Optional[torch.Tensor] = None,
        negative_prompt_embeds: Optional[torch.Tensor] = None,
        output_type: Optional[str] = "np",
        return_dict: bool = True,
        max_sequence_length: int = 512,
    ):
        if self.llm == "":
            raise ValueError("LLM model not loaded.")
        if self.tts == "":
            raise ValueError("TTS model not loaded.")
        if self.video == "":
            raise ValueError("Video model not loaded.")
        output = self.llm(prompt)
        output = self.video(output)

        frames = ov_pipe(
            image,
            num_inference_steps=4,
            motion_bucket_id=60,
            num_frames=8,
            height=320,
            width=512,
            generator=torch.manual_seed(12342),
        ).frames[0]
        out_path = Path("../../output/svd.mp4")

        export_to_video(frames, str(out_path), fps=7)
