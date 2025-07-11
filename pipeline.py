from openvino import OVWanPipeline, OVStableVideoDiffusionPipeline, OVOuteTTS

from transformers import AutoTokenizer, CLIPImageProcessor
from pathlib import Path
from diffusers.utils import load_image, export_to_video
from openvino.stable_video_diffusion.lcm_scheduler import (
    AnimateLCMSVDStochasticIterativeScheduler,
)
import openvino as ov
import torch
from optimum.intel.openvino import OVModelForCausalLM
from movie import concat_video_audio


class OVPipeline:
    def __init__(self, device: str = "GPU") -> None:
        self.device = device

        # self._load_llm()
        self._load_tts()
        self._load_t2v()
        self._load_i2v()

    def _load_llm(self):
        model_id = "OpenVINO/qwen2.5-7b-instruct-int4-ov"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.llm = OVModelForCausalLM.from_pretrained(model_id)

    def _load_tts(self):
        model_dir = Path("./models/outetts_ov")
        self.tts = OVOuteTTS(model_dir, self.device)

    def _load_t2v(self):
        device_map = {
            "transformer": self.device,
            "text_encoder": self.device,
            "vae": self.device,
        }
        self.t2v = OVWanPipeline(model_dir, device_map)

    def _load_i2v(self):
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
        # if self.llm == "":
        #     raise ValueError("LLM model not loaded.")
        # if self.tts == "":
        #     raise ValueError("TTS model not loaded.")
        # if self.t2v == "":
        #     raise ValueError("Video model not loaded.")
        # llm
        # inputs = self.tokenizer(prompt, return_tensors="pt")
        # outputs = self.llm.generate(**inputs, max_length=200)
        # text = self.tokenizer.batch_decode(outputs)[0]
        # print(text)
        # tts
        tts_output = self.tts.generate(
            text=prompt,
            temperature=0.1,
            repetition_penalty=1.1,
            max_length=4096,
        )
        tts_output.save("./output/outetts_output.wav")
        # t2v
        output = self.t2v(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=480,
            width=832,
            num_frames=20,
            guidance_scale=1.0,
            num_inference_steps=4,
        ).frames[0]
        export_to_video(output, "./output/wan2_1.mp4", fps=10)
        # i2v
        frames = self.i2v(
            image,
            num_inference_steps=4,
            motion_bucket_id=60,
            num_frames=8,
            height=320,
            width=512,
            generator=torch.manual_seed(12342),
        ).frames[0]
        out_path = Path("./output/svd.mp4")

        export_to_video(frames, str(out_path), fps=7)
        return text
