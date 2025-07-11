from ov_stable_video_diffusion_helper import OVStableVideoDiffusionPipeline
from transformers import CLIPImageProcessor
from pathlib import Path
from diffusers.utils import load_image, export_to_video
from ov_stable_video_diffusion_helper import (
    VAE_ENCODER_PATH,
    VAE_DECODER_PATH,
    MODEL_DIR,
    UNET_PATH,
    IMAGE_ENCODER_PATH,
)
from lcm_scheduler import AnimateLCMSVDStochasticIterativeScheduler
import openvino as ov
import torch

core = ov.Core()
device = "GPU"
image = load_image("demo.jpg")

vae_encoder = core.compile_model(VAE_ENCODER_PATH, device)
image_encoder = core.compile_model(IMAGE_ENCODER_PATH, device)
unet = core.compile_model(UNET_PATH, device)
vae_decoder = core.compile_model(VAE_DECODER_PATH, device)
scheduler = AnimateLCMSVDStochasticIterativeScheduler.from_pretrained(
    MODEL_DIR / "scheduler"
)
feature_extractor = CLIPImageProcessor.from_pretrained(MODEL_DIR / "feature_extractor")
ov_pipe = OVStableVideoDiffusionPipeline(
    vae_encoder, image_encoder, unet, vae_decoder, scheduler, feature_extractor
)
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
