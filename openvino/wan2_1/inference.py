from ov_wan_helper import OVWanPipeline
from diffusers.utils import export_to_video


device = "GPU"
device_map = {
    "transformer": device,
    "text_encoder": device,
    "vae": device,
}

ov_pipe = OVWanPipeline(model_dir, device_map)

prompt = "A cat walks on the grass, realistic"
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

output = ov_pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    num_frames=20,
    guidance_scale=1.0,
    num_inference_steps=4,
).frames[0]
export_to_video(output, "../../output/wan2_1.mp4", fps=10)
