import nncf
from ov_wan_helper import convert_pipeline

model_id = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
model_dir = "../../models/wan2_1_ov"
weights_compression_config = {
    "mode": nncf.CompressWeightsMode.INT4_ASYM,
    "group_size": 64,
    "ratio": 1.0,
}
convert_pipeline(model_id, model_dir, compression_config=weights_compression_config)
