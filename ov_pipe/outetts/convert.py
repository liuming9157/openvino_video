from cmd_helper import optimum_cli
from pathlib import Path

model_id = "OuteAI/OuteTTS-0.1-350M"
model_dir = Path("../../models/outetts_ov")

if not (model_dir / "openvino_model.xml").exists():
    optimum_cli(
        model_id, model_dir, additional_args={"task": "text-generation-with-past"}
    )
