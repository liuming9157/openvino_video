from ov_outetts_helper import InterfaceOV, OVHFModel
from pathlib import Path

# 运行推理会自动下载OuteAI/wavtokenizer-large-75token-interface模型，请保证能连接HF

device = "GPU"
model_dir = Path("../../models/outetts_ov")
interface = InterfaceOV(model_dir, device)
tts_output = interface.generate(
    text="Hello, I'm working!", temperature=0.1, repetition_penalty=1.1, max_length=4096
)
tts_output.save("../../output/outetts_output.wav")
