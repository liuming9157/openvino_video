from ov_outetts_helper import InterfaceOV, OVHFModel

device = "GPU"
interface = InterfaceOV(model_dir, device)
tts_output = interface.generate(
    text="Hello, I'm working!", temperature=0.1, repetition_penalty=1.1, max_length=4096
)
tts_output.save("../../output/outetts_output.wav")
