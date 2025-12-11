import genie_tts as genie

# Currently, convert_to_onnx can only convert v2 or v2ProPlus models.
genie.convert_to_onnx(
    torch_pth_path=r"<PATH_TO_TORCH_PTH_FILE>",  # Replace with the path to your .pth model file
    torch_ckpt_path=r"<PATH_TO_TORCH_CKPT_FILE>",  # Replace with the path to your .ckpt checkpoint file
    output_dir=r"<OUTPUT_DIRECTORY>"  # Replace with the directory where the ONNX model should be saved
)
