import torch
import torch.nn as nn
import torch.quantization

from dataset import BATCH_SIZE, WORKERS
from dataset import train_set


# DataLoader for calibration, loading from the train_set with specified batch size and worker count.
CAL_LOADER = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)

class QuantizedNet(nn.Module):
    """
    A wrapper class for quantizing neural networks. It encapsulates the process of quantization and dequantization
    for a given floating-point (fp32) model.
    """
    def __init__(self, model_fp32: nn.Module):
        """
        Initializes the QuantizedNet model.

        Args:
            model_fp32 (nn.Module): The original floating-point (FP32) model to be quantized.
        """
        super(QuantizedNet, self).__init__()
        self.quant = torch.quantization.QuantStub()  # Quantization stub for input
        self.dequant = torch.quantization.DeQuantStub()  # Dequantization stub for output
        self.model_fp32 = model_fp32  # The original FP32 model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the quantized model.

        Args:
            x (torch.Tensor): The input tensor for the model.

        Returns:
            torch.Tensor: The output tensor after processing by the quantized model.
        """
        x = self.quant(x)  # Quantize the input
        x = self.model_fp32(x)  # Apply the original model
        x = self.dequant(x)  # Dequantize the output
        return x

def quantize_model(model_fp32: nn.Module, calibrate: bool = True) -> nn.Module:
    """
    Quantizes a given FP32 model, optionally calibrating it using a calibration data loader.

    Args:
        model_fp32 (nn.Module): The FP32 model to be quantized.
        calibrate (bool, optional): If True, calibrate the model using the CAL_LOADER. Defaults to True.

    Returns:
        nn.Module: The quantized model.
    """
    # Move the model to CPU (quantization currently supports only CPU).
    model_fp32 = model_fp32.cpu()

    # Ensure the model is in eval mode.
    model_fp32.eval()

    # Fuse model layers where applicable. This step depends on the model architecture and
    # must be implemented in the model's fuse_model() method.
    model_fp32.fuse_model()

    # Create an instance of QuantizedNet wrapper with the FP32 model.
    quantized_model = QuantizedNet(model_fp32)

    # Set the quantization configuration for the model.
    quantized_model.qconfig = torch.quantization.get_default_qconfig('x86')

    # Prepare the model for quantization. This inserts quant/dequant stubs.
    torch.quantization.prepare(quantized_model, inplace=True)

    if calibrate:
        # Calibrate the quantized model using the calibration data loader.
        with torch.no_grad():
            for inputs, _ in CAL_LOADER:
                inputs = inputs.cpu()  # Ensure inputs are on CPU.
                quantized_model(inputs)  # Run calibration step.

    # Convert the prepared model to a fully quantized version.
    torch.quantization.convert(quantized_model, inplace=True)

    return quantized_model