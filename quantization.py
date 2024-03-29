import torch
import torch.nn as nn
import torch.quantization

from dataset import BATCH_SIZE, WORKERS
from dataset import train_set


CAL_LOADER = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS)

class QuantizedNet(nn.Module):
    def __init__(self, model_fp32):
        super(QuantizedNet, self).__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        self.model_fp32 = model_fp32
    
    def forward(self, x):
        x = self.quant(x)
        x = self.model_fp32(x)
        x = self.dequant(x)
        return x

def quantize_model(model_fp32, calibrate=True):
    model_fp32 = model_fp32.cpu()

    model_fp32.eval()

    # Fuse the model
    model_fp32.fuse_model()

    # Create a quantized model instance
    quantized_model = QuantizedNet(model_fp32)

    # Specify the quantization configuration for weights
    quantized_model.qconfig = torch.quantization.get_default_qconfig('x86')

    # Prepare the model
    torch.quantization.prepare(quantized_model, inplace=True)

    if calibrate:
        # Calibrate the quantized model
        with torch.no_grad():
            for inputs, _ in CAL_LOADER:
                inputs = inputs.cpu()
                quantized_model(inputs)

    torch.quantization.convert(quantized_model, inplace=True)

    return quantized_model
