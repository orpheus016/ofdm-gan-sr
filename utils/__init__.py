# =============================================================================
# CWGAN-GP for OFDM Signal Reconstruction
# Utilities Package Initialization
# =============================================================================

from .ofdm_utils import (
    QAMModulator,
    OFDMModulator, 
    ChannelModel,
    ImageOFDMConverter
)

from .dataset import (
    OFDMDataset,
    SyntheticOFDMDataset,
    create_dataloader,
    generate_test_samples
)

from .quantization import (
    QuantizationConfig,
    compute_scale,
    quantize_tensor,
    dequantize_tensor,
    FakeQuantize,
    QuantizedConv1d,
    export_weights_fpga,
    compute_layer_crc
)

__all__ = [
    # OFDM utilities
    'QAMModulator',
    'OFDMModulator',
    'ChannelModel',
    'ImageOFDMConverter',
    
    # Dataset utilities
    'OFDMDataset',
    'SyntheticOFDMDataset',
    'create_dataloader',
    'generate_test_samples',
    
    # Quantization utilities
    'QuantizationConfig',
    'compute_scale',
    'quantize_tensor',
    'dequantize_tensor',
    'FakeQuantize',
    'QuantizedConv1d',
    'export_weights_fpga',
    'compute_layer_crc'
]
