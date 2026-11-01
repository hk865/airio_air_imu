from .net import ModelBase
from .cnn import CNNPOS
from .others import Identity, ParamNet
from .code import *
from .model_freq import FreqHybridModel
from .model_freq_improved import ImprovedFreqHybridModel
from .model_freq_improved2 import ImprovedFreqHybridModelWithUpdate
from .model_freq_fusion import EarlyFusionFreqModel
from .model_freq_wavelet_only import WaveletOnlyModel
from .model_wavenet_v1 import WaveletOnlyModel as WaveNetV1
from .model_phase_aware import PhaseAwareFreqHybridModel
from .model_ronin_phase_aware import PhaseAwareVelocityModel

net_dict = {
    'codeposenet': CodePoseNet,
    'codenetkitti': CodeNetKITTI,
    'iden': Identity,
    'cnnpos': CNNPOS,
    'codenet': CodeNet,
    'param': ParamNet,
    'freqnet':FreqHybridModel,
    'freqnet_improved': ImprovedFreqHybridModel,
    'freqnet_improved2': ImprovedFreqHybridModelWithUpdate,
    'freqnet_fusion': EarlyFusionFreqModel,
    'freqnet_wavelet': WaveletOnlyModel,
    'wavenet_v1': WaveNetV1,
    'freqnet_phase_aware': PhaseAwareFreqHybridModel,
    'ronin_phase_aware_velocity': PhaseAwareVelocityModel,
}
