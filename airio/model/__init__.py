from .code import *
from .model_freq import FreqHybridModel, FreqHybridModelwithRot
from .model_freq_no_interp import FreqHybridModelNoInterp, FreqHybridModelwithRotNoInterp
from .model_ekf import EkfMotionNetwork, EkfMotionNetworkwithRot, SimpleTransformerIMUModel
from .nodel_wavenet_v1 import WaveletOnlyModel, WaveletOnlyModelwithRot
from .model_freq_improved import ImprovedFreqHybridModel, ImprovedFreqHybridModelwithRot
from .model_freq_improved_no_interp import ImprovedFreqHybridModelNoInterp, ImprovedFreqHybridModelwithRotNoInterp
from .model_freq_medium import FreqHybridModelMedium, FreqHybridModelMediumRot
from .model_freq_medium_no_interp import FreqHybridModelMediumNoInterp, FreqHybridModelMediumRotNoInterp
from .model_freq_complex import FreqHybridModelMedium as FreqHybridModelMediumComplex, FreqHybridModelMediumRot as FreqHybridModelMediumRotComplex
from .model_freq_lstm import FreqLSTMModel, FreqLSTMModelwithRot
#from .model_freq_simple import SimpleFreqPhaseModel, SimpleFreqPhaseModelwithRot

net_dict = {
    'codenetmotion': CodeNetMotion,
    'codewithrot': CodeNetMotionwithRot,
    # Frequency-based hybrid models
    'freqhybrid': FreqHybridModel,
    'freqhybridrot': FreqHybridModelwithRot,
    # No interpolation versions
    'freqhybridnointerp': FreqHybridModelNoInterp,
    'freqhybridrotnointerp': FreqHybridModelwithRotNoInterp,
    # EKF-based models
    'ekfmotion': EkfMotionNetwork,
    'ekfmotionrot': EkfMotionNetworkwithRot,
    'simpletransformer': SimpleTransformerIMUModel,
    # Wavelet-based models
    'waveletonly': WaveletOnlyModel,
    'waveletonlyrot': WaveletOnlyModelwithRot,
    # Improved frequency-based models
    'freqimproved': ImprovedFreqHybridModel,
    'freqimprovedrot': ImprovedFreqHybridModelwithRot,
    # Improved no interpolation versions
    'freqimprovednointerp': ImprovedFreqHybridModelNoInterp,
    'freqimprovedrotnointerp': ImprovedFreqHybridModelwithRotNoInterp,
    # Medium frequency-based models
    'freqmedium': FreqHybridModelMedium,
    'freqmediumrot': FreqHybridModelMediumRot,
    # Medium no interpolation versions
    'freqmediumnointerp': FreqHybridModelMediumNoInterp,
    'freqmediumrotnointerp': FreqHybridModelMediumRotNoInterp,
    # LSTM frequency-based models
    'freqlstm': FreqLSTMModel,
    'freqlstmrot': FreqLSTMModelwithRot,
    # Complex wavelet-based models (DTCWT + LeakyReLU)
    'freqcomplex': FreqHybridModelMediumComplex,
    'freqcomplexrot': FreqHybridModelMediumRotComplex,
}
