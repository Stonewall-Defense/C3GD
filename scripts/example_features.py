###############################################################################
# Certus Imports
###############################################################################
from AudioMlSpecTools import load_wav, FeatureChannel, FeatureSource
from audio_tensor_plotter import plot_with_time_domain


###############################################################################
# Constants
###############################################################################
SAMPLE_RATE = 48_000
AUDIO_DURATION_SEC = 1

N_FFT = 1024
HOP_LEN = N_FFT // 4
N_MELS = N_FFT // 8


###############################################################################
# Setup
###############################################################################
CHANNELS = [
    FeatureChannel(SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LEN, n_filters=N_MELS, is_logarithmic=True, is_mel=True),
    FeatureChannel(SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LEN, n_filters=N_MELS, is_logarithmic=True, is_mel=False)
]
FEATURE_SOURCE = FeatureSource(CHANNELS)

AUDIO, _ = load_wav("scripts/res/380ACP-example-audio.wav", target_sr=SAMPLE_RATE, duration_secs=AUDIO_DURATION_SEC)


###############################################################################
# ! MAIN
###############################################################################
def main():
    specs = FEATURE_SOURCE.forward(AUDIO)
    print(specs.shape)

    plot_with_time_domain(specs, AUDIO, SAMPLE_RATE, "Feature Generation Example")


if __name__ == "__main__":
    main()
