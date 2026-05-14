###############################################################################
# Certus Imports
###############################################################################
from AudioMlSpecTools import HighPassFilter
from gsclipper import AmplitudeGunshotExtractor, OnsetChecker, SnrChecker, SharpnessChecker, RoughnessChecker


###############################################################################
# Constants
###############################################################################
SAMPLE_RATE = 44100
CLIP_DURATION_SECS = 1

AUDIO = "./scripts/res/LILYPAD_h0a15a3bcfjs_9mm124tmcgr_15_DOM2-USB.wav"


###############################################################################
# Setup
###############################################################################
extractor = AmplitudeGunshotExtractor(
    relative_audio_cutoff=0.5,
    sample_rate=SAMPLE_RATE,
    clip_duration_seconds=CLIP_DURATION_SECS,
    trim_rewind_secs=0.1,
    show_parent_file_plot=False,
    show_clip_plot=False,
    print_debug=False,

    preproc=HighPassFilter(sample_rate=SAMPLE_RATE, cutoff_freq=7000, rolloff_db=6),
    postproc=[
        OnsetChecker(sample_rate=SAMPLE_RATE),
        SnrChecker(),
        SharpnessChecker(sample_rate=SAMPLE_RATE),
        RoughnessChecker(),
    ],
)


###############################################################################
# ! Main
###############################################################################
def main():
    gunshot_clips = extractor(AUDIO)
    print(f"Num clips: {len(gunshot_clips)}")
    print("Doubt votes: ", [c.doubt_votes for c in gunshot_clips])


if __name__ == "__main__":
    main()
