# Seismic feature calculation workflow
There are two types (Type A and Type B) \
or five sets (Benford's Law, waveform, spectrum, spectrogram, and network sets) seismic features will be calculated

# How to run this code?
1, Change the path for

if platform.system() == "Darwin": # your local PC name
    sys.path.append('/Users/qizhou/#python/functions/')
elif platform.system() == "Linux":  # your remote server name
    sys.path.append('/storage/vast-gfz-hpc-01/home/qizhou/2python/functions/')
