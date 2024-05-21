# Seismic feature calculation workflow
There are two types (Type A and Type B) <br>
or five sets (Benford's Law, waveform, spectrum, spectrogram, and network sets) seismic features will be calculated

# How to run this code?
## 1, Prepare your seismic data
### 1.1 Format your data as 
Make sure your seismic data are stored as the following structure <br>
~/2019/ILL13/EHZ/9S.ILL13.EHZ.2019.138 <br>
~/Year/Station/Component/Network.Station.Component.Year.Julday <br>
### 1.2 Link you seismic data
Change the file name in [1cal_TypeA_TypeB.py](1cal_TypeA_TypeB.py)
```python
if platform.system() == "Darwin": # your local PC name
    sys.path.append('/Users/qizhou/#python/functions/')
elif platform.system() == "Linux":  # your remote server name
    sys.path.append('/storage/vast-gfz-hpc-01/home/qizhou/2python/functions/')
```

## 1, Change the path for

### 1.1 In [1cal_TypeA_TypeB.py](1cal_TypeA_TypeB.py)
```python
if platform.system() == "Darwin": # your local PC name
    sys.path.append('/Users/qizhou/#python/functions/')
elif platform.system() == "Linux":  # your remote server name
    sys.path.append('/storage/vast-gfz-hpc-01/home/qizhou/2python/functions/')
```

Inline math: $E = mc^2$

# This is an italicized title using asterisks

## _This is an italicized subtitle using underscores_

### *This is another italicized subtitle using asterisks*
