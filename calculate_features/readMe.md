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
    if year in [2013, 2014]:  # 2013-2014 data
        OUTPUT_DIR, SAC_DIR = set_in_out_path(input_year, input_station, input_component, input_window_size)
        SPS = 200
        RULER = 1e3
        miniSEED = f"GM.{input_station}.{input_component}.{input_year}"
    elif year not in [2013, 2014]:  # 2017-2020 data
        OUTPUT_DIR, SAC_DIR = set_in_out_path(year, input_station, input_component, input_window_size)
        SPS = 100
        RULER = 1e2
        miniSEED = f"9S.{input_station}.{input_component}.{input_year}"
    else:
        print("error in OUTPUT_DIR, SAC_DIR = set_in_out_path")
```

## 2, Change the input and out path

### 2.1 In [1cal_TypeA_TypeB.py](1cal_TypeA_TypeB.py)
```python
if platform.system() == "Darwin": # your local PC name
    sys.path.append('/Users/qizhou/#python/functions/')
elif platform.system() == "Linux":  # your remote server name
    sys.path.append('/storage/vast-gfz-hpc-01/home/qizhou/2python/functions/')
```

then run the function
```python
set_in_out_path (input_year, input_station, input_component, input_window_size)
```

### 2.2 In [submitStep1.sh](submitStep1.sh)
Request your resources and change the parameters
```sh
#SBATCH --output /home/qizhou/1projects/dataForML/out60/logs/step1/2017-2020/out_%A_%a_%x.txt 		# Standard Output Log File (for Job Arrays)
#SBATCH --error  /home/qizhou/1projects/dataForML/out60/logs/step1/2017-2020/err_%A_%a_%x.txt 		# Standard Error Log File (for Job Arrays)
```

### 2.3 In [2cal_TypeB_network.py](2cal_TypeB_network.py)
Chnage the path
```python
OUTPUT_DIR = "/home/qizhou/1projects/dataForML/out60/" + str(input_year) + "/"
```

### 2.4 In [submitStep2.sh](submitStep2.sh)
Request your resources and change the parameters
```sh
#SBATCH --output /home/qizhou/1projects/dataForML/out60/logs/step2/out_%A_%a_%x.txt 		# Standard Output Log File (for Job Arrays)
#SBATCH --error  /home/qizhou/1projects/dataForML/out60/logs/step2/err_%A_%a_%x.txt 		# Standard Error Log File (for Job Arrays)
```
