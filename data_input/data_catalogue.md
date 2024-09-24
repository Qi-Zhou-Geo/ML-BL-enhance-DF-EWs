## Data Source
All training (2017-2019) and testing (2017-2019) data is open access.  <br>
However, there are some data need permiserion to access from Qi Zhou.

## How to access the online seismic data?
download data from online server <br>
[data_from_FDSN_server.py](fetch_raw_seismic_data/data_from_FDSN_server.py) <br>

## How to access the internal GFZ seismic data?
download data from online server <br>
[data_from_GFZ_server.py](fetch_raw_seismic_data/data_from_GFZ_server.py)

This data is stored in <br>
/storage/vast-gfz-hpc-01/project/seismic_data_qi/seismic



### 2, 2023 Luding data
2 stations (WD01, WD02) and 3 components (BHE, BHN, BHZ) <br>
/storage/vast-gfz-hpc-01/project/seismic_data_qi/seismic/CN/Luding/ <br>
*WD01 data is ready, WD02 may need to change* <br>

### 3, 2022 Ergou data
2 stations (EG01, EG02) and 3 components (BHE, BHN, BHZ) <br>
/storage/vast-gfz-hpc-01/project/seismic_data_qi/seismic/CN/2022/ErGou/ <br>
*Ready*

### 4, 2023 Dongchuan
3 stations (FT01, FT02) and 3 components (BHE, BHN, BHZ) <br>
/storage/vast-gfz-hpc-01/project/seismic_data_qi/seismic/CN/2023/Dongchuan/ <br>
*Ready*

### 5, 2022 Fotangbagou data
2 stations (FT01, FT02) and 3 components (BHE, BHN, BHZ) <br>
/storage/vast-gfz-hpc-01/project/seismic_data_qi/seismic/CN/2022/ErGou/ <br>
*NOT Ready*

### 6, 2022 Goulinping
1 station and 1 component <br>
/storage/vast-gfz-hpc-01/project/seismic_data_qi/seismic/CN/2022/Goulinping/ <br>
*NOT ready*


---------------
## Model Training and Testing Data Summary
### 1, 2017-2019 Illgraben Training Dataset, EU
Seismic Network Code: 9S <br>
Seismic Stations Used: ILL18, ILL12, ILL13 <br>
Available Components: EHE, EHN, EHZ <br>
Number of Observed Debris Flow Events: 20 <br>

### 2, 2020 Illgraben Testing Dataset, EU
Seismic Network Code: 9S <br>
Seismic Stations Used: ILL18, ILL12, ILL13 <br>
Available Components: EHE, EHN, EHZ <br>
Number of Observed Debris Flow Events: 12 <br>

---------------
## Trained Model Dual Testing Data Summary
### 1, 2022 Illgraben Dataset, EU
Seismic Network Code: 9S <br>
Seismic Stations Used: ILL18, ILL12, ILL13 <br>
Available Components: EHE, EHN, EHZ <br>
Number of Observed Debris Flow Events: 4 <br>

### 2, 2021 Museum Fire Dataset, USA
Seismic Network Code: 1A <br>
Seismic Stations Used: E19A, COCB <br>
Available Components: CHE, CHN, CHZ <br>
Number of Observed Debris Flow Events: 2 <br>

