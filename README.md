# ML-BL-enhance-DF-EWs
Codes for “Enhancing debris flow warning via seismic feature selection and machine learning model evaluation” <br>
[Preprint](https://doi.org/10.22541/essoar.172183990.09256095/v1), DOI: https://doi.org/10.22541/essoar.172183990.09256095/v1 <br>
Email: qi.zhou@gfz-potsdam.de

# Note (2024-10-22)
The project has been refactored, the output was moved out of this project.


## Workflow
### 1, seismic data prepare
you can download the raw seismic data as [fetch_raw_seismic_data](./data_input/fetch_raw_seismic_data) <br>
and make sure your data was acrivhed as 

```sh
catchment_name/
├── year1/
│   └── station1/
│       └── component1/
│           └── 9J.IGB01.HHZ.2014.002.mseed
│           └── 9J.IGB01.HHZ.2014.003.mseed
│       └── component2/
│       └── component3/
├── year2/
├── meta_data/
│   └── NetworkCode_year1_year2.xml/
```
then put your seismic data directory in [config_dir](./config/config_dir.py) <br>

### 2, calculate features
different seismic source have differnt remove sensor response methods [remove_sensor_response](./calculate_features/remove_sensor_response.py) <br>
and prepare the sbatch file to submit your job as jod array, please refer [2021USA](./calculate_features/2021USA) <br>

### 3, train and test the model
please double check the [config_dir](./config/config_dir.py) and make sure the input-output directory <br>
and prepare the sbatch file as [run_ensemble_model](.functions/sbatch/run_ensemble_model.sh) <br>
run [tree_ensemble_main](./functions/tree_ensemble_main.py) for ensemble model <br>
run [lstm_main](./functions/lstm_main.py) for LSTM model <br>


### 3, dual test the trained model
dual test the trained model for robustness, <br>
the seismic data outside the Illgraben catchment can be found at [data_catalogue](./data_input/data_catalogue.md)