### plot.py
This file contains the plotting function used.

### calculate_features.py
This file contains the BL calculation and also run the loop for 60 second time window. The output is a pandas dataframe of the data which is not saved. This dataframe is used for the plot.

### utils.py
This file contains some functions that are used repetitively.

### run.py
This is the main run file that works with command line interface for the user to generate plots

#### eg usage:
```bash
python run.py --ruler 300 --scaling 9 --index 7
```
or 
```bash
python run.py --ruler 300 --scaling 9 --data_start "2018-08-05 00:00:00" --data_end "2018-08-10 00:00:00"
```