# Setup Guide

### Setup Code
1. Download the source code:
```bash
git clone https://github.com/mateuszlerke2024/uavs_analysis
```

2. Download all necessary dependencies:
```bash
cd scripts
pip install -e .
```

### Folders Structure
```
~/Documents/UAVs Analysis
|--data
|  |--flights
|  |  |--1.csv
|  |  |--2.csv
|  |  |-- ...
|  |--parameters.csv
|
|--results
|  |-- ...
```

### Required Training Data
Prepare model training data. The data should consist of:
1. A `flights` folder of files in CSV format, containing records of the flights, named `1.csv`, `2.csv`, etc.
2. A single `parameters.csv` file, listing all the files in the `flights` directory, categorised by some 
   parameters. Those parameters are not required, and are only used to optionally limit the training pool
   to a certain subset of the flights.

For example, if some of the flights had an additional payload attached, the `parameters.csv` file should contain in 
the first column the ID of the flight, and in the second column the payload of the flight.

To use the proposed model, all `flight` files should contain:
- `time` - time in seconds,
- `voltage` - voltage in volts,
- `current` - current in amperes,
- `vx_anemometer`, `vy_anemometer` - horizontal airspeed in m/s,
- `vz_imu` - vertical speed in m/s,
- `total_mass` - total drone's mass in grams (mass of the drone, battery and payload).

To forecast the power consumption using the proposed model, a file describing the planned flight must contain all 
of the above, except `voltage` and `current`.


### Rodrigues Specific

If you are using Rodrigues' data, preparation of the training data is handled automatically.

1. Download data from [here](https://kilthub.cmu.edu/articles/dataset/Data_Collected_with_Package_Delivery_Quadcopter_Drone/12683453)

2. Place `parameters.csv` file and extracted `flights.zip` folder in the `~/Documents/UAVs Analysis/data` folder (as shown above)

3. Rename `flights` folder to `raw_flights`

4. Run python module `flights_transformator.py` located in `scripts/rodrigues_utils` folder

5. The flights data will be transformed and the resulting files will be saved in the `~/Documents/UAVs Analysis/data/flights` folder


More helpful scripts, which are dedicated to Rodrigues' data, can be found in `scripts/rodrigues_utils`

### Running the Code
1. Open `main.py`.
2. Configure the training data pool by adjusting the `conditions` dict at the beginning of the file. To use the 
   entire data set, delete the existing condition and leave the dict empty:
```
conditions: dict[str, Callable[[Any], bool]] = {}
```

3. Optionally, select the flights to use as test cases for verification of the model's quality:
```
test_ids: list[int] = []
```

4. Optionally, select the flights to forecast:
```
forecast_ids: list[int] = []
```

5. Execute `main.py`.

6. Summaries and plots will be saved in the `results` folder (see [**Folders Structure**](#folders-structure)).
