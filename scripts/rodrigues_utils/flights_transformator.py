import os
import pandas as pd

from tools.toolbox import Toolbox
from tools.project_paths import ProjectPaths
from tools.rodrigues_toolbox_adapter import RodriguesToolboxAdapter

# Read all available flights
data: pd.DataFrame = Toolbox.read_csv(ProjectPaths.parameters)

# Read all available payloads and initial mass
payloads: list[int] = [0, 250, 500, 750]
INITIAL_MASS: int = 3680

for payload in payloads:
    # Select flights - payload
    flights_ids: list[int] = Toolbox.get_flights_ids(data=data, conditions={"payload": (lambda x: x == payload)})

    # Check if any flights meet the conditions
    if len(flights_ids) == 0:
        print(f"No flights found for payload: {payload}")
        continue

    target_path = ProjectPaths.flights_data

    # Check if the folder for processed flights exists
    if not os.path.exists(target_path):
        os.makedirs(target_path)

    # Convert data from local coordinates to global coordinates - (0, 0, 0) is the initial position of the drone.
    # The x-axis is along the line of latitude, the y-axis is along the line of longitude, and the z-axis is height.
    for flight_id in flights_ids:
        # Read data from csv
        flight_data: pd.DataFrame = Toolbox.read_csv(ProjectPaths.raw_data / f"{flight_id}.csv")

        # Get time
        time: pd.Series = flight_data["time"]

        # Get orientations
        roll, pitch, yaw = RodriguesToolboxAdapter.quaternions2roll_pitch_yaw(flight_data)

        # Get velocities measured by IMU
        vx_imu, vy_imu, vz_imu = RodriguesToolboxAdapter.imu_velocities2global_velocities(flight_data, roll, pitch, yaw)

        # Convert wind from local to global
        vx_wind, vy_wind = RodriguesToolboxAdapter.wind_local2global(flight_data, vx_imu, vy_imu)

        vx_anemometer, vy_anemometer = RodriguesToolboxAdapter.anemometr_local2global(flight_data)

        x, y, z = RodriguesToolboxAdapter.lat_long_height2xyz(flight_data)

        # Create a new pd.DataFrame with processed data
        flight_data_processed = pd.DataFrame({
            "time": time,

            "voltage": flight_data["battery_voltage"],
            "current": flight_data["battery_current"],

            "is_moving": flight_data["battery_current"] > 5,

            "x_gps": x,
            "y_gps": y,
            "z_gps": z,

            "vx_wind": vx_wind,
            "vy_wind": vy_wind,

            "vx_anemometer": vx_anemometer,
            "vy_anemometer": vy_anemometer,

            "vx_imu": vx_imu,
            "vy_imu": vy_imu,
            "vz_imu": vz_imu,

            "total_mass": INITIAL_MASS + payload,
        })

        flight_data_processed.to_csv(target_path / f"{flight_id}.csv", index=False)
