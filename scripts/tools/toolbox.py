from pathlib import Path
from typing import Any, Callable

from tools.project_paths import ProjectPaths

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


class Toolbox:
    @staticmethod
    def read_csv(filename: str) -> pd.DataFrame:
        try:
            data = pd.read_csv(filename)
        except FileNotFoundError:
            print(f"File '{filename}' not found")
            exit(1)

        return data

    @staticmethod
    def print_columns_names(data: pd.DataFrame) -> None:
        column_names = data.columns
        to_print: str = "Columns: |"

        for column_name in column_names:
            to_print += f" {column_name} |"

        print(to_print)

    @staticmethod
    def print_column_values(data: pd.DataFrame, column_name: str) -> None:
        print(f"Column {column_name} values:")
        print(data[column_name])

    @staticmethod
    def velocities_local2global(
            imu_vx: pd.Series,
            imu_vy: pd.Series,
            imu_vz: pd.Series,
            roll: pd.Series,
            pitch: pd.Series,
            yaw: pd.Series
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Convert IMU-measured local velocities to global velocities using roll, pitch, and yaw angles.

        Parameters
        ----------
        imu_vx : pd.Series
            Velocities in the x-direction measured by IMU in local coordinates.
        imu_vy : pd.Series
            Velocities in the y-direction measured by IMU in local coordinates.
        imu_vz : pd.Series
            Velocities in the z-direction measured by IMU in local coordinates.
        roll : pd.Series
            Roll angles for rotation in degrees.
        pitch : pd.Series
            Pitch angles for rotation in degrees.
        yaw : pd.Series
            Yaw angles for rotation in degrees.

        Returns
        -------
        tuple[pd.Series, pd.Series, pd.Series]
            Global velocities in the x, y, and z directions.
        """

        vel_local = np.vstack([imu_vx.values, imu_vy.values, imu_vz.values]).T

        # Create rotation matrix
        r = R.from_euler('xyz', np.vstack([roll, pitch, yaw]).T, degrees=True)

        # Apply rotation
        vel_global = r.apply(vel_local)

        return (
            pd.Series(vel_global[:, 0], index=imu_vx.index),
            pd.Series(vel_global[:, 1], index=imu_vy.index),
            pd.Series(vel_global[:, 2], index=imu_vz.index)
        )

    @staticmethod
    def wind_local2global(
            wind_speed: pd.Series,
            wind_angle: pd.Series,
            drone_velocity_x: pd.Series,
            drone_velocity_y: pd.Series
    ) -> tuple[pd.Series, pd.Series]:
        """
        Converts local wind speed and angle to global wind speed components.

        Parameters
        ----------
        wind_speed : pd.Series
            `Series` representing the wind speed measured by the anemometer in meters per second.
        wind_angle : pd.Series
            `Series` representing the wind angle in degrees with respect to the north (clockwise).
        drone_velocity_x : pd.Series
            `Series` representing the drone's velocity in the x direction in global coordinates.
        drone_velocity_y : pd.Series
            `Series` representing the drone's velocity in the y direction in global coordinates.

        Returns
        -------
        tuple[pd.Series, pd.Series]
            Wind speed components in the global x and y directions, adjusted for the drone's velocity.
        """
        # Convert the wind angle from degrees to radians
        wind_direction: pd.Series = pd.Series(np.deg2rad(wind_angle))

        # Calculate wind speed in x and y directions (global coordinates)
        wind_speed_x: pd.Series = - wind_speed * np.sin(wind_direction)
        wind_speed_y: pd.Series = - wind_speed * np.cos(wind_direction)

        return wind_speed_x + drone_velocity_x, wind_speed_y + drone_velocity_y

    @staticmethod
    def anemometr_local2global(
            wind_speed: pd.Series,
            wind_angle: pd.Series,
    ) -> tuple[pd.Series, pd.Series]:
        # Convert the wind angle from degrees to radians
        wind_direction: pd.Series = pd.Series(np.deg2rad(wind_angle))

        # Calculate wind speed in x and y directions (global coordinates)
        vx_anemometr: pd.Series = - wind_speed * np.sin(wind_direction)
        vy_anemometr: pd.Series = - wind_speed * np.cos(wind_direction)

        return vx_anemometr, vy_anemometr

    @staticmethod
    def velocities2positions(
            global_vx: pd.Series,
            global_vy: pd.Series,
            global_vz: pd.Series,
            time: pd.Series
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate positions by integrating velocities over time.

        Parameters
        ----------
        global_vx : pd.Series
            Velocities in the x direction in global coordinates.
        global_vy : pd.Series
            Velocities in the y direction in global coordinates.
        global_vz : pd.Series
            Velocities in the z direction in global coordinates.
        time : pd.Series
            Time

        Returns
        -------
        tuple[pd.Series, pd.Series, pd.Series]
            x, y, z positions calculated from velocities.
        """

        d_time: pd.Series = time.diff()
        x: pd.Series = (global_vx * d_time).cumsum()
        y: pd.Series = (global_vy * d_time).cumsum()
        z: pd.Series = (global_vz * d_time).cumsum()

        return x, y, z

    @staticmethod
    def quaternions2roll_pitch_yaw(
            orientation_x: pd.Series,
            orientation_y: pd.Series,
            orientation_z: pd.Series,
            orientation_w: pd.Series
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Converts quaternions to roll, pitch, and yaw angles.

        Parameters
        ----------
        orientation_x : pd.Series
            x component of quaternion
        orientation_y : pd.Series
            y component of quaternion
        orientation_z : pd.Series
            z component of quaternion
        orientation_w : pd.Series
            w component of quaternion

        Returns
        -------
        roll : pd.Series
            Roll angle in degrees
        pitch : pd.Series
            Pitch angle in degrees
        yaw : pd.Series
            Yaw angle in degrees
        """
        roll = []
        pitch = []
        yaw = []

        length = len(orientation_x)

        for i in range(length):
            quat = [
                orientation_x[i],
                orientation_y[i],
                orientation_z[i],
                orientation_w[i]
            ]

            r = R.from_quat(quat)
            angles = r.as_euler('xyz', degrees=True)

            roll.append(angles[0])
            pitch.append(angles[1])
            yaw.append(angles[2])

        return (
            pd.Series(roll, index=orientation_x.index),
            pd.Series(pitch, index=orientation_y.index),
            pd.Series(yaw, index=orientation_z.index)
        )

    @staticmethod
    def get_flights_ids(
            data: pd.DataFrame,
            conditions: dict[str, Callable[[Any], bool]] = None
    ) -> list[int]:
        """
        Filters given data by given conditions and returns the list of unique flight IDs.

        Parameters
        ----------
        data : pd.DataFrame
            Data to be filtered.
        conditions : dict[str, Callable[[Any], bool]]
            Dictionary where key is the name of the column and value is a callable that takes an element of this column and returns a boolean.

        Returns
        -------
        list[int]
            The list of unique flight IDs after filtering the data.
        """
        filtered_data = data.copy()

        if conditions:
            for column, condition in conditions.items():
                if column not in filtered_data.columns:
                    raise ValueError
                filtered_data = filtered_data[filtered_data[column].apply(condition)]
                if filtered_data.empty:
                    break

        return list(filtered_data["flight"].unique())

    @staticmethod
    def lat_long_height2xyz(
            position_x: pd.Series,
            position_y: pd.Series,
            position_z: pd.Series,
            earth_radius: int = 6371000
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Converts latitude, longitude, and height to a Cartesian coordinate system.

        Parameters
        ----------
        position_x : pd.Series
            Latitude in degrees.
        position_y : pd.Series
            Longitude in degrees.
        position_z : pd.Series
            Height in meters.
        earth_radius : int, optional
            The radius of the Earth in meters. Defaults to 6,371,000.

        Returns
        -------
        tuple[pd.Series, pd.Series, pd.Series]
            x, y, z coordinates of the position in meters - (0, 0, 0 - the initial position of the drone).
        """
        # Total height
        Rh: pd.Series = position_z + earth_radius

        # Get initial position (x,y - degrees, z - meters)
        y0: float = position_y[0]
        x0: float = position_x[0]
        z0: float = position_z[0]

        # Convert degrees to radians
        # Changes the position from degrees to radians
        delta_x_rad = np.deg2rad(position_x - x0)
        delta_y_rad = np.deg2rad(position_y - y0)

        # Read position
        position_y_rad = np.deg2rad(position_y)

        # Convert to cartesian - (0, 0, 0) is the drone's initial position
        x = delta_x_rad * np.cos(position_y_rad) * Rh
        y = delta_y_rad * Rh
        z = position_z - z0

        return x, y, z

    @staticmethod
    def get_all_paths_from_dir(target_dir: Path) -> list[Path]:
        return [f for f in target_dir.iterdir() if f.is_file()]

    @staticmethod
    def create_dataframes_list(train_ids_list: list[int]) -> list[pd.DataFrame]:
        """Reads the flight data of specified flights from their respective files and returns that data as a list of pandas `DataFrames`."""
        dfs_list: list[pd.DataFrame] = []

        for flight_id in train_ids_list:
            df = Toolbox.read_csv(ProjectPaths.flights_data / f"{flight_id}.csv")
            dfs_list.append(df)

        return dfs_list
