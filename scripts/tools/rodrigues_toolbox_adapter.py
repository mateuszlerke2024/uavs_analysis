import pandas as pd

from .toolbox import Toolbox


class RodriguesToolboxAdapter:
    @staticmethod
    def quaternions2roll_pitch_yaw(data: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
        return Toolbox.quaternions2roll_pitch_yaw(data["orientation_x"], data["orientation_y"], data["orientation_z"], data["orientation_w"])

    @staticmethod
    def imu_velocities2global_velocities(
            data: pd.DataFrame,
            roll: pd.Series,
            pitch: pd.Series,
            yaw: pd.Series
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        return Toolbox.velocities_local2global(data["velocity_x"], data["velocity_y"], data["velocity_z"], roll, pitch, yaw)

    @staticmethod
    def velocities2positions(global_vx: pd.Series, global_vy: pd.Series, global_vz: pd.Series, time: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
        return Toolbox.velocities2positions(global_vx, global_vy, global_vz, time)

    @staticmethod
    def lat_long_height2xyz(data: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
        return Toolbox.lat_long_height2xyz(data["position_x"], data["position_y"], data["position_z"])

    @staticmethod
    def wind_local2global(data: pd.DataFrame, velocity_x: pd.Series, velocity_y: pd.Series) -> tuple[pd.Series, pd.Series]:
        return Toolbox.wind_local2global(data["wind_speed"], data["wind_angle"], velocity_x, velocity_y)

    @staticmethod
    def anemometr_local2global(data: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        return Toolbox.anemometr_local2global(data["wind_speed"], data["wind_angle"])
