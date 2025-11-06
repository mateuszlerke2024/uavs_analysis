from models.abc_predictive_model import APredictiveModel

import numpy as np
import pandas as pd

class FinalModel(APredictiveModel):
    def __init__(self, delta_time: float = 0.17):
        """
        Parameters
        ----------
        delta_time : float, optional
            The time step in seconds between consecutive records in the flight data.
            Defaults to 0.17.

        Notes
        -----
        This model is the final model that forecasts power consumption.
        The power consumption in the moving phase is forecast as the linear combination of the following predictors:

        - total_mass
        - force_z
        - force_xy
        - velocity_xy_factor
        - velocity_z_factor
        The power consumption in the stationary phase is forecast as the mean usage in the stationary phase.
        """
        super().__init__()
        self._delta_time: float = delta_time

    @property
    def predictors_list(self) -> list[str]:
        """
        List of predictors used in this model.

        Returns
        -------
        list[str]
            List of predictors

        Notes
        -----
        Overridden method.
        """
        return ['total_mass', 'force_z', 'force_xy', 'velocity_xy_factor', 'velocity_z_factor']

    def transform_power(self, power: pd.Series) -> pd.Series:
        """
        Method that transforms power [W] to power indicator.

        Parameters
        ----------
        power : pd.Series
            Power [W] to be transformed

        Returns
        -------
        pd.Series
            Transformed power indicator

        Notes
        -----
        Overridden method.
        """
        return power ** (2 / 3)

    def retransform_power(self, power: pd.Series) -> pd.Series:
        """
        Method that retransforms power indicator to power [W] for a given flight.

        Parameters
        ----------
        power : pd.Series
            Power indicator to be retransformed

        Returns
        -------
        pd.Series
            Retransformed power [W]

        Notes
        -----
        Overridden method.
        """
        return power ** (3 / 2)

    def spec_model_calc(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates specific features according to the model.

        Parameters
        ----------
        df : pd.DataFrame
            `DataFrame` with flight data

        Returns
        -------
        pd.DataFrame
            `DataFrame` with calculated features

        Notes
        -----
        Overridden method.
        """
        # Calculate additional features specific to this model
        df['xy_air_speed'] = np.sqrt(df['vx_anemometer'] ** 2 + df['vy_anemometer'] ** 2)
        df['xy_air_acceleration'] = np.sqrt((df['vx_anemometer'].diff() / self._delta_time) ** 2 + (df['vy_anemometer'].diff() / self._delta_time) ** 2)
        df['z_acceleration'] = df['vz_imu'].diff() / self._delta_time

        df['force_z'] = df['z_acceleration'] * df['total_mass']
        df['force_xy'] = df['xy_air_acceleration'] * df['total_mass']

        df['velocity_xy_factor'] = df['xy_air_speed'] ** 2 * df['total_mass'] ** (2 / 3)
        df['velocity_z_factor'] = df['vz_imu'] ** 2 * df['total_mass'] ** (2 / 3)

        # Drop NaN values and fix indexes
        df = df.dropna()
        df = df.reset_index(drop=True)
        df.loc[-1] = [0.0] * len(df.columns)
        df.index += 1

        return df
