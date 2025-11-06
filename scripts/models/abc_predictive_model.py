from abc import ABC, abstractmethod
from typing import Union, final

import numpy as np
import pandas as pd


class APredictiveModel(ABC):
    @abstractmethod
    def __init__(self):
        self._beta: Union[np.ndarray, None] = None
        self._default_usage: float | None = None

    @property
    @final
    def beta(self) -> np.ndarray:
        """
        The beta vector of the model.

        Returns
        -------
        np.ndarray
            The beta vector of the model.

        Raises
        ------
        RuntimeError
            If the model is not trained (beta is not initialized).
        """
        if self._beta is None:
            raise RuntimeError("Beta is not initialized. Train model first.")

        return self._beta

    @property
    @final
    def default_usage(self) -> float:
        """
        The default usage value of the model in the stationary phase.

        Returns
        -------
        float
            The default usage value of the model in the stationary phase
        
        Raises
        ------
        RuntimeError
            If the model is not trained (beta is not initialized).
        """
        if self._default_usage is None:
            raise RuntimeError("Default usage is not initialized. Train model first.")

        return self._default_usage

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
        Virtual method: override in subclasses if needed.
        """
        return []

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
        Virtual method: override in subclasses if needed.
        """
        return df

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
        Virtual method: override in subclasses if needed.
        """
        return power

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
        Virtual method: override in subclasses if needed.
        """
        return power

    @final
    def train(self, dfs_list: list[pd.DataFrame]) -> None:
        """
        Train the model using a list of `DataFrames` with flight data

        Parameters
        ----------
        dfs_list : list[pd.DataFrame]
            List of `DataFrames` with flight data

        Returns
        -------
        None

        Notes
        -----
        Does the following steps:

        1. Transforms each `DataFrame` in the list using the `spec_model_calc` method
        2. Merges the transformed `DataFrames` into a single one
        3. Calculates the instantaneous power indicator for each row in the merged `DataFrame`
        4. Splits the merged `DataFrame` into two parts based on the current value
        5. Calculates the beta vector using the least squares method on the moving phase part
        6. Calculates the default usage using the mean of the power indicator in the stationary phase part
        """
        # Transform dataframes extracting features according to a specific model
        transformed_dfs_list: list[pd.DataFrame] = []
        for df in dfs_list:
            df = self.spec_model_calc(df)
            transformed_dfs_list.append(df)

        # Merge dataframes
        try:
            dfs: pd.DataFrame = pd.concat(transformed_dfs_list, ignore_index=True)
        except ValueError:
            print("\033[91mTraining set is empty, unable to train model. Check your conditions (main.py, config parameters section).\033[0m")
            exit(-1)

        # Calculate instantaneous power indicator (transform power [voltage * current] -> power_indicator)
        dfs['power_indicator'] = self.transform_power(dfs['voltage'] * dfs['current'])

        # Copy and divide dataframes
        df1 = dfs[dfs['is_moving'] == True]
        df2 = dfs[dfs['is_moving'] == False]

        # * MOVING PHASE *
        # Prepare X and y vectors
        X = df1[self.predictors_list].values
        y = df1['power_indicator'].values

        # Add intercept
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        # Solve for beta vector using least squares method
        self._beta, *_ = np.linalg.lstsq(X, y, rcond=None)


        # * STATIONARY PHASE *
        # Calculate default usage
        self._default_usage = df2['power_indicator'].mean()

    @final
    def forecast(self, df: pd.DataFrame) -> pd.Series:
        """
        Forecast power consumption based on the given `DataFrame`.

        Parameters
        ----------
        df : pd.DataFrame
            `DataFrame` with flight data

        Returns
        -------
        pd.Series
            Power consumption forecast
        """
        # Calculate additional features specific to this model
        df = self.spec_model_calc(df)

        # Initialise power vector with default usage
        power: pd.Series = pd.Series(self._default_usage, index=df.index)

        # Create mask for moving phase
        mask: pd.Series = df['is_moving'] == True

        # Set power to default usage (intercept) for moving phase intervals
        power[mask] = self.beta[0]

        # Add contributions from each predictor
        for i, predictor in enumerate(self.predictors_list):
            power[mask] += df.loc[mask, predictor] * self.beta[i + 1]

        # Re-transform power (power indicator -> power)
        return self.retransform_power(power)

    @final
    def print_training_results(self) -> None:
        """Prints training results to the console."""
        if self._beta is None or self._default_usage is None:
            raise RuntimeError("Model is not trained. Train model first.")

        print("Moving phase:")

        predictors_list: list[str] = self.predictors_list
        predictors_list.insert(0, 'intercept')
        for (beta, predictor) in zip(self.beta, predictors_list):
            print(f"beta: {beta}, predictor: {predictor}")

        print("Stationary phase:")
        print(self.default_usage)
