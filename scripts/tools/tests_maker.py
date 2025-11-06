import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from os import mkdir

from models.abc_predictive_model import APredictiveModel
from tools.project_paths import ProjectPaths
from tools.toolbox import Toolbox
from tools.battery import joules2SoC

# ! -------- CONFIG PARAMETERS --------
font_size: int = 20

# ! ----------------------------------

class TestsMaker:
    def __init__(self, model: APredictiveModel, test_ids: list[int]) -> None:
        """
        Creates a list of flight ids for the given route and payload, and a list to store results.
        Resets the result list at the start of each test.

        Parameters
        ----------
        model: APredictiveModel
            instance of the model to be tested
        test_ids: list[int]
            list of flight ids to test on

        Returns
        -------
        None
        """
        self.tests: list[int] = test_ids

        self.model: APredictiveModel = model

        try:
            mkdir(ProjectPaths.results)
        except FileExistsError:
            pass

        try:
            mkdir(ProjectPaths.energy_results)
        except FileExistsError:
            pass
        
        # APE list
        self.results: list[np.ndarray] = []

        self.time: np.ndarray = np.ndarray([])

        # Initial battery state
        self.batt_state = 1

        # True values
        self.power: np.ndarray = np.ndarray([])
        self.energy: np.ndarray = np.ndarray([])
        self.energy_cum: np.ndarray = np.ndarray([])

        # Predicted values
        self.power_pred: np.ndarray = np.ndarray([])
        self.energy_pred: np.ndarray = np.ndarray([])
        self.energy_pred_cum: np.ndarray = np.ndarray([])

    def _reset(self) -> None:
        """
        Resets all the class variables at the start of each test.

        Returns
        -------
        None
        """
        # Time vector
        self.time = np.ndarray([])

        # Initial battery state        
        self.batt_state = 1

        # True values
        self.power = np.ndarray([])
        self.energy = np.ndarray([])
        self.energy_cum = np.ndarray([])

        # Predicted values
        self.power_pred = np.ndarray([])
        self.energy_pred = np.ndarray([])
        self.energy_pred_cum = np.ndarray([])

    def execute(self) -> None:
        """
        Executes all the tests in the list.
        Finally, the list of results is printed.

        Returns
        -------
        None
        """
        for test in self.tests:
            self._execute_test(test)

        print(f"APE across the tests: {self.results}")

    def _model_evaluation(self) -> None:
        """
        Evaluates the model for a single test.

        Calculates the following metrics:

        - Mean Absolute Error (MAE) and Mean Absolute Percentage Error (MAPE) for energy per time step
        - MAE and MAPE for cumulative energy
        - Coefficient of determination (R-squared) for energy per time step
        - R-squared for cumulative energy
        - Absolute Error (AE) and Absolute Percentage Error (APE) for the difference between last values of cumulative energy true vs predicted
        - Battery state after flight

        Prints the results and stores the APE in the list of results.

        Returns
        -------
        None
        """
        # MAE and MAPE - energy per time step
        mae = np.mean(np.abs(self.energy - self.energy_pred))  # ! Values used in the article
        mask = self.energy != 0
        mape = np.mean(np.abs((self.energy[mask] - self.energy_pred[mask]) / self.energy[mask])) * 100

        # MAE and MAPE - cumulative energy
        mae_cum = np.mean(np.abs(self.energy_cum - self.energy_pred_cum))
        mask = self.energy_cum != 0
        mape_cum = np.mean(np.abs((self.energy_cum[mask] - self.energy_pred_cum[mask]) / self.energy_cum[mask])) * 100

        # R-squared - energy per time step
        ss_res: float = np.sum((self.energy - self.energy_pred) ** 2)
        ss_tot = np.sum((self.energy - np.mean(self.energy)) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan  # ! Values used in the article

        # R-squared - cumulative energy
        ss_res_cum: float = np.sum((self.energy_cum - self.energy_pred_cum) ** 2)
        ss_tot_cum = np.sum((self.energy_cum - np.mean(self.energy_cum)) ** 2)
        r2_cum = 1 - ss_res_cum / ss_tot_cum if ss_tot_cum != 0 else np.nan

        # AE and APE - difference between last values of cumulative energy true vs predicted
        ae = np.abs(self.energy_cum[self.energy_cum.shape[0] - 1] - self.energy_pred_cum[self.energy_cum.shape[0] - 1])
        ape = ae / self.energy_cum[self.energy_cum.shape[0] - 1] * 100  # ! Values used in the article

        # Store results
        self.results.append(ape)

        # Battery state after flight
        self.batt_state = joules2SoC(1, self.energy_pred_cum, wear_capacity_coefficient=1)

        # Print results
        print(f"MAE: {mae:.2f}")
        print(f"MAPE: {mape:.2f}%")

        print(f"MAE cum: {mae_cum:.2f}")
        print(f"MAPE cum: {mape_cum:.2f}%")

        print(f"R² (energy per time interval): {r2:.2f}")

        print(f"R² (accumulated energy): {r2_cum:.2f}")

        print(f"AE: {ae:.2f}")

        print(f"APE: {ape:.2f}%")

        print(f"Forecasted battery state after flight: {self.batt_state.iloc[-1] * 100:.2f}%")

    def _draw_plots(self, test: int) -> None:
        """
        Draws plots of battery state in time, power and energy (instantaneous and cumulative)

        Parameters
        ----------
        test: int
            Number of the test to be executed

        Returns
        -------
        None
        """
        # Set font size
        plt.rcParams['font.size'] = font_size

        # Plot battery state in time        
        plt.figure()
        plt.plot(self.time, self.batt_state * 100)
        plt.xlabel("Time [s]")
        plt.ylabel("Battery state [%]")
        plt.tight_layout()  # auto-adjusts to fit labels
        plt.savefig(ProjectPaths.energy_results / f"battery_soc_{test}.pdf")
        print("\033[92m" + f"Plot saved in {ProjectPaths.energy_results / f'battery_soc_{test}.pdf'}" + "\033[0m")

        # Plot power and energy
        _, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)

        # Left subplot - Power
        axes[0].plot(self.time, self.power, label='Instantaneous power')
        axes[0].plot(self.time, self.power_pred, label='Predicted power')
        axes[0].set_xlabel("Time [s]")
        axes[0].set_ylabel("Power [W]")
        axes[0].legend()

        # Right subplot - Energy
        axes[1].plot(self.time, self.energy_cum / 1000, label='Consumed energy')
        axes[1].plot(self.time, self.energy_pred_cum / 1000, label='Predicted energy')
        axes[1].set_xlabel("Time [s]")
        axes[1].set_ylabel("Cumulative Energy [kJ]")
        axes[1].legend()

        plt.tight_layout()
        plt.savefig(ProjectPaths.energy_results / f"consumed_energy_{test}.pdf")
        print("\033[92m" + f"Plot saved in {ProjectPaths.energy_results / f'consumed_energy_{test}.pdf'}" + "\033[0m")

    def _execute_test(self, test: int) -> None:
        """
        Executes a single test.

        Parameters
        ----------
        test: int
            Number of the test to be executed

        Returns
        -------
        None
        """
        # Read data
        df: pd.DataFrame = Toolbox.read_csv(ProjectPaths.flights_data / f"{test}.csv")

        # Prepare data
        df['delta_time'] = df['time'].diff()
        df = df.dropna()
        df = df.reset_index(drop=True)

        # Calculate predictions
        self.power_pred = self.model.forecast(df)
        self.energy_pred = self.power_pred * df['delta_time']
        self.energy_pred_cum = self.energy_pred.cumsum()

        # Calculate true values
        self.power = (df['voltage'] * df['current']).values
        self.energy = (df['delta_time'] * self.power).values
        self.energy_cum = self.energy.cumsum()

        # Get time vector
        self.time = df['time'].values

        # Calculate metrics - print and plot results
        print(f"------------- TEST: {test} ---------------")

        self._model_evaluation()

        self._draw_plots(test)

        # Reset - ready for the next test
        self._reset()
