from models.abc_predictive_model import APredictiveModel


class BaseLineModel(APredictiveModel):
    def __init__(self, delta_time: float = 0.17):
        """
        Parameters
        ----------
        delta_time : float, optional
            The time step in seconds between consecutive records in the flight data.
            Defaults to 0.17.

        Notes
        -----
        This model is a baseline model that forecasts power consumption.
        The power consumption in moving phase and stationary phase is forecast as the mean usage.
        """
        super().__init__()
        self._delta_time: float = delta_time