from scripts.models.abc_predictive_model import APredictiveModel
from scripts.models.final_model import FinalModel
from scripts.tools.toolbox import Toolbox
from scripts.tools.tests_maker import TestsMaker
from scripts.tools.project_paths import ProjectPaths
from typing import Callable, Any

# ! -------- CONFIG PARAMETERS --------
# -------------- Model ----------------
model: APredictiveModel = FinalModel()

# ------- Limit Training Pool ---------
conditions: dict[str, Callable[[Any], bool]] = {
    'payload': lambda x: x >= 0
}

# -------- Select Test Flight ---------
test_ids: list[int] = []

# ------ Select Forecast Flight -------
forecast_ids: list[int] = []

# ! ----------- END CONFIG ------------


# ------ Example Training Script ------
if __name__ == '__main__':
    param_data = Toolbox.read_csv(ProjectPaths.parameters)

    # * ----------- Train -------------
    # Get flights ids
    train_ids_list = Toolbox.get_flights_ids(param_data, conditions)

    # Prepare data
    dfs_list = Toolbox.create_dataframes_list(train_ids_list)

    # Train model
    model.train(dfs_list)

    # Print training results - coefficients
    model.print_training_results()

    # * ----------- Test --------------
    if test_ids:
        test = TestsMaker(model, test_ids)
        test.execute()

    # * ----------- Use ---------------
    if forecast_ids:
        forecast_dfs = Toolbox.create_dataframes_list(forecast_ids)
        power_forecast = [model.forecast(df) for df in forecast_dfs]
