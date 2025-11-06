import matplotlib.pyplot as plt
import pandas as pd
from os import mkdir

from tools.toolbox import Toolbox
from tools.project_paths import ProjectPaths
from tools.rodrigues_toolbox_adapter import RodriguesToolboxAdapter

font_size: int = 20


def create_figure(x: pd.Series, y: pd.Series, z: pd.Series, time: pd.Series, route: str, flight_id: int) -> None:
    """
    Create a figure with two plots: (x, y) trajectory with start and end points and (time, height) plot.

    Parameters
    ----------
    x : pd.Series
        X coordinates of the trajectory
    y : pd.Series
        Y coordinates of the trajectory
    z : pd.Series
        Height of the flight
    time : pd.Series
        Time of the flight
    route : str
        Route of the flight
    flight_id : int
        Id of the flight
    """
    # Configure global plots settings
    plt.rc('font', size=font_size)
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot trajectory (x, y) with start and end points
    ax1.plot(x, y)
    ax1.plot(x[0], y[0], marker="o", color="lightgreen")
    n: int = len(x)
    ax1.plot(x[n - 1], y[n - 1], marker="o", color="red")

    # Plot trajectory (time, height)
    ax2.plot(time, z)

    # Configure first plots
    ax1.set_ylabel("y [m]")
    ax1.set_xlabel("x [m]")
    ax1.legend(["trajectory", "start", "end"])
    ax1.grid(True)

    # Configure second plot
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("height [m]")
    ax2.grid(True)

    # Configure plots
    plt.tight_layout()

    # Save figure  
    try:
        mkdir(ProjectPaths.routes_specific)
    except FileExistsError:
        pass
    plt.savefig(ProjectPaths.routes_specific / f"{route}_{flight_id}.pdf")
    print("\033[92m" + f"Plot saved in {ProjectPaths.routes_specific / f'{route}_{flight_id}.pdf'}" + "\033[0m")


if __name__ == '__main__':
    # Read data
    data: pd.DataFrame = Toolbox.read_csv(ProjectPaths.parameters)
    routes = data["route"].unique()

    # Create examples trajectories plots for each route
    for route in routes:
        # Read data
        flight_id: list[int] = Toolbox.get_flights_ids(data=data, conditions={"route": (lambda x: str(x) == route)})
        flight_data: pd.DataFrame = Toolbox.read_csv(ProjectPaths.raw_data / f"{flight_id[0]}.csv")

        # Convert GPS data lat, long, height to local frame coordinates x, y, z
        (x, y, z) = RodriguesToolboxAdapter.lat_long_height2xyz(flight_data)

        # Create a trajectory figure
        create_figure(x, y, z, flight_data["time"], route, len(flight_id))
