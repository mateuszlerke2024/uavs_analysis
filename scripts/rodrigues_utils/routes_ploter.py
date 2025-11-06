import matplotlib.pyplot as plt
import pandas as pd
from os import mkdir

from tools.toolbox import Toolbox
from tools.project_paths import ProjectPaths

# ! --- CONFIG PARAMETERS ---
route: str = "R5"
limit: int = 50
font_size: int = 20
add_legend: bool = True
# ! -------------------------

if __name__ == '__main__':
    # Read data
    data: pd.DataFrame = Toolbox.read_csv(ProjectPaths.parameters)
    flights_ids: list[int] = Toolbox.get_flights_ids(data=data, conditions={"route": (lambda x: str(x) == route)})

    # Configure a global font before creating plots
    plt.rc('font', size=font_size)

    # Create a figure with two subplots
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    count: int = 0
    for flight_id in flights_ids:
        # Check maximum number of flights (to increase clarity)
        if count >= limit:
            break

        # Read data
        fd: pd.DataFrame = Toolbox.read_csv(ProjectPaths.flights_data / f"{flight_id}.csv")

        # Integrate to get the trajectory
        dt = fd['time'].diff().mean()
        x = (fd['vx_imu'] * dt).cumsum()
        y = (fd['vy_imu'] * dt).cumsum()
        z = (fd['vz_imu'] * dt).cumsum()

        # Plot trajectory (x, y)
        ax1.plot(x, y, label=f"Flight {flight_id}")

        # Plot trajectory (time, height)
        ax2.plot(fd["time"], z, label=f"Flight {flight_id}")

        # Increase flight counter
        count += 1

    # Add labels to (x, y) plot
    ax1.set_xlabel("x [m]")
    ax1.set_ylabel("y [m]")
    ax1.grid(True)

    # Add labels to (time, height) plot
    ax2.set_xlabel("time [s]")
    ax2.set_ylabel("height [m]")
    ax2.grid(True)

    # Add legend (if the number of flights is too high, it is recommended to change `add_legend` to False)
    if add_legend:
        ax1.legend()
        ax2.legend()

    # Configure plot parameter
    plt.tight_layout()

    # Save figure
    try:
        mkdir(ProjectPaths.results)
    except FileExistsError:
        pass
    
    try:
        mkdir(ProjectPaths.routes_all)
    except FileExistsError:
        pass

    plt.savefig(ProjectPaths.routes_all / f"{route}_all.pdf")
    print("\033[92m" + f"Plot saved in {ProjectPaths.routes_all / f'{route}_all.pdf'}" + "\033[0m")
