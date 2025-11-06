import matplotlib.pyplot as plt
import pandas as pd

from tools.toolbox import Toolbox
from tools.project_paths import ProjectPaths

from os import mkdir

# ! --- CONFIG PARAMETERS ---
font_size: int = 18
# ! -------------------------

if __name__ == '__main__':
    # Read data
    data: pd.DataFrame = Toolbox.read_csv(ProjectPaths.parameters)

    # Count flights per payload
    flights_payloads: list[int] = [0, 250, 500]
    flights_payloads_counts: dict[int, int] = {}

    for flight_payload in flights_payloads:
        flights_payloads_counts[flight_payload] = len(
            Toolbox.get_flights_ids(
                data=data,
                conditions={"route": (lambda x: str(x) == 'R1'),
                            "payload": (lambda x: x == flight_payload)}
            )
        )

    # Print results
    print(f"Flights payload 0: {flights_payloads_counts[0]}, payload 250: {flights_payloads_counts[250]}, payload 500: {flights_payloads_counts[500]}")

    # Plot results
    _, ax = plt.subplots(figsize=(6, 6))
    plt.rcParams['font.size'] = font_size

    ax.pie([flights_payloads_counts[0], flights_payloads_counts[250], flights_payloads_counts[500]],
           labels=[f'0g ({flights_payloads_counts[0]})',
                   f'250g ({flights_payloads_counts[250]})',
                   f'500g ({flights_payloads_counts[500]})'],
           autopct='%1.1f%%',
           startangle=90)
    ax.axis('equal')

    # Configure plot
    plt.tight_layout()

    # Save figure
    try:
        mkdir(ProjectPaths.results)
    except FileExistsError:
        pass

    plt.savefig(ProjectPaths.results / "flights_per_payload.pdf")
    print("\033[92m" + f"Plot saved in {ProjectPaths.results / 'flights_per_payload.pdf'}" + "\033[0m")
