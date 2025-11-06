from platformdirs import user_documents_dir
from pathlib import Path

class ProjectPaths:
    base_dir = Path(user_documents_dir()) / "UAVs Analysis"
    data = base_dir / "data"
    flights_data = data / "flights"
    parameters = data / "parameters.csv"
    results = base_dir / "results"
    energy_results = results / "energy"
    # Rodrigues
    raw_data = data / "raw_flights"
    routes_all = results / "routes_all"
    routes_specific = results / "routes_specific"
    speed_analysis = results / "speed_analysis"
