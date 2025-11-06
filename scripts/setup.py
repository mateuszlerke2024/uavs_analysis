from setuptools import setup, find_packages
from os import mkdir
from platformdirs import user_documents_dir
from pathlib import Path
from tools.project_paths import ProjectPaths

setup(
    name="drone_flights",
    version="1.1",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "numpy",
        "matplotlib",
        "scipy",
        "platformdirs",
    ],
)

try:
    mkdir(ProjectPaths.base_dir)
except FileExistsError:
    pass

try:
    mkdir(ProjectPaths.data)
except FileExistsError:
    pass

try:
    mkdir(ProjectPaths.results)
except FileExistsError:
    pass