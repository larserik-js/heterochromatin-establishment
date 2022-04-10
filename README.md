# Heterochromatin establishment
> Developing a 3D polymer model for the establishment of heterochromatin on DNA.

The repository contains a variety of modules used to simulate a 3D polymer, which represents the bead-on-a-string structure of nucleosomes in DNA. Data can be gathered on the behavior of the different monomer states.

## Setup (macOS & Linux)
The necessary libraries for this application can be installed in the following way:
* Navigate into the repository directory, i.e. the location of ``requirements.txt``
* Set up a virtual environment, e.g. via:
```sh
python3 -m venv env
```

* Activate the environment via:
```sh
source env/bin/activate
```
* Install libraries via:
```sh
pip3 install -r requirements.txt
```
The libraries listed in the document are compatible with Python 3.8.10-3.9.12, and were last available for download on 5th April 2022.

## Usage example
The best way to start exploring the repository is probably by generating one or more animations, to see how the polymer behaves in space. The following command starts generating the animation (the files will take up a total of approximately 77 MB, and the process should take around 5 minutes in total):
```sh
python3 main.py --animate=1 --t_total=100000 --cenH_size=8
```
The generated files can be found in ``output/animations/``.

To generate several animations in parallel (using multiprocessing with different seeds), use the multiprocessing argument. E.g. to generate 10 animations, run:
```sh
python3 main.py --animate=1 --t_total=100000 --cenH_size=8 --n_processes=10
```

More usage examples will be added as better input functionality is implemented for plotting.

## Meta
Lars Erik J. Skjegstad - lars_erik_skjegstad@hotmail.com

This project is not open source, and thus does not come with a license.
