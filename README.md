# Master's Thesis
> Developing a physical model for the establishment of heterochromatin on DNA.

The repository contains a variety of modules used to simulate a 3D polymer, which represents the bead-on-a-string structure of nucleosomes in DNA. Data can be gathered on the behavior of the different monomer states.

## Setup (macOS & Linux)
The necessary libraries for this application can be installed in the following way:
* Navigate into the directory of ``requirements.txt``.
* Set up a virtual environment.
* Install libraries via:
```sh
pip install -r requirements.txt
```

## Usage example
The best way to start exploring the repository is probably by generating one or more animations, to see how the polymer behaves in space. The following command starts generating the animation (the files will take up a total of approximately 77 MB, and the process should take around 5 minutes in total):
```sh
python3 main.py --animate=1 --t_total=100000 --cenH_size=8
```
The files can be found in ``output/animations/``.

To generate several animations at once (using multiprocessing with different seeds), use the multiprocessing argument. E.g. to generate 10 animations, run:
```sh
python3 main.py --animate=1 --t_total=100000 --cenH_size=8 --n_processes=10
```

More usage examples will be added as better input functionality is implemented for plotting.

## Meta
Lars Erik J. Skjegstad - lars_erik_skjegstad@hotmail.com

This project is not open source, and thus comes without a license.
