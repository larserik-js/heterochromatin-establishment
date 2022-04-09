import argparse


# The function returns an integer, a NoneType object, or raises an error message
def _none_or_int(value):
    if value == 'None':
        return None
    else:
        try:
            int_value = int(value)
        # Raising a TypeError will cause the parser.add_argument function
        # to display a formatted error message
        except:
            raise TypeError
        else:
            return int_value


def _get_parser_args():
    # Argparser from command line
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument(
        '--model',
        type=str,
        default='CMOL',
        choices=['CMOL', 'S_magnetic', 'S_A_magnetic'],
        help=('Which physical model to use. '
              + 'The choices are: "CMOL", "S_magnetic", and "S_A_magnetic".')
    )

    # Run on Cell computers
    parser.add_argument(
        '--run_on_cell',
        type=int,
        default=0,
        help='Run on NBI Cell computers or not.')

    # Multiprocessing
    parser.add_argument(
        '--n_processes',
        type=int,
        default=1,
        help='Total number of processes.')

    parser.add_argument(
        '--pool_size',
        type=int,
        default=25,
        help='The number of workers.')

    parser.add_argument(
        '--multiprocessing_param',
        type=str,
        default='seed',
        choices=['seed', 'alpha_1', 'rms'],
        help=('The parameter different in each process '
              + 'when using multiprocessing. '
              + 'The choices are: "seed", "alpha_1", and "rms".')
    )

    # Additionally generates and saves an animation
    parser.add_argument(
        '--animate',
        type=int,
        default=0,
        help='If true, creates an animation.')

    # Seeding
    parser.add_argument(
        '--set_seed',
        type=int,
        default=1,
        help='If true, sets random seed for Numpy, Numba, and Torch.')

    # Random seed
    parser.add_argument(
        '--min_seed',
        type=int,
        default=0,
        help='Random seed for Pytorch, Numpy and Numba. '
             + 'The minimum value if multiple values are used.')

    # No. of monomers
    parser.add_argument(
        '--N',
        type=int,
        default=40,
        help='The number of monomers in the system.')

    # Equilibrium spring length
    parser.add_argument(
        '--l0',
        type=float,
        default=1,
        help='Equilibrium spring length.')

    # Noise
    parser.add_argument(
        '--noise',
        type=float,
        default=0.5,
        help=('The dynamic noise level. '
              + 'Should be the product of some constant and the parameter l0.')
    )

    # Time-step size
    parser.add_argument(
        '--dt',
        type=float,
        default=0.02,
        help='The time step size.')

    # No. of time-steps
    parser.add_argument(
        '--t_total',
        type=int,
        default=1000,
        help='The number of time-steps.')

    # Potential weight
    parser.add_argument(
        '--interaction_size',
        type=float,
        default=50,
        help='Scales the strength of the two-interaction potential.')

    # rms
    parser.add_argument(
        '--rms',
        type=float,
        default=2.0,
        help=('The root-mean-square of the distances of the monomers '
              + 'to the center-of-mass. '
              + 'This value translates directly into a pressure_size value.')
    )

    ## State parameters
    # Allow states to change
    parser.add_argument(
        '--allow_state_change',
        type=int,
        default=1,
        help='Enables the monomer states to change.')

    # Initial polymer state
    parser.add_argument(
        '--initial_state',
        type=str,
        default='A',
        help='The initial polymer state.')

    # Allow cell division
    parser.add_argument(
        '--cell_division',
        type=int,
        default=0,
        help='Enables cell division.')

    # Include cenH region
    parser.add_argument(
        '--cenH_size',
        type=int,
        default=0,
        help='The size of the cenH region.')

    parser.add_argument(
        '--cenH_init_idx',
        type=int,
        default=16,
        help='The index of the first monomer of the cenH region.')

    parser.add_argument(
        '--write_cenH_data',
        type=int,
        default=0,
        help='Collects data on the spreading of the silent state.')

    # ATF1
    # The position of a single silent monomer
    parser.add_argument(
        '--ATF1_idx',
        type=_none_or_int,
        default=None,
        help=('The index of the position of the ATF1 protein. '
              + 'For the purpose of this script it is realized as '
              + 'one constantly silent monomer. '
              + 'The default values is "None", which results in no ATF1.')
    )

    ## Recruited conversion probabilities
    # Towards S
    parser.add_argument(
        '--alpha_1',
        type=float,
        default=0.07,
        help=('The reaction rate for the recruited conversions of '
              + 'A to U and U to S.')
    )

    parser.add_argument(
        '--alpha_1_const',
        type=float,
        default=1,
        help=('The reaction rates for the recruited conversions of'
              + 'A to U and U to S.')
    )

    # Towards A
    parser.add_argument(
        '--alpha_2',
        type=float,
        default=0.1,
        help=('The reaction rate for the recruited conversions of '
              + 'S to U and U to A.')
    )

    ## Noisy conversion probability
    parser.add_argument(
        '--beta',
        type=float,
        default=0.004,
        help='The reaction rate for the noisy conversions of all states.')

    # Gather all arguments in a namespace object
    params = parser.parse_args()
    # Transform to dictionary
    params = vars(params)
    
    return params

# Extract all parameters
params = _get_parser_args()