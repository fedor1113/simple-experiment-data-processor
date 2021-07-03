#!/usr/bin/env python3


from math import log, log10, floor, ceil
from inspect import getfullargspec
from functools import partial
import argparse


# Values for Student's t-distribution for up to 30 degrees of freedom
# For 'zero' degrees value for infinitely many (i.e. for the normal
# distribution) is given
# Values are given for 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998
# and 0.999 credence.
# The table is as follows: STUDENT_COEFFICIENTS[degrees][probability]
# with [0..10] keys corresponding to 0.5 through 0.999 probabilities
STUDENT_COEFFICIENTS = [[0.674, 0.842, 1.036, 1.282, 1.645, 1.96,
                         2.326, 2.576, 2.807, 3.09, 3.291],
                        [1.0, 1.376, 1.963, 3.078, 6.314, 12.71,
                        31.82, 63.66, 127.3, 318.3, 636.6],
                        [0.816, 1.08, 1.386, 1.886, 2.92, 4.303,
                         6.965, 9.925, 14.09, 22.33, 31.6],
                        [0.765, 0.978, 1.25, 1.638, 2.353, 3.182,
                         4.541, 5.841, 7.453, 10.21, 12.92],
                        [0.741, 0.941, 1.19, 1.533, 2.132, 2.776,
                         3.747, 4.604, 5.598, 7.173, 8.61],
                        [0.727, 0.92, 1.156, 1.476, 2.015, 2.571,
                         3.365, 4.032, 4.773, 5.893, 6.869],
                        [0.718, 0.906, 1.134, 1.44, 1.943, 2.447,
                         3.143, 3.707, 4.317, 5.208, 5.959],
                        [0.711, 0.896, 1.119, 1.415, 1.895, 2.365,
                         2.998, 3.499, 4.029, 4.785, 5.408],
                        [0.706, 0.889, 1.108, 1.397, 1.86, 2.306,
                         2.896, 3.355, 3.833, 4.501, 5.041],
                        [0.703, 0.883, 1.1, 1.383, 1.833, 2.262,
                         2.821, 3.25, 3.69, 4.297, 4.781],
                        [0.7, 0.879, 1.093, 1.372, 1.812, 2.228,
                         2.764, 3.169, 3.581, 4.144, 4.587],
                        [0.697, 0.876, 1.088, 1.363, 1.796, 2.201,
                         2.718, 3.106, 3.497, 4.025, 4.437],
                        [0.695, 0.873, 1.083, 1.356, 1.782, 2.179,
                         2.681, 3.055, 3.428, 3.93, 4.318],
                        [0.694, 0.87, 1.079, 1.35, 1.771, 2.16,
                         2.65, 3.012, 3.372, 3.852, 4.221],
                        [0.692, 0.868, 1.076, 1.345, 1.761, 2.145,
                         2.624, 2.977, 3.326, 3.787, 4.14],
                        [0.691, 0.866, 1.074, 1.341, 1.753, 2.131,
                         2.602, 2.947, 3.286, 3.733, 4.073],
                        [0.69, 0.865, 1.071, 1.337, 1.746, 2.12,
                         2.583, 2.921, 3.252, 3.686, 4.015],
                        [0.689, 0.863, 1.069, 1.333, 1.74, 2.11,
                         2.567, 2.898, 3.222, 3.646, 3.965],
                        [0.688, 0.862, 1.067, 1.33, 1.734, 2.101,
                         2.552, 2.878, 3.197, 3.61, 3.922],
                        [0.688, 0.861, 1.066, 1.328, 1.729,
                         2.093, 2.539, 2.861, 3.174, 3.579, 3.883],
                        [0.687, 0.86, 1.064, 1.325, 1.725, 2.086,
                         2.528, 2.845, 3.153, 3.552, 3.85],
                        [0.686, 0.859, 1.063, 1.323, 1.721, 2.08,
                         2.518, 2.831, 3.135, 3.527, 3.819],
                        [0.686, 0.858, 1.061, 1.321, 1.717, 2.074,
                         2.508, 2.819, 3.119, 3.505, 3.792],
                        [0.685, 0.858, 1.06, 1.319, 1.714, 2.069,
                         2.5, 2.807, 3.104, 3.485, 3.767],
                        [0.685, 0.857, 1.059, 1.318, 1.711, 2.064,
                         2.492, 2.797, 3.091, 3.467, 3.745],
                        [0.684, 0.856, 1.058, 1.316, 1.708, 2.06,
                         2.485, 2.787, 3.078, 3.45, 3.725],
                        [0.684, 0.856, 1.058, 1.315, 1.706, 2.056,
                         2.479, 2.779, 3.067, 3.435, 3.707],
                        [0.684, 0.855, 1.057, 1.314, 1.703, 2.052,
                         2.473, 2.771, 3.057, 3.421, 3.69],
                        [0.683, 0.855, 1.056, 1.313, 1.701, 2.048,
                         2.467, 2.763, 3.047, 3.408, 3.674],
                        [0.683, 0.854, 1.055, 1.311, 1.699, 2.045,
                         2.462, 2.756, 3.038, 3.396, 3.659],
                        [0.683, 0.854, 1.055, 1.31, 1.697, 2.042,
                         2.457, 2.75, 3.03, 3.385, 3.64],
                        ]

PROBABILITY_LOOKUP = {0.5: 0, 0.6: 1, 0.7: 2, 0.8: 3,
                      0.9: 4, 0.95: 5, 0.98: 6, 0.99: 7, 0.995: 8,
                      0.998: 9, 0.999: 10}


def student_t(alpha, n):
    """Calculate t(α, n-1) for intervals
       following Student's distribution.
       Reliable for up to 30 data points.
       Uses hardcoded t-values from STUDENT_COEFFICIENTS
       list and lookupval to find correct value there
       using given probability.
       Input: float (from PROBABILITY_LOOKUP), int
       Output: float
    """

    lookupval = PROBABILITY_LOOKUP
    if n == float('inf') or n >= len(STUDENT_COEFFICIENTS):
        n = 0

    return STUDENT_COEFFICIENTS[n-1][lookupval[alpha]]


def sample_size(*variables):
    """Return the size of a sample.
       Input: *float (all variables as arguments)
       Output: int
    """

    return len(variables)


def range_of_sample(*variables):
    """Return the range of a sample.
       Input: *float (all variables as arguments)
       Output: int
    """

    return max(variables) - min(variables)


def Cornfeld_variance(*variables):
    """Return the variance of a sample.
       Input: *float (all variables as arguments)
       Output: int
    """

    return range_of_sample(*variables) / 2


def Cornfeld_mean(*variables):
    """Return the mean of a sample using Cornfeld's method.
       Input: *float (all variables as arguments)
       Output: int
    """

    return (max(variables) + min(variables)) / 2


def cornfeld_probability(sample_size):
    """Return the probability for Cornfeld's method.
       Input: int
       Output: float
    """

    return 1 - (0.5**(sample_size-1))


def arithmetic_mean(*variables):
    """Return the arithmetic mean of a sample.
       Input: *float (all variables as arguments)
       Output: int
    """

    # print(*variables)
    return sum(variables) / sample_size(*variables)


def integer_digit_length(integer):
    """Return the quantity of digits of an integer.
       Input: int
       Output: int
    """

    return ceil(log10(integer + 1))


def calculate_if_not_given(func, value, calc_f):
    """Modify a function to calculate a value if it is passed as None
       (using the calc_f function).
       Input: function, *, function
       Output: function
    """

    def inner(*args, value=value, **kwargs):
        """Modified function to be returned.
        """

        # print('Kwargs', kwargs, args)
        try:
            name, value = value, kwargs[value]
        except KeyError:
            return func(*args, **kwargs)
        f_args = [[], []]
        if value is None:
            del kwargs[name]
            f_params = getfullargspec(calc_f)
            if f_params.varargs is not None:
                f_args[0] += args
            if f_params.varkw is not None:
                f_args[1] = kwargs
            value = calc_f(*f_args[0], **f_args[1])
            kwargs[name] = value
        return func(*args, **kwargs)

    return inner


calculate_length_of_num_if_not_given = partial(calculate_if_not_given,
                                               value="length",
                                               calc_f=integer_digit_length)

calculate_mean_of_sample_if_not_given = partial(calculate_if_not_given,
                                                value="mean",
                                                calc_f=arithmetic_mean)


@calculate_length_of_num_if_not_given
def round_to_last_digit(integer, length=None):
    """Round an integer to its last digit.
       Its amount of digits can be given
       but is not required.
       Input: int, int
       Output: int
    """

    return int(round(integer, -(length-1)))


@calculate_length_of_num_if_not_given
def last_digit(integer, length=None):
    """Get the last digit of an integer.
       Its amount of digits can be given
       but is not required.
       Input: int, int
       Output: int
    """

    return integer // 10**(length-1)


@calculate_mean_of_sample_if_not_given
def standard_deviation_of_a_sample(*variables, mean=None):
    """Calculate the standard deviation of a sample.
       Its arithmetic mean can be given
       but is not required.
       Input: *float (all data points as arguments), float
       Output: float
    """

    sigma, sample_size = 0, 0
    for x in variables:
        sigma += (x - mean)**2
        sample_size += 1

    if sample_size < 2:
        return 0

    sigma = (sigma / (sample_size-1))**0.5

    return sigma


@calculate_mean_of_sample_if_not_given
def standard_error_of_the_mean(*variables, mean=None):
    """Calculate the standard error for a sample.
       Its arithmetic mean can be given
       but is not required.
       Input: *float (all data points as arguments), float
       Output: float
    """

    sigma = standard_deviation_of_a_sample(*variables, mean=mean)
    sample_size_sqrt = sample_size(*variables)**0.5

    return sigma / sample_size_sqrt


@calculate_mean_of_sample_if_not_given
def student_error(*variables, mean=None, probability=0.8):
    """Calculate the Student error of a sample.
       Its arithmetic mean can be given
       but is not required.
       Probability can be: 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98,
       0.99, 0.995, 0.998 and 0.999 (do not pass any other value!)
       0.8 is default prob.
       Input: *float (all data points as arguments), float, float
       Output: float
    """

    smple_size = sample_size(*variables)
    SEM = standard_error_of_the_mean(*variables, mean=mean)
    # res = SEM * STUDENT_COEFFICIENTS[smple_size-1][probability]
    t = student_t(probability, smple_size)
    res = SEM * t
    print('t(α, n-1) =', t)

    return res


def formatted_output_of_an_experiment_result(
        res,
        error,
        scientific_notation_threshold=2
):
    """Format the numeric result with error as a str for output more
       or less sanely.
       A quantity of digits of the error after which result is shown
       in standard form can specified (default one is 2 decimal digits).
       If the error is only a fraction of the unit of measurement,
       scientific notation is not used.
       Input: float, float, int
       Output: str
    """

    # Zero error is an obvious (even though hardly feasible)
    # corner case better handled separately.
    if error == 0:
        # Nothing fancy here. Just the result, as is.
        # Allegedly verified...
        return str(res)

    """Now, two cases are apparent for the non-zero error: it can be
       only a fraction of, or it can be equal to or bigger than
       the unit of measurement.
       These two cases will be processed separately.
    """
    if error >= 1:
        # Note that we can use floor safely on error
        # as it is non-negative
        error_int = round(error)
        error_len = integer_digit_length(error_int)
        res_significant = floor(round(res, -error_len+1))

        error_significant = round(error_int, -error_len+1)

        # Now, result and error (read the interval of values)
        # will be displayed in scientific notation,
        # that is in standard form (normalised scientific notation)
        # for the error and with the same power of 10 factored out
        # for the result. Note: the result is not normalised!
        # (unless both are less than 10 in magnitude
        # in which case no standard form is useful at all).
        res_first_digit = last_digit(res_significant, length=error_len)
        error_first_digit = last_digit(error_significant, length=error_len)
        error_significand = error_first_digit
        res_significand = res_first_digit
        error_exponent = error_len-1
        # res_exponent = error_exponent
        if error_len >= scientific_notation_threshold:
            return (f'({res_significand} ± ' +
                    f'{error_significand}) ' +
                    f'* 10^{error_exponent}')
        else:
            return (f'{res_significant} ' +
                    f'± {error_significant}')

    # Error is smaller than the unit of measurement.
    # No scientific notation here.
    # Just display stuff the old-fashioned straight-forward
    # way.
    # error = float(f'{error:.1g}')
    # error_significant_digits = ceil(log(error, 0.1))
    error_significant_digits = ceil(abs(log(error, 10)))
    if floor(error * (10**error_significant_digits)) == 1:
        error_significant_digits += 1
    error_significant = round(error, error_significant_digits)
    res_significant = round(res, error_significant_digits)
    return (f'{res_significant:.{error_significant_digits}f} ' +
            f'± {error_significant:.{error_significant_digits}f}'
            )

    # print(f'{AM:.{numberlen + error_significant_digits}g} ± {error:.1g}')


def main():
    RESULTS_READOUT = """how to read the output:
n = sample size
<x> = mean (arithmetic)
max = maximal value of the sample
min = minimal value of the sample
Δ(x) = absolute stochastic error
Ɛ(x) = relative error
Result: x = (<x> ± Δ(x))
α = probability
    """

    # Get command line arguments passed to the program
    dscrptn = 'This is a small utility to help you process\n'
    dscrptn += 'small data samples from experiments. '
    dscrptn += 'It can \nuse either Cornfeld\'s or Student\'s method.\n'
    dscrptn += 'However, it always uses the arithmetic mean,\n'
    dscrptn += 'as it is more precise.'
    dscrptn += ' Probability can be: \n0.5, 0.6, 0.7, 0.8, 0.9, 0.95,'
    dscrptn += '0.98, 0.99, 0.995, \n0.998 and 0.999'
    dscrptn += ' (do not pass any other value)! \nDefault probability'
    dscrptn += ' for Student\'s is 0.8.\n'
    warn = """Warning! It has precise values for Student's
t-distributions only for n up to 30.
Then assumes n->inf.\n"""
    dscrptn += warn
    dscrptn += """\nTo process data normally, just run it and enter
data."""

    # Initiate the parser
    parser = argparse.ArgumentParser(
        description=dscrptn,
        epilog=RESULTS_READOUT,
        formatter_class=argparse.RawDescriptionHelpFormatter
       )
    parser.add_argument('-p', '--probability',
                        help='set probability for Student\'s distribution',
                        type=float)
    parser.add_argument('-d', '--data', nargs='*', type=float,
                        help='enter data as arguments', default=[])
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-C', '--cornfeld', help='use Cornfeld\'s method',
                        action='store_true')
    group.add_argument('-S', '--student', help='use Student\'s method',
                        action='store_true')

    # Read arguments from the command line
    args = parser.parse_args()
    if args.cornfeld and (args.probability):
        msg = 'Probability is for Student\'s t-distribution.'
        msg += ' Does not make sense with Cornfeld\'s method.'
        raise ValueError(msg)

    # Enter experiment results (sample data):
    try:
        if not args.data:
            variables = list(map(float, input().split()))
        else:
            variables = args.data

        n = sample_size(*variables)

        if args.probability:
            alpha = args.probability
        elif args.cornfeld:
            alpha = cornfeld_probability(n)
        else:
            alpha = 0.8

        assert n >= 1  # sample_size should be at least one
    except Exception as e:
        if isinstance(e, ValueError) or isinstance(e, AssertionError):
            raise ValueError('Please, enter numerical data only!') from e
        else:
            raise e

    if not args.cornfeld:
        try:
            assert alpha in PROBABILITY_LOOKUP.keys()
        except AssertionError as e:
            msg = 'Probability can only be in '
            msg += str(tuple(PROBABILITY_LOOKUP.keys()))
            raise ValueError(msg) from e

    # Display sample size
    print(f'n = {n}')
    # Find the arithmetic mean of the results:
    AM = arithmetic_mean(*variables)
    # Then calculate the stochastic error:
    if args.cornfeld:
        # using Cornfeld's method
        error = Cornfeld_variance(*variables)
    else:
        # using Student's method
        error = student_error(*variables, mean=AM, probability=alpha)

    # Now display the numbers that characterise the sample:
    print('<x> =', AM)  # Arithmetic mean
    print('max =', max(variables))  # Maximum
    print('min =', min(variables))  # Minimum
    print('Δ(x) =', error)  # Absolute error of measurement

    # Relative error is, of course, just absolute error
    # divided by the result (i.e. AM)
    rel_error = error/AM
    # print('Ɛ(x) =', rel_error)  # Relative error
    # print('Ɛ(x) =', 1/rel_error)  # Precision

    # Relative error in percentages (with 2 digits after
    # the floating point - fixed p.).
    # (It is tacitly assumed that it won't be too little to
    # vanish with such display precision)
    if rel_error <= 0.00005:
        print('Ɛ(x) << 1%')
    else:
        print('Ɛ(x) =', f'{rel_error:.2%}')

    # Aaaand, the most important part... The result of the experiment
    # is displayed (as conventionally recorded).
    res = formatted_output_of_an_experiment_result(AM, error)
    print(f'Result: x = (' + res + ')')

    # Display α
    print(f'α = {alpha}')


if __name__ == '__main__':
    main()
