import argparse
import sys
from rvg import NumPyRVG

parser = argparse.ArgumentParser(
    description='rvg - Random Values Generator',
    epilog='''NOTE: rvg can be run with no flags, which is equivalent to running `rvg --numpy float32 -limits 0 1`
    (i.e. sampling of the uniform(0, 1) distribution)
    '''
)

parser.add_argument('--numpy',
    type=str,
    metavar='DTYPE',
    help='request a numpy random value of type DTYPE'
)

parser.add_argument('-s', '--samples',
    type=int,
    help='number of samples to produce, one per line (0 or 1 both mean a single value)',
    default=0
)

parser.add_argument('-l', '--limits',
    type=int,
    nargs='+',
    help='''if 1 positive integer is given, `limit`: define lower and upper numerical limit for produced values
    as (-limit, limit) for signed types or (0, limit) for unsigned values\n
    if 2 integers are given, `a` and `b`, where a < b: define lower and upper numerical limit for produced values
    as (a, b) for signed types or (max(a, 0), b) for unsigned values, in which case b must be a positive integer
    ''',
    default=[0, 1]
)

def cli():

    # default behavior
    if not len(sys.argv) > 1:
        import numpy as np
        rand = NumPyRVG(0, 1)
        print(rand(np.float32))
        return

    args = parser.parse_args()
    if args.numpy is None:
        sys.stderr.write('Please provide a data type\n')
        exit(1)

    if args.numpy:
        import numpy as np
        rand = NumPyRVG(*args.limits)
        try:
            vals = rand(eval(f'np.{args.numpy}'), args.samples)
        except AttributeError:
            sys.stderr.write(f'ERROR: numpy does not have the type `{args.numpy}`\n')
            exit(1)
        if args.samples == 0:
            print(vals)
        else:
            for val in vals:
                print(val)
