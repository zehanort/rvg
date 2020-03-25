#  struct knode {
#          int location;
#          int indices [DEFAULT_ORDER + 1];
#          int  keys [DEFAULT_ORDER + 1];
#          bool is_leaf;
#          int num_keys;
#  } knode;

import numpy as np
import rvg

DEFAULT_ORDER = 5

knode = np.dtype([
    ('location', int),
    ('indices', (int, DEFAULT_ORDER + 1)),
    ('keys', (int, DEFAULT_ORDER + 1)),
    ('is_leaf', int),
    ('num_keys', int)
])

knode_params = {
    'location'  : (0, 10),
    'indices'   : 42,
    'keys'      : 117,
    'is_leaf'   : (0, 2),
    'num_keys'  : (0, DEFAULT_ORDER + 1)
}

knodes = rvg.random(knode, knode_params, 5)
print(knodes)

