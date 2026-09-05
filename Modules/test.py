
try:
    import numpy
    print("numpy finns")
except ImportError:
    print("numpy finns inte")
try:
    import matplotlib
    print("matplotlib finns")
except ImportError:
    print("matplotlib finns inte")
try:
    import scipy as sp
    print("scipy finns")
except ImportError:
    print("scipy finns inte")
