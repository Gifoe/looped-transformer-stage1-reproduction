"""CPU solver/library identity only; does not construct or solve an LP."""
import sys,platform
def versions():
 import numpy,scipy
 from scipy.optimize._highspy import _core
 return dict(python='.'.join(map(str,sys.version_info[:3])),numpy=numpy.__version__,scipy=scipy.__version__,HiGHS='.'.join(str(getattr(_core,'HIGHS_VERSION_'+k)) for k in ['MAJOR','MINOR','PATCH']),platform=platform.platform(),executable=sys.executable)
