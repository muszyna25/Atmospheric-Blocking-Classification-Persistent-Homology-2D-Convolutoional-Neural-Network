import numpy as np
from ripser import ripser

data = np.random.random((100,2))
diagrams = ripser(data)['dgms']

