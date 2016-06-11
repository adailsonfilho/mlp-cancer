import numpy as np
from oversampler import Oversampler
from enums import Oversampling
data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
target = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
over = Oversampler(Oversampling.Repeat,data, target,True)
over_data, over_target = over.balance()