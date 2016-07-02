from enum import Enum

class Oversampling(Enum):

    SmoteRegular, SmoteSVM, Repeat, DontUse = range(4)

class Undersampling(Enum):

    ClusterCentroids, SMOTEENN = range(2)
    