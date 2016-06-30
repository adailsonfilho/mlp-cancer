from enum import Enum

class Oversampling(Enum):

    SmoteRegular, SmoteSVM, Repeat, DontUse = range(4)