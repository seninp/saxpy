import doctest
from saxpy import paa, sax, znorm

doctest.testmod(paa, optionflags=doctest.NORMALIZE_WHITESPACE)
doctest.testmod(sax, optionflags=doctest.NORMALIZE_WHITESPACE)
doctest.testmod(znorm, optionflags=doctest.NORMALIZE_WHITESPACE)