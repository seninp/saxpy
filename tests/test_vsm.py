"""Testing SAX implementation."""
import numpy as np
from saxpy.sax import sax_via_window
from saxpy.saxvsm import series_to_wordbag
from saxpy.saxvsm import manyseries_to_wordbag


def test_series_to_wordbag():
    """Test TS to vector."""
    dat = np.array([0., 0., 0., 0., 0., -0.270340178359072, -0.367828308500142,
                    0.666980581124872, 1.87088147328446, 2.14548907684624,
                    -0.480859313143032, -0.72911654245842, -0.490308602315934,
                    -0.66152028906509, -0.221049033806403, 0.367003418871239,
                    0.631073992586373, 0.0487728723414486, 0.762655178750436,
                    0.78574757843331, 0.338239686422963, 0.784206454089066,
                    -2.14265084073625, 2.11325193044223, 0.186018356196443,
                    0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.519132472499234,
                    -2.604783141655, -0.244519550114012, -1.6570790528784,
                    3.34184602886343, 2.10361226260999, 1.9796808733979,
                    -0.822247322003058, 1.06850578033292, -0.678811824405992,
                    0.804225748913681, 0.57363964388698, 0.437113583759113,
                    0.437208643628268, 0.989892093383503, 1.76545983424176,
                    0.119483882364649, -0.222311941138971, -0.74669456611669,
                    -0.0663660879732063, 0., 0., 0., 0., 0.])

    sax_none = sax_via_window(dat, 6, 3, 3, "none", 0.01)
    wordbag = series_to_wordbag(dat, 6, 3, 3, "none", 0.01)
    wordbag2 = manyseries_to_wordbag(np.matrix([dat, dat]),
                                     6, 3, 3, "none", 0.01)

    frequencies = {}
    for k, v in sax_none.items():
        frequencies[k] = len(v)

    for k, v in wordbag.items():
        assert v == frequencies[k]

    for k, v in wordbag2.items():
        assert v == frequencies[k] * 2
