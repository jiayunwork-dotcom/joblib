import glob, os, tempfile
import numpy as np
from joblib import Memory

with tempfile.TemporaryDirectory() as tmp:
    mem = Memory(location=tmp, verbose=0)
    @mem.cache
    def make(x):
        return np.array([11, x])
    make(0)
    for m in glob.glob(os.path.join(tmp, "**", "generation"), recursive=True):
        os.unlink(m)
    shelf = make.call_and_shelve(0)
    h1 = shelf.pin(); h2 = shelf.pin()
    np.testing.assert_array_equal(h1.get(), [11, 0])
    np.testing.assert_array_equal(h2.get(), [11, 0])
    h1.close(); h2.close()
    print("ok repeated pin on sealed legacy shelf")
