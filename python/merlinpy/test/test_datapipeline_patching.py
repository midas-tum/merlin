import unittest
import numpy as np
from merlinpy.datapipeline.patching import patching, unpatching

class PatchingTest(unittest.TestCase):
    def test_patching2D(self):
        imgin = np.ones((120,120,30))
        patches2D = patching(imgin,[64, 32], 0.5)
        imgunpatch2D = unpatching(patches2D, [64, 32], 0.5, np.shape(imgin))

        self.assertTrue(np.sum(imgin - imgunpatch2D) == 0)

    def test_patching3D(self):
        imgin = np.ones((120, 120, 30))
        patches3D = patching(imgin, [32, 32, 32], 0.5)
        imgunpatch3D = unpatching(patches3D, [32, 32, 32], 0.5, np.shape(imgin))

        self.assertTrue(np.sum(imgin - imgunpatch3D) == 0)

    def test_patching3D_2D(self):
        imgin = np.ones((120, 120, 30))
        patches3D_2 = patching(imgin, [32, 32, 1], 0)
        imgunpatch3D_2 = unpatching(patches3D_2, [32, 32, 1], 0, np.shape(imgin))

        self.assertTrue(np.sum(imgin - imgunpatch3D_2) == 0)


if __name__ == '__main__':
    unittest.main()

    #patchShape_2 = compute_patchedshape(imgin, [32, 32, 1], 0)
    #patchShape_21 = compute_patchedshape(np.shape(imgin), [32, 32, 1], 0)
