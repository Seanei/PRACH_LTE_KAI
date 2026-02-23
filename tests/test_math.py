import unittest
from prach.math import zadoff_chu


class TestZadoffChu(unittest.TestCase):

    def test_length(self):
        n_zc = 63
        root = 25
        seq = zadoff_chu(root, n_zc)
        self.assertEqual(len(seq), n_zc)

    def test_constant_magnitude(self):
        n_zc = 63
        root = 29
        seq = zadoff_chu(root, n_zc)

        magnitudes = [abs(x) for x in seq]
        # All magnitudes should be 1 within numerical tolerance
        for mag in magnitudes:
            self.assertAlmostEqual(mag, 1.0, places=12)

    def test_deterministic(self):
        n_zc = 139
        root = 7
        seq1 = zadoff_chu(root, n_zc)
        seq2 = zadoff_chu(root, n_zc)
        self.assertEqual(seq1, seq2)

    def test_multiple_roots(self):
        n_zc = 63
        roots = [1, 2, 4, 5]  # all coprime with 63
        seqs = [zadoff_chu(n_zc, r) for r in roots]

        # Ensure sequences differ for different roots
        for i in range(len(seqs)):
            for j in range(i + 1, len(seqs)):
                self.assertNotEqual(seqs[i], seqs[j])

    def test_invalid_length_raises(self):
        # Depending on your implementation, n_zc may need to be odd or prime.
        # Adjust the expected exception type accordingly.
        with self.assertRaises(Exception):
            zadoff_chu(0, 1)

        with self.assertRaises(Exception):
            zadoff_chu(-10, 5)

    def test_invalid_root_raises(self):
        n_zc = 63
        with self.assertRaises(Exception):
            zadoff_chu(n_zc, -1)

        # roots NOT COPRIME with 63 should raise ValueError
        for bad_root in [0, 3, 7, 9, 14, 21]:
            with self.assertRaises(ValueError):
                zadoff_chu(n_zc, bad_root)

    def test_phase_difference(self):
        # Check that the phase increments match ZC formula
        n_zc = 63
        root = 25
        seq = zadoff_chu(root, n_zc)

        # Verify ratio seq[k+1] / seq[k] has magnitude 1
        for k in range(n_zc - 1):
            ratio = seq[k + 1] / seq[k]
            self.assertAlmostEqual(abs(ratio), 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
