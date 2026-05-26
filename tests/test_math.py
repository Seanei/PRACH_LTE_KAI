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


class TestIDFT(unittest.TestCase):
    def assertComplexAlmostEqual(self, actual, expected, places=12):
        self.assertAlmostEqual(actual.real, expected.real, places=places)
        self.assertAlmostEqual(actual.imag, expected.imag, places=places)

        def test_idft_vectors(self):
            test_vectors = [
                {
                    "input": [1 + 0j, 2 + 0j, 3 + 0j, 4 + 0j],
                    "expected": [
                        2.50000000000000 + 0.00000000000000j,
                        -0.500000000000000 - 0.500000000000000j,
                        -0.500000000000000 + 0.00000000000000j,
                        -0.500000000000000 + 0.500000000000000j,
                    ],
                },
                {
                    "input": [1 + 0j, 0 + 0j, 0 + 0j, 0 + 0j],
                    "expected": [
                        0.250000000000000 + 0j,
                        0.250000000000000 + 0j,
                        0.250000000000000 + 0j,
                        0.250000000000000 + 0j,
                    ],
                },
                {
                    "input": [0 + 0j, 1 + 0j, 0 + 0j, 0 + 0j],
                    "expected": [
                        0.250000000000000 + 0.00000000000000j,
                        0.00000000000000 + 0.250000000000000j,
                        -0.250000000000000 + 0.00000000000000j,
                        0.00000000000000 - 0.250000000000000j,
                    ],
                },
                {
                    "input": [1 + 0j, -1 + 0j, 1 + 0j, -1 + 0j],
                    "expected": [0 + 0j, 0 + 0j, 1 + 0j, 0 + 0j],
                },
                {
                    "input": [3.5 + 0j, -2 + 0j, 0.25 + 0j, 7.125 + 0j],
                    "expected": [
                        2.21875000000000 + 0.00000000000000j,
                        0.812500000000000 - 2.28125000000000j,
                        -0.343750000000000 + 0.00000000000000j,
                        0.812500000000000 + 2.28125000000000j,
                    ],
                },
                {
                    "input": [1 + 2j, 3 - 4j, -1 + 0.5j, 0 - 2j],
                    "expected": [
                        0.750000000000000 - 0.875000000000000j,
                        1.00000000000000 + 1.12500000000000j,
                        -0.750000000000000 + 2.12500000000000j,
                        0.00000000000000 - 0.375000000000000j,
                    ],
                },
                {
                    "input": [
                        0 + 0j,
                        1 + 0j,
                        2 + 0j,
                        3 + 0j,
                        4 + 0j,
                        5 + 0j,
                        6 + 0j,
                        7 + 0j,
                    ],
                    "expected": [
                        3.50000000000000 + 0.00000000000000j,
                        -0.500000000000000 - 1.20710678118655j,
                        -0.500000000000000 - 0.500000000000000j,
                        -0.500000000000000 - 0.207106781186548j,
                        -0.500000000000000 + 0.00000000000000j,
                        -0.500000000000000 + 0.207106781186548j,
                        -0.500000000000000 + 0.500000000000000j,
                        -0.500000000000000 + 1.20710678118655j,
                    ],
                },
                {
                    "input": [
                        0.53766713954610001 + 0j,
                        1.8338850145950865 + 0j,
                        -2.2588468610036481 + 0j,
                        0.86217332036812055 + 0j,
                        0.31876523985898081 + 0j,
                        -1.3076882963052734 + 0j,
                        -0.43359202230568356 + 0j,
                        0.34262446653864992 + 0j,
                    ],
                    "expected": [
                        -0.0131264998384584 + 0.00000000000000j,
                        0.259119146714865 + 0.0954436838418589j,
                        0.443608907839302 - 0.0848251335771197j,
                        -0.204393671793085 + 0.551757393516350j,
                        -0.445875126137604 + 0.00000000000000j,
                        -0.204393671793085 - 0.551757393516350j,
                        0.443608907839302 + 0.0848251335771197j,
                        0.259119146714865 - 0.0954436838418589j,
                    ],
                },
                {
                    "input": [
                        3.5783969397257605 - 0.12414434821631191j,
                        2.7694370298848772 + 1.4896976077854649j,
                        -1.3498869401565212 + 1.4090344898004792j,
                        3.0349234663318545 + 1.4171924134296139j,
                        0.72540422494610557 + 0.6714971336080805j,
                        -0.063054873189656191 - 1.2074869226850378j,
                        0.71474290382609584 + 0.71723865132883846j,
                        -0.20496605829977463 + 1.6302352891647292j,
                    ],
                    "expected": [
                        1.15062458663359 + 0.750408039276982j,
                        0.0145712307467947 + 0.436424036674775j,
                        0.963020277312026 - 0.212811950884301j,
                        0.259538595575278 + 0.438121114239696j,
                        -0.233460304548233 - 0.0820015576467105j,
                        0.525727988330209 - 1.15149186812653j,
                        0.271716022938547 - 0.181918138050086j,
                        0.626658542737546 - 0.120874023700140j,
                    ],
                },
            ]


if __name__ == "__main__":
    unittest.main()
