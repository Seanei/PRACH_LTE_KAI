import unittest
from prach.pipeline import CommonData
from subcarrier_mapping import SubcarrierMappingBlock 

def read_complex_file(path): #чтение компл числа из файла
    data = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip() #убираем пробелы

            if 'j' in line: # считываем комп число
                data.append(complex(line))
            else: #иначе
                real, imag = map(float, line.split())
                data.append(complex(real, imag))

    return data

class TestSubcarrierMapping(unittest.TestCase):

    def test_mapping(self):
        # читаем вход и эталон
        input_data = read_complex_file("input_data.txt")
        expected_output = read_complex_file("output_data.txt")

        # формируем CommonData  
        data = CommonData(
            meta={
                "after_dft": input_data, #это то, что идет на вход блока
                "n_ul_rb": 6,
                "n_ra_prb_offset": 0
            }
        )

        # запускаем блок
        block = SubcarrierMappingBlock()
        result = block.process(data)

        output = result.meta["Subcarrier_Mapping"] #результаты через написанный код

        # проверка длины
        self.assertEqual(len(output), len(expected_output))

        # поэлементное сравнение
        #Сравниваем действительную часть (real)
        # Сравниваем мнимую часть (imag)
        #  places=7 точность ошибки до 10^-7
        for i in range(len(output)):
            self.assertAlmostEqual(output[i].real, expected_output[i].real, places=7)
            self.assertAlmostEqual(output[i].imag, expected_output[i].imag, places=7)

if __name__ == "__main__":
    unittest.main()