import unittest
from unittest.mock import patch
import sys
sys.path.append('../src/generate proteins program/glmfp_main_program.py')



class TestGenerateProtein(unittest.TestCase):
    def setUp(self):
        # Setup a test model similar to what `generate_protein` expects
        self.test_model = {
            'start_amino_acid_probs': {'A': 0.5, 'B': 0.5},
            'bigram_model': {'A': {'A': 0.5, 'B': 0.5}, 'B': {'A': 0.5, 'B': 0.5}},
            # Add similar structures for 'trigram_model', 'start_2mer_probs', etc., as needed
        }

    @patch('main.random.randint')
    @patch('main.random.choices')
    def test_generate_protein_length_and_content(self, mock_choices, mock_randint):
        # Mocking random to return a fixed sequence length and specific amino acids
        mock_randint.return_value = 5  # Assuming we want to generate a protein of length 5
        mock_choices.side_effect = [('A',), ('B',)] * 3  # Alternating between 'A' and 'B' for simplicity

        # Call the function with the test model and assert expectations
        protein = main.generate_protein(self.test_model, 5, 10, '2mer')

        # Verify the length of the generated protein
        self.assertEqual(len(protein), 5, "Generated protein does not have the expected length.")

        # Verify the content of the generated protein (simple case)
        expected_protein = 'ABABA'
        self.assertEqual(protein, expected_protein, "Generated protein sequence does not match expected.")

        # Add more tests as needed, for example, testing different model types ('3mer', '5mer', '6mer')


if __name__ == '__main__':
    unittest.main()
