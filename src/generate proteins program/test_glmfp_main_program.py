import unittest
from unittest.mock import patch, MagicMock, mock_open
import re
from glmfp_main_program import *

# PATH for test data
TEST_DATA_PATH = '../../data/test data'
TEST_DATA_FILE_PATH = '../../data/test data/test_proteins.fasta'


class ModelLoadingTestCases(unittest.TestCase):
    def test_load_2mer_model(self):
        model_path = NGRAM_MODEL_PATH + '2mer_model.pkl'
        model = load_ngram_model(model_path)

        # Verify the model contains the correct keys
        self.assertIn('2mer_model', model, "2mer_model key is missing in the loaded model")
        self.assertIn('start_amino_acids', model, "start_amino_acids key is missing in the loaded model")
        self.assertIn('start_amino_acid_probs', model, "start_amino_acid_probs key is missing in the "
                                                       "loaded model")

    def test_load_3mer_model(self):
        model_path = NGRAM_MODEL_PATH + '3mer_model.pkl'
        model = load_ngram_model(model_path)

        # Verify the model contains the correct keys
        self.assertIn('3mer_model', model, "3mer_model key is missing in the loaded model")
        self.assertIn('start_2mers', model, "start_2mers key is missing in the loaded model")
        self.assertIn('start_2mer_probs', model, "start_2mer_probs key is missing in the loaded model")

    def test_load_5mer_model(self):
        model_path = NGRAM_MODEL_PATH + '5mer_model.pkl'
        model = load_ngram_model(model_path)

        # Verify the model contains the correct keys
        self.assertIn('5mer_model', model, "5mer_model key is missing in the loaded model")
        self.assertIn('start_4mer_probs', model, "start_4mer_probs key is missing in the loaded model")

    def test_load_6mer_model(self):
        model_path = NGRAM_MODEL_PATH + '6mer_model.pkl'
        model = load_ngram_model(model_path)

        # Verify the model contains the correct keys
        self.assertIn('6mer_model', model, "6mer_model key is missing in the loaded model")
        self.assertIn('start_5mer_probs', model, "start_5mer_probs key is missing in the loaded model")

    def test_load_nn_model_and_encoder(self):
        model_path = NN_MODEL_PATH + 'nn_model.pt'
        encoder_path = NN_MODEL_PATH + 'nn_label_encoder.pkl'
        model, label_encoder = load_nn_model_and_encoder(model_path, encoder_path)

        # Check if the model has the expected methods and encoder is loaded
        self.assertTrue(hasattr(model, 'forward'), "Loaded model is missing the 'forward' method.")
        self.assertTrue(callable(getattr(model, 'forward')), "Model's 'forward' attribute is not callable.")
        self.assertTrue(hasattr(label_encoder, 'transform'), "Label encoder does not have 'transform' method.")
        self.assertTrue(callable(getattr(label_encoder, 'transform')),"Label encoder's 'transform' method is not "
                                                                      "callable.")

    def test_load_trans_model_and_encoder(self):
        model_path = TRANS_MODEL_PATH + 'transformer_model.pt'
        encoder_path = TRANS_MODEL_PATH + 'transformer_label_encoder.pkl'
        model, label_encoder = load_trans_model_and_encoder(model_path, encoder_path)

        # Check if the model has the expected methods and encoder is loaded
        self.assertTrue(hasattr(model, 'forward'), "Loaded model is missing the 'forward' method.")
        self.assertTrue(callable(getattr(model, 'forward')), "Model's 'forward' attribute is not callable.")
        self.assertTrue(hasattr(label_encoder, 'transform'), "Label encoder does not have 'transform' method.")
        self.assertTrue(callable(getattr(label_encoder, 'transform')), "Label encoder's 'transform' method is not"
                                                                       " callable.")


class ProteinGenerationTestCases(unittest.TestCase):
    def setUp(self):
        # Load models and encoders before each test case
        self.ngram_models = {
            '2mer': load_ngram_model("../../data/models/n-gram/2mer_model.pkl"),
            '3mer': load_ngram_model("../../data/models/n-gram/3mer_model.pkl"),
            '5mer': load_ngram_model("../../data/models/n-gram/5mer_model.pkl"),
            '6mer': load_ngram_model("../../data/models/n-gram/6mer_model.pkl")
        }
        self.nn_model, self.nn_encoder = load_nn_model_and_encoder('../../data/models/neural network/nn_model.pt',
                                                                   '../../data/models/neural network/nn_label_encoder.pkl')
        self.trans_model, self.trans_encoder = load_trans_model_and_encoder(
            '../../data/models/transformer/transformer_model.pt',
            '../../data/models/transformer/transformer_label_encoder.pkl')

    def test_ngram_generated_protein_validity(self):
        """ Ensure generated proteins consist only of valid amino acids. """
        for name, model in self.ngram_models.items():
            protein = generate_ngram_protein(model, 30, 50, name)
            self.assertTrue(re.match('^[ACDEFGHIKLMNPQRSTVWY]+$', protein),
                            f"Invalid characters found in protein from {name} model.")

    def test_ngram_protein_length(self):
        # Test generating a protein sequence using the 2mer model
        for name, model in self.ngram_models.items():
            protein = generate_ngram_protein(model, 30, 50, name)
            self.assertTrue(30 <= len(protein) <= 50, "Protein length not within the specified range for 2mer.")

    def test_ngram_repeatability(self):
        """ Check if protein generation is repeatable with a fixed seed. """
        random.seed(42)  # Fix seed for repeatability
        expected_protein = generate_ngram_protein(self.ngram_models['3mer'], 30, 50, '3mer')
        random.seed(42)  # Reset seed to same state
        protein = generate_ngram_protein(self.ngram_models['3mer'], 30, 50, '3mer')
        self.assertEqual(expected_protein, protein, "Protein generation is not repeatable.")

    def test_ngram_error_handling_negative_length(self):
        """ Test that generating protein with negative length raises an error. """
        with self.assertRaises(ValueError):
            generate_ngram_protein(self.ngram_models['2mer'], -1, -5, '2mer')

    def test_nn_generated_protein_validity(self):
        """ Ensure generated proteins consist only of valid amino acids. """
        protein = generate_complex_protein(self.nn_model, self.nn_encoder, 50, 200)
        self.assertTrue(re.match('^[ACDEFGHIKLMNPQRSTVWY]+$', protein),f"Invalid characters found in "
                                                                       f"protein from NN (LSTM) model.")

    def test_nn_protein_length(self):
        # Test generating a protein sequence using LSTM model
        protein = generate_complex_protein(self.nn_model, self.nn_encoder, 100, 400)
        self.assertTrue(100 <= len(protein) <= 400, "Protein length not within the specified range for LSTM.")

    def test_nn_error_handling_negative_length(self):
        """ Test that generating protein with negative length raises an error. """
        with self.assertRaises(ValueError):
             generate_complex_protein(self.nn_model, self.nn_encoder, -10, -20)

    def test_trans_generated_protein_validity(self):
        """ Ensure generated proteins consist only of valid amino acids. """
        protein = generate_complex_protein(self.trans_model, self.nn_encoder, 50, 200)
        self.assertTrue(re.match('^[ACDEFGHIKLMNPQRSTVWY]+$', protein),f"Invalid characters found in "
                                                                       f"protein from Transformer model.")

    def test_trans_protein_length(self):
        # Test generating a protein sequence using Transformer model
        protein = generate_complex_protein(self.trans_model, self.trans_encoder, 100, 200)
        self.assertTrue(100 <= len(protein) <= 200, "Protein length not within the specified range for "
                                                    "Transformer.")


class AnalysisToolsTestCases(unittest.TestCase):
    def setUp(self):
        # Real path to the test data
        self.fasta_file = TEST_DATA_FILE_PATH

    def test_calculate_physicochemical_properties(self):
        """ Test physicochemical properties calculation for real protein sequences. """
        sequence = "ACDEFGHIKLMNPQRSTVWY"
        mw, ip = calculate_physicochemical_properties(sequence)
        expected_mw = 2395.71
        expected_ip = 6.78
        self.assertAlmostEqual(mw, expected_mw, places=2)
        self.assertAlmostEqual(ip, expected_ip, places=2)

    def test_calculate_amino_acid_composition(self):
        """ Calculate amino acid composition using a real protein sequence. """
        sequences = parse_fasta(self.fasta_file)  # Using actual parsing function
        composition = calculate_amino_acid_composition(sequences)
        # Assert that certain amino acids are present and their composition is reasonable
        self.assertAlmostEqual(composition['A'], 8.60, 2)
        self.assertAlmostEqual(composition['C'], 1.23, 2)
        self.assertAlmostEqual(composition['M'], 2.37, 2)

    def test_calculate_amino_acid_composition_is_valid(self):
        """ Validates amino acid composition contains only valid amino acids. """
        sequences = list(parse_fasta(self.fasta_file))  # Using actual parsing function
        composition = calculate_amino_acid_composition(sequences)
        # Just ensure that it returns a dictionary containing amino acid keys
        self.assertIsInstance(composition, dict)
        self.assertTrue(all(aa in 'ACDEFGHIKLMNPQRSTVWY' for aa in composition.keys()),
                        "Composition should only contain valid amino acids.")

    @patch('matplotlib.pyplot.figure')
    def test_plot_aa_composition(self, mock_figure):
        """ Ensure amino acid composition plotting does not raise errors. """
        composition = {'A': 20, 'C': 30, 'D': 50}
        try:
            plot_aa_composition(composition, "Test Plot",
                                None)  # We do not actually save the plot to keep the test non-writing
            self.assertTrue(True, "Plotting function should execute without error.")
        except Exception as e:
            self.fail(f"Plotting function raised an exception {e}")

    def test_calculate_shannon_entropy(self):
        """ Test Shannon entropy calculation on a simple sequence. """
        sequence = "AACCGGTT"
        entropy = calculate_shannon_entropy(sequence)
        self.assertEqual(entropy, 2.0, "Entropy not properly calculated")
        # Expected entropy for equal distribution

    def test_clean_sequence(self):
        """ Ensure sequence cleaning removes 'X' and other unwanted characters correctly. """
        test_sequence = "MGMTPRLGLESLLEAAGAMKXMX"  # Including 'X' as a common placeholder in sequences
        cleaned_sequence = clean_sequence(test_sequence)
        self.assertNotIn('X', cleaned_sequence, "Cleaned sequence should not contain 'X'.")


class IntegrationTestCase(unittest.TestCase):
    def setUp(self):
        self.diamond_nr_db_path = "../../data/diamond db/nr.dmnd"
        self.diamond_swissprot_db_path = "../../data/diamond db/uniprot_sprot.dmnd"
        self.fasta_file = TEST_DATA_FILE_PATH
        self.email = "example@example.com"  # Use a mock email for testing

    @patch('subprocess.run')
    def test_diamond_blastp_nr_database(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        run_diamond_blastp(self.fasta_file, self.diamond_nr_db_path, 'nr')
        mock_run.assert_called_once()

    @patch('subprocess.run')
    def test_diamond_blastp_swissprot_database(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0)
        run_diamond_blastp(self.fasta_file, self.diamond_swissprot_db_path, 'swissprot')
        mock_run.assert_called_once()

    @patch('subprocess.run')
    @patch('os.path.exists')
    @patch('os.rename')
    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data='{"results":[]}')
    def test_interpro_scan_api(self, mock_open, mock_rename, mock_exists, mock_run):
        mock_exists.return_value = True
        mock_run.return_value = MagicMock(returncode=0, stdout="InterProScan analysis complete.")
        run_interpro_scan(self.fasta_file, self.email)
        mock_run.assert_called_once()
        mock_rename.assert_called_once()


if __name__ == '__main__':
    unittest.main()
