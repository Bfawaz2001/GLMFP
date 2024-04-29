import unittest
from glmfp_main_program import *

# File path to the models
NGRAM_MODEL_PATH = "../../data/models/n-gram/"
NN_MODEL_PATH = '../../data/models/neural network/'
TRANS_MODEL_PATH = '../../data/models/transformer/'

# File Paths to results directories
GENERATED_PROTEINS_RESULTS_PATH = "../../results/generated proteins/"
INTERPRO_RESULTS_PATH = "../../results/interpro results/"
ANALYSIS_SUMMARY_PATH = "../../results/analysis summary/"
DIAMOND_RESULTS_PATH = "../../results/diamond blastp results/"

# Diamond Database paths for DIAMOND BLASTp
DIAMOND_NR_DB_PATH = "../../data/diamond db/nr.dmnd"
DIAMOND_SwissProt_DB_PATH = "../../data/diamond db/uniprot_sprot.dmnd"

# InterProScan script path and user email address
IPRSCAN5_PATH = "../../data/interpro script/iprscan5.py"
EMAIL = "b.fawaz2001@gmail.com"

# class TestGLMFPMainProgram(unittest.TestCase):
# Add the setUp method to load models, etc., if necessary

class ModelLoadingTestCases(unittest.TestCase):
    def test_load_2mer_model(self):
        model_path = NGRAM_MODEL_PATH + '2mer_model.pkl'
        model = load_ngram_model(model_path)

        # Verify the model contains the correct keys
        self.assertIn('2mer_model', model)
        self.assertIn('start_amino_acids', model)
        self.assertIn('start_amino_acid_probs', model)

    def test_load_3mer_model(self):
        model_path = NGRAM_MODEL_PATH + '3mer_model.pkl'
        model = load_ngram_model(model_path)

        # Verify the model contains the correct keys
        self.assertIn('3mer_model', model)
        self.assertIn('start_2mers', model)
        self.assertIn('start_2mer_probs', model)

    def test_load_5mer_model(self):
        model_path = NGRAM_MODEL_PATH + '5mer_model.pkl'
        model = load_ngram_model(model_path)

        # Verify the model contains the correct keys
        self.assertIn('5mer_model', model)
        self.assertIn('start_4mer_probs', model)

    def test_load_6mer_model(self):
        model_path = NGRAM_MODEL_PATH + '6mer_model.pkl'
        model = load_ngram_model(model_path)

        # Verify the model contains the correct keys
        self.assertIn('6mer_model', model)
        self.assertIn('start_5mer_probs', model)


class ProteinGenerationTestCases(unittest.TestCase):
    def test_ngram_protein_generation(self):
        model = load_ngram_model('path/to/2mer_model.pkl')
        protein = generate_ngram_protein(model, 50, 100, '2mer')
        self.assertTrue(50 <= len(protein) <= 100)
        # Add more assertions related to amino acid frequencies, etc.


class AnalysisToolsTestCases(unittest.TestCase):
    def test_ngram_protein_generation(self):
        model = load_ngram_model('path')


class IntegrationTestCase(unittest.TestCase):
    def test_protein_generation_to_blastp(self):
        # Assuming you have a mock or small-scale test for diamond blastp
        model, encoder = load_nn_model_and_encoder('path/to/nn_model.pt', 'path/to/encoder.pkl')
        protein = generate_complex_protein(model, encoder, 50, 100)
        # Use protein to run a simulated BLASTp search
        # Assert that the BLASTp results meet certain conditions


if __name__ == '__main__':
    unittest.main()
