"""
ClinVar Cancer Detection and Type Classification
Detects cancer-related variants and predicts specific cancer types with confidence percentages
"""

import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from features import preprocess_variant
from trainer import ClinVarNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _find_latest_model(output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(__file__), 'output_clinvar')
    if os.path.exists(output_dir):
        pt_files = glob.glob(os.path.join(output_dir, 'clinvar_model_*.pt'))
        if pt_files:
            return max(pt_files, key=os.path.getmtime)
    return None

# Cancer Gene Database - Maps genes to cancer types with confidence scores
CANCER_GENE_DATABASE = {
    # Breast/Ovarian Cancer Genes
    'BRCA1': {'types': {'Breast Cancer': 0.95, 'Ovarian Cancer': 0.90, 'Pancreatic Cancer': 0.70, 'Prostate Cancer': 0.75}},
    'BRCA2': {'types': {'Breast Cancer': 0.95, 'Ovarian Cancer': 0.85, 'Prostate Cancer': 0.80, 'Pancreatic Cancer': 0.75}},
    'TP53': {'types': {'Breast Cancer': 0.85, 'Ovarian Cancer': 0.80, 'Lung Cancer': 0.75, 'Colorectal Cancer': 0.70,
                       'Liver Cancer': 0.80, 'Gastric Cancer': 0.75, 'Bladder Cancer': 0.75, 'Pancreatic Cancer': 0.70}},
    'PTEN': {'types': {'Breast Cancer': 0.80, 'Endometrial Cancer': 0.85, 'Prostate Cancer': 0.75, 'Thyroid Cancer': 0.75, 'Cowden Syndrome': 0.90}},

    # Lung Cancer Genes
    'EGFR': {'types': {'Lung Cancer': 0.95, 'Lung Adenocarcinoma': 0.90}},
    'KRAS': {'types': {'Lung Cancer': 0.90, 'Colorectal Cancer': 0.85, 'Pancreatic Cancer': 0.85}},
    'ALK': {'types': {'Lung Cancer': 0.95, 'Lymphoma': 0.70}},
    'ROS1': {'types': {'Lung Cancer': 0.90}},

    # Colorectal Cancer Genes
    'APC': {'types': {'Colorectal Cancer': 0.95, 'Familial Adenomatous Polyposis': 0.98}},
    'MLH1': {'types': {'Colorectal Cancer': 0.90, 'Lynch Syndrome': 0.95, 'Endometrial Cancer': 0.85}},
    'MSH2': {'types': {'Colorectal Cancer': 0.90, 'Lynch Syndrome': 0.95}},
    'MSH6': {'types': {'Colorectal Cancer': 0.85, 'Endometrial Cancer': 0.85}},
    'BRAF': {'types': {'Colorectal Cancer': 0.85, 'Melanoma': 0.90}},

    # Melanoma Genes
    'CDKN2A': {'types': {'Melanoma': 0.90, 'Pancreatic Cancer': 0.70}},
    'CDK4': {'types': {'Melanoma': 0.85}},
    'NRAS': {'types': {'Melanoma': 0.85, 'Thyroid Cancer': 0.70}},

    # Liver Cancer Genes
    'CTNNB1': {'types': {'Liver Cancer': 0.85, 'Hepatocellular Carcinoma': 0.90}},
    'AXIN1': {'types': {'Liver Cancer': 0.80}},

    # Thyroid Cancer Genes
    'RET': {'types': {'Medullary Thyroid Cancer': 0.95, 'Papillary Thyroid Cancer': 0.70}},

    # Prostate Cancer Genes
    'CHEK2': {'types': {'Prostate Cancer': 0.70, 'Breast Cancer': 0.65}},

    # Pancreatic Cancer Genes
    # (covered by BRCA1, BRCA2, CDKN2A, TP53 above)

    # Endometrial Cancer Genes
    # (covered by PTEN, MLH1, MSH6 above)

    # Gastric Cancer Genes
    'CDH1': {'types': {'Gastric Cancer': 0.90, 'Hereditary Diffuse Gastric Cancer': 0.95}},

    # Lymphoma Genes
    'KMT2D': {'types': {'Lymphoma': 0.80, 'Diffuse Large B-cell Lymphoma': 0.85}},
    'CREBBP': {'types': {'Lymphoma': 0.75, 'Diffuse Large B-cell Lymphoma': 0.80}},

    # Renal Cancer Genes
    'VHL': {'types': {'Renal Cell Carcinoma': 0.95, 'Clear Cell RCC': 0.95}},
    'BAP1': {'types': {'Renal Cell Carcinoma': 0.85}},

    # Additional Cancer Genes
    'RB1': {'types': {'Retinoblastoma': 0.95, 'Bladder Cancer': 0.70, 'Lung Cancer': 0.65}},
    'ATM': {'types': {'Breast Cancer': 0.75, 'Lymphoma': 0.70, 'Leukemia': 0.65}},
    'PALB2': {'types': {'Breast Cancer': 0.85, 'Pancreatic Cancer': 0.70, 'Ovarian Cancer': 0.75}},
    'RAD51C': {'types': {'Ovarian Cancer': 0.85, 'Breast Cancer': 0.75}},
    'RAD51D': {'types': {'Ovarian Cancer': 0.85, 'Breast Cancer': 0.70}},
    'STK11': {'types': {'Lung Cancer': 0.75, 'Colorectal Cancer': 0.70, 'Pancreatic Cancer': 0.80}},
    'NF1': {'types': {'Neurofibromatosis': 0.95, 'Glioma': 0.80, 'Breast Cancer': 0.65}},
    'NF2': {'types': {'Neurofibromatosis': 0.95, 'Mesothelioma': 0.75}},
    'SMAD4': {'types': {'Pancreatic Cancer': 0.85, 'Colorectal Cancer': 0.75}},
    'BMPR1A': {'types': {'Colorectal Cancer': 0.80, 'Juvenile Polyposis': 0.90}},
    'MUTYH': {'types': {'Colorectal Cancer': 0.85, 'Familial Adenomatous Polyposis': 0.80}},
    'EPCAM': {'types': {'Colorectal Cancer': 0.85, 'Lynch Syndrome': 0.90}},
    'PMS2': {'types': {'Colorectal Cancer': 0.85, 'Lynch Syndrome': 0.90}},
    'POLE': {'types': {'Colorectal Cancer': 0.80, 'Endometrial Cancer': 0.85}},
    'POLD1': {'types': {'Colorectal Cancer': 0.80, 'Endometrial Cancer': 0.80}},
    'GREM1': {'types': {'Colorectal Cancer': 0.75}},
    'NBN': {'types': {'Breast Cancer': 0.70, 'Lymphoma': 0.65}},
    'BARD1': {'types': {'Breast Cancer': 0.75, 'Ovarian Cancer': 0.70}},
    'BRIP1': {'types': {'Ovarian Cancer': 0.80, 'Breast Cancer': 0.65}},
    'RAD51': {'types': {'Breast Cancer': 0.70, 'Ovarian Cancer': 0.65}},
    'XRCC2': {'types': {'Breast Cancer': 0.65}},
    'MECOM': {'types': {'Leukemia': 0.80}},
    'GATA2': {'types': {'Leukemia': 0.85, 'Myelodysplastic Syndrome': 0.90}},
    'RUNX1': {'types': {'Leukemia': 0.90}},
    'CEBPA': {'types': {'Leukemia': 0.85}},
    'DDX41': {'types': {'Leukemia': 0.85, 'Myelodysplastic Syndrome': 0.80}},
    'ANKRD26': {'types': {'Leukemia': 0.75, 'Thrombocytopenia': 0.85}},
    'ETV6': {'types': {'Leukemia': 0.80, 'Thyroid Cancer': 0.65}},
    'PAX5': {'types': {'Leukemia': 0.80}},
}


class ClinVarCancerDetector:
    """Cancer detection and type classification using trained ClinVar model"""

    def __init__(self, model_path=None):
        """Initialize detector with trained model"""
        if model_path is None:
            model_path = _find_latest_model()
            if model_path is None:
                raise FileNotFoundError(
                    "No trained model found in output_clinvar/. "
                    "Please train a model first."
                )

        self.device = device
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.label_encoders = {}
        self.input_dim = None
        self.gene_db = CANCER_GENE_DATABASE

        self._load_model()
        print("ClinVar Cancer Detector initialized")
        print(f"  Device: {device}")
        print(f"  Model: {model_path}")
        print(f"  Cancer genes in database: {len(self.gene_db)}")

    def _load_model(self):
        """Load trained model from file"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at: {self.model_path}")

        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)

        self.input_dim = checkpoint.get('input_dim')
        if self.input_dim is None:
            raise ValueError("Checkpoint does not contain 'input_dim' key")

        self.scaler = checkpoint.get('scaler')
        if self.scaler is None:
            raise ValueError("Checkpoint does not contain 'scaler' key")

        self.label_encoders = checkpoint.get('label_encoders')
        if self.label_encoders is None:
            raise ValueError("Checkpoint does not contain 'label_encoders' key")
        if not isinstance(self.label_encoders, dict):
            raise ValueError("'label_encoders' in checkpoint is not a dictionary")
        # Additional validation: check that the dict contains the expected keys
        required_keys = ['VariationType', 'GeneSymbol']
        for key in required_keys:
            if key not in self.label_encoders:
                raise ValueError(f"Label encoder dictionary is missing required key: '{key}'")

        config = checkpoint.get('config', {})
        hidden_dims = config.get('hidden_dims', [128, 64, 32])
        dropout = config.get('dropout', 0.3)
        self.model = ClinVarNet(input_dim=self.input_dim, hidden_dims=hidden_dims, dropout=dropout).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        print(f"  Model loaded successfully (input_dim={self.input_dim})")

    def preprocess_variant(self, chromosome, position, gene='UNKNOWN', variant_type='SNP',
                           ref_allele='', alt_allele='', variation_id=0,
                           chr_grch38=0, start_grch38=0, stop_grch38=0):
        return preprocess_variant(
            chromosome=chromosome, position=position, gene=gene,
            variant_type=variant_type, ref_allele=ref_allele, alt_allele=alt_allele,
            variation_id=variation_id, chr_grch38=chr_grch38,
            start_grch38=start_grch38, stop_grch38=stop_grch38,
            label_encoders=self.label_encoders
        )

    def predict_cancer_probability(self, features):
        if self.scaler is None:
            raise ValueError("Scaler not loaded.")
        if features.shape[1] != self.input_dim:
            raise ValueError(
                f"Feature dimension mismatch: got {features.shape[1]}, "
                f"model expects {self.input_dim}. "
                f"Please retrain the model to match the current feature set."
            )
        features_scaled = self.scaler.transform(features)

        X_tensor = torch.FloatTensor(features_scaled).to(self.device)

        with torch.no_grad():
            prob = self.model(X_tensor).cpu().numpy().flatten()[0]

        return float(prob)

    def predict_cancer_type(self, gene, cancer_probability):
        """
        Predict cancer type(s) based on gene and model probability

        Returns:
            dict: Cancer types with confidence percentages
        """
        gene_upper = str(gene).upper().strip()

        if gene_upper not in self.gene_db:
            return {
                'gene': gene,
                'cancer_types': {},
                'top_cancer_type': 'Unknown',
                'top_confidence': 0.0,
                'note': f'Gene {gene_upper} not in cancer database'
            }

        # Get cancer types and confidences for this gene
        cancer_types = self.gene_db[gene_upper]['types']

        # Weight by model prediction confidence
        weighted_types = {}
        for cancer_type, base_confidence in cancer_types.items():
            # Combine gene database confidence with model probability
            weighted_conf = base_confidence * cancer_probability * 100
            weighted_types[cancer_type] = round(weighted_conf, 2)

        # Sort by confidence
        sorted_types = sorted(weighted_types.items(), key=lambda x: x[1], reverse=True)

        return {
            'gene': gene,
            'cancer_types': dict(sorted_types),
            'top_cancer_type': sorted_types[0][0] if sorted_types else 'Unknown',
            'top_confidence': sorted_types[0][1] if sorted_types else 0.0
        }

    def get_risk_level(self, probability):
        """Determine risk level from probability"""
        if probability >= 0.7:
            return 'HIGH'
        elif probability >= 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'

    def detect(self, chromosome, position, gene='UNKNOWN', variant_type='SNP',
               ref_allele='', alt_allele='', variation_id=0,
               chr_grch38=0, start_grch38=0, stop_grch38=0,
               clinical_significance=''):
        """
        Complete cancer detection with type classification

        Parameters:
        -----------
        chromosome : str or int
            Chromosome number/name
        position : int
            Genomic position
        gene : str
            Gene symbol
        variant_type : str
            Type of variant (SNP, deletion, insertion, etc.)
        ref_allele : str
            Reference allele sequence
        alt_allele : str
            Alternate allele sequence
        variation_id : int
            ClinVar variation ID
        chr_grch38 : int
            Chromosome from GRCh38
        start_grch38 : int
            Start position from GRCh38
        stop_grch38 : int
            Stop position from GRCh38
        clinical_significance : str
            ClinVar clinical significance

        Returns:
        --------
        dict : Complete detection results
        """
        # Preprocess
        features = self.preprocess_variant(
            chromosome=chromosome,
            position=position,
            gene=gene,
            variant_type=variant_type,
            ref_allele=ref_allele,
            alt_allele=alt_allele,
            variation_id=variation_id,
            chr_grch38=chr_grch38,
            start_grch38=start_grch38,
            stop_grch38=stop_grch38
        )

        # Get model prediction
        cancer_prob = self.predict_cancer_probability(features)
        is_pathogenic = cancer_prob >= 0.5
        risk_level = self.get_risk_level(cancer_prob)

        # Predict cancer type(s)
        cancer_type_pred = self.predict_cancer_type(gene, cancer_prob)

        return {
            'variant_info': {
                'chromosome': chromosome,
                'position': position,
                'gene': gene,
                'variant_type': variant_type,
                'ref_allele': ref_allele,
                'alt_allele': alt_allele,
                'clinical_significance': clinical_significance
            },
            'cancer_detection': {
                'is_pathogenic': is_pathogenic,
                'cancer_probability': round(cancer_prob * 100, 2),
                'risk_level': risk_level
            },
            'cancer_type_prediction': cancer_type_pred
        }

    def detect_from_csv(self, csv_path, output_path=None):
        """
        Process a CSV file of variants and detect cancer types

        Parameters:
        -----------
        csv_path : str
            Path to input CSV file
        output_path : str, optional
            Path to save results CSV

        Returns:
        --------
        pd.DataFrame : Results with cancer predictions
        """
        df = pd.read_csv(csv_path)
        results = []

        for idx, row in df.iterrows():
            result = self.detect(
                chromosome=row.get('Chromosome', row.get('CHR', 1)),
                position=row.get('Position', row.get('POS', 0)),
                gene=row.get('Gene', row.get('GeneSymbol', 'UNKNOWN')),
                variant_type=row.get('Variant_Type', row.get('VariationType', 'SNP')),
                ref_allele=row.get('Ref_Allele', ''),
                alt_allele=row.get('Alt_Allele', ''),
                variation_id=row.get('VariationID', 0),
                chr_grch38=row.get('CHR_GRCh38', 0),
                start_grch38=row.get('Start_GRCh38', 0),
                stop_grch38=row.get('Stop_GRCh38', 0),
                clinical_significance=row.get('Clinical_Significance', '')
            )

            results.append({
                'Chromosome': result['variant_info']['chromosome'],
                'Position': result['variant_info']['position'],
                'Gene': result['variant_info']['gene'],
                'Variant_Type': result['variant_info']['variant_type'],
                'Cancer_Probability_%': result['cancer_detection']['cancer_probability'],
                'Risk_Level': result['cancer_detection']['risk_level'],
                'Is_Pathogenic': result['cancer_detection']['is_pathogenic'],
                'Top_Cancer_Type': result['cancer_type_prediction']['top_cancer_type'],
                'Top_Confidence_%': round(result['cancer_type_prediction']['top_confidence'], 2)
            })

        results_df = pd.DataFrame(results)

        if output_path:
            results_df.to_csv(output_path, index=False)
            print(f"Results saved to: {output_path}")

        return results_df

    def print_detailed_report(self, result):
        """Print formatted detection report"""
        print("\n" + "=" * 80)
        print("CLINVAR CANCER DETECTION REPORT")
        print("=" * 80)

        vi = result['variant_info']
        print(f"\nVariant: {vi['gene']} | chr{vi['chromosome']}:{vi['position']}")
        print(f"Type: {vi['variant_type']} | Ref: {vi['ref_allele'] or 'N/A'} | Alt: {vi['alt_allele'] or 'N/A'}")
        if vi['clinical_significance']:
            print(f"Clinical Significance: {vi['clinical_significance']}")

        cd = result['cancer_detection']
        print(f"\n--- CANCER RISK ASSESSMENT ---")
        print(f"Probability: {cd['cancer_probability']:.1f}%")
        print(f"Risk Level: {cd['risk_level']}")
        print(f"Pathogenic: {'YES' if cd['is_pathogenic'] else 'NO'}")

        ct = result['cancer_type_prediction']
        print(f"\n--- PREDICTED CANCER TYPE(S) ---")
        if ct['cancer_types']:
            print(f"\nTop Prediction: {ct['top_cancer_type']} ({ct['top_confidence']:.1f}% confidence)")
            print("\nAll Predicted Cancer Types:")
            for cancer_type, confidence in list(ct['cancer_types'].items())[:5]:
                bar = '#' * int(confidence / 5)
                print(f"  {cancer_type}: {confidence:.1f}% {bar}")
        else:
            print(f"Note: {ct.get('note', 'No predictions available')}")

        print("\n" + "=" * 80)


def main():
    """Demo: Cancer detection with type classification"""

    print("\n" + "=" * 80)
    print("CLINVAR CANCER DETECTION AND TYPE CLASSIFICATION SYSTEM")
    print("=" * 80)

    # Initialize detector (auto-find latest model)
    detector = ClinVarCancerDetector()

    print("\n" + "=" * 80)
    print("CANCER DETECTION EXAMPLES")
    print("=" * 80)

    # Example 1: BRCA1 mutation
    print("\n[Example 1] BRCA1 Pathogenic Variant")
    print("-" * 80)
    result1 = detector.detect(
        chromosome='17',
        position=43044295,
        gene='BRCA1',
        variant_type='Missense',
        ref_allele='G',
        alt_allele='A',
        variation_id=12345,
        chr_grch38=17,
        start_grch38=43044295,
        stop_grch38=43044295,
        clinical_significance='Pathogenic'
    )
    detector.print_detailed_report(result1)

    # Example 2: EGFR mutation (Lung Cancer)
    print("\n[Example 2] EGFR Mutation (Lung Cancer Driver)")
    print("-" * 80)
    result2 = detector.detect(
        chromosome='7',
        position=55086714,
        gene='EGFR',
        variant_type='Missense',
        ref_allele='C',
        alt_allele='T',
        variation_id=23456,
        chr_grch38=7,
        start_grch38=55086714,
        stop_grch38=55086714,
        clinical_significance='Pathogenic'
    )
    detector.print_detailed_report(result2)

    # Example 3: APC mutation (Colorectal Cancer)
    print("\n[Example 3] APC Mutation (Colorectal Cancer)")
    print("-" * 80)
    result3 = detector.detect(
        chromosome='5',
        position=112045052,
        gene='APC',
        variant_type='Frameshift',
        ref_allele='GAA',
        alt_allele='G',
        variation_id=34567,
        chr_grch38=5,
        start_grch38=112045052,
        stop_grch38=112045052,
        clinical_significance='Pathogenic'
    )
    detector.print_detailed_report(result3)

    # Example 4: VHL mutation (Renal Cell Carcinoma)
    print("\n[Example 4] VHL Mutation (Renal Cell Carcinoma)")
    print("-" * 80)
    result4 = detector.detect(
        chromosome='3',
        position=10183915,
        gene='VHL',
        variant_type='Missense',
        ref_allele='T',
        alt_allele='C',
        variation_id=45678,
        chr_grch38=3,
        start_grch38=10183915,
        stop_grch38=10183915,
        clinical_significance='Pathogenic'
    )
    detector.print_detailed_report(result4)

    # Example 5: TP53 mutation (Multiple cancer types)
    print("\n[Example 5] TP53 Mutation (Li-Fraumeni Syndrome - Multiple Cancer Risks)")
    print("-" * 80)
    result5 = detector.detect(
        chromosome='17',
        position=7577121,
        gene='TP53',
        variant_type='Missense',
        ref_allele='G',
        alt_allele='A',
        variation_id=56789,
        chr_grch38=17,
        start_grch38=7577121,
        stop_grch38=7577121,
        clinical_significance='Pathogenic'
    )
    detector.print_detailed_report(result5)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    examples = [
        ('BRCA1', result1, 'Breast/Ovarian Cancer'),
        ('EGFR', result2, 'Lung Cancer'),
        ('APC', result3, 'Colorectal Cancer'),
        ('VHL', result4, 'Renal Cell Carcinoma'),
        ('TP53', result5, 'Multiple Cancer Types')
    ]

    print(f"\n{'Gene':<10} {'Cancer Prob':<15} {'Risk':<10} {'Top Cancer Type':<35} {'Confidence':<12}")
    print("-" * 82)
    for gene, result, expected in examples:
        cd = result['cancer_detection']
        ct = result['cancer_type_prediction']
        print(f"{gene:<10} {cd['cancer_probability']:>6.1f}%         {cd['risk_level']:<10} {ct['top_cancer_type']:<35} {ct['top_confidence']:>6.1f}%")

    print("\n" + "=" * 80)
    print("Detection complete!")
    print("=" * 80 + "\n")


if __name__ == '__main__':
    main()
