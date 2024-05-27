import pandas as pd
from sdv.metadata import SingleTableMetadata
from sdv.single_table import TVAESynthesizer, CTGANSynthesizer
from FinDiffSynthesizer import FinDiffSynthesizer
from tabula_middle_padding import Tabula

class DataSynthesizer:
    def __init__(self, data: pd.DataFrame, synthesizer_type: str):#, categorical_columns=None):
        #categorical_columns only needed if using FinDiff
        self.data = data
        self.synthesizer_type = synthesizer_type
        self.synthsizer = None
        #self.categorical_columns = categorical_columns

        self.synthesizers = {
            'tvae': {
                'create': self.create_tvae,
                'fit': self.fit_tvae,
                'sample': self.sample_tvae
            },
            'ctgan': {
                'create': self.create_ctgan,
                'fit': self.fit_ctgan,
                'sample': self.sample_ctgan
            },
            'findiff': {
                'create': self.create_findiff,
                'fit': self.fit_findiff,
                'sample': self.sample_findiff
            },
            'tabula': {
                'create': self.create_tabula,
                'fit': self.fit_tabula,
                'sample': self.sample_tabula
            }
        }
        if synthesizer_type not in self.synthesizers:
            raise ValueError(f"Unknown synthesizer type: {synthesizer_type}")
        
        self.synthesizers[synthesizer_type]['create']()

    
    def create_tvae(self):
        print("Initializing TVAE synthesizer")
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(self.data)
        self.synthesizer = TVAESynthesizer(metadata)

    def fit_tvae(self):
        print("Fitting TVAE synthesizer")
        self.synthesizer.fit(self.data)

    def sample_tvae(self, num_samples: int):
        print("Sampling from TVAE synthesizer")
        synthetic_data = self.synthesizer.sample(num_rows=num_samples)
        return synthetic_data
    
    def create_ctgan(self):
        print("Initializing CTGAN synthesizer")
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(self.data)
        self.synthesizer = CTGANSynthesizer(metadata)
    
    def fit_ctgan(self):
        print("Fitting CTGAN synthesizer")
        self.synthesizer.fit(self.data)
    
    def sample_ctgan(self, num_samples: int):
        print("Sampling from CTGAN synthesizer")
        synthetic_data = self.synthesizer.sample(num_rows=num_samples)
        return synthetic_data
    
    def create_findiff(self):
        print("Initializing FinDiff synthesizer")
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(self.data)
        categorical_columns = metadata.get_column_names(sdtype='categorical')
        self.synthesizer = FinDiffSynthesizer(self.data, categorical_columns)

    def fit_findiff(self):
        print("Fitting FinDiff synthesizer")
        self.synthesizer.fit()
    
    def sample_findiff(self, num_samples: int):
        print("Sampling from FinDiff synthesizer")
        synthetic_data = self.synthesizer.sample(num_samples)
        return synthetic_data
        
    def create_tabula(self):
        print("Initializing Tabula synthesizer")
        self.synthesizer = Tabula(llm='distilgpt2', experiment_dir = "rice", batch_size=32, epochs=50)

    def fit_tabula(self):
        print("Fitting Tabula synthesizer")
        self.synthesizer.fit(self.data, conditional_col = self.data.columns[0])
    
    def sample_tabula(self, num_samples: int):
        print("Sampling from Tabula synthesizer")
        synthetic_data = self.synthesizer.sample(num_samples)
        return synthetic_data
    
    def fit(self):
        self.synthesizers[self.synthesizer_type]['fit']()
    
    def sample(self, num_samples: int):
        return self.synthesizers[self.synthesizer_type]['sample'](num_samples)
