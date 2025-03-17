import pandas as pd
from snsql import *

pums = pd.read_csv("https://raw.githubusercontent.com/opendifferentialprivacy/dp-test-datasets/master/data/PUMS_california_demographics_1000/data.csv")

print(pums)


# Adopted from https://github.com/opendp/smartnoise-sdk/blob/main/datasets/PUMS.yaml
metadata = {
  "PUMS": {
    "PUMS": {
      "PUMS": {
        "row_privacy": True,
        "rows": 1000,
        "age": {
          "type": "int",
          "lower": 0,
          "upper": 100
        },
        "sex": {
          "type": "string"
        },
        "educ": {
          "type": "string"
        },
        "race": {
          "type": "string"
        },
        "income": {
          "type": "int",
          "lower": 0,
          "upper": 500000
        },
        "married": {
          "type": "string"
        }
      }
    }
  }
}

private_reader = from_connection(
    pums, metadata=metadata,
    privacy=Privacy(epsilon=1.0, delta=1/1000)
)

query = 'SELECT married, COUNT(*) AS n FROM PUMS.PUMS GROUP BY married'

result_dp = private_reader.execute_df(query)
print(result_dp)

INFO:SingleTableSynthesizer:{'EVENT': 'Instance', 'TIMESTAMP': datetime.datetime(2025, 3, 4, 18, 13, 1, 473775), 'SYNTHESIZER CLASS NAME': 'CTGANSynthesizer', 'SYNTHESIZER ID': 'CTGANSynthesizer_1.18.0_55e08d1c6774470186c56914c428e6f5'}
INFO:SingleTableSynthesizer:{'EVENT': 'Fit', 'TIMESTAMP': datetime.datetime(2025, 3, 4, 18, 13, 1, 474341), 'SYNTHESIZER CLASS NAME': 'CTGANSynthesizer', 'SYNTHESIZER ID': 'CTGANSynthesizer_1.18.0_55e08d1c6774470186c56914c428e6f5', 'TOTAL NUMBER OF TABLES': 1, 'TOTAL NUMBER OF ROWS': 9114, 'TOTAL NUMBER OF COLUMNS': 14}
INFO:sdv.data_processing.data_processor:Fitting table  metadata
INFO:sdv.data_processing.data_processor:Fitting formatters for table 
INFO:sdv.data_processing.data_processor:Fitting constraints for table 
INFO:sdv.data_processing.data_processor:Setting the configuration for the ``HyperTransformer`` for table 
INFO:sdv.data_processing.data_processor:Fitting HyperTransformer for table 
INFO:SingleTableSynthesizer:{'EVENT': 'Fit processed data', 'TIMESTAMP': datetime.datetime(2025, 3, 4, 18, 13, 1, 535920), 'SYNTHESIZER CLASS NAME': 'CTGANSynthesizer', 'SYNTHESIZER ID': 'CTGANSynthesizer_1.18.0_55e08d1c6774470186c56914c428e6f5', 'TOTAL NUMBER OF TABLES': 1, 'TOTAL NUMBER OF ROWS': 9114, 'TOTAL NUMBER OF COLUMNS': 14}
INFO:rdt.transformers.null:Guidance: There are no missing values in column Credit_Limit. Extra column not created.
INFO:rdt.transformers.null:Guidance: There are no missing values in column Avg_Open_To_Buy. Extra column not created.
INFO:rdt.transformers.null:Guidance: There are no missing values in column Total_Trans_Amt. Extra column not created.
INFO:rdt.transformers.null:Guidance: There are no missing values in column Total_Trans_Ct. Extra column not created.
INFO:rdt.transformers.null:Guidance: There are no missing values in column Customer_Age. Extra column not created.
INFO:rdt.transformers.null:Guidance: There are no missing values in column Months_on_book. Extra column not created.
INFO:rdt.transformers.null:Guidance: There are no missing values in column Total_Amt_Chng_Q4_Q1. Extra column not created.

The Kernel crashed while executing code in the current cell or a previous cell. 
Please review the code in the cell(s) to identify a possible cause of the failure. 
Click <a href='https://aka.ms/vscodeJupyterKernelCrash'>here</a> for more info. 
View Jupyter <a href='command:jupyter.viewOutput'>log</a> for further details.