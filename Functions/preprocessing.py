import pandas as pd

import functions as fcn

__author__ = "bc. Petra Nemčeková"
__supervisor__ = "Ing. et Ing. Jana Schwarzerová MSc"
__license__ = "MIT"
__maintainer__ = "bc. Petra Nemčeková"
__email__ = "xnemce05@vutbr.cz"


"""Loading data"""

genome = pd.read_csv('./Data/SNP.csv')

metabolome1 = pd.read_csv('./Data/Metabolome_day1.csv').T

metabolites1_names = metabolome1.index[1:].values

"""Preprocessing"""

"""removing not matching data"""

metabolome1_dataframe_intersected, genome1_dataframe_intersected = fcn.getInterselectedWithFamilies(
    fcn.replaceColumnsWith1stRowValues(metabolome1), fcn.replaceIndexWith1stColumnValues(genome))

"""replace NaN values by 0"""

met1_preprocessed = fcn.replaceMetabolomeValuesWithNaN(
    metabolome1_dataframe_intersected)

"""export preprocessed data"""

met1_preprocessed.to_csv('metabolome1_preprocessed.csv')
genome1_dataframe_intersected.to_csv('SNP_gen1_preprocessed.csv')
