# Master-Thesis

Hordeum vulgare, like many other crops, suffers from the reduction of genetic diversity
caused by climate changes. Therefore, it is necessary to improve the performance of its
breeding. Nowadays, the area of interest in current research focuses on indirect selection
methods based on computational prediction modeling. This thesis deals with dynamic
metabolomic prediction based on genomic data consisting of 33,005 single nucleotide
polymorphisms. Metabolomic data include 128 metabolites belonging to 25 Halle exotic
barley families. The main goal of this thesis is creating dynamic metabolomic predictions
using different approaches chosen upon various publications. The created models will
be helpful for the prediction of phenotype or for revealing important traits of Hordeum
vulgare.

The study by Gemmer et al. [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7274421/] presented metabolomic information which is related to theHEB-25 population. The metabolomic information was extracted by GC-MS 2.1 and processed using MassHunter Qualitative Analysis software. As the result, 158 metabolites concentrations of 1419 lines were obtained.

The genomic dataset related to the above-mentioned metabolomic dataset used in this thesis was taken from the study by Maurer et al. [https://doi.ipk-gatersleben.de/DOI/0420f485-23ad-4dfa-9959-1a3e6807438b/436f7ff4-f37a-4c55-95e9-00f882326bf6/2]. The genomic information was extracted using the 50k Illumina Infinium iSelect 9k SNP chip. The genotyping resulted in 1429 lines of SNPs on 37 chromosomes.

5 multivariate and univariate models were optimized and created using the SVR, LASSO, sPLS, random forest and BLUP methods.
Before running predictions script, preprocessing of the data needs to be done by running the script preprocessing.py.
