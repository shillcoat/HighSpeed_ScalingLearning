# IT-PI: Dimensionless learning based on information

Information-theoretic, data-driven method to identify the most predictive dimensionless inputs for non-dimensional quantities. Developed by Yuan Yuan & Adrian Lozano-Duran: https://doi.org/10.1038/s41467-025-64425-8

Code for parallelized version provided by Gonzalo

## Notes on use

Due to random sampling of parameter space solutions can correspond to local rather than global minima. Best practice is to run ~30 times and take the solution with the largest absolute value of mutual information (MI)

Dimensions must be consistent within a case, however different dimension systems can be used for different cases