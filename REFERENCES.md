# References

Scientific references for soilspec package.

## Data Access

**Albinet, F. et al. (2024)**. soilspecdata: Python package for accessing soil spectral libraries.
GitHub: https://github.com/franckalbinet/soilspecdata

- We integrate with this package for OSSL data loading
- See `soilspec.datasets.OSSLDataset`

## Experimental Features

**Albinet, F., Nkwain Nkemih, T., Bonte, P., van der Ha, J., Mandiaye, S.M., Verbelen, T., Cornelis, W. (2023)**. Prediction of exchangeable potassium in soil through mid-infrared spectroscopy and deep learning.
GitHub: https://github.com/franckalbinet/lssm

- Implements GADF + transfer learning approach (ResNet/ViT from ImageNet)
- We provide similar functionality in `soilspec.experimental.GADFTransformer`
- **Note:** We recommend 1D CNNs over GADF for most use cases

**Wang, Z. & Oates, T. (2015)**. Encoding Time Series as Images for Visual Inspection and Classification Using Tiled Convolutional Neural Networks. AAAI Workshop on Learning Rich Representations from Low-Level Sensors.

- Original GADF method for time series
- Adapted to soil spectroscopy by Albinet et al.

## Traditional Models

### Memory-Based Learning (MBL)

**Ramirez-Lopez, L., Behrens, T., Schmidt, K., Stevens, A., Demattê, J.A.M., Scholten, T. (2013)**. The spectrum-based learner: A new local approach for modeling soil vis-NIR spectra of complex datasets. *Geoderma* 195:268-279.
DOI: 10.1016/j.geoderma.2012.12.014

**Ramirez-Lopez, L., Schmidt, K., Behrens, T., van Wesemael, B., Demattê, J.A.M., Scholten, T. (2014)**. Sampling optimal calibration sets in soil infrared spectroscopy. *Geoderma* 226:140-150.
DOI: 10.1016/j.geoderma.2014.02.002

### Cubist

**Quinlan, J.R. (1992)**. Learning with continuous classes. *Proceedings of the 5th Australian Joint Conference on Artificial Intelligence*, pp. 343-348.

**Quinlan, J.R. (1993)**. Combining instance-based and model-based learning. *Proceedings of the Tenth International Conference on Machine Learning*, pp. 236-243.

**Kuhn, M. & Johnson, K. (2013)**. Applied Predictive Modeling. Springer.

### OSSL (Open Soil Spectral Library)

**Sanderman, J., Savage, K., Dangal, S.R.S., Midwood, A.J., Wills, S. (2020)**. Mid-infrared spectroscopy for prediction of soil health indicators in the United States. *Soil Science Society of America Journal* 84(1):251-261.
DOI: 10.1002/saj2.20009

**Hengl, T., Miller, M.A.E., Križan, J., et al. (2021)**. African soil properties and nutrients mapped at 30 m spatial resolution using two-scale ensemble machine learning. *Scientific Reports* 11(1):6130.
DOI: 10.1038/s41598-021-85639-y

**Viscarra Rossel, R.A., Behrens, T., Ben-Dor, E., et al. (2016)**. A global spectral library to characterize the world's soil. *Earth-Science Reviews* 155:198-230.
DOI: 10.1016/j.earscirev.2016.01.012

**Wijewardane, N.K., Ge, Y., Wills, S., Loecke, T. (2018)**. Predicting physical and chemical properties of US soils with a mid-infrared reflectance spectral library. *Soil Science Society of America Journal* 82(3):722-731.
DOI: 10.2136/sssaj2017.10.0361

## Spectral Interpretation

**Soriano-Disla, J.M., Janik, L.J., Viscarra Rossel, R.A., Macdonald, L.M., McLaughlin, M.J. (2014)**. The performance of visible, near-, and mid-infrared reflectance spectroscopy for prediction of soil physical, chemical, and biological properties. *Applied Spectroscopy Reviews* 49(2):139-186.
DOI: 10.1080/05704928.2013.811081

**Margenot, A.J., Calderón, F.J., Goyne, K.W., Mukome, F.N.D., Parikh, S.J. (2017)**. Infrared spectroscopy, soil analysis applications. *Encyclopedia of Spectroscopy and Spectrometry*, pp. 448-454.
DOI: 10.1016/B978-0-12-409547-2.12170-5

**Tinti, A., Tugnoli, V., Bonora, S., Francioso, O. (2015)**. Recent applications of vibrational mid-infrared (IR) spectroscopy for studying soil components: a review. *Journal of Central European Agriculture* 16(1):1-22.
DOI: 10.5513/JCEA01/16.1.1535

**Nguyen, T.T., Janik, L.J., Raupach, M. (1991)**. Diffuse reflectance infrared Fourier transform (DRIFT) spectroscopy in soil studies. *Soil Research* 29(1):49-67.
DOI: 10.1071/SR9910049

**Reeves III, J.B. (2010)**. Near- versus mid-infrared diffuse reflectance spectroscopy for soil analysis emphasizing carbon and laboratory versus on-site analysis: Where are we and what needs to be done? *Geoderma* 158:3-14.
DOI: 10.1016/j.geoderma.2009.04.005

## Preprocessing

**Savitzky, A. & Golay, M.J.E. (1964)**. Smoothing and differentiation of data by simplified least squares procedures. *Analytical Chemistry* 36(8):1627-1639.
DOI: 10.1021/ac60214a047

**Barnes, R.J., Dhanoa, M.S., Lister, S.J. (1989)**. Standard normal variate transformation and de-trending of near-infrared diffuse reflectance spectra. *Applied Spectroscopy* 43(5):772-777.
DOI: 10.1366/0003702894202201

**Geladi, P., MacDougall, D., Martens, H. (1985)**. Linearization and scatter-correction for near-infrared reflectance spectra of meat. *Applied Spectroscopy* 39(3):491-500.
DOI: 10.1366/0003702854248656

**Rinnan, Å., van den Berg, F., Engelsen, S.B. (2009)**. Review of the most common pre-processing techniques for near-infrared spectra. *TrAC Trends in Analytical Chemistry* 28(10):1201-1222.
DOI: 10.1016/j.trac.2009.07.007

**Donoho, D.L. (1995)**. De-noising by soft-thresholding. *IEEE Transactions on Information Theory* 41(3):613-627.
DOI: 10.1109/18.382009

## Evaluation Metrics

**Williams, P.C. (1987)**. Variables affecting near-infrared reflectance spectroscopy analysis. In: *Near-infrared technology in the agricultural and food industries*. AACC, St. Paul, MN.

**Chang, C.W., Laird, D.A., Mausbach, M.J., Hurburgh, C.R. (2001)**. Near-infrared reflectance spectroscopy-principal components regression analyses of soil properties. *Soil Science Society of America Journal* 65:480-490.

**Bellon-Maurel, V., Fernandez-Ahumada, E., Palagos, B., Roger, J.M., McBratney, A. (2010)**. Critical review of chemometric indicators commonly used for assessing the quality of the prediction of soil attributes by NIR spectroscopy. *TrAC Trends in Analytical Chemistry* 29(9):1073-1081.
DOI: 10.1016/j.trac.2010.05.006

## Deep Learning

**Padarian, J., Minasny, B., McBratney, A.B. (2019)**. Using deep learning for digital soil mapping. *Soil* 5(1):79-89.
DOI: 10.5194/soil-5-79-2019

**Tsakiridis, N.L., Keramaris, K.D., Theocharis, J.B., Zalidis, G.C. (2020)**. Simultaneous prediction of soil properties from VNIR-SWIR spectra using a localized multi-channel 1-D convolutional neural network. *Geoderma* 367:114208.
DOI: 10.1016/j.geoderma.2020.114208

**Liu, L., Ji, M., Buchroithner, M. (2019)**. Transferability of a visible and near-infrared model for soil organic carbon estimation in riparian landscapes. *Remote Sensing* 11(20):2438.
DOI: 10.3390/rs11202438

## Other Methods

**Wold, S., Sjöström, M., Eriksson, L. (2001)**. PLS-regression: a basic tool of chemometrics. *Chemometrics and Intelligent Laboratory Systems* 58(2):109-130.
DOI: 10.1016/S0169-7439(01)00155-1

**Mayerhöfer, T.G., Mutschke, H., Popp, J. (2020)**. The Bouguer-Beer-Lambert Law: Shining light on the obscure. *ChemPhysChem* 21(18):2029-2046.
DOI: 10.1002/cphc.202000464

---

## How to Cite This Package

If you use soilspec in your research, please cite:

```
soilspec: Evidence-Based Machine Learning for Soil Spectroscopy (2024)
https://github.com/[username]/soilspec
```

And cite the specific methods you use:
- MBL: Ramirez-Lopez et al. (2013)
- Cubist: Quinlan (1992, 1993), Sanderman et al. (2020)
- OSSL data: Via soilspecdata (Albinet et al. 2024)
- GADF (if used): Wang & Oates (2015), Albinet et al. (2023)
- 1D CNNs: Tsakiridis et al. (2020), Padarian et al. (2019)
