<h1 align="center">High Mass X-Ray Binaries Catalogs</h1>


Abstract

In this work, we address the problem of missing data in astronomical catalogs (Fortin, Neumann, Malacaria, etc.) by evaluating and comparing different imputation strategies tailored for datasets with both numerical and categorical variables. Classical methods such as mean/mode imputation are used as baselines, while more sophisticated techniques including k-nearest neighbors (kNN), Bayesian methods via Markov Chain Monte Carlo (e.g., emcee), and low-rank matrix completion (SoftImpute, SVD) are applied to numeric variables. For categorical variables, probabilistic models and decision-tree-based methods are considered. Furthermore, integrated approaches—such as Autoencoders and Generative Adversarial Imputation Networks (GAIN)—are explored to jointly handle mixed-type variables by learning latent representations. The performance of these imputers is quantitatively assessed using metrics such as RMSE, MAE, and categorical agreement, with model selection guided by the proportion of missing data per variable, the total dataset size, and the expected computational cost.

Besides, we can impute/obtain the missing values of the categorical & numerical parameters together or separately. For the decision making of the approach, we can take in consideration the missing value percentage per variable, the total amount of data, the nature of the missing values and an acceptable computational complexity:

    Separately:

        1. we can use OneHotEncoder and then use KNN for imputing the mode of the correspondent cluster of the Class / Categorical variable:
            - Mode
            - Probabilistic Models (Naive Bayes)
            - Decision Trees


        2. we can obtain the numerical data alone via 2 methoc:
            * Bayesian Aproach: them can be used weather to estimate the galactic/stellar parameters (stellar mass, age, metalicity, extinction by MCMC), modeling of light curves or Cosmologic Inference (matter density, Hubble Constant or spectral index):
                - Using EMCEE package and its libraries.
                - PyMC: general probabilistic modeling
                - Dynesty / Multinest: nested sampling for calculating of bayesian evidence.
                - Stan / CmdStanPy: ver powerful for complex hierarchical models.
                - Bilby (Bayesian Inference library): used in gravitational waves.
            * MICE / Iterative Imputer 
            * Machine Learning tools (from the [paper](https://peerj.com/articles/cs-619/)): 
                Here we can use the metrics RMSE, NRMSE and MAE for evaluating weather the imputed values fit well the original values:
                - KNN, Mean, SoftImpute, SVDimpute, Iterative Imputation, EMI, DMI, KDMI, KEMI, KEMI+, KI & FCKI.
                Also, we can compare the methods in respect to the accuracy value of a classification model as Decision Tree Classifier.

    Or together: 

        1. Autoencoders (DAE, VAE):
        2. GANS for imputation:
        3. Integrated models as DMI, EMI, KEMI









## Bibliography

### To understand the meaning of Hardness
- [Variable structures in the stellar wind of the HMXB Vela X-1](https://arxiv.org/pdf/2410.21456)
- [Observing the onset of the accretion wake in Vela X-1](https://www.aanda.org/articles/aa/pdf/2023/06/aa45708-22.pdf)

### Bonsai Tool (for estimating the Spectral-type mass)

- [The BONNSAI project: web-service](https://www.astro.uni-bonn.de/stars/bonnsai/)
- [Bonnsai: a Bayesian tool for comparing stars with stellar evolution models](https://ui.adsabs.harvard.edu/abs/2014A%26A...570A..66S/abstract)
- [The VLT-FLAMES Tarantula Survey. XXIX. Massive star formation in the local 30 Doradus starburst](https://ui.adsabs.harvard.edu/abs/2018A%26A...618A..73S/abstract)

### For Spectroscopic mass

- [An X-ray-quiet black hole born with a negligible kick in a massive binary within the Large Magellanic Cloud](https://ui.adsabs.harvard.edu/abs/2022NatAs...6.1085S/abstract)

### Catalogs

- Neumann: [XRBcats: Galactic High Mass X-ray Binary Catalogue★](https://ui.adsabs.harvard.edu/abs/2023A%26A...677A.134N/abstract) and [Online Catalog](http://astro.uni-tuebingen.de/~xrbcat/HMXBcat.html)
- Fortin: [A catalogue of high-mass X-ray binaries in the Galaxy: from the INTEGRAL to the Gaia era](https://ui.adsabs.harvard.edu/abs/2023A%26A...671A.149F/abstract) and [Online Catalog](https://binary-revolution.github.io/HMXBwebcat/catalog.html)
- Malacaria: [Malacaria +2020](https://ui.adsabs.harvard.edu/search/p_=1&q=author%3A%22^malacaria%2Cc%22&sort=date%20desc%2C%20bibcode%20desc) and [Online Catalog](https://iopscience.iop.org/article/10.3847/1538-4357/ab855c)
- Kim: [Catalog of the Galactic Population of X-Ray Pulsars in High-mass X-Ray Binary Systems](https://ui.adsabs.harvard.edu/abs/2023ApJS..268...21K/abstract) and [Online Catalog](https://iopscience.iop.org/article/10.3847/1538-4365/ace68f#apjsace68fapp1)

### Six systems like my own ones (for imputation of Spectral type, p.e.)

- [The stellar and wind parameters of six prototypical HMXBs and their evolutionary status](https://ui.adsabs.harvard.edu/abs/2020A%26A...634A..49H/abstract)

### Some adittional information

- [Venn Diagram](https://en.wikipedia.org/wiki/Venn_diagram) and also [some of this kind](https://upload.wikimedia.org/wikipedia/commons/5/56/6-set_Venn_diagram_SMIL.svg)
- [Swift Sentitivity](https://heasarc.gsfc.nasa.gov/W3Browse/swift/swbat105m.html)
- [Interactive Kendall matrix and dendogram](https://ipywidgets.readthedocs.io/en/8.1.5/examples/Using%20Interact.html)

### To involve machine learning tools

- Complex fractal analysis: [AleRCE broker](https://alerce.science), [Alert Classification for the ALeRCE Broker System: The Light Curve Classifier](https://arxiv.org/pdf/2008.03311), [Evolution of fractality in space plasmas of interest to geomagnetic activity](https://ui.adsabs.harvard.edu/abs/2018NPGeo..25..207M/abstract) and [Complex Network Study of Solar Magnetograms](https://ui.adsabs.harvard.edu/abs/2022Entrp..24..753M/abstract).


### To impute the nan values for numeric columns i can use the Mean Impute or the KNN impute:

- [Applied Machine-Learning Models to Identify Spectral Sub-Types of M
Dwarfs from Photometric Surveys](https://iopscience.iop.org/article/10.1088/1538-3873/acc974/pdf) 
- [And other paper](https://arxiv.org/pdf/2111.13806)

and for the imputation of the spectral type and the Class (both columns of fortin catalogue) i can use 

