<h1 align="center">High Mass X-Ray Binaries Catalogs Missing values Imputation</h1>



> Abstract

> In this work, we address the problem of missing data in astronomical catalogs (Fortin, Neumann, Malacaria, etc.) by evaluating and comparing different imputation strategies tailored for datasets with both numerical and categorical variables. Classical methods such as mean/mode imputation are used as baselines, while more sophisticated techniques including k-nearest neighbors (kNN), Bayesian methods via Markov Chain Monte Carlo (e.g., emcee), and low-rank matrix completion (SoftImpute, SVD) are applied to numeric variables. For categorical variables, probabilistic models and decision-tree-based methods are considered. Furthermore, integrated approaches—such as Autoencoders and Generative Adversarial Imputation Networks (GAIN)—are explored to jointly handle mixed-type variables by learning latent representations. The performance of these imputers is quantitatively assessed using metrics such as RMSE, MAE, and categorical agreement, with model selection guided by the proportion of missing data per variable, the total dataset size, and the expected computational cost.

Obviously, we can use Interpolation, but it's more useful when it's applied to spatial or temporal data (not this case). 

Some questions: ¿Is it necessary to use oversampling, subsampling? ¿Do we wanna scale some variables?

For the decision making of the approach, we can take in consideration 
    - the missing value percentage per variable (see this papers for 10-25% level of missing values: [the first](https://arxiv.org/abs/2109.04227) and [the second one](https://arxiv.org/abs/2403.14687), while for 20%-80% level see this [another one](https://pmc.ncbi.nlm.nih.gov/articles/PMC8426774/?utm_source=chatgpt.com))
    - the total amount of data 
    - the nature of the missing values and 
    - an acceptable computational complexity

Besides, we can impute/obtain the missing values of the categorical & numerical parameters together or separately. 

    Separately:

        1. we can use OneHotEncoder and then use KNN for imputing the mode of the correspondent cluster of the Class / Categorical variable:
            - Mode
            - Probabilistic Models (Naive Bayes)
            - Decision Trees


        2. we can obtain the numerical data alone via 2 methods:
            - Bayesian Aproach: them can be used weather to estimate the galactic/stellar parameters (stellar mass, age, metalicity, extinction by MCMC), modeling of light curves or Cosmologic Inference (matter density, Hubble Constant or spectral index):
                - Using EMCEE package and its libraries.
                - PyMC: general probabilistic modeling
                - Dynesty / Multinest: nested sampling for calculating of bayesian evidence.
                - Stan / CmdStanPy: ver powerful for complex hierarchical models.
                - Bilby (Bayesian Inference library): used in gravitational waves.
            - MICE / Iterative Imputer 
            - Machine Learning tools (from the [paper](https://peerj.com/articles/cs-619/) ): 
                Here we can use the metrics RMSE, NRMSE and MAE for evaluating weather the imputed values fit well the original values:
                - KNN, Mean, SoftImpute, SVDimpute, Iterative Imputation, EMI, DMI, KDMI, KEMI, KEMI+, KI & FCKI.
                Also, we can compare the methods in respect to the accuracy value of a classification model as Decision Tree Classifier.

    Or together: 

        1. Autoencoders (DAE, VAE):
        2. GANS for imputation:
        3. Integrated models as DMI, EMI, KEMI









## Bibliography

### To understand 

The meaning of Hardness:
- [Variable structures in the stellar wind of the HMXB Vela X-1](https://arxiv.org/pdf/2410.21456)
- [Observing the onset of the accretion wake in Vela X-1](https://www.aanda.org/articles/aa/pdf/2023/06/aa45708-22.pdf)

The Spectroscopic mass:

- [An X-ray-quiet black hole born with a negligible kick in a massive binary within the Large Magellanic Cloud](https://ui.adsabs.harvard.edu/abs/2022NatAs...6.1085S/abstract)

Six systems like my own ones (for imputation of Spectral type, p.e.):

- [The stellar and wind parameters of six prototypical HMXBs and their evolutionary status](https://ui.adsabs.harvard.edu/abs/2020A%26A...634A..49H/abstract)

Swift Telescope:

- [Swift Sentitivity](https://heasarc.gsfc.nasa.gov/W3Browse/swift/swbat105m.html)

Some tools:

- [Venn Diagram](https://en.wikipedia.org/wiki/Venn_diagram) and also [some of this kind](https://upload.wikimedia.org/wikipedia/commons/5/56/6-set_Venn_diagram_SMIL.svg)
- [Interactive Kendall matrix and dendogram](https://ipywidgets.readthedocs.io/en/8.1.5/examples/Using%20Interact.html)


### Catalogs

- Neumann: [XRBcats: Galactic High Mass X-ray Binary Catalogue★](https://ui.adsabs.harvard.edu/abs/2023A%26A...677A.134N/abstract) and [Online Catalog](http://astro.uni-tuebingen.de/~xrbcat/HMXBcat.html)
- Fortin: [A catalogue of high-mass X-ray binaries in the Galaxy: from the INTEGRAL to the Gaia era](https://ui.adsabs.harvard.edu/abs/2023A%26A...671A.149F/abstract) and [Online Catalog](https://binary-revolution.github.io/HMXBwebcat/catalog.html)
- Malacaria: [Malacaria +2020](https://ui.adsabs.harvard.edu/search/p_=1&q=author%3A%22^malacaria%2Cc%22&sort=date%20desc%2C%20bibcode%20desc) and [Online Catalog](https://iopscience.iop.org/article/10.3847/1538-4357/ab855c)
- Kim: [Catalog of the Galactic Population of X-Ray Pulsars in High-mass X-Ray Binary Systems](https://ui.adsabs.harvard.edu/abs/2023ApJS..268...21K/abstract) and [Online Catalog](https://iopscience.iop.org/article/10.3847/1538-4365/ace68f#apjsace68fapp1)


### For imputation tasks:

For percentage of missing values:

- [On the Performance of Imputation Techniques for Missing Values on Healthcare Datasets,Luke Oluwaseye Joel, Wesley Doorsamy, Babu Sena Paul](https://arxiv.org/abs/2403.14687)
- [Evaluation of imputation techniques with varying percentage of missing data Seema Sangari, Herman E. Ray](https://arxiv.org/abs/2109.04227)

- [Evaluation of Multiple Imputation with Large Proportions of Missing Data: How Much Is Too Much?, Jin Hyuk Lee, J Charles Huber Jr.](https://pmc.ncbi.nlm.nih.gov/articles/PMC8426774/?utm_source=chatgpt.com)

For imputation

- [Applied Machine-Learning Models to Identify Spectral Sub-Types of M Dwarfs from Photometric Surveys](https://iopscience.iop.org/article/10.1088/1538-3873/acc974/pdf) 
- [Advanced methods for missing values imputation based on similarity learning, Khaled M. Fouad, Mahmoud M. Ismail, Ahmad Taher Azar and Mona M. Arafa](https://peerj.com/articles/cs-619/)

EMCEE:

- [Original Proposal -> Data analysis recipes: Fitting a model to data, David W. Hogg, Jo Bovy, Dustin Lang (2010)](https://arxiv.org/pdf/2111.13806)
- [Formalized Affine Invariant Ensemble Sampler Algorithm -> emcee: The MCMC Hammer, DANIEL FOREMAN-MACKEY, DAVID W. HOGG, DUSTIN LANG, & JONATHAN GOODMAN](https://iopscience.iop.org/article/10.1086/670067)

Bonsai Tool (for estimating the Spectral-type mass)

- [The BONNSAI project: web-service](https://www.astro.uni-bonn.de/stars/bonnsai/)
- [Bonnsai: a Bayesian tool for comparing stars with stellar evolution models](https://ui.adsabs.harvard.edu/abs/2014A%26A...570A..66S/abstract)
- [The VLT-FLAMES Tarantula Survey. XXIX. Massive star formation in the local 30 Doradus starburst](https://ui.adsabs.harvard.edu/abs/2018A%26A...618A..73S/abstract)


### Other options

AleRCE:

- [AleRCE broker](https://alerce.science)
- [Alert Classification for the ALeRCE Broker System: The Light Curve Classifier](https://arxiv.org/pdf/2008.03311)

Complex fractal analysis:

- [Evolution of fractality in space plasmas of interest to geomagnetic activity](https://ui.adsabs.harvard.edu/abs/2018NPGeo..25..207M/abstract) and [Complex Network Study of Solar Magnetograms](https://ui.adsabs.harvard.edu/abs/2022Entrp..24..753M/abstract).

----

Questions that have guided my understanding:

- In the case of the distances from GAIA DR3 do i have to query with the GAIA DR3 ID from neumann catalog?
- In the case I use the BAT fluxes from actualized Neumann Catalog, do i take the BAT_min_flux or the BAT_max_flux? 
- In the case I use the Swift Fluxes, Do I have to make a query to match all the objects in the Neumann+Fortin Catalog, right?
- How do we usually select the priors for the observables in the prediction of the models (i.e. parameters) in the bayesian Tool?
- Have you used the BONNSAI tool? For which cases? How do you use the BONNSAI tool for multiple systems simultaneously?
- Which of the three kinds of masses I'm supposed to obtain from the catalogs (as the observables from Fortin+Neumann catalog), since i don't know the uncertainties of the effective temperature neither the uncertainties from the surface gravity (neither the mean of this), neither luminosities? 
- Or do I have to obtain the luminosities from the Flux and then use them to predict the mass with BONNSAI?
- When we talk about the luminosity of the HMXBs, we are talking about the luminosity of both objects like one object? Or we are talking about the Companion of the Neutron star alone?
- Why in the Fig 5 Of Fortin the quantity of detected or identified spectral type increases? e.g. Be and sg.?
- I was thinking, it's the imputing of the median an inadequate form of using the data for the kendall correlation matrix and the Dendograms because of the definition of the median when there are a even amount of data for a determined parameter as the effective temperature. Right?
- It's possible to use the Bootstrap technique in some cases to determine the kendall Tau's coefficient with it's uncertainties? (e.g. with replacement)
- How can i take into account the absortion in the line of sight?
- There have to be a maximum distance because its contrary the flux have a minimum? When we say that a pulsar have a minimum flux, we are talking about a specific band in the Swift sensitivity?
- What's the definition of near-infrared?
- It's fine that the regime of the luminosities is for 14-195 keV?
- It's fine to take the aritmetic mean of the min and max flux from bat in neumann catalog?
- It's always the same to talk about the regime/energy band in the context of the X-ray energy emission? 
- Why the near infrarred it's named like that?
- Do make any sense if i want to take a Color-Magnitude diagram from the Neumann Catalog aparent magnitudes?
- Why the near infrarred/X-ray/optical bands are called the counterparts?
- How valid is it to extrapolate the terminal velocity and the mass loss rate?
- For example, is it perhaps more valid than interpolating some known value from the parameters of the HMXBs themselves, such as the mass of the compact object?
- Are we more interested in the Mass of the compact object or in the mass of the companion?
- Does the terminal velocity refer to the terminal velocity of the pulsar traveling through the medium, or is it more related to the terminal velocity of the stellar wind from the companion star?
- Which are the esential conditions to determine wether a compact object is a black hole?
- Why were the number of Be in relation to Sg companions biased in the context of the Catalogs?
- What we understand when we talk about the poblational studies?
- In a catalog of these kind, are we any time interested in some kind of time series (as in the spin up-down of the systems) or in the graphics of flux versus wavelenght?
- It's normal that the terminal velocity correlates the most with the mass loss rate in the super giant than in the Be stars like systems?
- What is MJ2000? It's like the currently calendar of astronomers? Why?
- What's the difference between the transient and Persistent HMXBs? Does it have relation with the persistent/transient accretion systems? So for example, in the Neumann and Fortin Catalogs are they transient and/or persistent HMXBs?
- Can i extrapolate also the espectroscopic mass, radius *, log luminosity (log L) and T* from the six prototypical HMXBs?
- Did we use the geometric mean between the max/min fluxes because they can be like 10 to the 10 power, and 10 to the 20?
- The reason that the 10 to the 50 power isn't possible for the luminosity is because of the eddington limit? (which consist in the luminosity of a star when its gas pressure overpower the force of gravity?) Do the stars consequently, have a maximum effective temperature?
- The Wolf-Rayet binary systems are O or B stars (in respect to the companion)
- The reason for the difference in the correlation coefficients (spin period-orbital period) in the correlation matrix and Corbet diagram could be due to the fact that the corbet diagram is for all Fortin systems, while in the correlation matrix only the systems that are present in Neumann and Fortin (?) In other words, the coefficient is different because we are taking a subset of systems. Maybe i've introduced some bias in the analysis. 
- When extrapolating the Spectral type should i take into account the fact that it has spectral class, subclass, luminosity class and aditional characteristics? For example, O8.5Ib-II(f)p where O is the spectral class, 8.5 is the subclass, Ib-II is the luminosity class and f & p are the aditional characteristics. So i have to consider this fact to make the extrapolation in first place. 
- Its normal that for example some systems have 0.6 * 10**-12 and others 6.123.123.123.123.123 ergs/s/cm**2 in the XRT_min_flux and XRT_max_flux? That's probably the the reason of why i couln't reproduce Fortin Figure. 

