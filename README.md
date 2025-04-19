<h1 align="center">High Mass X-Ray Binaries Catalogs</h1>


Here i use the Data_and_Catalogs_proyect.ipynb file to import the 4 catalogs (Fortin, Neumann actualized and not actualized both of them called cat_neuman_2 and cat_neuman respectively, Malacaria and Kim). 
Also, in the interactive_part_proyect.ipynb file I display all the graphics that require to the user the election of one or more parameters, giving the posibility of selecting the type of Companion (sg, Be, etc)
and the parameters of them, showing the Kendall correlation matrix, the venn diagram, dendograms for positive and negative correlation coefficients, etc. 

Finally, in the non_interactive_part_proyect.ipynb file I add all the graphics that don't require any election by the user (equatorial and galactic distribution of all HMXB's in a like comparation between the 4 catalogs, Corbet diagram, luminosity distribution 
using Swift/BAT fluxes from Neumann and distances from Fortin, the maximum distance for a minimum detectable flux for different luminosities and the distribution of the geometric mean of the min/max Soft Flux columns 
from Neumann catalog).

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