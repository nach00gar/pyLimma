<h1 align="center">pyLimma</h1>

<div align="center">
  <strong>Differential expression analysis for microarray and RNA-seq data using linear models and empirical Bayes.</strong>
</div>
<div align="center">
  Python implementation of the basics of R's limma package <sup>[1]</sup> including new features as Multiclass DEGs extraction via Coverage parameter <sup>[2]</sup> and Scikit-Learn integration for ML enriched pipelines.
</div>

<br />



<div align="center">
  <sub>Built as part of MSc thesis on treatment of RNASeq data by
  <a href="https://www.linkedin.com/in/ignacio-garach-vélez-7a6970218/">Ignacio Garach</a>
</div>



## Example
```python
import pandas 
from pyLimma import pyLimmaDEG

df = pd.read_csv("HeadNeckExpressionMatrix.csv").set_index("external_gene_name").T
labels = pd.read_csv("HeadNeckLabels.csv")

deg_extractor = pyLimmaDEG(lfc=1.5, pvalue=0.01, coverage=1)

deg_extractor.fit(df, labels)
degs_df = deg_extractor.transform(df, labels)

print(deg_extractor.stats) # Here you can extract stats for the non-selected genes. Useful for Volcano Plots!
```



## Installation
```sh
$ pip install pylimma
```

## See Also
- [Inteligent RNASeq analysis pipeline in Python](https://github.com/nach00gar/pyseqML) 
- [Surrogate Variable Analysis](https://github.com/nach00gar/SurrogateVariableAnalysisPython) - Unsupervised Batch Effects Removal

## References
<a id="1">[1]</a> 
Ritchie ME, Phipson B, Wu D, Hu Y, Law CW, Shi W, Smyth GK (2015). “limma powers differential expression analyses for RNA-sequencing and microarray studies.” Nucleic Acids Research, 43(7), e47. doi:10.1093/nar/gkv007.

<a id="1">[2]</a> 
Castillo, D., Galvez, J. M., Herrera, L. J., Rojas, F., Valenzuela, O., Caba, O., Prados, J., & Rojas, I. (2019). Leukemia multiclass assessment and classification from Microarray and RNA-seq technologies integration at gene expression level. PloS one, 14(2), e0212127. https://doi.org/10.1371/journal.pone.0212127


## License
[MIT](https://tldrlegal.com/license/mit-license)
