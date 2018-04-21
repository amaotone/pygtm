# PyGTM

A python implementation of Generative Topographic Mapping.

**This is beta release.**
For example, this project has no test as you can see.

## Requirements

- numpy
- scipy
- scikit-learn

## Getting Started

To install PyGTM, use `pip`

```bash
$ pip install -U pygtm
```

The pygtm package inherits scikit-learn classes.

```python
from pygtm import GTM
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

iris = load_iris()
model = make_pipeline(
    StandardScaler(),
    GTM(n_components=2)
)
embedding = model.fit_transform(iris.data)
```

## References

- [GTM: The Generative Topographic Mapping](https://www.microsoft.com/en-us/research/publication/gtm-the-generative-topographic-mapping/)
- [Development of the Generative Topographic Mapping](https://www.microsoft.com/en-us/research/publication/developments-of-the-generative-topographic-mapping/)