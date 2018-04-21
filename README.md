# PyGTM

A python implementation of Generative Topographic Mapping.

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
from sklearn.datasets import load_digits

digits = load_digits()
embddding = GTM().fit_transform(digits.data)
```

## References

- [GTM: The Generative Topographic Mapping](https://www.microsoft.com/en-us/research/publication/gtm-the-generative-topographic-mapping/)
- [Development of the Generative Topographic Mapping](https://www.microsoft.com/en-us/research/publication/developments-of-the-generative-topographic-mapping/)