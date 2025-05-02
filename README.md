# ML2Solidity

[![Python Version](https://img.shields.io/badge/python-3.7%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

ML2Solidity is a Python library that allows you to export trained scikit-learn Random Forest models (both classification and regression) to optimized Solidity smart contracts for blockchain deployment.

## Features

- Export `RandomForestClassifier` models to Solidity contracts
- Export `RandomForestRegressor` models to Solidity contracts
- Assembly-optimized code for efficient on-chain execution
- Fixed-point representation with configurable precision
- Support for feature names and custom contract naming
- Gas-efficient implementation for blockchain deployment

## Installation

```bash
pip install ml2solidity
```

Or install from source:

```bash
git clone https://github.com/yourusername/ml2solidity.git
cd ml2solidity
pip install -e .
```

## Quick Start

```python
from sklearn.ensemble import RandomForestClassifier
from ml2solidity import ML2Solidity

# Train your model
clf = RandomForestClassifier(n_estimators=10, max_depth=5)
clf.fit(X_train, y_train)

# Export to Solidity
feature_names = ['temperature', 'humidity', 'pressure', 'wind_speed']
converter = ML2Solidity(clf, feature_names=feature_names, contract_name="WeatherClassifier")
solidity_code = converter.export(output_file="WeatherClassifier.sol")
```

## Detailed Usage

### Classification Example

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from ml2solidity import ML2Solidity

# Generate a synthetic dataset
X, y = make_classification(n_samples=1000, n_features=4, n_informative=3, 
                           n_redundant=1, n_classes=3, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a random forest classifier
clf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42)
clf.fit(X_train, y_train)

# Define feature names
feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4']

# Export the model to Solidity
converter = ML2Solidity(clf, feature_names=feature_names, contract_name="MyClassifier")
solidity_code = converter.export(output_file="MyClassifier.sol")
```

### Regression Example

```python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from ml2solidity import ML2Solidity

# Generate a synthetic dataset
X, y = make_regression(n_samples=1000, n_features=4, n_informative=3, noise=0.1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a random forest regressor
regr = RandomForestRegressor(n_estimators=5, max_depth=3, random_state=42)
regr.fit(X_train, y_train)

# Define feature names
feature_names = ['feature_1', 'feature_2', 'feature_3', 'feature_4']

# Export the model to Solidity
converter = ML2Solidity(regr, feature_names=feature_names, contract_name="MyRegressor")
solidity_code = converter.export(output_file="MyRegressor.sol")
```

### Custom Precision

By default, ML2Solidity uses 6 decimal places for fixed-point representation. You can customize this:

```python
# Use 8 decimal places for higher precision
converter = ML2Solidity(
    model, 
    feature_names=feature_names, 
    contract_name="HighPrecisionModel",
    precision=8
)
```

## Using the Generated Solidity Contracts

The generated Solidity contracts include a `forest_predict` function that takes an array of feature values and returns the prediction:

```solidity
// For classifier
int256 class = myClassifier.forest_predict(features);

// For regressor
int256 prediction = myRegressor.forest_predict(features);
```

For fixed-point representation, input feature values and output predictions are scaled by 10^precision:

- Input: A value of 1.5 with precision=6 would be represented as 1500000
- Output: A prediction of 2500000 with precision=6 would represent 2.5

## How It Works

ML2Solidity converts each tree in the Random Forest to an optimized Solidity function using assembly-level operations for efficiency. The main features include:

1. **Tree structure conversion**: Each decision tree is converted to a series of if-else statements
2. **Fixed-point arithmetic**: Decimal values are scaled for Solidity compatibility
3. **Assembly optimization**: Low-level assembly code for efficient gas usage
4. **Voting mechanism**: Classification models use majority voting, regression models use averaging

## Performance Considerations

- **Contract Size**: Each tree increases the contract size. For large models, consider reducing the number of trees or their depth.
- **Gas Usage**: The `forest_predict` function will consume gas proportional to the complexity of your forest model.
- **Precision**: Higher precision values increase the accuracy of decimal representation but may lead to overflow in extreme cases.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contribution

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use ML2Solidity in your research, please cite:

```
@misc{ml2solidity,
  author = {Your Name},
  title = {ML2Solidity: Export scikit-learn Random Forest models to Solidity},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yourusername/ml2solidity}}
}
```