"""
ML2Solidity - A library for exporting sklearn Random Forest models to Solidity contracts
======================================================================================

This library provides utilities to export trained sklearn Random Forest models 
(both classification and regression) to optimized Solidity smart contracts.

Features:
- Export RandomForestClassifier models to Solidity contracts
- Export RandomForestRegressor models to Solidity contracts
- Optimized assembly code for efficient on-chain execution
- Support for feature names and custom contract naming
- Fixed-point number handling for decimal values
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class ML2Solidity:
    """Main class for converting machine learning models to Solidity contracts."""
    
    def __init__(self, model, feature_names=None, contract_name=None, precision=6):
        """
        Initialize the ML2Solidity converter.
        
        Args:
            model: A trained sklearn model (currently supports RandomForestClassifier and RandomForestRegressor)
            feature_names: List of feature names (optional)
            contract_name: Name for the generated Solidity contract (optional)
            precision: Number of decimal places for fixed-point representation (default=6)
        """
        self.model = model
        self.feature_names = feature_names
        self.precision = precision
        self.scaling_factor = 10 ** precision
        
        # Determine model type and set appropriate contract name if not provided
        if isinstance(model, RandomForestClassifier):
            self.model_type = "classifier"
            self.contract_name = contract_name or "RandomForestClassifier"
        elif isinstance(model, RandomForestRegressor):
            self.model_type = "regressor"
            self.contract_name = contract_name or "RandomForestRegressor"
        else:
            raise ValueError("Unsupported model type. Currently only supports RandomForestClassifier and RandomForestRegressor.")
        
        # If feature names not provided, use generic names
        if self.feature_names is None and hasattr(model, 'n_features_in_'):
            self.feature_names = [f'feature_{i}' for i in range(model.n_features_in_)]
    
    def export(self, output_file=None):
        """
        Export the model to Solidity code.
        
        Args:
            output_file: Path to save the generated Solidity code (optional)
            
        Returns:
            str: The generated Solidity code
        """
        if self.model_type == "classifier":
            solidity_code = self._export_classifier()
        else:  # regressor
            solidity_code = self._export_regressor()
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(solidity_code)
                
        return solidity_code
    
    def _export_classifier(self):
        """Export RandomForestClassifier to Solidity."""
        return export_forest_to_solidity(
            self.model, 
            feature_names=self.feature_names, 
            contract_name=self.contract_name,
            precision=self.precision
        )
    
    def _export_regressor(self):
        """Export RandomForestRegressor to Solidity."""
        return export_regression_forest_to_solidity(
            self.model, 
            feature_names=self.feature_names, 
            contract_name=self.contract_name,
            precision=self.precision
        )


def export_tree_to_solidity_assembly(tree, feature_names=None, tree_index=0, precision=6):
    """
    Convert a single decision tree classifier to a Solidity function using assembly.
    
    Args:
        tree: A trained decision tree classifier
        feature_names: List of feature names
        tree_index: Index of this tree in the forest
        precision: Number of decimal places for fixed-point representation
        
    Returns:
        str: Solidity code for the tree function
    """
    tree_ = tree.tree_
    scaling_factor = 10 ** precision
    
    # If feature names not provided, use generic names
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(tree_.n_features)]
    
    # Track all variables we'll need to declare
    var_declarations = set([
        "featureValue", "thresholdValue", "isFeatureNegative", 
        "absThreshold", "absFeature", "ltThreshold"
    ])
    
    # Start building the function
    solidity_code = []
    solidity_code.append(f"    /**")
    solidity_code.append(f"     * @dev Tree {tree_index} prediction function")
    solidity_code.append(f"     * @param features Array of feature values with the following indices:")
    
    # Document each feature and its index
    for i, name in enumerate(feature_names):
        solidity_code.append(f"     *   - features[{i}]: {name}")
    
    solidity_code.append(f"     * @return Predicted class")
    solidity_code.append(f"     */")
    solidity_code.append(f"    function tree{tree_index}_predict(int256[] memory features) private pure returns (int256) {{")
    solidity_code.append("        int256 result;")
    solidity_code.append("        assembly {")
    
    # Add variable declarations at the beginning
    solidity_code.append("            // Variable declarations")
    for var in var_declarations:
        solidity_code.append(f"            let {var} := 0")
    
    # Function to recursively build the assembly code for a node and its children
    def recurse(node, depth, code):
        indent = "            " + "    " * depth
        
        if tree_.feature[node] != -2:  # Internal node
            feature_idx = tree_.feature[node]
            threshold = tree_.threshold[node]
            feature_name = feature_names[feature_idx]
            
            # Convert threshold to fixed point
            threshold_fixed = int(abs(threshold) * scaling_factor)
            is_negative = threshold < 0
            
            # Compare feature value with threshold, handling negative values
            code.append(f"{indent}// Check if {feature_name} (features[{feature_idx}]) <= {threshold}")
            code.append(f"{indent}featureValue := mload(add(features, mul({feature_idx}, 0x20)))")
            
            # Handle comparison differently based on if threshold is negative
            if is_negative:
                # For negative thresholds: x <= threshold means x is negative or x's absolute value >= threshold's absolute value
                code.append(f"{indent}// Threshold is negative: {threshold}")
                code.append(f"{indent}isFeatureNegative := slt(featureValue, 0)")
                code.append(f"{indent}absThreshold := {threshold_fixed} // Absolute value of threshold * 10^{precision}")
                code.append(f"{indent}absFeature := featureValue")
                code.append(f"{indent}if isFeatureNegative {{ absFeature := sub(0, featureValue) }}")
                code.append(f"{indent}ltThreshold := or(isFeatureNegative, lt(absFeature, absThreshold))")
                code.append(f"{indent}switch ltThreshold")
            else:
                # For positive thresholds: simple comparison
                code.append(f"{indent}// Threshold is positive: {threshold}")
                code.append(f"{indent}thresholdValue := {threshold_fixed} // Threshold * 10^{precision}")
                code.append(f"{indent}switch slt(featureValue, thresholdValue)")
            
            code.append(f"{indent}case 1 {{")
            
            # Left branch (true)
            recurse(tree_.children_left[node], depth + 1, code)
            
            code.append(f"{indent}}}")
            code.append(f"{indent}case 0 {{")
            
            # Right branch (false)
            recurse(tree_.children_right[node], depth + 1, code)
            
            code.append(f"{indent}}}")
            
        else:  # Leaf node
            class_values = tree_.value[node][0]
            predicted_class = np.argmax(class_values)
            code.append(f"{indent}// Leaf node - return class {predicted_class}")
            code.append(f"{indent}result := {predicted_class}")
    
    # Build the tree recursively
    asm_code = []
    recurse(0, 0, asm_code)
    solidity_code.extend(asm_code)
    
    solidity_code.append("        }")
    solidity_code.append("        return result;")
    solidity_code.append("    }")
    
    return "\n".join(solidity_code)


def export_regression_tree_to_solidity_assembly(tree, feature_names=None, tree_index=0, precision=6):
    """
    Convert a single regression tree to a Solidity function using assembly.
    
    Args:
        tree: A trained decision tree regressor
        feature_names: List of feature names
        tree_index: Index of this tree in the forest
        precision: Number of decimal places for fixed-point representation
        
    Returns:
        str: Solidity code for the tree function
    """
    tree_ = tree.tree_
    scaling_factor = 10 ** precision
    
    # If feature names not provided, use generic names
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(tree_.n_features)]
    
    # Track all variables we'll need to declare
    var_declarations = set([
        "featureValue", "thresholdValue", "isFeatureNegative", 
        "absThreshold", "absFeature", "ltThreshold"
    ])
    
    # Start building the function
    solidity_code = []
    solidity_code.append(f"    /**")
    solidity_code.append(f"     * @dev Tree {tree_index} regression function")
    solidity_code.append(f"     * @param features Array of feature values with the following indices:")
    
    # Document each feature and its index
    for i, name in enumerate(feature_names):
        solidity_code.append(f"     *   - features[{i}]: {name}")
    
    solidity_code.append(f"     * @return Predicted value (scaled by 10^{precision})")
    solidity_code.append(f"     */")
    solidity_code.append(f"    function tree{tree_index}_predict(int256[] memory features) private pure returns (int256) {{")
    solidity_code.append("        int256 result;")
    solidity_code.append("        assembly {")
    
    # Add variable declarations at the beginning
    solidity_code.append("            // Variable declarations")
    for var in var_declarations:
        solidity_code.append(f"            let {var} := 0")
    
    # Function to recursively build the assembly code for a node and its children
    def recurse(node, depth, code):
        indent = "            " + "    " * depth
        
        if tree_.feature[node] != -2:  # Internal node
            feature_idx = tree_.feature[node]
            threshold = tree_.threshold[node]
            feature_name = feature_names[feature_idx]
            
            # Convert threshold to fixed point
            threshold_fixed = int(abs(threshold) * scaling_factor)
            is_negative = threshold < 0
            
            # Compare feature value with threshold, handling negative values
            code.append(f"{indent}// Check if {feature_name} (features[{feature_idx}]) <= {threshold}")
            code.append(f"{indent}featureValue := mload(add(features, mul({feature_idx}, 0x20)))")
            
            # Handle comparison differently based on if threshold is negative
            if is_negative:
                # For negative thresholds
                code.append(f"{indent}// Threshold is negative: {threshold}")
                code.append(f"{indent}isFeatureNegative := slt(featureValue, 0)")
                code.append(f"{indent}absThreshold := {threshold_fixed} // Absolute value of threshold * 10^{precision}")
                code.append(f"{indent}absFeature := featureValue")
                code.append(f"{indent}if isFeatureNegative {{ absFeature := sub(0, featureValue) }}")
                code.append(f"{indent}ltThreshold := or(isFeatureNegative, lt(absFeature, absThreshold))")
                code.append(f"{indent}switch ltThreshold")
            else:
                # For positive thresholds
                code.append(f"{indent}// Threshold is positive: {threshold}")
                code.append(f"{indent}thresholdValue := {threshold_fixed} // Threshold * 10^{precision}")
                code.append(f"{indent}switch slt(featureValue, thresholdValue)")
            
            code.append(f"{indent}case 1 {{")
            
            # Left branch (true)
            recurse(tree_.children_left[node], depth + 1, code)
            
            code.append(f"{indent}}}")
            code.append(f"{indent}case 0 {{")
            
            # Right branch (false)
            recurse(tree_.children_right[node], depth + 1, code)
            
            code.append(f"{indent}}}")
            
        else:  # Leaf node - For regression, return the predicted value
            value = float(tree_.value[node][0][0])  # Get the regression value
            # Convert the regression value to fixed point
            value_fixed = int(value * scaling_factor)
            
            # Handle negative values
            if value < 0:
                code.append(f"{indent}// Leaf node - return value {value} (negative)")
                code.append(f"{indent}result := sub(0, {abs(value_fixed)}) // Value * 10^{precision}")
            else:
                code.append(f"{indent}// Leaf node - return value {value}")
                code.append(f"{indent}result := {value_fixed} // Value * 10^{precision}")
    
    # Build the tree recursively
    asm_code = []
    recurse(0, 0, asm_code)
    solidity_code.extend(asm_code)
    
    solidity_code.append("        }")
    solidity_code.append("        return result;")
    solidity_code.append("    }")
    
    return "\n".join(solidity_code)


def export_forest_to_solidity(forest, feature_names=None, contract_name="RandomForestClassifier", precision=6):
    """
    Convert a random forest classifier to a Solidity contract.
    
    Args:
        forest: A trained RandomForestClassifier
        feature_names: List of feature names
        contract_name: Name for the Solidity contract
        precision: Number of decimal places for fixed-point representation
        
    Returns:
        str: Complete Solidity code for the forest contract
    """
    solidity_code = []
    
    # Add SPDX license and pragma
    solidity_code.append("// SPDX-License-Identifier: MIT")
    solidity_code.append("pragma solidity ^0.8.0;")
    solidity_code.append("")
    
    # Add comments explaining the fixed-point representation and features
    solidity_code.append("/**")
    solidity_code.append(" * @title Random Forest Classifier")
    solidity_code.append(" * @dev This contract implements a random forest classifier with assembly-optimized tree functions")
    solidity_code.append(f" * @notice All feature values must be provided as fixed-point integers with {precision} decimal places")
    solidity_code.append(f" *         For example, 1.5 should be represented as 1.5 * 10^{precision} = {int(1.5 * 10**precision)}")
    solidity_code.append(" *")
    solidity_code.append(" * Feature mapping:")
    
    if feature_names is not None:
        for i, name in enumerate(feature_names):
            solidity_code.append(f" * - features[{i}]: {name}")
    
    solidity_code.append(" */")
    
    # Start contract
    solidity_code.append(f"contract {contract_name} {{")
    
    # Export each tree to Solidity with assembly
    for i, tree in enumerate(forest.estimators_):
        solidity_code.append("")
        solidity_code.append(export_tree_to_solidity_assembly(tree, feature_names, i, precision))
    
    # Create a prediction function without assembly
    solidity_code.append("")
    solidity_code.append("    mapping(int256 => uint256) classCounts;")
    solidity_code.append("    /**")
    solidity_code.append("     * @dev Predict class using majority voting across all trees")
    solidity_code.append(f"     * @param features Array of feature values (fixed point with {precision} decimals) with the following indices:")
    
    if feature_names is not None:
        for i, name in enumerate(feature_names):
            solidity_code.append(f"     *   - features[{i}]: {name}")
    
    solidity_code.append("     * @return Predicted class")
    solidity_code.append("     */")
    solidity_code.append("    function forest_predict(int256[] memory features) public returns (int256) {")
    solidity_code.append("        // Get predictions from all trees")
    solidity_code.append("        int256[] memory predictions = new int256[](" + str(len(forest.estimators_)) + ");")
    
    # Collect predictions from all trees
    for i in range(len(forest.estimators_)):
        solidity_code.append(f"        predictions[{i}] = tree{i}_predict(features);")
    
    # Majority voting for classification
    solidity_code.append("")
    solidity_code.append("        // Count votes for each class")
    solidity_code.append("        uint256 maxCount = 0;")
    solidity_code.append("        int256 maxClass = 0;")
    solidity_code.append("")
    solidity_code.append("        // Reset counts for classes before counting votes")
    solidity_code.append("        for (uint256 i = 0; i < predictions.length; i++) {")
    solidity_code.append("            classCounts[predictions[i]] = 0;")
    solidity_code.append("        }")
    solidity_code.append("")
    solidity_code.append("        // Find the class with the most votes")
    solidity_code.append("        for (uint256 i = 0; i < predictions.length; i++) {")
    solidity_code.append("            int256 pred = predictions[i];")
    solidity_code.append("            classCounts[pred] += 1;")
    solidity_code.append("")
    solidity_code.append("            if (classCounts[pred] > maxCount) {")
    solidity_code.append("                maxCount = classCounts[pred];")
    solidity_code.append("                maxClass = pred;")
    solidity_code.append("            }")
    solidity_code.append("        }")
    
    solidity_code.append("")
    solidity_code.append("        return maxClass;")
    solidity_code.append("    }")
    
    # Close contract
    solidity_code.append("}")
    
    return "\n".join(solidity_code)


def export_regression_forest_to_solidity(forest, feature_names=None, contract_name="RandomForestRegressor", precision=6):
    """
    Convert a random forest regressor to a Solidity contract.
    
    Args:
        forest: A trained RandomForestRegressor
        feature_names: List of feature names
        contract_name: Name for the Solidity contract
        precision: Number of decimal places for fixed-point representation
        
    Returns:
        str: Complete Solidity code for the forest contract
    """
    solidity_code = []
    
    # Add SPDX license and pragma
    solidity_code.append("// SPDX-License-Identifier: MIT")
    solidity_code.append("pragma solidity ^0.8.0;")
    solidity_code.append("")
    
    # Add comments explaining the fixed-point representation and features
    solidity_code.append("/**")
    solidity_code.append(" * @title Random Forest Regressor")
    solidity_code.append(" * @dev This contract implements a random forest regressor with assembly-optimized tree functions")
    solidity_code.append(f" * @notice All feature values must be provided as fixed-point integers with {precision} decimal places")
    solidity_code.append(f" *         For example, 1.5 should be represented as 1.5 * 10^{precision} = {int(1.5 * 10**precision)}")
    solidity_code.append(f" * @notice The prediction result is also a fixed-point integer with {precision} decimal places")
    solidity_code.append(" *")
    solidity_code.append(" * Feature mapping:")
    
    if feature_names is not None:
        for i, name in enumerate(feature_names):
            solidity_code.append(f" * - features[{i}]: {name}")
    
    solidity_code.append(" */")
    
    # Start contract
    solidity_code.append(f"contract {contract_name} {{")
    
    # Export each tree to Solidity with assembly
    for i, tree in enumerate(forest.estimators_):
        solidity_code.append("")
        solidity_code.append(export_regression_tree_to_solidity_assembly(tree, feature_names, i, precision))
    
    # Create a prediction function for regression
    solidity_code.append("")
    solidity_code.append("    /**")
    solidity_code.append("     * @dev Predict value by averaging predictions from all trees")
    solidity_code.append(f"     * @param features Array of feature values (fixed point with {precision} decimals) with the following indices:")
    
    if feature_names is not None:
        for i, name in enumerate(feature_names):
            solidity_code.append(f"     *   - features[{i}]: {name}")
    
    solidity_code.append(f"     * @return Predicted value (fixed point with {precision} decimals)")
    solidity_code.append("     */")
    solidity_code.append("    function forest_predict(int256[] memory features) public pure returns (int256) {")
    
    # Get the number of trees
    num_trees = len(forest.estimators_)
    
    # Generate tree prediction calls
    solidity_code.append("        // Get predictions from all trees")
    solidity_code.append(f"        int256[] memory predictions = new int256[]({num_trees});")
    
    for i in range(num_trees):
        solidity_code.append(f"        predictions[{i}] = tree{i}_predict(features);")
    
    # For regression, we average the predictions
    solidity_code.append("")
    solidity_code.append("        // Average all tree predictions")
    solidity_code.append("        int256 sum = 0;")
    solidity_code.append("        for (uint256 i = 0; i < predictions.length; i++) {")
    solidity_code.append("            sum += predictions[i];")
    solidity_code.append("        }")
    
    # Calculate average - for exact division in Solidity
    solidity_code.append("")
    solidity_code.append(f"        // Divide by number of trees ({num_trees})")
    solidity_code.append(f"        int256 result = sum / {num_trees};")
    solidity_code.append("")
    solidity_code.append("        return result;")
    solidity_code.append("    }")
    
    # Close contract
    solidity_code.append("}")
    
    return "\n".join(solidity_code)