"""
Utility validators for data schema validation
"""

from typing import Dict, List, Any, Union
import re


def validate_data_schema(data: Any, schema: Dict[str, Any]) -> List[str]:
    """
    Validate data against a schema definition.
    
    Args:
        data: Data to validate
        schema: Schema definition
        
    Returns:
        List of validation error messages
    """
    errors = []
    
    try:
        if not schema:
            return errors
        
        # Handle DataFrame
        if hasattr(data, 'columns'):
            return _validate_dataframe_schema(data, schema)
        
        # Handle list/array data
        elif isinstance(data, (list, tuple)) and len(data) > 0:
            return _validate_array_schema(data, schema)
        
        # Handle dict data
        elif isinstance(data, dict):
            return _validate_dict_schema(data, schema)
        
        return errors
        
    except Exception as e:
        errors.append(f"Schema validation error: {str(e)}")
        return errors


def _validate_dataframe_schema(data, schema: Dict[str, Any]) -> List[str]:
    """Validate DataFrame against schema."""
    errors = []
    
    # Check required columns
    for column_name, column_spec in schema.items():
        if column_name not in data.columns:
            errors.append(f"Missing required column: {column_name}")
            continue
        
        # Validate column data
        column_errors = _validate_column_data(data[column_name], column_spec, column_name)
        errors.extend(column_errors)
    
    return errors


def _validate_array_schema(data: List, schema: Dict[str, Any]) -> List[str]:
    """Validate array data against schema."""
    errors = []
    
    # For array data, assume schema applies to each item
    if isinstance(data[0], dict):
        # Array of dictionaries
        for i, item in enumerate(data[:10]):  # Check first 10 items
            item_errors = _validate_dict_schema(item, schema)
            for error in item_errors:
                errors.append(f"Item {i}: {error}")
    
    return errors


def _validate_dict_schema(data: Dict, schema: Dict[str, Any]) -> List[str]:
    """Validate dictionary against schema."""
    errors = []
    
    # Check required fields
    for field_name, field_spec in schema.items():
        if field_name not in data:
            errors.append(f"Missing required field: {field_name}")
            continue
        
        # Validate field value
        field_errors = _validate_field_value(data[field_name], field_spec, field_name)
        errors.extend(field_errors)
    
    return errors


def _validate_column_data(column_data, column_spec, column_name: str) -> List[str]:
    """Validate column data against specification."""
    errors = []
    
    try:
        if isinstance(column_spec, str):
            # Simple type specification
            errors.extend(_validate_simple_type(column_data, column_spec, column_name))
        elif isinstance(column_spec, dict):
            # Detailed specification
            errors.extend(_validate_detailed_spec(column_data, column_spec, column_name))
    
    except Exception as e:
        errors.append(f"Column {column_name} validation error: {str(e)}")
    
    return errors


def _validate_field_value(value: Any, field_spec, field_name: str) -> List[str]:
    """Validate field value against specification."""
    errors = []
    
    try:
        if isinstance(field_spec, str):
            # Simple type specification  
            if not _check_simple_type(value, field_spec):
                errors.append(f"Field {field_name} type mismatch: expected {field_spec}")
        elif isinstance(field_spec, dict):
            # Detailed specification
            errors.extend(_validate_detailed_field_spec(value, field_spec, field_name))
    
    except Exception as e:
        errors.append(f"Field {field_name} validation error: {str(e)}")
    
    return errors


def _validate_simple_type(column_data, type_spec: str, column_name: str) -> List[str]:
    """Validate column against simple type specification."""
    errors = []
    
    # Extract type and constraints from specification
    if '[' in type_spec and ']' in type_spec:
        base_type = type_spec.split('[')[0]
        constraints = type_spec.split('[')[1].rstrip(']')
    else:
        base_type = type_spec
        constraints = None
    
    # Check base type
    if base_type == 'integer':
        if not _is_integer_column(column_data):
            errors.append(f"Column {column_name} is not integer type")
    elif base_type == 'float':
        if not _is_numeric_column(column_data):
            errors.append(f"Column {column_name} is not numeric type")
    elif base_type == 'string' or base_type == 'text':
        if not _is_string_column(column_data):
            errors.append(f"Column {column_name} is not string type")
    elif base_type == 'categorical':
        if not _is_categorical_column(column_data):
            errors.append(f"Column {column_name} is not categorical type")
    elif base_type == 'boolean':
        if not _is_boolean_column(column_data):
            errors.append(f"Column {column_name} is not boolean type")
    elif base_type == 'datetime':
        if not _is_datetime_column(column_data):
            errors.append(f"Column {column_name} is not datetime type")
    
    # Check constraints
    if constraints:
        constraint_errors = _validate_constraints(column_data, constraints, column_name)
        errors.extend(constraint_errors)
    
    return errors


def _validate_detailed_spec(column_data, spec: Dict, column_name: str) -> List[str]:
    """Validate column against detailed specification."""
    errors = []
    
    # Check type
    if 'type' in spec:
        type_errors = _validate_simple_type(column_data, spec['type'], column_name)
        errors.extend(type_errors)
    
    # Check constraints
    if 'min' in spec or 'max' in spec:
        if _is_numeric_column(column_data):
            if 'min' in spec and column_data.min() < spec['min']:
                errors.append(f"Column {column_name} has values below minimum {spec['min']}")
            if 'max' in spec and column_data.max() > spec['max']:
                errors.append(f"Column {column_name} has values above maximum {spec['max']}")
    
    # Check allowed values
    if 'values' in spec:
        allowed_values = set(spec['values'])
        actual_values = set(column_data.unique())
        invalid_values = actual_values - allowed_values
        if invalid_values:
            errors.append(f"Column {column_name} has invalid values: {invalid_values}")
    
    return errors


def _validate_detailed_field_spec(value: Any, spec: Dict, field_name: str) -> List[str]:
    """Validate field value against detailed specification."""
    errors = []
    
    # Check type
    if 'type' in spec:
        if not _check_simple_type(value, spec['type']):
            errors.append(f"Field {field_name} type mismatch: expected {spec['type']}")
    
    # Check constraints
    if 'min' in spec and isinstance(value, (int, float)):
        if value < spec['min']:
            errors.append(f"Field {field_name} value {value} below minimum {spec['min']}")
    
    if 'max' in spec and isinstance(value, (int, float)):
        if value > spec['max']:
            errors.append(f"Field {field_name} value {value} above maximum {spec['max']}")
    
    if 'values' in spec and value not in spec['values']:
        errors.append(f"Field {field_name} value {value} not in allowed values {spec['values']}")
    
    return errors


def _check_simple_type(value: Any, type_spec: str) -> bool:
    """Check if value matches simple type specification."""
    base_type = type_spec.split('[')[0] if '[' in type_spec else type_spec
    
    if base_type == 'integer':
        return isinstance(value, int)
    elif base_type == 'float':
        return isinstance(value, (int, float))
    elif base_type in ['string', 'text']:
        return isinstance(value, str)
    elif base_type == 'boolean':
        return isinstance(value, bool)
    elif base_type == 'categorical':
        return isinstance(value, str)
    else:
        return True  # Unknown types pass


def _is_integer_column(column_data) -> bool:
    """Check if column contains integer data."""
    try:
        import pandas as pd
        return pd.api.types.is_integer_dtype(column_data)
    except:
        return all(isinstance(x, int) for x in column_data if x is not None)


def _is_numeric_column(column_data) -> bool:
    """Check if column contains numeric data."""
    try:
        import pandas as pd
        return pd.api.types.is_numeric_dtype(column_data)
    except:
        return all(isinstance(x, (int, float)) for x in column_data if x is not None)


def _is_string_column(column_data) -> bool:
    """Check if column contains string data."""
    try:
        import pandas as pd
        return pd.api.types.is_string_dtype(column_data) or pd.api.types.is_object_dtype(column_data)
    except:
        return all(isinstance(x, str) for x in column_data if x is not None)


def _is_categorical_column(column_data) -> bool:
    """Check if column contains categorical data."""
    try:
        import pandas as pd
        return pd.api.types.is_categorical_dtype(column_data) or len(column_data.unique()) <= 20
    except:
        unique_values = len(set(column_data))
        return unique_values <= 20


def _is_boolean_column(column_data) -> bool:
    """Check if column contains boolean data."""
    try:
        import pandas as pd
        return pd.api.types.is_bool_dtype(column_data)
    except:
        return all(isinstance(x, bool) for x in column_data if x is not None)


def _is_datetime_column(column_data) -> bool:
    """Check if column contains datetime data."""
    try:
        import pandas as pd
        return pd.api.types.is_datetime64_any_dtype(column_data)
    except:
        return False


def _validate_constraints(column_data, constraints: str, column_name: str) -> List[str]:
    """Validate column constraints."""
    errors = []
    
    try:
        # Parse range constraints like "1:100" or "0.0:1.0"
        if ':' in constraints:
            min_val, max_val = constraints.split(':')
            
            try:
                min_val = float(min_val)
                max_val = float(max_val)
                
                if _is_numeric_column(column_data):
                    if column_data.min() < min_val:
                        errors.append(f"Column {column_name} has values below {min_val}")
                    if column_data.max() > max_val:
                        errors.append(f"Column {column_name} has values above {max_val}")
            except ValueError:
                # Non-numeric constraints - treat as categorical values
                allowed_values = constraints.split(':')
                actual_values = set(column_data.unique())
                invalid_values = actual_values - set(allowed_values)
                if invalid_values:
                    errors.append(f"Column {column_name} has invalid values: {invalid_values}")
        
    except Exception as e:
        errors.append(f"Constraint validation error for {column_name}: {str(e)}")
    
    return errors