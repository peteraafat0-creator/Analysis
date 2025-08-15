"""
Helper functions and utilities
"""
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Any, Tuple
import io
import base64
from datetime import datetime, timedelta
import json


def download_dataframe_as_csv(df: pd.DataFrame, filename: str = "data.csv") -> str:
    """Generate download link for dataframe as CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
    return href


def download_dataframe_as_excel(df: pd.DataFrame, filename: str = "data.xlsx") -> str:
    """Generate download link for dataframe as Excel"""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
    
    excel_data = output.getvalue()
    b64 = base64.b64encode(excel_data).decode()
    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Download Excel</a>'
    return href


def format_number(value: float, format_type: str = "auto") -> str:
    """Format numbers for display"""
    if pd.isna(value):
        return "N/A"
    
    if format_type == "currency":
        return f"${value:,.2f}"
    elif format_type == "percentage":
        return f"{value:.1%}"
    elif format_type == "integer":
        return f"{int(value):,}"
    elif format_type == "decimal":
        return f"{value:.2f}"
    else:  # auto
        if abs(value) >= 1000000:
            return f"{value/1000000:.1f}M"
        elif abs(value) >= 1000:
            return f"{value/1000:.1f}K"
        elif value == int(value):
            return f"{int(value):,}"
        else:
            return f"{value:.2f}"


def calculate_kpis(df: pd.DataFrame, kpi_config: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate KPIs based on configuration"""
    kpis = {}
    
    for kpi_name, config in kpi_config.items():
        try:
            if config['type'] == 'count':
                if 'filter' in config:
                    filtered_df = apply_filter(df, config['filter'])
                    value = len(filtered_df)
                else:
                    value = len(df)
            
            elif config['type'] == 'sum':
                column = config['column']
                if column in df.columns:
                    if 'filter' in config:
                        filtered_df = apply_filter(df, config['filter'])
                        value = filtered_df[column].sum()
                    else:
                        value = df[column].sum()
                else:
                    value = 0
            
            elif config['type'] == 'mean':
                column = config['column']
                if column in df.columns:
                    if 'filter' in config:
                        filtered_df = apply_filter(df, config['filter'])
                        value = filtered_df[column].mean()
                    else:
                        value = df[column].mean()
                else:
                    value = 0
            
            elif config['type'] == 'unique':
                column = config['column']
                if column in df.columns:
                    if 'filter' in config:
                        filtered_df = apply_filter(df, config['filter'])
                        value = filtered_df[column].nunique()
                    else:
                        value = df[column].nunique()
                else:
                    value = 0
            
            elif config['type'] == 'percentage':
                numerator_filter = config['numerator_filter']
                denominator_filter = config.get('denominator_filter', {})
                
                numerator_df = apply_filter(df, numerator_filter)
                denominator_df = apply_filter(df, denominator_filter) if denominator_filter else df
                
                value = len(numerator_df) / len(denominator_df) * 100 if len(denominator_df) > 0 else 0
            
            else:
                value = 0
            
            # Format the value
            format_type = config.get('format', 'auto')
            formatted_value = format_number(value, format_type)
            
            kpis[kpi_name] = {
                'value': formatted_value,
                'raw_value': value,
                'delta': config.get('delta', None)
            }
            
        except Exception as e:
            st.error(f"Error calculating KPI {kpi_name}: {str(e)}")
            kpis[kpi_name] = {'value': 'Error', 'raw_value': 0}
    
    return kpis


def apply_filter(df: pd.DataFrame, filter_config: Dict[str, Any]) -> pd.DataFrame:
    """Apply filter configuration to dataframe"""
    filtered_df = df.copy()
    
    for column, condition in filter_config.items():
        if column not in df.columns:
            continue
        
        if isinstance(condition, dict):
            if 'equals' in condition:
                filtered_df = filtered_df[filtered_df[column] == condition['equals']]
            elif 'in' in condition:
                filtered_df = filtered_df[filtered_df[column].isin(condition['in'])]
            elif 'greater_than' in condition:
                filtered_df = filtered_df[filtered_df[column] > condition['greater_than']]
            elif 'less_than' in condition:
                filtered_df = filtered_df[filtered_df[column] < condition['less_than']]
            elif 'between' in condition:
                min_val, max_val = condition['between']
                filtered_df = filtered_df[
                    (filtered_df[column] >= min_val) & (filtered_df[column] <= max_val)
                ]
        else:
            # Simple equality filter
            filtered_df = filtered_df[filtered_df[column] == condition]
    
    return filtered_df


def detect_data_types(df: pd.DataFrame) -> Dict[str, str]:
    """Detect and suggest data types for columns"""
    suggestions = {}
    
    for column in df.columns:
        col_data = df[column].dropna()
        
        if col_data.empty:
            suggestions[column] = 'unknown'
            continue
        
        # Check if it's already a datetime
        if pd.api.types.is_datetime64_any_dtype(col_data):
            suggestions[column] = 'datetime'
            continue
        
        # Check if it's numeric
        if pd.api.types.is_numeric_dtype(col_data):
            if col_data.dtype == 'int64':
                suggestions[column] = 'integer'
            else:
                suggestions[column] = 'float'
            continue
        
        # Check if it looks like a date
        sample_values = col_data.head(10).astype(str)
        if any(is_date_like(val) for val in sample_values):
            suggestions[column] = 'datetime'
            continue
        
        # Check if it's categorical (limited unique values)
        unique_ratio = col_data.nunique() / len(col_data)
        if unique_ratio < 0.1 and col_data.nunique() < 50:
            suggestions[column] = 'category'
        else:
            suggestions[column] = 'text'
    
    return suggestions


def is_date_like(value: str) -> bool:
    """Check if a string looks like a date"""
    import re
    
    date_patterns = [
        r'\d{4}-\d{2}-\d{2}',
        r'\d{2}/\d{2}/\d{4}',
        r'\d{2}-\d{2}-\d{4}',
        r'\d{4}/\d{2}/\d{2}',
        r'\d{1,2}/\d{1,2}/\d{4}',
        r'\d{1,2}-\d{1,2}-\d{4}'
    ]
    
    return any(re.match(pattern, str(value)) for pattern in date_patterns)


def generate_data_profile(df: pd.DataFrame) -> Dict[str, Any]:
    """Generate comprehensive data profile"""
    profile = {
        'basic_info': {
            'rows': len(df),
            'columns': len(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'missing_cells': df.isnull().sum().sum(),
            'missing_percentage': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
        },
        'column_info': {},
        'data_quality': {
            'duplicate_rows': df.duplicated().sum(),
            'empty_rows': (df.isnull().all(axis=1)).sum(),
            'empty_columns': (df.isnull().all(axis=0)).sum()
        }
    }
    
    # Column-level analysis
    for column in df.columns:
        col_data = df[column]
        col_info = {
            'dtype': str(col_data.dtype),
            'non_null_count': col_data.count(),
            'null_count': col_data.isnull().sum(),
            'null_percentage': (col_data.isnull().sum() / len(df)) * 100,
            'unique_count': col_data.nunique(),
            'unique_percentage': (col_data.nunique() / len(df)) * 100
        }
        
        if pd.api.types.is_numeric_dtype(col_data):
            col_info.update({
                'min': col_data.min(),
                'max': col_data.max(),
                'mean': col_data.mean(),
                'median': col_data.median(),
                'std': col_data.std(),
                'zeros': (col_data == 0).sum()
            })
        
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            col_info.update({
                'min_date': col_data.min(),
                'max_date': col_data.max(),
                'date_range_days': (col_data.max() - col_data.min()).days if col_data.min() and col_data.max() else 0
            })
        
        else:  # Object/categorical
            value_counts = col_data.value_counts()
            col_info.update({
                'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                'most_frequent_count': value_counts.iloc[0] if len(value_counts) > 0 else 0,
                'least_frequent': value_counts.index[-1] if len(value_counts) > 0 else None,
                'least_frequent_count': value_counts.iloc[-1] if len(value_counts) > 0 else 0
            })
        
        profile['column_info'][column] = col_info
    
    return profile


def create_data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Create a data quality report"""
    quality_metrics = []
    
    for column in df.columns:
        col_data = df[column]
        
        metrics = {
            'Column': column,
            'Data Type': str(col_data.dtype),
            'Total Records': len(df),
            'Non-Null Records': col_data.count(),
            'Null Records': col_data.isnull().sum(),
            'Null Percentage': f"{(col_data.isnull().sum() / len(df)) * 100:.1f}%",
            'Unique Values': col_data.nunique(),
            'Unique Percentage': f"{(col_data.nunique() / len(df)) * 100:.1f}%"
        }
        
        # Add type-specific metrics
        if pd.api.types.is_numeric_dtype(col_data):
            metrics.update({
                'Min Value': col_data.min(),
                'Max Value': col_data.max(),
                'Mean': f"{col_data.mean():.2f}" if not pd.isna(col_data.mean()) else "N/A",
                'Zeros': (col_data == 0).sum()
            })
        
        quality_metrics.append(metrics)
    
    return pd.DataFrame(quality_metrics)


def export_analysis_report(data: Dict[str, Any], filename: str = "analysis_report.json") -> str:
    """Export analysis results to JSON"""
    # Convert non-serializable objects
    serializable_data = {}
    
    for key, value in data.items():
        if isinstance(value, pd.DataFrame):
            serializable_data[key] = value.to_dict('records')
        elif isinstance(value, (np.integer, np.floating)):
            serializable_data[key] = float(value)
        elif isinstance(value, np.ndarray):
            serializable_data[key] = value.tolist()
        elif isinstance(value, (datetime, pd.Timestamp)):
            serializable_data[key] = value.isoformat()
        else:
            try:
                json.dumps(value)  # Test if serializable
                serializable_data[key] = value
            except (TypeError, ValueError):
                serializable_data[key] = str(value)
    
    # Add metadata
    serializable_data['_metadata'] = {
        'export_timestamp': datetime.now().isoformat(),
        'version': '1.0'
    }
    
    json_str = json.dumps(serializable_data, indent=2, ensure_ascii=False)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:application/json;base64,{b64}" download="{filename}">Download Analysis Report</a>'
    return href


def validate_uploaded_file(uploaded_file, max_size_mb: int = 200) -> Tuple[bool, str]:
    """Validate uploaded file"""
    if uploaded_file is None:
        return False, "No file uploaded"
    
    # Check file size
    file_size_mb = uploaded_file.size / (1024 * 1024)
    if file_size_mb > max_size_mb:
        return False, f"File size ({file_size_mb:.1f}MB) exceeds maximum allowed size ({max_size_mb}MB)"
    
    # Check file extension
    allowed_extensions = ['csv', 'xlsx', 'xls']
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension not in allowed_extensions:
        return False, f"File type '{file_extension}' not supported. Allowed types: {', '.join(allowed_extensions)}"
    
    return True, "File is valid"


def create_sample_data_config() -> Dict[str, Any]:
    """Create sample configuration for KPIs and charts"""
    return {
        'kpis': {
            'Total Records': {
                'type': 'count',
                'format': 'integer'
            },
            'Open Items': {
                'type': 'count',
                'filter': {'status': 'Open'},
                'format': 'integer'
            },
            'Completion Rate': {
                'type': 'percentage',
                'numerator_filter': {'status': 'Closed'},
                'format': 'percentage'
            },
            'Average Score': {
                'type': 'mean',
                'column': 'score',
                'format': 'decimal'
            }
        },
        'charts': {
            'status_distribution': {
                'type': 'pie',
                'data': 'status',
                'title': 'Status Distribution'
            },
            'trend_over_time': {
                'type': 'line',
                'x': 'date',
                'y': 'count',
                'title': 'Trend Over Time'
            },
            'category_breakdown': {
                'type': 'bar',
                'x': 'category',
                'y': 'count',
                'title': 'Category Breakdown'
            }
        }
    }