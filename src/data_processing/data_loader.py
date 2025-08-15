"""
Data loading and preprocessing utilities
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
import streamlit as st
from datetime import datetime
import re
import os
from config.settings import DATE_FORMATS, ALLOWED_FILE_TYPES


class DataLoader:
    """Handles data loading and initial preprocessing"""
    
    def __init__(self):
        self.data_cache = {}
        
    def load_file(self, file_path: str, file_type: str = None) -> pd.DataFrame:
        """Load data from various file formats"""
        try:
            if file_type is None:
                file_type = file_path.split('.')[-1].lower()
            
            if file_type == 'csv':
                # Try different encodings
                encodings = ['utf-8', 'latin-1', 'cp1256', 'iso-8859-1']
                for encoding in encodings:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError("Could not decode CSV file with any supported encoding")
                    
            elif file_type in ['xlsx', 'xls']:
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
            return df
            
        except Exception as e:
            st.error(f"Error loading file {file_path}: {str(e)}")
            return pd.DataFrame()
    
    def load_uploaded_file(self, uploaded_file) -> pd.DataFrame:
        """Load data from Streamlit uploaded file"""
        try:
            file_type = uploaded_file.name.split('.')[-1].lower()
            
            if file_type == 'csv':
                df = pd.read_csv(uploaded_file)
            elif file_type in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
                
            return df
            
        except Exception as e:
            st.error(f"Error loading uploaded file: {str(e)}")
            return pd.DataFrame()
    
    def load_multiple_files(self, file_paths: List[str]) -> Dict[str, pd.DataFrame]:
        """Load multiple data files"""
        datasets = {}
        
        for file_path in file_paths:
            file_name = os.path.basename(file_path).split('.')[0]
            df = self.load_file(file_path)
            if not df.empty:
                datasets[file_name] = df
                
        return datasets
    
    def get_sample_data(self) -> Dict[str, pd.DataFrame]:
        """Generate sample data for demonstration purposes"""
        np.random.seed(42)
        
        # Sample inspection data
        inspection_data = {
            'inspection_id': range(1, 101),
            'date': pd.date_range('2023-01-01', periods=100, freq='D'),
            'department': np.random.choice(['Safety', 'Quality', 'Operations', 'Maintenance'], 100),
            'inspector': np.random.choice(['John Doe', 'Jane Smith', 'Mike Johnson', 'Sarah Wilson'], 100),
            'category': np.random.choice(['Equipment', 'Process', 'Documentation', 'Training'], 100),
            'severity': np.random.choice(['Low', 'Medium', 'High', 'Critical'], 100),
            'status': np.random.choice(['Open', 'In Progress', 'Closed', 'Closed - Close'], 100),
            'description': [f'Inspection finding {i}' for i in range(1, 101)],
            'corrective_action': [f'Action required {i}' for i in range(1, 101)]
        }
        
        # Sample incident data
        incident_data = {
            'incident_id': range(1, 51),
            'date': pd.date_range('2023-01-01', periods=50, freq='2D'),
            'type': np.random.choice(['Near Miss', 'Accident', 'Property Damage', 'Environmental'], 50),
            'severity': np.random.choice(['Minor', 'Moderate', 'Major', 'Severe'], 50),
            'department': np.random.choice(['Safety', 'Quality', 'Operations', 'Maintenance'], 50),
            'cost': np.random.uniform(100, 10000, 50),
            'days_lost': np.random.randint(0, 30, 50),
            'status': np.random.choice(['Open', 'Under Investigation', 'Closed'], 50)
        }
        
        # Sample risk assessment data
        risk_data = {
            'risk_id': range(1, 76),
            'date': pd.date_range('2023-01-01', periods=75, freq='3D'),
            'risk_category': np.random.choice(['Operational', 'Financial', 'Strategic', 'Compliance'], 75),
            'probability': np.random.choice(['Very Low', 'Low', 'Medium', 'High', 'Very High'], 75),
            'impact': np.random.choice(['Negligible', 'Minor', 'Moderate', 'Major', 'Catastrophic'], 75),
            'risk_score': np.random.randint(1, 25, 75),
            'mitigation_status': np.random.choice(['Not Started', 'In Progress', 'Completed'], 75),
            'owner': np.random.choice(['Team A', 'Team B', 'Team C', 'Team D'], 75)
        }
        
        return {
            'inspections': pd.DataFrame(inspection_data),
            'incidents': pd.DataFrame(incident_data),
            'risks': pd.DataFrame(risk_data)
        }


class DataCleaner:
    """Handles data cleaning and standardization"""
    
    def __init__(self):
        self.cleaning_log = []
    
    def clean_dataset(self, df: pd.DataFrame, dataset_name: str = "") -> pd.DataFrame:
        """Comprehensive data cleaning pipeline"""
        original_shape = df.shape
        cleaned_df = df.copy()
        
        # Remove completely empty rows and columns
        cleaned_df = self._remove_empty_rows_columns(cleaned_df)
        
        # Standardize column names
        cleaned_df = self._standardize_column_names(cleaned_df)
        
        # Handle missing values
        cleaned_df = self._handle_missing_values(cleaned_df)
        
        # Standardize text data
        cleaned_df = self._standardize_text_data(cleaned_df)
        
        # Parse and standardize dates
        cleaned_df = self._standardize_dates(cleaned_df)
        
        # Remove duplicates
        cleaned_df = self._remove_duplicates(cleaned_df)
        
        # Log cleaning results
        final_shape = cleaned_df.shape
        self.cleaning_log.append({
            'dataset': dataset_name,
            'original_shape': original_shape,
            'final_shape': final_shape,
            'rows_removed': original_shape[0] - final_shape[0],
            'columns_removed': original_shape[1] - final_shape[1]
        })
        
        return cleaned_df
    
    def _remove_empty_rows_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove completely empty rows and columns"""
        # Remove empty columns
        df = df.dropna(axis=1, how='all')
        # Remove empty rows
        df = df.dropna(axis=0, how='all')
        return df
    
    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names"""
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('[^a-zA-Z0-9_]', '', regex=True)
        return df
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values based on column type"""
        for column in df.columns:
            if df[column].dtype == 'object':
                # For text columns, fill with 'Unknown' or most frequent value
                if df[column].isnull().sum() > 0:
                    mode_value = df[column].mode()
                    if len(mode_value) > 0:
                        df[column] = df[column].fillna(mode_value[0])
                    else:
                        df[column] = df[column].fillna('Unknown')
            elif df[column].dtype in ['int64', 'float64']:
                # For numeric columns, fill with median
                df[column] = df[column].fillna(df[column].median())
        
        return df
    
    def _standardize_text_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize text data"""
        text_columns = df.select_dtypes(include=['object']).columns
        
        for column in text_columns:
            # Strip whitespace
            df[column] = df[column].astype(str).str.strip()
            
            # Standardize common status values
            status_mapping = {
                'closed - close': 'closed',
                'in-progress': 'in progress',
                'inprogress': 'in progress',
                'not started': 'not_started',
                'not-started': 'not_started'
            }
            
            df[column] = df[column].str.lower().replace(status_mapping).str.title()
        
        return df
    
    def _standardize_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parse and standardize date columns"""
        potential_date_columns = []
        
        # Identify potential date columns
        for column in df.columns:
            if 'date' in column.lower() or 'time' in column.lower():
                potential_date_columns.append(column)
            elif df[column].dtype == 'object':
                # Check if column contains date-like strings
                sample_values = df[column].dropna().head(10).astype(str)
                if any(self._is_date_like(val) for val in sample_values):
                    potential_date_columns.append(column)
        
        # Parse date columns
        for column in potential_date_columns:
            df[column] = self._parse_dates(df[column])
        
        return df
    
    def _is_date_like(self, value: str) -> bool:
        """Check if a string looks like a date"""
        date_patterns = [
            r'\d{4}-\d{2}-\d{2}',
            r'\d{2}/\d{2}/\d{4}',
            r'\d{2}-\d{2}-\d{4}',
            r'\d{4}/\d{2}/\d{2}'
        ]
        
        return any(re.match(pattern, str(value)) for pattern in date_patterns)
    
    def _parse_dates(self, series: pd.Series) -> pd.Series:
        """Parse dates using multiple formats"""
        for date_format in DATE_FORMATS:
            try:
                return pd.to_datetime(series, format=date_format, errors='coerce')
            except:
                continue
        
        # If no format works, try pandas' automatic parsing
        try:
            return pd.to_datetime(series, errors='coerce')
        except:
            return series
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows"""
        return df.drop_duplicates()
    
    def get_cleaning_summary(self) -> pd.DataFrame:
        """Get summary of cleaning operations"""
        return pd.DataFrame(self.cleaning_log)


class DataIntegrator:
    """Handles data integration and relationship mapping"""
    
    def __init__(self):
        self.relationships = {}
    
    def integrate_datasets(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Integrate multiple datasets and establish relationships"""
        integrated_data = {}
        
        for name, df in datasets.items():
            # Add dataset identifier
            df['source_dataset'] = name
            integrated_data[name] = df
        
        # Identify potential relationships
        self._identify_relationships(integrated_data)
        
        return integrated_data
    
    def _identify_relationships(self, datasets: Dict[str, pd.DataFrame]):
        """Identify potential relationships between datasets"""
        dataset_names = list(datasets.keys())
        
        for i, name1 in enumerate(dataset_names):
            for name2 in dataset_names[i+1:]:
                df1, df2 = datasets[name1], datasets[name2]
                
                # Look for common columns
                common_columns = set(df1.columns) & set(df2.columns)
                common_columns.discard('source_dataset')
                
                if common_columns:
                    self.relationships[f"{name1}_{name2}"] = {
                        'common_columns': list(common_columns),
                        'potential_keys': self._find_potential_keys(df1, df2, common_columns)
                    }
    
    def _find_potential_keys(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                           common_columns: set) -> List[str]:
        """Find potential foreign keys between datasets"""
        potential_keys = []
        
        for column in common_columns:
            # Check if column values in df1 exist in df2
            if column in df1.columns and column in df2.columns:
                overlap = set(df1[column].unique()) & set(df2[column].unique())
                if len(overlap) > 0:
                    potential_keys.append(column)
        
        return potential_keys
    
    def create_master_dataset(self, datasets: Dict[str, pd.DataFrame], 
                            join_keys: Dict[str, str] = None) -> pd.DataFrame:
        """Create a master dataset by joining multiple datasets"""
        if not datasets:
            return pd.DataFrame()
        
        # Start with the first dataset
        master_df = list(datasets.values())[0].copy()
        dataset_names = list(datasets.keys())
        
        # Join with other datasets
        for i, (name, df) in enumerate(list(datasets.items())[1:], 1):
            if join_keys and name in join_keys:
                key = join_keys[name]
                if key in master_df.columns and key in df.columns:
                    master_df = master_df.merge(df, on=key, how='outer', suffixes=('', f'_{name}'))
            else:
                # Try to find common columns for joining
                common_cols = set(master_df.columns) & set(df.columns)
                common_cols.discard('source_dataset')
                
                if common_cols:
                    key = list(common_cols)[0]  # Use first common column
                    master_df = master_df.merge(df, on=key, how='outer', suffixes=('', f'_{name}'))
        
        return master_df