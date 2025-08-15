"""
Chart creation utilities for the dashboard
"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Any, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
from config.settings import DEFAULT_COLOR_PALETTE, CHART_HEIGHT, CHART_WIDTH


class ChartGenerator:
    """Generates various types of charts for the dashboard"""
    
    def __init__(self):
        self.color_palette = DEFAULT_COLOR_PALETTE
        self.default_height = CHART_HEIGHT
        self.default_width = CHART_WIDTH
    
    def create_kpi_cards(self, kpis: Dict[str, Any]) -> None:
        """Create KPI cards display"""
        cols = st.columns(len(kpis))
        
        for i, (label, value) in enumerate(kpis.items()):
            with cols[i]:
                if isinstance(value, dict):
                    st.metric(
                        label=label,
                        value=value.get('value', 0),
                        delta=value.get('delta', None)
                    )
                else:
                    st.metric(label=label, value=value)
    
    def create_bar_chart(self, df: pd.DataFrame, x: str, y: str, 
                        title: str = "", color: str = None, 
                        orientation: str = 'v') -> go.Figure:
        """Create interactive bar chart"""
        if orientation == 'h':
            fig = px.bar(df, x=y, y=x, title=title, color=color,
                        color_discrete_sequence=self.color_palette,
                        orientation='h')
        else:
            fig = px.bar(df, x=x, y=y, title=title, color=color,
                        color_discrete_sequence=self.color_palette)
        
        fig.update_layout(
            height=self.default_height,
            showlegend=True if color else False,
            hovermode='x unified'
        )
        
        return fig
    
    def create_line_chart(self, df: pd.DataFrame, x: str, y: str, 
                         title: str = "", color: str = None) -> go.Figure:
        """Create interactive line chart"""
        fig = px.line(df, x=x, y=y, title=title, color=color,
                     color_discrete_sequence=self.color_palette)
        
        fig.update_layout(
            height=self.default_height,
            showlegend=True if color else False,
            hovermode='x unified'
        )
        
        fig.update_traces(mode='lines+markers')
        
        return fig
    
    def create_pie_chart(self, df: pd.DataFrame, values: str, names: str,
                        title: str = "") -> go.Figure:
        """Create interactive pie chart"""
        fig = px.pie(df, values=values, names=names, title=title,
                    color_discrete_sequence=self.color_palette)
        
        fig.update_layout(height=self.default_height)
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        return fig
    
    def create_scatter_plot(self, df: pd.DataFrame, x: str, y: str,
                           title: str = "", color: str = None, 
                           size: str = None) -> go.Figure:
        """Create interactive scatter plot"""
        fig = px.scatter(df, x=x, y=y, title=title, color=color, size=size,
                        color_discrete_sequence=self.color_palette)
        
        fig.update_layout(
            height=self.default_height,
            showlegend=True if color else False
        )
        
        return fig
    
    def create_heatmap(self, df: pd.DataFrame, x: str, y: str, z: str,
                      title: str = "") -> go.Figure:
        """Create interactive heatmap"""
        # Pivot data for heatmap
        pivot_df = df.pivot_table(values=z, index=y, columns=x, aggfunc='mean')
        
        fig = px.imshow(pivot_df, title=title, aspect='auto',
                       color_continuous_scale='RdYlBu_r')
        
        fig.update_layout(height=self.default_height)
        
        return fig
    
    def create_correlation_heatmap(self, df: pd.DataFrame, 
                                  title: str = "Correlation Matrix") -> go.Figure:
        """Create correlation heatmap for numeric columns"""
        numeric_df = df.select_dtypes(include=[np.number])
        
        if numeric_df.empty:
            st.warning("No numeric columns found for correlation analysis")
            return go.Figure()
        
        corr_matrix = numeric_df.corr()
        
        fig = px.imshow(corr_matrix, 
                       title=title,
                       color_continuous_scale='RdBu_r',
                       aspect='auto',
                       text_auto=True)
        
        fig.update_layout(height=self.default_height)
        
        return fig
    
    def create_histogram(self, df: pd.DataFrame, x: str, title: str = "",
                        bins: int = 30, color: str = None) -> go.Figure:
        """Create interactive histogram"""
        fig = px.histogram(df, x=x, title=title, nbins=bins, color=color,
                          color_discrete_sequence=self.color_palette)
        
        fig.update_layout(
            height=self.default_height,
            showlegend=True if color else False
        )
        
        return fig
    
    def create_box_plot(self, df: pd.DataFrame, x: str, y: str,
                       title: str = "", color: str = None) -> go.Figure:
        """Create interactive box plot"""
        fig = px.box(df, x=x, y=y, title=title, color=color,
                    color_discrete_sequence=self.color_palette)
        
        fig.update_layout(
            height=self.default_height,
            showlegend=True if color else False
        )
        
        return fig
    
    def create_treemap(self, df: pd.DataFrame, path: List[str], values: str,
                      title: str = "") -> go.Figure:
        """Create interactive treemap"""
        fig = px.treemap(df, path=path, values=values, title=title,
                        color_discrete_sequence=self.color_palette)
        
        fig.update_layout(height=self.default_height)
        
        return fig
    
    def create_sunburst(self, df: pd.DataFrame, path: List[str], values: str,
                       title: str = "") -> go.Figure:
        """Create interactive sunburst chart"""
        fig = px.sunburst(df, path=path, values=values, title=title,
                         color_discrete_sequence=self.color_palette)
        
        fig.update_layout(height=self.default_height)
        
        return fig
    
    def create_gauge_chart(self, value: float, title: str = "",
                          min_val: float = 0, max_val: float = 100,
                          threshold_ranges: List[Tuple[float, float, str]] = None) -> go.Figure:
        """Create gauge chart for KPIs"""
        if threshold_ranges is None:
            threshold_ranges = [
                (0, 30, "red"),
                (30, 70, "yellow"),
                (70, 100, "green")
            ]
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            gauge={
                'axis': {'range': [None, max_val]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [r[0], r[1]], 'color': r[2]}
                    for r in threshold_ranges
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': max_val * 0.9
                }
            }
        ))
        
        fig.update_layout(height=self.default_height)
        
        return fig
    
    def create_time_series(self, df: pd.DataFrame, x: str, y: str,
                          title: str = "", color: str = None,
                          show_trend: bool = True) -> go.Figure:
        """Create time series chart with trend line"""
        fig = px.line(df, x=x, y=y, title=title, color=color,
                     color_discrete_sequence=self.color_palette)
        
        if show_trend and color is None:
            # Add trend line
            from sklearn.linear_model import LinearRegression
            from sklearn.preprocessing import LabelEncoder
            
            # Convert dates to numeric for trend calculation
            df_copy = df.copy()
            if pd.api.types.is_datetime64_any_dtype(df_copy[x]):
                df_copy[x + '_numeric'] = df_copy[x].astype(np.int64) // 10**9
                X = df_copy[x + '_numeric'].values.reshape(-1, 1)
            else:
                le = LabelEncoder()
                X = le.fit_transform(df_copy[x]).reshape(-1, 1)
            
            y_vals = df_copy[y].values
            
            # Fit trend line
            reg = LinearRegression().fit(X, y_vals)
            trend_y = reg.predict(X)
            
            fig.add_trace(go.Scatter(
                x=df[x],
                y=trend_y,
                mode='lines',
                name='Trend',
                line=dict(dash='dash', color='red')
            ))
        
        fig.update_layout(
            height=self.default_height,
            hovermode='x unified'
        )
        
        return fig
    
    def create_multi_axis_chart(self, df: pd.DataFrame, x: str, 
                               y1: str, y2: str, title: str = "") -> go.Figure:
        """Create chart with dual y-axes"""
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add first trace
        fig.add_trace(
            go.Scatter(x=df[x], y=df[y1], name=y1, line=dict(color=self.color_palette[0])),
            secondary_y=False,
        )
        
        # Add second trace
        fig.add_trace(
            go.Scatter(x=df[x], y=df[y2], name=y2, line=dict(color=self.color_palette[1])),
            secondary_y=True,
        )
        
        # Set y-axes titles
        fig.update_yaxes(title_text=y1, secondary_y=False)
        fig.update_yaxes(title_text=y2, secondary_y=True)
        
        fig.update_layout(
            title_text=title,
            height=self.default_height,
            hovermode='x unified'
        )
        
        return fig
    
    def create_waterfall_chart(self, categories: List[str], values: List[float],
                              title: str = "") -> go.Figure:
        """Create waterfall chart"""
        fig = go.Figure(go.Waterfall(
            name="",
            orientation="v",
            measure=["relative"] * (len(categories) - 1) + ["total"],
            x=categories,
            textposition="outside",
            text=[f"{v:+.0f}" for v in values],
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
            title=title,
            height=self.default_height,
            showlegend=False
        )
        
        return fig


class DashboardLayout:
    """Handles dashboard layout and organization"""
    
    def __init__(self):
        self.chart_generator = ChartGenerator()
    
    def create_kpi_section(self, df: pd.DataFrame, kpi_config: Dict[str, Any]) -> None:
        """Create KPI section based on configuration"""
        kpis = {}
        
        for kpi_name, config in kpi_config.items():
            if config['type'] == 'count':
                value = len(df)
            elif config['type'] == 'sum':
                value = df[config['column']].sum() if config['column'] in df.columns else 0
            elif config['type'] == 'mean':
                value = df[config['column']].mean() if config['column'] in df.columns else 0
            elif config['type'] == 'unique':
                value = df[config['column']].nunique() if config['column'] in df.columns else 0
            else:
                value = 0
            
            kpis[kpi_name] = {
                'value': f"{value:,.0f}" if isinstance(value, (int, float)) else str(value),
                'delta': config.get('delta', None)
            }
        
        self.chart_generator.create_kpi_cards(kpis)
    
    def create_filter_sidebar(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create interactive filter sidebar"""
        filters = {}
        
        st.sidebar.header("🔍 Filters")
        
        # Date range filter
        date_columns = df.select_dtypes(include=['datetime64']).columns
        if len(date_columns) > 0:
            date_col = st.sidebar.selectbox("Select Date Column", date_columns)
            if date_col:
                min_date = df[date_col].min().date()
                max_date = df[date_col].max().date()
                date_range = st.sidebar.date_input(
                    "Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                filters['date_range'] = {
                    'column': date_col,
                    'range': date_range
                }
        
        # Categorical filters
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns[:5]:  # Limit to first 5 categorical columns
            unique_values = df[col].unique()
            if len(unique_values) <= 20:  # Only show filter if reasonable number of options
                selected_values = st.sidebar.multiselect(
                    f"Filter by {col.title()}",
                    options=unique_values,
                    default=unique_values
                )
                filters[col] = selected_values
        
        # Numeric range filters
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns[:3]:  # Limit to first 3 numeric columns
            min_val = float(df[col].min())
            max_val = float(df[col].max())
            if min_val != max_val:
                range_values = st.sidebar.slider(
                    f"{col.title()} Range",
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val)
                )
                filters[f"{col}_range"] = {
                    'column': col,
                    'range': range_values
                }
        
        return filters
    
    def apply_filters(self, df: pd.DataFrame, filters: Dict[str, Any]) -> pd.DataFrame:
        """Apply filters to dataframe"""
        filtered_df = df.copy()
        
        for filter_name, filter_value in filters.items():
            if filter_name == 'date_range' and filter_value:
                date_col = filter_value['column']
                date_range = filter_value['range']
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    filtered_df = filtered_df[
                        (filtered_df[date_col].dt.date >= start_date) &
                        (filtered_df[date_col].dt.date <= end_date)
                    ]
            elif filter_name.endswith('_range') and filter_value:
                col = filter_value['column']
                range_vals = filter_value['range']
                filtered_df = filtered_df[
                    (filtered_df[col] >= range_vals[0]) &
                    (filtered_df[col] <= range_vals[1])
                ]
            elif isinstance(filter_value, list) and filter_value:
                filtered_df = filtered_df[filtered_df[filter_name].isin(filter_value)]
        
        return filtered_df