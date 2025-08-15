"""
Advanced Data Analysis Dashboard - Main Streamlit Application
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processing.data_loader import DataLoader, DataCleaner, DataIntegrator
from visualization.charts import ChartGenerator, DashboardLayout
from chatbot.gemini_bot import GeminiDataAnalyst
from utils.helpers import (
    download_dataframe_as_csv, download_dataframe_as_excel,
    calculate_kpis, generate_data_profile, create_data_quality_report,
    validate_uploaded_file, create_sample_data_config
)
from config.settings import APP_TITLE, APP_DESCRIPTION, MAX_FILE_SIZE_MB

# Page configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .kpi-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .chat-container {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)


class AdvancedDashboard:
    """Main dashboard application class"""
    
    def __init__(self):
        self.data_loader = DataLoader()
        self.data_cleaner = DataCleaner()
        self.data_integrator = DataIntegrator()
        self.chart_generator = ChartGenerator()
        self.dashboard_layout = DashboardLayout()
        self.gemini_analyst = GeminiDataAnalyst()
        
        # Initialize session state
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'datasets' not in st.session_state:
            st.session_state.datasets = {}
        if 'current_data' not in st.session_state:
            st.session_state.current_data = None
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        if 'filters' not in st.session_state:
            st.session_state.filters = {}
    
    def run(self):
        """Main application runner"""
        # Header
        st.markdown(f'<h1 class="main-header">{APP_TITLE}</h1>', unsafe_allow_html=True)
        st.markdown(f'<p style="text-align: center; color: #7f8c8d;">{APP_DESCRIPTION}</p>', unsafe_allow_html=True)
        
        # Sidebar navigation
        page = self._create_sidebar_navigation()
        
        # Main content based on selected page
        if page == "📊 Dashboard":
            self._render_dashboard_page()
        elif page == "📈 Advanced Analytics":
            self._render_analytics_page()
        elif page == "🤖 AI Assistant":
            self._render_chatbot_page()
        elif page == "📁 Data Management":
            self._render_data_management_page()
        elif page == "📋 Data Quality":
            self._render_data_quality_page()
    
    def _create_sidebar_navigation(self):
        """Create sidebar navigation"""
        st.sidebar.title("🧭 Navigation")
        
        pages = [
            "📊 Dashboard",
            "📈 Advanced Analytics", 
            "🤖 AI Assistant",
            "📁 Data Management",
            "📋 Data Quality"
        ]
        
        selected_page = st.sidebar.selectbox("Select Page", pages)
        
        # Data upload section
        st.sidebar.markdown("---")
        st.sidebar.header("📤 Data Upload")
        
        uploaded_files = st.sidebar.file_uploader(
            "Upload Data Files",
            type=['csv', 'xlsx', 'xls'],
            accept_multiple_files=True,
            help=f"Maximum file size: {MAX_FILE_SIZE_MB}MB"
        )
        
        if uploaded_files:
            self._handle_file_upload(uploaded_files)
        
        # Load sample data option
        if st.sidebar.button("📋 Load Sample Data"):
            self._load_sample_data()
        
        # Data status
        st.sidebar.markdown("---")
        st.sidebar.header("📊 Data Status")
        
        if st.session_state.data_loaded:
            st.sidebar.success(f"✅ {len(st.session_state.datasets)} dataset(s) loaded")
            for name, df in st.session_state.datasets.items():
                st.sidebar.info(f"📄 {name}: {df.shape[0]} rows, {df.shape[1]} columns")
        else:
            st.sidebar.warning("⚠️ No data loaded")
        
        return selected_page
    
    def _handle_file_upload(self, uploaded_files):
        """Handle uploaded files"""
        datasets = {}
        
        for uploaded_file in uploaded_files:
            # Validate file
            is_valid, message = validate_uploaded_file(uploaded_file, MAX_FILE_SIZE_MB)
            
            if not is_valid:
                st.sidebar.error(f"❌ {uploaded_file.name}: {message}")
                continue
            
            # Load file
            try:
                df = self.data_loader.load_uploaded_file(uploaded_file)
                if not df.empty:
                    # Clean data
                    cleaned_df = self.data_cleaner.clean_dataset(df, uploaded_file.name)
                    datasets[uploaded_file.name.split('.')[0]] = cleaned_df
                    st.sidebar.success(f"✅ {uploaded_file.name} loaded successfully")
                else:
                    st.sidebar.error(f"❌ Failed to load {uploaded_file.name}")
            
            except Exception as e:
                st.sidebar.error(f"❌ Error loading {uploaded_file.name}: {str(e)}")
        
        if datasets:
            # Integrate datasets
            integrated_data = self.data_integrator.integrate_datasets(datasets)
            st.session_state.datasets = integrated_data
            st.session_state.data_loaded = True
            
            # Set context for AI assistant
            self.gemini_analyst.set_data_context(integrated_data)
            
            st.sidebar.success(f"🎉 Successfully loaded {len(datasets)} dataset(s)")
    
    def _load_sample_data(self):
        """Load sample data for demonstration"""
        try:
            sample_datasets = self.data_loader.get_sample_data()
            
            # Clean sample data
            cleaned_datasets = {}
            for name, df in sample_datasets.items():
                cleaned_df = self.data_cleaner.clean_dataset(df, name)
                cleaned_datasets[name] = cleaned_df
            
            # Integrate datasets
            integrated_data = self.data_integrator.integrate_datasets(cleaned_datasets)
            st.session_state.datasets = integrated_data
            st.session_state.data_loaded = True
            
            # Set context for AI assistant
            self.gemini_analyst.set_data_context(integrated_data)
            
            st.sidebar.success("🎉 Sample data loaded successfully!")
            
        except Exception as e:
            st.sidebar.error(f"❌ Error loading sample data: {str(e)}")
    
    def _render_dashboard_page(self):
        """Render main dashboard page"""
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please upload data or load sample data to view the dashboard.")
            return
        
        # Create filters
        st.markdown('<h2 class="section-header">🔍 Filters</h2>', unsafe_allow_html=True)
        
        # Get first dataset for filtering
        first_dataset_name = list(st.session_state.datasets.keys())[0]
        first_dataset = st.session_state.datasets[first_dataset_name]
        
        filters = self.dashboard_layout.create_filter_sidebar(first_dataset)
        filtered_data = self.dashboard_layout.apply_filters(first_dataset, filters)
        
        # KPI Section
        st.markdown('<h2 class="section-header">📊 Key Performance Indicators</h2>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="kpi-container">', unsafe_allow_html=True)
            
            # Calculate KPIs
            kpi_config = {
                'Total Records': {'type': 'count', 'format': 'integer'},
                'Unique Categories': {'type': 'unique', 'column': list(filtered_data.select_dtypes(include=['object']).columns)[0] if len(filtered_data.select_dtypes(include=['object']).columns) > 0 else 'source_dataset', 'format': 'integer'},
                'Data Quality Score': {'type': 'percentage', 'numerator_filter': {}, 'format': 'percentage'},
                'Latest Update': {'type': 'count', 'format': 'integer'}
            }
            
            # Adjust KPI config based on available columns
            numeric_cols = filtered_data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                kpi_config['Average Value'] = {'type': 'mean', 'column': numeric_cols[0], 'format': 'decimal'}
            
            kpis = calculate_kpis(filtered_data, kpi_config)
            self.chart_generator.create_kpi_cards(kpis)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Charts Section
        st.markdown('<h2 class="section-header">📈 Visualizations</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Distribution chart
            categorical_cols = filtered_data.select_dtypes(include=['object']).columns
            if len(categorical_cols) > 0:
                col_for_dist = categorical_cols[0]
                dist_data = filtered_data[col_for_dist].value_counts().reset_index()
                dist_data.columns = [col_for_dist, 'count']
                
                fig_pie = self.chart_generator.create_pie_chart(
                    dist_data, 'count', col_for_dist,
                    title=f"Distribution of {col_for_dist.title()}"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Bar chart
            if len(categorical_cols) > 0:
                col_for_bar = categorical_cols[0]
                bar_data = filtered_data[col_for_bar].value_counts().reset_index()
                bar_data.columns = [col_for_bar, 'count']
                
                fig_bar = self.chart_generator.create_bar_chart(
                    bar_data, col_for_bar, 'count',
                    title=f"Count by {col_for_bar.title()}"
                )
                st.plotly_chart(fig_bar, use_container_width=True)
        
        # Time series if date column exists
        date_cols = filtered_data.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            st.markdown('<h3 class="section-header">📅 Time Series Analysis</h3>', unsafe_allow_html=True)
            
            date_col = date_cols[0]
            # Create daily counts
            daily_counts = filtered_data.groupby(filtered_data[date_col].dt.date).size().reset_index()
            daily_counts.columns = ['date', 'count']
            daily_counts['date'] = pd.to_datetime(daily_counts['date'])
            
            fig_time = self.chart_generator.create_time_series(
                daily_counts, 'date', 'count',
                title="Records Over Time"
            )
            st.plotly_chart(fig_time, use_container_width=True)
        
        # Correlation heatmap for numeric data
        numeric_data = filtered_data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 1:
            st.markdown('<h3 class="section-header">🔥 Correlation Analysis</h3>', unsafe_allow_html=True)
            
            fig_corr = self.chart_generator.create_correlation_heatmap(numeric_data)
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Data table
        st.markdown('<h2 class="section-header">📋 Data Table</h2>', unsafe_allow_html=True)
        
        # Display options
        col1, col2, col3 = st.columns(3)
        with col1:
            show_rows = st.selectbox("Rows to display", [10, 25, 50, 100], index=1)
        with col2:
            if st.button("📥 Download CSV"):
                csv_link = download_dataframe_as_csv(filtered_data, "filtered_data.csv")
                st.markdown(csv_link, unsafe_allow_html=True)
        with col3:
            if st.button("📊 Download Excel"):
                excel_link = download_dataframe_as_excel(filtered_data, "filtered_data.xlsx")
                st.markdown(excel_link, unsafe_allow_html=True)
        
        st.dataframe(filtered_data.head(show_rows), use_container_width=True)
    
    def _render_analytics_page(self):
        """Render advanced analytics page"""
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please upload data or load sample data to view analytics.")
            return
        
        st.markdown('<h2 class="section-header">📈 Advanced Analytics</h2>', unsafe_allow_html=True)
        
        # Dataset selection
        dataset_names = list(st.session_state.datasets.keys())
        selected_dataset = st.selectbox("Select Dataset for Analysis", dataset_names)
        
        if selected_dataset:
            df = st.session_state.datasets[selected_dataset]
            
            # Analytics tabs
            tab1, tab2, tab3, tab4 = st.tabs(["📊 Statistical Summary", "📈 Advanced Charts", "🔍 Data Exploration", "🎯 Custom Analysis"])
            
            with tab1:
                st.subheader("Statistical Summary")
                
                # Basic statistics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Numeric Columns Summary**")
                    numeric_summary = df.describe()
                    if not numeric_summary.empty:
                        st.dataframe(numeric_summary)
                    else:
                        st.info("No numeric columns found")
                
                with col2:
                    st.write("**Categorical Columns Summary**")
                    categorical_cols = df.select_dtypes(include=['object']).columns
                    if len(categorical_cols) > 0:
                        for col in categorical_cols[:5]:  # Show first 5 categorical columns
                            st.write(f"**{col}**")
                            value_counts = df[col].value_counts().head(10)
                            st.write(value_counts)
                    else:
                        st.info("No categorical columns found")
            
            with tab2:
                st.subheader("Advanced Visualizations")
                
                # Chart type selection
                chart_type = st.selectbox(
                    "Select Chart Type",
                    ["Histogram", "Box Plot", "Scatter Plot", "Heatmap", "Treemap", "Sunburst"]
                )
                
                if chart_type == "Histogram":
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) > 0:
                        selected_col = st.selectbox("Select Column", numeric_cols)
                        bins = st.slider("Number of Bins", 10, 100, 30)
                        
                        fig = self.chart_generator.create_histogram(df, selected_col, bins=bins)
                        st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Box Plot":
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    categorical_cols = df.select_dtypes(include=['object']).columns
                    
                    if len(numeric_cols) > 0 and len(categorical_cols) > 0:
                        y_col = st.selectbox("Select Numeric Column (Y-axis)", numeric_cols)
                        x_col = st.selectbox("Select Categorical Column (X-axis)", categorical_cols)
                        
                        fig = self.chart_generator.create_box_plot(df, x_col, y_col)
                        st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Scatter Plot":
                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    
                    if len(numeric_cols) >= 2:
                        x_col = st.selectbox("Select X Column", numeric_cols)
                        y_col = st.selectbox("Select Y Column", numeric_cols)
                        color_col = st.selectbox("Color by (optional)", [None] + list(df.columns))
                        
                        fig = self.chart_generator.create_scatter_plot(df, x_col, y_col, color=color_col)
                        st.plotly_chart(fig, use_container_width=True)
                
                elif chart_type == "Heatmap":
                    fig = self.chart_generator.create_correlation_heatmap(df)
                    st.plotly_chart(fig, use_container_width=True)
            
            with tab3:
                st.subheader("Data Exploration")
                
                # Column analysis
                selected_column = st.selectbox("Select Column for Analysis", df.columns)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Analysis of {selected_column}**")
                    st.write(f"Data Type: {df[selected_column].dtype}")
                    st.write(f"Non-null Count: {df[selected_column].count()}")
                    st.write(f"Null Count: {df[selected_column].isnull().sum()}")
                    st.write(f"Unique Values: {df[selected_column].nunique()}")
                    
                    if pd.api.types.is_numeric_dtype(df[selected_column]):
                        st.write(f"Min: {df[selected_column].min()}")
                        st.write(f"Max: {df[selected_column].max()}")
                        st.write(f"Mean: {df[selected_column].mean():.2f}")
                        st.write(f"Median: {df[selected_column].median():.2f}")
                
                with col2:
                    if df[selected_column].nunique() <= 20:
                        st.write("**Value Counts**")
                        value_counts = df[selected_column].value_counts()
                        st.write(value_counts)
                        
                        # Create chart for value counts
                        if len(value_counts) > 0:
                            fig = self.chart_generator.create_bar_chart(
                                value_counts.reset_index(),
                                'index', selected_column,
                                title=f"Distribution of {selected_column}"
                            )
                            st.plotly_chart(fig, use_container_width=True)
            
            with tab4:
                st.subheader("Custom Analysis")
                
                st.write("Create custom visualizations by selecting your parameters:")
                
                # Custom chart builder
                chart_type = st.selectbox(
                    "Chart Type",
                    ["Bar Chart", "Line Chart", "Pie Chart", "Scatter Plot", "Histogram"]
                )
                
                if chart_type in ["Bar Chart", "Line Chart", "Scatter Plot"]:
                    x_col = st.selectbox("X-axis Column", df.columns)
                    y_col = st.selectbox("Y-axis Column", df.columns)
                    color_col = st.selectbox("Color by (optional)", [None] + list(df.columns))
                    
                    if st.button("Generate Chart"):
                        try:
                            if chart_type == "Bar Chart":
                                # Group data if needed
                                if df[x_col].dtype == 'object':
                                    grouped_df = df.groupby(x_col)[y_col].sum().reset_index()
                                    fig = self.chart_generator.create_bar_chart(grouped_df, x_col, y_col, color=color_col)
                                else:
                                    fig = self.chart_generator.create_bar_chart(df, x_col, y_col, color=color_col)
                            
                            elif chart_type == "Line Chart":
                                fig = self.chart_generator.create_line_chart(df, x_col, y_col, color=color_col)
                            
                            elif chart_type == "Scatter Plot":
                                fig = self.chart_generator.create_scatter_plot(df, x_col, y_col, color=color_col)
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error creating chart: {str(e)}")
                
                elif chart_type == "Pie Chart":
                    names_col = st.selectbox("Categories Column", df.select_dtypes(include=['object']).columns)
                    values_col = st.selectbox("Values Column", df.select_dtypes(include=[np.number]).columns)
                    
                    if st.button("Generate Chart"):
                        try:
                            # Group data for pie chart
                            grouped_df = df.groupby(names_col)[values_col].sum().reset_index()
                            fig = self.chart_generator.create_pie_chart(grouped_df, values_col, names_col)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error creating chart: {str(e)}")
                
                elif chart_type == "Histogram":
                    col = st.selectbox("Column", df.select_dtypes(include=[np.number]).columns)
                    bins = st.slider("Number of Bins", 10, 100, 30)
                    
                    if st.button("Generate Chart"):
                        try:
                            fig = self.chart_generator.create_histogram(df, col, bins=bins)
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error creating chart: {str(e)}")
    
    def _render_chatbot_page(self):
        """Render AI assistant chatbot page"""
        st.markdown('<h2 class="section-header">🤖 AI Data Assistant</h2>', unsafe_allow_html=True)
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please upload data or load sample data to use the AI assistant.")
            return
        
        # Chat interface
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                st.markdown(f"**You:** {message['content']}")
            else:
                st.markdown(f"**AI Assistant:** {message['content']}")
                
                # Display chart if available
                if "chart" in message and message["chart"]:
                    st.plotly_chart(message["chart"], use_container_width=True)
        
        # Chat input
        user_input = st.text_input("Ask me anything about your data:", key="chat_input")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("Send", key="send_button"):
                if user_input:
                    self._process_chat_message(user_input)
        
        with col2:
            if st.button("Clear Chat", key="clear_button"):
                st.session_state.chat_history = []
                self.gemini_analyst.clear_conversation()
                st.rerun()
        
        with col3:
            if st.button("Get Data Insights", key="insights_button"):
                insights = self.gemini_analyst.get_data_insights()
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": insights
                })
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Quick questions
        st.markdown("### 💡 Quick Questions")
        quick_questions = [
            "What are the key trends in this data?",
            "Show me the distribution of categories",
            "What correlations exist in the numeric data?",
            "Are there any data quality issues?",
            "What insights can you provide about this dataset?"
        ]
        
        cols = st.columns(len(quick_questions))
        for i, question in enumerate(quick_questions):
            with cols[i]:
                if st.button(question, key=f"quick_{i}"):
                    self._process_chat_message(question)
    
    def _process_chat_message(self, user_input: str):
        """Process chat message and get AI response"""
        # Add user message to history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })
        
        # Get AI response
        response = self.gemini_analyst.chat(user_input)
        
        # Add AI response to history
        ai_message = {
            "role": "assistant",
            "content": response["text_response"]
        }
        
        if response["chart"]:
            ai_message["chart"] = response["chart"]
        
        st.session_state.chat_history.append(ai_message)
        
        # Rerun to update the display
        st.rerun()
    
    def _render_data_management_page(self):
        """Render data management page"""
        st.markdown('<h2 class="section-header">📁 Data Management</h2>', unsafe_allow_html=True)
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please upload data or load sample data to manage datasets.")
            return
        
        # Dataset overview
        st.subheader("📊 Dataset Overview")
        
        for name, df in st.session_state.datasets.items():
            with st.expander(f"📄 {name} ({df.shape[0]} rows, {df.shape[1]} columns)"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Basic Information**")
                    st.write(f"Shape: {df.shape}")
                    st.write(f"Memory Usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
                    st.write(f"Missing Values: {df.isnull().sum().sum()}")
                
                with col2:
                    st.write("**Column Types**")
                    type_counts = df.dtypes.value_counts()
                    for dtype, count in type_counts.items():
                        st.write(f"{dtype}: {count} columns")
                
                # Data preview
                st.write("**Data Preview**")
                st.dataframe(df.head(), use_container_width=True)
                
                # Download options
                col1, col2 = st.columns(2)
                with col1:
                    csv_link = download_dataframe_as_csv(df, f"{name}.csv")
                    st.markdown(csv_link, unsafe_allow_html=True)
                with col2:
                    excel_link = download_dataframe_as_excel(df, f"{name}.xlsx")
                    st.markdown(excel_link, unsafe_allow_html=True)
        
        # Data cleaning summary
        st.subheader("🧹 Data Cleaning Summary")
        
        cleaning_summary = self.data_cleaner.get_cleaning_summary()
        if not cleaning_summary.empty:
            st.dataframe(cleaning_summary, use_container_width=True)
        else:
            st.info("No cleaning operations performed yet.")
        
        # Data integration info
        st.subheader("🔗 Data Integration")
        
        if hasattr(self.data_integrator, 'relationships') and self.data_integrator.relationships:
            st.write("**Identified Relationships:**")
            for rel_name, rel_info in self.data_integrator.relationships.items():
                st.write(f"- {rel_name}: {rel_info['common_columns']}")
        else:
            st.info("No relationships identified between datasets.")
    
    def _render_data_quality_page(self):
        """Render data quality assessment page"""
        st.markdown('<h2 class="section-header">📋 Data Quality Assessment</h2>', unsafe_allow_html=True)
        
        if not st.session_state.data_loaded:
            st.warning("⚠️ Please upload data or load sample data to assess data quality.")
            return
        
        # Dataset selection for quality assessment
        dataset_names = list(st.session_state.datasets.keys())
        selected_dataset = st.selectbox("Select Dataset for Quality Assessment", dataset_names)
        
        if selected_dataset:
            df = st.session_state.datasets[selected_dataset]
            
            # Generate data profile
            profile = generate_data_profile(df)
            
            # Overall quality metrics
            st.subheader("📊 Overall Quality Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", f"{profile['basic_info']['rows']:,}")
            with col2:
                st.metric("Total Columns", profile['basic_info']['columns'])
            with col3:
                st.metric("Missing Cells", f"{profile['basic_info']['missing_cells']:,}")
            with col4:
                st.metric("Missing %", f"{profile['basic_info']['missing_percentage']:.1f}%")
            
            # Data quality issues
            st.subheader("⚠️ Data Quality Issues")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Duplicate Rows", profile['data_quality']['duplicate_rows'])
            with col2:
                st.metric("Empty Rows", profile['data_quality']['empty_rows'])
            with col3:
                st.metric("Empty Columns", profile['data_quality']['empty_columns'])
            
            # Column-level quality report
            st.subheader("📋 Column-Level Quality Report")
            
            quality_report = create_data_quality_report(df)
            st.dataframe(quality_report, use_container_width=True)
            
            # Download quality report
            if st.button("📥 Download Quality Report"):
                csv_link = download_dataframe_as_csv(quality_report, f"{selected_dataset}_quality_report.csv")
                st.markdown(csv_link, unsafe_allow_html=True)
            
            # Missing data visualization
            st.subheader("📊 Missing Data Visualization")
            
            missing_data = df.isnull().sum().reset_index()
            missing_data.columns = ['Column', 'Missing_Count']
            missing_data = missing_data[missing_data['Missing_Count'] > 0]
            
            if not missing_data.empty:
                fig = self.chart_generator.create_bar_chart(
                    missing_data, 'Column', 'Missing_Count',
                    title="Missing Values by Column"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("✅ No missing values found in the dataset!")


def main():
    """Main function to run the application"""
    dashboard = AdvancedDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()