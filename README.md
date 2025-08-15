# Advanced Data Analysis Dashboard

A comprehensive, AI-powered data analysis and visualization platform built with Streamlit, featuring interactive dashboards, advanced analytics, and intelligent chatbot assistance.

## 🚀 Features

### 📊 Main Dashboard
- **Interactive KPI Cards**: Real-time key performance indicators
- **Dynamic Visualizations**: Pie charts, bar charts, time series, and correlation heatmaps
- **Advanced Filtering**: Multi-dimensional data filtering with date ranges, categories, and numeric ranges
- **Data Export**: Download filtered data as CSV or Excel files

### 📈 Advanced Analytics
- **Statistical Analysis**: Comprehensive statistical summaries for numeric and categorical data
- **Advanced Charts**: Histograms, box plots, scatter plots, treemaps, and sunburst charts
- **Data Exploration**: Column-level analysis with detailed statistics
- **Custom Visualizations**: Build custom charts with user-selected parameters

### 🤖 AI Assistant
- **Gemini-Powered Chatbot**: Intelligent data analysis assistant using Google's Gemini API
- **Visual Responses**: AI generates relevant charts and visualizations based on questions
- **Context-Aware**: Understands your data structure and provides relevant insights
- **Quick Questions**: Pre-built questions for common analysis tasks

### 📁 Data Management
- **Multi-Format Support**: CSV, Excel (XLSX, XLS) file uploads
- **Automatic Data Cleaning**: Handles missing values, standardizes formats, removes duplicates
- **Data Integration**: Automatically identifies relationships between datasets
- **File Validation**: Ensures data quality and file integrity

### 📋 Data Quality Assessment
- **Comprehensive Profiling**: Detailed analysis of data quality metrics
- **Missing Data Visualization**: Visual representation of data completeness
- **Column-Level Reports**: Individual column statistics and quality indicators
- **Quality Scoring**: Overall data quality assessment

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Analysis
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure environment variables**:
   - Copy `.env.example` to `.env`
   - Add your Gemini API key to enable AI features:
     ```
     GEMINI_API_KEY=your_api_key_here
     ```

4. **Run the application**:
   ```bash
   streamlit run app.py --server.port 12000 --server.address 0.0.0.0
   ```

## 📖 Usage

### Getting Started
1. **Upload Data**: Use the sidebar to upload CSV or Excel files
2. **Load Sample Data**: Click "Load Sample Data" to explore with demo data
3. **Navigate**: Use the sidebar navigation to explore different features

### Dashboard Navigation
- **📊 Dashboard**: Main overview with KPIs and key visualizations
- **📈 Advanced Analytics**: Deep-dive analysis tools and custom charts
- **🤖 AI Assistant**: Chat with AI about your data
- **📁 Data Management**: View and manage uploaded datasets
- **📋 Data Quality**: Assess and improve data quality

### AI Assistant Usage
- Ask natural language questions about your data
- Request specific visualizations
- Get insights and recommendations
- Use quick questions for common analysis tasks

Example questions:
- "What are the main trends in this data?"
- "Show me the correlation between variables"
- "Create a chart showing sales by category"
- "What data quality issues should I be aware of?"

## 🏗️ Architecture

### Project Structure
```
Analysis/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── config/
│   └── settings.py       # Configuration settings
├── src/
│   ├── data_processing/  # Data loading and cleaning
│   ├── visualization/    # Chart generation and layouts
│   ├── chatbot/         # AI assistant functionality
│   └── utils/           # Helper functions
├── data/                # Data storage directory
├── assets/              # Static assets
└── tests/               # Test files
```

### Key Components

#### Data Processing Pipeline
1. **DataLoader**: Handles multiple file formats and encodings
2. **DataCleaner**: Standardizes data, handles missing values, removes duplicates
3. **DataIntegrator**: Identifies relationships and integrates multiple datasets

#### Visualization Engine
1. **ChartGenerator**: Creates interactive Plotly charts
2. **DashboardLayout**: Manages dashboard layout and filtering
3. **Advanced Analytics**: Statistical analysis and custom visualizations

#### AI Assistant
1. **GeminiDataAnalyst**: Integrates with Google's Gemini API
2. **Context-Aware Responses**: Understands data structure and user intent
3. **Visual Intelligence**: Generates relevant charts based on conversations

## 🔧 Configuration

### Environment Variables
- `GEMINI_API_KEY`: Google Gemini API key for AI features
- `APP_TITLE`: Application title
- `APP_DESCRIPTION`: Application description
- `DEBUG_MODE`: Enable debug mode

### Customization
- **Color Palette**: Modify `DEFAULT_COLOR_PALETTE` in `config/settings.py`
- **Chart Dimensions**: Adjust `CHART_HEIGHT` and `CHART_WIDTH`
- **File Limits**: Change `MAX_FILE_SIZE_MB` for upload limits

## 📊 Data Requirements

### Supported Formats
- **CSV**: Comma-separated values with various encodings
- **Excel**: XLSX and XLS formats
- **Multiple Files**: Upload and integrate multiple datasets

### Data Preparation Tips
- Ensure consistent date formats
- Use clear column names
- Include data dictionaries for complex datasets
- Remove or handle special characters in text fields

## 🤖 AI Features

### Gemini API Integration
The AI assistant uses Google's Gemini API to provide:
- Natural language understanding of data questions
- Intelligent chart recommendations
- Contextual insights and analysis
- Visual response generation

### Setting Up AI Features
1. Get a Gemini API key from Google AI Studio
2. Add the key to your `.env` file
3. Restart the application
4. The AI assistant will be fully functional

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py --server.port 12000 --server.address 0.0.0.0
```

### Production Deployment
- Configure environment variables
- Set up proper authentication
- Use a production WSGI server
- Enable HTTPS for security

## 🧪 Testing

Run tests with:
```bash
python -m pytest tests/
```

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Check the documentation
- Review the example data and configurations

## 🔄 Updates and Roadmap

### Recent Updates
- ✅ Multi-dataset support
- ✅ AI-powered chatbot
- ✅ Advanced filtering
- ✅ Data quality assessment
- ✅ Export functionality

### Planned Features
- 🔄 Real-time data streaming
- 🔄 Advanced ML models
- 🔄 Custom dashboard templates
- 🔄 API integration
- 🔄 User authentication

---

**Built with ❤️ using Streamlit, Plotly, and Google Gemini AI**