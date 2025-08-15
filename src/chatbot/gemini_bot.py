"""
Gemini AI-powered chatbot for data analysis
"""
import google.generativeai as genai
import pandas as pd
import streamlit as st
from typing import Dict, List, Optional, Any, Tuple
import json
import plotly.express as px
import plotly.graph_objects as go
from config.settings import GEMINI_API_KEY
from src.visualization.charts import ChartGenerator


class GeminiDataAnalyst:
    """AI-powered data analyst using Gemini API"""
    
    def __init__(self):
        self.chart_generator = ChartGenerator()
        self.conversation_history = []
        self.current_data = None
        self.data_summary = None
        
        # Configure Gemini API
        if GEMINI_API_KEY:
            genai.configure(api_key=GEMINI_API_KEY)
            self.model = genai.GenerativeModel('gemini-pro')
        else:
            self.model = None
            st.warning("⚠️ Gemini API key not configured. Chatbot functionality will be limited.")
    
    def set_data_context(self, data: Dict[str, pd.DataFrame]):
        """Set the current data context for analysis"""
        self.current_data = data
        self.data_summary = self._generate_data_summary(data)
    
    def _generate_data_summary(self, data: Dict[str, pd.DataFrame]) -> str:
        """Generate a summary of the available data"""
        summary_parts = []
        
        for dataset_name, df in data.items():
            summary_parts.append(f"\nDataset: {dataset_name}")
            summary_parts.append(f"- Shape: {df.shape[0]} rows, {df.shape[1]} columns")
            summary_parts.append(f"- Columns: {', '.join(df.columns.tolist())}")
            
            # Add column types
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            date_cols = df.select_dtypes(include=['datetime']).columns.tolist()
            
            if numeric_cols:
                summary_parts.append(f"- Numeric columns: {', '.join(numeric_cols)}")
            if categorical_cols:
                summary_parts.append(f"- Categorical columns: {', '.join(categorical_cols)}")
            if date_cols:
                summary_parts.append(f"- Date columns: {', '.join(date_cols)}")
        
        return '\n'.join(summary_parts)
    
    def chat(self, user_message: str) -> Dict[str, Any]:
        """Process user message and return response with potential visualizations"""
        if not self.model:
            return {
                'text_response': "Chatbot is not available. Please configure the Gemini API key.",
                'chart': None,
                'data': None
            }
        
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Create context-aware prompt
        system_prompt = self._create_system_prompt()
        full_prompt = f"{system_prompt}\n\nUser Question: {user_message}"
        
        try:
            # Generate response from Gemini
            response = self.model.generate_content(full_prompt)
            ai_response = response.text
            
            # Add AI response to conversation history
            self.conversation_history.append({"role": "assistant", "content": ai_response})
            
            # Analyze if visualization is needed
            chart_suggestion = self._analyze_for_visualization(user_message, ai_response)
            
            return {
                'text_response': ai_response,
                'chart': chart_suggestion.get('chart'),
                'data': chart_suggestion.get('data'),
                'chart_type': chart_suggestion.get('chart_type')
            }
            
        except Exception as e:
            error_message = f"Error generating response: {str(e)}"
            return {
                'text_response': error_message,
                'chart': None,
                'data': None
            }
    
    def _create_system_prompt(self) -> str:
        """Create system prompt with data context"""
        base_prompt = """
        You are an expert data analyst AI assistant. You have access to the following datasets:
        
        {data_summary}
        
        Your role is to:
        1. Answer questions about the data accurately and insightfully
        2. Provide statistical analysis and interpretations
        3. Suggest relevant visualizations when appropriate
        4. Identify trends, patterns, and anomalies in the data
        5. Provide actionable insights and recommendations
        
        Guidelines:
        - Be concise but comprehensive in your responses
        - Use specific numbers and statistics from the data when possible
        - Suggest visualizations that would help illustrate your points
        - If asked about trends, provide time-based analysis
        - If asked about relationships, suggest correlation or comparison analysis
        - Always consider the business context of the data
        
        When suggesting visualizations, use these keywords:
        - "CHART_SUGGESTION:" followed by chart type (bar, line, pie, scatter, heatmap, etc.)
        - "CHART_DATA:" followed by the specific columns to use
        - "CHART_TITLE:" followed by a descriptive title
        
        Previous conversation context:
        {conversation_history}
        """.format(
            data_summary=self.data_summary or "No data currently loaded",
            conversation_history=self._format_conversation_history()
        )
        
        return base_prompt
    
    def _format_conversation_history(self) -> str:
        """Format conversation history for context"""
        if not self.conversation_history:
            return "No previous conversation"
        
        formatted_history = []
        for message in self.conversation_history[-6:]:  # Last 6 messages for context
            role = "User" if message["role"] == "user" else "Assistant"
            formatted_history.append(f"{role}: {message['content'][:200]}...")
        
        return '\n'.join(formatted_history)
    
    def _analyze_for_visualization(self, user_message: str, ai_response: str) -> Dict[str, Any]:
        """Analyze if the response suggests a visualization"""
        visualization_keywords = {
            'trend': 'line',
            'over time': 'line',
            'distribution': 'histogram',
            'compare': 'bar',
            'comparison': 'bar',
            'correlation': 'scatter',
            'relationship': 'scatter',
            'breakdown': 'pie',
            'composition': 'pie',
            'heatmap': 'heatmap',
            'pattern': 'heatmap'
        }
        
        # Check if AI response contains chart suggestions
        if "CHART_SUGGESTION:" in ai_response:
            return self._parse_chart_suggestion(ai_response)
        
        # Auto-suggest based on keywords
        user_lower = user_message.lower()
        ai_lower = ai_response.lower()
        
        for keyword, chart_type in visualization_keywords.items():
            if keyword in user_lower or keyword in ai_lower:
                return self._auto_generate_chart(user_message, chart_type)
        
        return {'chart': None, 'data': None, 'chart_type': None}
    
    def _parse_chart_suggestion(self, ai_response: str) -> Dict[str, Any]:
        """Parse explicit chart suggestions from AI response"""
        try:
            lines = ai_response.split('\n')
            chart_info = {}
            
            for line in lines:
                if line.startswith("CHART_SUGGESTION:"):
                    chart_info['chart_type'] = line.replace("CHART_SUGGESTION:", "").strip()
                elif line.startswith("CHART_DATA:"):
                    chart_info['columns'] = line.replace("CHART_DATA:", "").strip().split(',')
                elif line.startswith("CHART_TITLE:"):
                    chart_info['title'] = line.replace("CHART_TITLE:", "").strip()
            
            if 'chart_type' in chart_info:
                return self._generate_suggested_chart(chart_info)
            
        except Exception as e:
            st.error(f"Error parsing chart suggestion: {e}")
        
        return {'chart': None, 'data': None, 'chart_type': None}
    
    def _auto_generate_chart(self, user_message: str, chart_type: str) -> Dict[str, Any]:
        """Auto-generate chart based on user message and suggested type"""
        if not self.current_data:
            return {'chart': None, 'data': None, 'chart_type': None}
        
        # Get the first dataset for simplicity
        dataset_name = list(self.current_data.keys())[0]
        df = self.current_data[dataset_name]
        
        try:
            if chart_type == 'line':
                # Find date and numeric columns
                date_cols = df.select_dtypes(include=['datetime']).columns
                numeric_cols = df.select_dtypes(include=['number']).columns
                
                if len(date_cols) > 0 and len(numeric_cols) > 0:
                    chart = self.chart_generator.create_time_series(
                        df, date_cols[0], numeric_cols[0],
                        title=f"{numeric_cols[0].title()} Over Time"
                    )
                    return {
                        'chart': chart,
                        'data': df[[date_cols[0], numeric_cols[0]]],
                        'chart_type': 'line'
                    }
            
            elif chart_type == 'bar':
                # Find categorical and numeric columns
                cat_cols = df.select_dtypes(include=['object']).columns
                numeric_cols = df.select_dtypes(include=['number']).columns
                
                if len(cat_cols) > 0 and len(numeric_cols) > 0:
                    # Group by categorical column and sum numeric column
                    grouped_df = df.groupby(cat_cols[0])[numeric_cols[0]].sum().reset_index()
                    chart = self.chart_generator.create_bar_chart(
                        grouped_df, cat_cols[0], numeric_cols[0],
                        title=f"{numeric_cols[0].title()} by {cat_cols[0].title()}"
                    )
                    return {
                        'chart': chart,
                        'data': grouped_df,
                        'chart_type': 'bar'
                    }
            
            elif chart_type == 'pie':
                # Find categorical column for breakdown
                cat_cols = df.select_dtypes(include=['object']).columns
                
                if len(cat_cols) > 0:
                    value_counts = df[cat_cols[0]].value_counts().reset_index()
                    value_counts.columns = [cat_cols[0], 'count']
                    chart = self.chart_generator.create_pie_chart(
                        value_counts, 'count', cat_cols[0],
                        title=f"Distribution of {cat_cols[0].title()}"
                    )
                    return {
                        'chart': chart,
                        'data': value_counts,
                        'chart_type': 'pie'
                    }
            
            elif chart_type == 'scatter':
                # Find two numeric columns
                numeric_cols = df.select_dtypes(include=['number']).columns
                
                if len(numeric_cols) >= 2:
                    chart = self.chart_generator.create_scatter_plot(
                        df, numeric_cols[0], numeric_cols[1],
                        title=f"{numeric_cols[1].title()} vs {numeric_cols[0].title()}"
                    )
                    return {
                        'chart': chart,
                        'data': df[[numeric_cols[0], numeric_cols[1]]],
                        'chart_type': 'scatter'
                    }
            
            elif chart_type == 'histogram':
                # Find numeric column
                numeric_cols = df.select_dtypes(include=['number']).columns
                
                if len(numeric_cols) > 0:
                    chart = self.chart_generator.create_histogram(
                        df, numeric_cols[0],
                        title=f"Distribution of {numeric_cols[0].title()}"
                    )
                    return {
                        'chart': chart,
                        'data': df[[numeric_cols[0]]],
                        'chart_type': 'histogram'
                    }
            
            elif chart_type == 'heatmap':
                # Create correlation heatmap
                chart = self.chart_generator.create_correlation_heatmap(df)
                return {
                    'chart': chart,
                    'data': df.select_dtypes(include=['number']),
                    'chart_type': 'heatmap'
                }
        
        except Exception as e:
            st.error(f"Error generating auto chart: {e}")
        
        return {'chart': None, 'data': None, 'chart_type': None}
    
    def _generate_suggested_chart(self, chart_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate chart based on explicit AI suggestion"""
        if not self.current_data:
            return {'chart': None, 'data': None, 'chart_type': None}
        
        chart_type = chart_info.get('chart_type', '').lower()
        columns = [col.strip() for col in chart_info.get('columns', [])]
        title = chart_info.get('title', '')
        
        # Get the first dataset that contains the required columns
        target_df = None
        for dataset_name, df in self.current_data.items():
            if all(col in df.columns for col in columns):
                target_df = df
                break
        
        if target_df is None:
            return {'chart': None, 'data': None, 'chart_type': None}
        
        try:
            if chart_type == 'bar' and len(columns) >= 2:
                chart = self.chart_generator.create_bar_chart(
                    target_df, columns[0], columns[1], title=title
                )
            elif chart_type == 'line' and len(columns) >= 2:
                chart = self.chart_generator.create_line_chart(
                    target_df, columns[0], columns[1], title=title
                )
            elif chart_type == 'pie' and len(columns) >= 2:
                chart = self.chart_generator.create_pie_chart(
                    target_df, columns[1], columns[0], title=title
                )
            elif chart_type == 'scatter' and len(columns) >= 2:
                chart = self.chart_generator.create_scatter_plot(
                    target_df, columns[0], columns[1], title=title
                )
            else:
                return {'chart': None, 'data': None, 'chart_type': None}
            
            return {
                'chart': chart,
                'data': target_df[columns],
                'chart_type': chart_type
            }
            
        except Exception as e:
            st.error(f"Error generating suggested chart: {e}")
            return {'chart': None, 'data': None, 'chart_type': None}
    
    def get_data_insights(self, dataset_name: str = None) -> str:
        """Generate automatic insights about the data"""
        if not self.current_data:
            return "No data available for analysis."
        
        if dataset_name and dataset_name in self.current_data:
            df = self.current_data[dataset_name]
            data_context = f"Dataset: {dataset_name}\n{self._generate_data_summary({dataset_name: df})}"
        else:
            data_context = self.data_summary
        
        insight_prompt = f"""
        Based on the following data summary, provide key insights and observations:
        
        {data_context}
        
        Please provide:
        1. Key statistics and patterns
        2. Notable trends or anomalies
        3. Data quality observations
        4. Recommendations for further analysis
        
        Keep the response concise but informative.
        """
        
        if self.model:
            try:
                response = self.model.generate_content(insight_prompt)
                return response.text
            except Exception as e:
                return f"Error generating insights: {str(e)}"
        else:
            return "Gemini API not configured. Cannot generate automatic insights."
    
    def clear_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []