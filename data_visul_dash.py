import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from io import StringIO
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Data Visualization Dashboard",
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
        padding: 1rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 10px;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e86ab;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #1f77b4;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Main header
    st.markdown('<div class="main-header">📊 Data Visualization Dashboard</div>', unsafe_allow_html=True)
    
    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("📁 Data Upload")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your dataset in CSV format"
        )
        
        st.markdown("---")
        st.header("⚙️ Settings")
        
        # Sample data option
        use_sample_data = st.checkbox("Use sample data for demo", value=False)
        
        if use_sample_data:
            st.info("Using built-in sample dataset")
        
        st.markdown("---")
        st.header("ℹ️ About")
        st.info("""
        This dashboard allows you to:
        - Upload and explore CSV data
        - Generate automatic visualizations
        - Analyze correlations
        - Identify missing values
        - Create interactive charts
        """)

    # Load data
    df = load_data(uploaded_file, use_sample_data)
    
    if df is not None:
        # Display basic dataset info
        display_dataset_info(df)
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Data Overview", 
            "📊 Visualizations", 
            "🔥 Correlation Analysis", 
            "❓ Missing Values",
            "🔍 Data Explorer"
        ])
        
        with tab1:
            display_data_overview(df)
        
        with tab2:
            display_visualizations(df)
        
        with tab3:
            display_correlation_analysis(df)
        
        with tab4:
            display_missing_values_analysis(df)
        
        with tab5:
            display_data_explorer(df)
    else:
        # Show instructions if no data is loaded
        show_instructions()

def load_data(uploaded_file, use_sample_data):
    """
    Load data from uploaded file or use sample data
    """
    if use_sample_data:
        # Create sample dataset
        np.random.seed(42)
        n_samples = 1000
        
        sample_data = {
            'Age': np.random.normal(35, 10, n_samples).astype(int),
            'Salary': np.random.normal(50000, 15000, n_samples).astype(int),
            'Experience': np.random.normal(8, 5, n_samples).astype(int),
            'Department': np.random.choice(['Sales', 'Marketing', 'IT', 'HR', 'Finance'], n_samples),
            'Performance_Score': np.random.normal(75, 15, n_samples).astype(int),
            'Satisfaction': np.random.uniform(1, 5, n_samples),
            'Projects_Completed': np.random.poisson(5, n_samples),
            'Training_Hours': np.random.normal(20, 5, n_samples).astype(int)
        }
        
        # Introduce some missing values
        df = pd.DataFrame(sample_data)
        mask = np.random.random(sample_data['Age'].shape) < 0.05
        df.loc[mask, 'Salary'] = np.nan
        mask = np.random.random(sample_data['Age'].shape) < 0.03
        df.loc[mask, 'Experience'] = np.nan
        
        return df
    
    elif uploaded_file is not None:
        try:
            # Read CSV file
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return None
    else:
        return None

def display_dataset_info(df):
    """
    Display basic information about the dataset
    """
    st.markdown('<div class="section-header">Dataset Information</div>', unsafe_allow_html=True)
    
    # Create metrics columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    
    with col2:
        st.metric("Total Columns", f"{len(df.columns):,}")
    
    with col3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        st.metric("Numeric Columns", f"{len(numeric_cols)}")
    
    with col4:
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        st.metric("Categorical Columns", f"{len(categorical_cols)}")
    
    # Display dataframe head
    st.subheader("Data Preview (First 10 rows)")
    st.dataframe(df.head(10), use_container_width=True)

def display_data_overview(df):
    """
    Display data overview including statistics and data types
    """
    st.markdown('<div class="section-header">Data Overview</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Types")
        dtype_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.values,
            'Non-Null Count': df.count().values
        })
        st.dataframe(dtype_info, use_container_width=True)
    
    with col2:
        st.subheader("Basic Statistics")
        st.dataframe(df.describe(), use_container_width=True)
    
    # Show unique values for categorical columns
    st.subheader("Categorical Values Summary")
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            with st.expander(f"📋 {col} - Unique Values"):
                unique_vals = df[col].value_counts()
                st.write(f"Total Unique Values: {len(unique_vals)}")
                st.dataframe(unique_vals, use_container_width=True)
    else:
        st.info("No categorical columns found in the dataset")

def display_visualizations(df):
    """
    Generate automatic visualizations based on data types
    """
    st.markdown('<div class="section-header">Automatic Visualizations</div>', unsafe_allow_html=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if len(numeric_cols) > 0:
        # Distribution plots for numeric columns
        st.subheader("Distribution of Numeric Variables")
        
        # Let user select column for distribution plot
        selected_num_col = st.selectbox(
            "Select numeric column for distribution:",
            numeric_cols,
            key="dist_select"
        )
        
        if selected_num_col:
            fig = px.histogram(
                df, 
                x=selected_num_col,
                title=f"Distribution of {selected_num_col}",
                marginal="box",
                nbins=50
            )
            st.plotly_chart(fig, use_container_width=True)
    
    if len(categorical_cols) > 0 and len(numeric_cols) > 0:
        # Box plots for categorical vs numeric
        st.subheader("Categorical vs Numeric Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            cat_col = st.selectbox(
                "Select categorical column:",
                categorical_cols,
                key="cat_select"
            )
        
        with col2:
            num_col = st.selectbox(
                "Select numeric column:",
                numeric_cols,
                key="num_select"
            )
        
        if cat_col and num_col:
            fig = px.box(
                df, 
                x=cat_col, 
                y=num_col,
                title=f"{num_col} by {cat_col}"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    if len(numeric_cols) >= 2:
        # Scatter plot
        st.subheader("Scatter Plot")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            x_col = st.selectbox("X-axis:", numeric_cols, key="x_scatter")
        
        with col2:
            y_col = st.selectbox("Y-axis:", numeric_cols, key="y_scatter")
        
        with col3:
            color_col = st.selectbox(
                "Color by (optional):", 
                [None] + categorical_cols,
                key="color_scatter"
            )
        
        if x_col and y_col:
            fig = px.scatter(
                df, 
                x=x_col, 
                y=y_col,
                color=color_col,
                title=f"{y_col} vs {x_col}",
                hover_data=df.columns.tolist()
            )
            st.plotly_chart(fig, use_container_width=True)

def display_correlation_analysis(df):
    """
    Display correlation heatmap and analysis
    """
    st.markdown('<div class="section-header">Correlation Analysis</div>', unsafe_allow_html=True)
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) > 1:
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr()
        
        # Create correlation heatmap
        fig = px.imshow(
            corr_matrix,
            title="Correlation Heatmap",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1
        )
        
        # Update layout
        fig.update_layout(
            width=800,
            height=600
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display correlation matrix as table
        with st.expander("View Correlation Matrix Table"):
            st.dataframe(corr_matrix.style.background_gradient(cmap='RdBu_r', vmin=-1, vmax=1), use_container_width=True)
        
        # Show top correlations
        st.subheader("Top Correlations")
        
        # Get upper triangle of correlation matrix
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find top positive and negative correlations
        correlations = []
        for col in upper_tri.columns:
            for idx in upper_tri.index:
                if not pd.isna(upper_tri.loc[idx, col]):
                    correlations.append({
                        'Variable 1': col,
                        'Variable 2': idx,
                        'Correlation': upper_tri.loc[idx, col]
                    })
        
        corr_df = pd.DataFrame(correlations)
        corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top Positive Correlations**")
            positive_corr = corr_df[corr_df['Correlation'] > 0].head(10)
            st.dataframe(positive_corr, use_container_width=True)
        
        with col2:
            st.write("**Top Negative Correlations**")
            negative_corr = corr_df[corr_df['Correlation'] < 0].head(10)
            st.dataframe(negative_corr, use_container_width=True)
    
    else:
        st.warning("Need at least 2 numeric columns for correlation analysis")

def display_missing_values_analysis(df):
    """
    Analyze and display missing values information
    """
    st.markdown('<div class="section-header">Missing Values Analysis</div>', unsafe_allow_html=True)
    
    # Calculate missing values
    missing_data = df.isnull().sum()
    missing_percent = (missing_data / len(df)) * 100
    
    missing_df = pd.DataFrame({
        'Column': df.columns,
        'Missing Count': missing_data,
        'Missing Percentage': missing_percent
    }).sort_values('Missing Count', ascending=False)
    
    # Display missing values summary
    total_missing = missing_data.sum()
    total_cells = df.size
    missing_percentage_total = (total_missing / total_cells) * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Missing Values", f"{total_missing:,}")
    
    with col2:
        st.metric("Total Data Cells", f"{total_cells:,}")
    
    with col3:
        st.metric("Overall Missing %", f"{missing_percentage_total:.2f}%")
    
    # Display missing values table
    st.subheader("Missing Values by Column")
    st.dataframe(
        missing_df[missing_df['Missing Count'] > 0], 
        use_container_width=True
    )
    
    # Visualize missing values
    if total_missing > 0:
        st.subheader("Missing Values Heatmap")
        
        # Create missing values heatmap
        fig = px.imshow(
            df.isnull(),
            title="Missing Values Heatmap (Yellow = Missing)",
            aspect="auto",
            color_continuous_scale=['lightblue', 'yellow']
        )
        fig.update_layout(
            xaxis_title="Columns",
            yaxis_title="Rows"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show columns with missing values
        cols_with_missing = missing_df[missing_df['Missing Count'] > 0]['Column'].tolist()
        st.warning(f"Columns with missing values: {', '.join(cols_with_missing)}")
    else:
        st.success("🎉 No missing values found in the dataset!")

def display_data_explorer(df):
    """
    Interactive data explorer with filtering capabilities
    """
    st.markdown('<div class="section-header">Interactive Data Explorer</div>', unsafe_allow_html=True)
    
    # Filter options
    st.subheader("Data Filtering")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Column selector for filtering
        filter_column = st.selectbox(
            "Select column to filter:",
            [None] + df.columns.tolist()
        )
    
    with col2:
        if filter_column:
            if df[filter_column].dtype in ['object']:
                # Categorical filter
                unique_vals = df[filter_column].unique()
                selected_vals = st.multiselect(
                    f"Select {filter_column} values:",
                    unique_vals,
                    default=unique_vals[:min(5, len(unique_vals))]
                )
                filtered_df = df[df[filter_column].isin(selected_vals)]
            else:
                # Numerical filter
                min_val = float(df[filter_column].min())
                max_val = float(df[filter_column].max())
                selected_range = st.slider(
                    f"Select {filter_column} range:",
                    min_val,
                    max_val,
                    (min_val, max_val)
                )
                filtered_df = df[
                    (df[filter_column] >= selected_range[0]) & 
                    (df[filter_column] <= selected_range[1])
                ]
        else:
            filtered_df = df
    
    # Display filtered data
    st.subheader("Filtered Data")
    st.dataframe(filtered_df, use_container_width=True)
    
    # Download filtered data
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name="filtered_data.csv",
        mime="text/csv"
    )

def show_instructions():
    """
    Show instructions when no data is loaded
    """
    st.markdown("""
    <div class="warning-box">
    <h3>🚀 Welcome to the Data Visualization Dashboard!</h3>
    <p>To get started:</p>
    <ol>
        <li><strong>Upload a CSV file</strong> using the sidebar, or</li>
        <li><strong>Check "Use sample data for demo"</strong> to explore with sample data</li>
    </ol>
    <p>The dashboard will automatically generate:</p>
    <ul>
        <li>📊 Data overview and statistics</li>
        <li>📈 Interactive visualizations</li>
        <li>🔥 Correlation heatmaps</li>
        <li>❓ Missing values analysis</li>
        <li>🔍 Interactive data explorer</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Sample data preview
    st.subheader("Sample Data Structure")
    sample_df = pd.DataFrame({
        'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'Salary': [50000, 60000, 70000],
        'Department': ['IT', 'HR', 'Finance'],
        'Score': [85, 92, 78]
    })
    st.dataframe(sample_df, use_container_width=True)
    st.info("💡 Your data should be in a similar CSV format with headers in the first row.")

if __name__ == "__main__":
    main()