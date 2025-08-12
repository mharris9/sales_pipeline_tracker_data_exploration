import streamlit as st
import pandas as pd
import time

def render_data_preview_section():
    st.header("👀 Data Preview & Exploration")
    
    if not st.session_state.state_manager.get_state('data.data_loaded', False):
        st.warning("No data loaded. Please upload data first.")
        return

    data_handler = st.session_state.state_manager.get_extension('data_handler')
    df = data_handler.get_current_df()
    column_types = data_handler.get_column_types()

    if df is None:
        st.error("No data available for preview.")
        return

    st.subheader("Dataset Overview")

    # Basic statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Rows", len(df))
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    with col4:
        missing_data = df.isnull().sum().sum()
        st.metric("Missing Values", missing_data)

    # Column information
    st.subheader("Column Information")
    
    column_info = []
    for col in df.columns:
        col_type = column_types.get(col, "Unknown")
        unique_count = df[col].nunique()
        missing_count = df[col].isnull().sum()
        
        column_info.append({
            "Column": col,
            "Type": col_type.value if hasattr(col_type, 'value') else str(col_type),
            "Unique Values": unique_count,
            "Missing Values": missing_count,
            "Sample Values": str(df[col].dropna().head(3).tolist())[:50] + "..." if len(str(df[col].dropna().head(3).tolist())) > 50 else str(df[col].dropna().head(3).tolist())
        })

    column_df = pd.DataFrame(column_info)
    st.dataframe(column_df, use_container_width=True)

    # Data preview tabs
    st.subheader("Data Preview")
    
    tab1, tab2, tab3 = st.tabs(["📊 Head", "📈 Tail", "🔍 Sample"])
    
    with tab1:
        st.write("**First 10 rows:**")
        st.dataframe(df.head(10), use_container_width=True)
    
    with tab2:
        st.write("**Last 10 rows:**")
        st.dataframe(df.tail(10), use_container_width=True)
    
    with tab3:
        st.write("**Random sample (10 rows):**")
        sample_size = min(10, len(df))
        st.dataframe(df.sample(n=sample_size), use_container_width=True)

    # Column type distribution
    st.subheader("Column Type Distribution")
    
    type_counts = {}
    for col_type in column_types.values():
        type_name = col_type.value if hasattr(col_type, 'value') else str(col_type)
        type_counts[type_name] = type_counts.get(type_name, 0) + 1

    if type_counts:
        type_df = pd.DataFrame(list(type_counts.items()), columns=['Data Type', 'Count'])
        st.bar_chart(type_df.set_index('Data Type'))

    # Missing data analysis
    if df.isnull().sum().sum() > 0:
        st.subheader("Missing Data Analysis")
        
        missing_df = pd.DataFrame({
            'Column': df.columns,
            'Missing Count': df.isnull().sum(),
            'Missing Percentage': (df.isnull().sum() / len(df) * 100).round(2)
        }).sort_values('Missing Count', ascending=False)
        
        missing_df = missing_df[missing_df['Missing Count'] > 0]
        st.dataframe(missing_df, use_container_width=True)

    # Data quality insights
    st.subheader("Data Quality Insights")
    
    insights = []
    
    # Check for duplicate rows
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        insights.append(f"⚠️ **{duplicate_count} duplicate rows** found")
    else:
        insights.append("✅ **No duplicate rows** found")
    
    # Check for columns with all null values
    all_null_cols = df.columns[df.isnull().all()].tolist()
    if all_null_cols:
        insights.append(f"⚠️ **{len(all_null_cols)} columns** have all null values: {', '.join(all_null_cols)}")
    else:
        insights.append("✅ **No columns** with all null values")
    
    # Check for columns with single value
    single_value_cols = []
    for col in df.columns:
        if df[col].nunique() == 1:
            single_value_cols.append(col)
    
    if single_value_cols:
        insights.append(f"⚠️ **{len(single_value_cols)} columns** have only one unique value: {', '.join(single_value_cols)}")
    else:
        insights.append("✅ **No columns** with single values")
    
    # Display insights
    for insight in insights:
        st.write(insight)

    # Data statistics by type
    st.subheader("Statistical Summary")
    
    # Numerical columns
    numerical_cols = [col for col, dtype in column_types.items() 
                     if hasattr(dtype, 'value') and dtype.value == 'NUMERICAL']
    
    if numerical_cols:
        st.write("**Numerical Columns:**")
        st.dataframe(df[numerical_cols].describe(), use_container_width=True)
    
    # Categorical columns
    categorical_cols = [col for col, dtype in column_types.items() 
                       if hasattr(dtype, 'value') and dtype.value == 'CATEGORICAL']
    
    if categorical_cols:
        st.write("**Categorical Columns:**")
        for col in categorical_cols[:5]:  # Show first 5 categorical columns
            st.write(f"**{col}:**")
            value_counts = df[col].value_counts().head(10)
            st.bar_chart(value_counts)

if __name__ == "__main__":
    render_data_preview_section()
