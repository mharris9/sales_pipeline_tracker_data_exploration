"""
Filters Page - Manage and apply data filters
"""
import streamlit as st
import pandas as pd
from config.settings import FILTER_MAX_CATEGORIES

def render_filters_section():
    st.title("Data Filters")

    # Sticky controls marker and bar (will pin the next container)
    st.markdown('<div class="sticky-controls"></div>', unsafe_allow_html=True)
    with st.container():
        # This container is sticky due to CSS rule in load_custom_css
        colA, colB = st.columns([1,1])
        with colA:
            apply_click = st.button("Apply Filters", use_container_width=True)
        with colB:
            clear_click = st.button("Clear All Filters", use_container_width=True)

    # Check if data is loaded
    if not st.session_state.data_loaded or st.session_state.current_df is None:
        st.warning("No data loaded. Please upload data first.")
        return

    df = st.session_state.current_df
    column_types = st.session_state.column_types  # mapping column -> string type

    if clear_click:
        # Reset widgets to defaults when available
        for col, dtype_str in column_types.items():
            if dtype_str == 'numerical':
                st.session_state.pop(f"min_{col}", None)
                st.session_state.pop(f"max_{col}", None)
            elif dtype_str == 'categorical':
                st.session_state.pop(f"cat_{col}", None)
            elif dtype_str == 'date':
                st.session_state.pop(f"start_{col}", None)
                st.session_state.pop(f"end_{col}", None)
            elif dtype_str == 'boolean':
                st.session_state.pop(f"bool_{col}", None)
        st.session_state.filtered_df = df.copy()
        st.toast("✅ All filters cleared!", icon="✅")
        st.rerun()

    # Compact, scrollable filters region
    st.markdown('<div class="filters-compact">', unsafe_allow_html=True)
    st.subheader("Configure Filters")

    # 3 columns for filters
    col_containers = st.columns(3)
    col_idx = 0

    # Render each filter widget group (no form; values commit immediately)
    for column, dtype_str in column_types.items():
        with col_containers[col_idx % 3]:
            st.caption(f"{column} · {dtype_str}")
            
            if dtype_str == 'numerical':
                series_numeric = pd.to_numeric(df[column], errors='coerce')
                default_min = float(series_numeric.min()) if series_numeric.notna().any() else 0.0
                default_max = float(series_numeric.max()) if series_numeric.notna().any() else 0.0
                c1, c2 = st.columns(2)
                with c1:
                    st.number_input(f"Min {column}", value=default_min, key=f"min_{column}")
                with c2:
                    st.number_input(f"Max {column}", value=default_max, key=f"max_{column}")

            elif dtype_str == 'categorical':
                unique_values = df[column].dropna().unique().tolist()
                if len(unique_values) > FILTER_MAX_CATEGORIES:
                    # High-cardinality: provide text search instead of massive multiselect
                    search_text = st.text_input(f"Search {column}", key=f"cat_search_{column}")
                    if search_text:
                        st.session_state[f"cat_{column}"] = [search_text]
                else:
                    st.multiselect(f"Select {column} values", unique_values, key=f"cat_{column}")

            elif dtype_str == 'date':
                series_date = pd.to_datetime(df[column], errors='coerce')
                default_start = series_date.min().date() if series_date.notna().any() else None
                default_end = series_date.max().date() if series_date.notna().any() else None
                c1, c2 = st.columns(2)
                with c1:
                    st.date_input(f"Start {column}", value=default_start, key=f"start_{column}")
                with c2:
                    st.date_input(f"End {column}", value=default_end, key=f"end_{column}")

            elif dtype_str == 'boolean':
                st.checkbox(f"{column} is True", key=f"bool_{column}")
        col_idx += 1

    st.markdown('</div>', unsafe_allow_html=True)

    # Apply click computes cumulative mask using committed widget values
    if apply_click:
        try:
            mask = pd.Series(True, index=df.index)
            any_active = False
            for column, dtype_str in column_types.items():
                if dtype_str == 'categorical':
                    values = st.session_state.get(f"cat_{column}") or []
                    # When using search_text, interpret as contains filter
                    if not values:
                        search_text = st.session_state.get(f"cat_search_{column}")
                        if search_text:
                            any_active = True
                            mask &= df[column].astype(str).str.contains(str(search_text), case=False, na=False)
                    else:
                        any_active = True
                        mask &= df[column].isin(values)
                elif dtype_str == 'numerical':
                    series_numeric = pd.to_numeric(df[column], errors='coerce')
                    default_min = float(series_numeric.min()) if series_numeric.notna().any() else 0.0
                    default_max = float(series_numeric.max()) if series_numeric.notna().any() else 0.0
                    min_val = float(st.session_state.get(f"min_{column}", default_min))
                    max_val = float(st.session_state.get(f"max_{column}", default_max))
                    if min_val > max_val:
                        min_val, max_val = max_val, min_val
                    if (min_val > default_min) or (max_val < default_max):
                        any_active = True
                        mask &= pd.to_numeric(df[column], errors='coerce').between(min_val, max_val, inclusive='both')
                elif dtype_str == 'date':
                    series_date = pd.to_datetime(df[column], errors='coerce')
                    default_start = series_date.min().date() if series_date.notna().any() else None
                    default_end = series_date.max().date() if series_date.notna().any() else None
                    start_date = st.session_state.get(f"start_{column}", default_start)
                    end_date = st.session_state.get(f"end_{column}", default_end)
                    if start_date and end_date and ((start_date != default_start) or (end_date != default_end)):
                        any_active = True
                        mask &= series_date.dt.date.between(start_date, end_date)
                elif dtype_str == 'boolean':
                    true_only = bool(st.session_state.get(f"bool_{column}", False))
                    if true_only:
                        any_active = True
                        mask &= df[column] == True
            
            filtered_df = df[mask] if any_active else df.copy()
            # Ensure Arrow compatibility
            from src.services.data_handler import DataHandler
            st.session_state.filtered_df = DataHandler.ensure_arrow_compatible(filtered_df)
            st.toast(f"✅ Filters applied: {len(st.session_state.filtered_df)} records selected", icon="✅")
            st.rerun()
        except Exception as e:
            st.toast(f"❌ Error applying filters: {str(e)}", icon="❌")

if __name__ == "__main__":
    render_filters_section()
