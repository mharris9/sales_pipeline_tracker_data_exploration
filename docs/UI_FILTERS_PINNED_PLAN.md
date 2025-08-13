### Pinned Metrics, Sticky Filter Controls, Compact 3‑Column Filters – Implementation Plan

- [ ] Sticky metrics header (Total/Selected/Memory)
  - [ ] Render immediately after data load (no filter dependency)
  - [ ] Selected Records = len(filtered_df) if present else len(current_df)
  - [ ] Pin using a robust sticky wrapper that targets Streamlit’s scroll container
  - [ ] Test across tabs for overlap/flicker

- [ ] Sticky filter control bar
  - [ ] Always visible "Apply Filters" and "Clear All Filters" buttons
  - [ ] No form; read widget values from session_state on click
  - [ ] Apply: build cumulative mask; set filtered_df; rerun
  - [ ] Clear: reset widget defaults and filtered_df; rerun

- [ ] Compact 3‑column filter layout
  - [ ] Create 3 columns once; distribute column filters round‑robin
  - [ ] Tight spacing (caption labels, 2-up rows for min/max and start/end)
  - [ ] Scrollable container directly under sticky controls
  - [ ] High-cardinality categorical handling: text search when unique_count > threshold

- [ ] Filter defaults and reset
  - [ ] On data load, compute and store serializable defaults per column (min/max; earliest/latest; empty set; false)
  - [ ] Clear All resets widget keys to defaults (and filtered_df)

- [ ] Serialization safety
  - [ ] Ensure `column_types` in session_state are strings (not enums)
  - [ ] Audit all consumers to treat types as strings

- [ ] Arrow compatibility hardening
  - [ ] Add `ensure_arrow_compatible(df)` utility in `DataHandler`
  - [ ] Apply after `_process_data` and before assigning `filtered_df`
  - [ ] Add defensive conversion in data preview before `st.dataframe`
  - [ ] Log columns coerced for Arrow

- [ ] Testing and acceptance
  - [ ] Manual test with large dataset; Region=South ≈ 7,800 selected; header updates
  - [ ] Verify sticky bars stay visible while scrolling dense 3‑column grid
  - [ ] Verify Clear All resets UI widgets and counts
  - [ ] Verify data preview renders without Arrow errors
  - [ ] Add quick unit for mask building with mixed types

- [ ] Configuration and docs
  - [ ] Add `FILTER_MAX_CATEGORIES` to `config/settings.py`
  - [ ] Document sticky behavior, filter defaults, high-cardinality strategy, and Arrow compatibility guard in `docs/LESSONS_LEARNED.md`
