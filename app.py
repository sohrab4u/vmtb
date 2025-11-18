import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO

# Use exact names as per your working sheet and summary export columns
SUMMARY_COLS = {
    "Population": "Population of Gram Panchayat (as per Dept. of Panchayati Raj)",
    "Presumptive": "No. of Presumptive TB Tests done in the GP from Jan 2025 to 31 October 2025 (as per Lab Register and NAAT Register)",
    "Diagnosed": "Number of TB Patient diagnosed in the GP from Jan 2025 to 31 October 2025 as per Nikshay/TB Register",
    "UDST": "No. of TB Patient's UDST done from (1st Jan 2025 to 31 October 2025 2025 )",
    "NPY_DBT": "No. of TB Patient received NPY -DBT (At least 1st installment )",
    "Poshan": "No. of TB Patient received Poshan Potaly support under PMTBMBA"
}

def clean_total_rows(df, village_col):
    df[village_col] = df[village_col].astype(str).str.strip()
    exclude_keywords = ['total', 'summary', 'grand total']
    mask = ~df[village_col].str.lower().isin(exclude_keywords)
    return df[mask]

def find_duplicates(series):
    dup = series[series.duplicated(keep=False)]
    return dup.str.lower().unique().tolist()

def generate_report(df1, df2):
    cols1 = list(df1.columns)
    try:
        tb_unit_col = [c for c in cols1 if "tb unit" in c.lower()][0]
        village_col_1 = [c for c in cols1 if "gram panchayat" in c.lower()][0]
    except IndexError as e:
        raise KeyError("Could not find TB Unit or Gram Panchayat columns. Found: " + str(cols1)) from e

    # All summary columns must exist
    for label, col in SUMMARY_COLS.items():
        if col not in cols1:
            raise KeyError(f"Column for {label} not found. Expected: '{col}'.\nFound columns: {cols1}")

    population_col = SUMMARY_COLS["Population"]
    presumptive_col = SUMMARY_COLS["Presumptive"]
    diagnosed_col = SUMMARY_COLS["Diagnosed"]
    udst_col = SUMMARY_COLS["UDST"]
    npy_dbt_col = SUMMARY_COLS["NPY_DBT"]
    poshan_col = SUMMARY_COLS["Poshan"]

    df1[village_col_1] = df1[village_col_1].astype(str).str.strip()
    df1 = df1[df1[village_col_1].notnull() & (df1[village_col_1] != '')]
    df1 = clean_total_rows(df1, village_col_1)
    df1 = df1.reset_index(drop=True)

    cols2 = list(df2.columns)
    village_col_2 = cols2[1]
    df2[village_col_2] = df2[village_col_2].astype(str).str.strip()
    df2 = df2[df2[village_col_2].notnull() & (df2[village_col_2] != '')]
    df2 = clean_total_rows(df2, village_col_2)
    df2 = df2.reset_index(drop=True)

    duplicates = find_duplicates(df1[village_col_1])
    total_villages = len(df1)
    villages_2_set = set(df2[village_col_2].str.lower())

    def match_status(row):
        village_name = row[village_col_1].lower()
        if village_name == '' or pd.isna(village_name):
            return 'Missing Village Name'
        elif village_name in villages_2_set:
            return 'Matched'
        else:
            return 'Not Matched'
    df1['Match Status'] = df1.apply(match_status, axis=1)

    matched_df   = df1[df1['Match Status'] == 'Matched'].copy()
    unmatched_df = df1[df1['Match Status'] == 'Not Matched'].copy()
    missing_df   = df1[df1['Match Status'] == 'Missing Village Name'].copy()

    # Convert all columns of interest to numeric
    for df in [matched_df, unmatched_df, missing_df]:
        for col in SUMMARY_COLS.values():
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    total_tb_units = df1[tb_unit_col].astype(str).str.strip().nunique()
    matched_village_tb_unit_wise = matched_df.groupby(tb_unit_col)[village_col_1].count().to_dict()

    # Only summary calculations below!
    report = {
        'Total Village in TB Unit': total_villages,
        'Total Number of TB Units': total_tb_units,
        'Total Matched Village': len(matched_df),
        'Total Not Matched Village': len(unmatched_df),
        'Total Missing Village Name': len(missing_df),
        f"{population_col} - Matched": matched_df[population_col].sum(),
        f"{population_col} - Not Matched": unmatched_df[population_col].sum(),
        f"{presumptive_col} - Matched": matched_df[presumptive_col].sum(),
        f"{presumptive_col} - Not Matched": unmatched_df[presumptive_col].sum(),
        f"{diagnosed_col} - Matched": matched_df[diagnosed_col].sum(),
        f"{diagnosed_col} - Not Matched": unmatched_df[diagnosed_col].sum(),
        f"{udst_col} - Matched": matched_df[udst_col].sum(),
        f"{udst_col} - Not Matched": unmatched_df[udst_col].sum(),
        f"{npy_dbt_col} - Matched": matched_df[npy_dbt_col].sum(),
        f"{npy_dbt_col} - Not Matched": unmatched_df[npy_dbt_col].sum(),
        f"{poshan_col} - Matched": matched_df[poshan_col].sum(),
        f"{poshan_col} - Not Matched": unmatched_df[poshan_col].sum(),
        'Matched Village TB Unit wise': matched_village_tb_unit_wise,
        'Duplicate Villages in Detailed TB Data': duplicates
    }

    export_cols = [tb_unit_col, village_col_1] + list(SUMMARY_COLS.values()) + ['Match Status']
    return report, matched_df, unmatched_df, missing_df, tb_unit_col, export_cols

def plot_pie(labels, sizes, title, colors=None):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140, shadow=True, textprops={'fontsize': 12})
    ax.axis('equal')
    plt.title(title)
    st.pyplot(fig)

def main():
    st.set_page_config(layout="wide")
    st.title("TB Village Matching and Analytics Dashboard")
    uploaded_file1 = st.file_uploader("Upload First Excel File (Detailed TB Data)", type=["xlsx"])
    uploaded_file2 = st.file_uploader("Upload Second Excel File (Village List)", type=["xlsx"])

    if uploaded_file1 and uploaded_file2:
        df1 = pd.read_excel(uploaded_file1, header=0)
        df2 = pd.read_excel(uploaded_file2, header=0)

        try:
            report, matched_df, unmatched_df, missing_df, tb_unit_col, export_cols = generate_report(df1, df2)
        except KeyError as e:
            st.write(f"**ERROR:** {e}")
            st.write(list(df1.columns))
            return

        st.sidebar.header("Filter by TB Unit")
        all_tb_units = ["All"] + sorted(df1[tb_unit_col].astype(str).str.strip().unique())
        selected_tb_unit = st.sidebar.selectbox("Select TB Unit", all_tb_units)

        # -------------------------------------------------
        # PLOT PIE CHARTS SECTION (restored!)
        # -------------------------------------------------
        st.header("Summary Statistics Pie Charts")
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Village Counts")
            labels = ["Matched", "Not Matched", "Missing"]
            sizes = [
                report["Total Matched Village"],
                report["Total Not Matched Village"],
                report["Total Missing Village Name"]
            ]
            colors = ['#4CAF50', '#F44336', '#FFC107']
            plot_pie(labels, sizes, "Village Counts Distribution", colors)

            st.subheader("Population Summary")
            labels = ["Matched", "Not Matched"]
            pop_matched = report[f"{SUMMARY_COLS['Population']} - Matched"]
            pop_unmatched = report[f"{SUMMARY_COLS['Population']} - Not Matched"]
            sizes = [pop_matched, pop_unmatched]
            colors = ['#2196F3', '#FF9800']
            plot_pie(labels, sizes, "Population Distribution", colors)
        with col2:
            st.subheader("Presumptive TB Tests Summary")
            labels = ["Matched", "Not Matched"]
            pres_matched = report[f"{SUMMARY_COLS['Presumptive']} - Matched"]
            pres_unmatched = report[f"{SUMMARY_COLS['Presumptive']} - Not Matched"]
            sizes = [pres_matched, pres_unmatched]
            colors = ['#9C27B0', '#E91E63']
            plot_pie(labels, sizes, "Presumptive TB Tests", colors)

            st.subheader("Patients Diagnosed Summary")
            labels = ["Matched", "Not Matched"]
            diag_matched = report[f"{SUMMARY_COLS['Diagnosed']} - Matched"]
            diag_unmatched = report[f"{SUMMARY_COLS['Diagnosed']} - Not Matched"]
            sizes = [diag_matched, diag_unmatched]
            colors = ['#3F51B5', '#00BCD4']
            plot_pie(labels, sizes, "Patients Diagnosed", colors)

        # -------------------------------------------------

        if selected_tb_unit != "All":
            filtered_matched = matched_df[matched_df[tb_unit_col] == selected_tb_unit]
            filtered_unmatched = unmatched_df[unmatched_df[tb_unit_col] == selected_tb_unit]
            filtered_missing = missing_df[missing_df[tb_unit_col] == selected_tb_unit]
        else:
            filtered_matched = matched_df
            filtered_unmatched = unmatched_df
            filtered_missing = missing_df

        st.header(f"Matched Villages Detail - {selected_tb_unit}")
        st.dataframe(filtered_matched)
        st.header(f"Not Matched Villages Detail - {selected_tb_unit}")
        st.dataframe(filtered_unmatched)
        st.header(f"Missing Village Name Detail - {selected_tb_unit}")
        st.dataframe(filtered_missing)

        st.header("Summary Report Details")
        for title, content in report.items():
            if isinstance(content, dict):
                st.markdown(f"**{title}:**")
                for key, val in content.items():
                    st.write(f"- {key}: {val}")
            elif isinstance(content, list):
                st.markdown(f"**{title} (Count: {len(content)}):**")
                for i, item in enumerate(content):
                    if i < 15:
                        st.write(f"- {item}")
                    elif i == 15:
                        st.write(f"... ({len(content) - 15} more not shown)")
                        break
            else:
                st.markdown(f"**{title}:** {content}")

        if st.button("Export Full Detail to Excel"):
            matched_export = filtered_matched[export_cols].copy()
            unmatched_export = filtered_unmatched[export_cols].copy()
            missing_export = filtered_missing[export_cols].copy()
            full_detail_df = pd.concat([matched_export, unmatched_export, missing_export])
            excel_bytes = BytesIO()
            with pd.ExcelWriter(excel_bytes, engine="xlsxwriter") as writer:
                full_detail_df.to_excel(writer, index=False, sheet_name="Details")
                matched_export.to_excel(writer, index=False, sheet_name="Matched Villages")
                unmatched_export.to_excel(writer, index=False, sheet_name="Not Matched Villages")
                missing_export.to_excel(writer, index=False, sheet_name="Missing Villages")
            excel_bytes.seek(0)
            st.download_button("Download Excel File", data=excel_bytes,
                               file_name="TB_Village_Detail_Report.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if __name__ == "__main__":
    main()