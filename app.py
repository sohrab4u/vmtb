import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF

def clean_total_rows(df, village_col='Matched Village'):
    """
    Remove summary/total rows from the df.
    Usually these rows have 'Total' or 'Summary' in the village column.
    """
    df[village_col] = df[village_col].astype(str).str.strip()
    exclude_keywords = ['total', 'summary', 'grand total']
    # Filter out rows where village name matches excluded keywords
    mask = ~df[village_col].str.lower().isin(exclude_keywords)
    return df[mask]

def find_duplicates(series):
    dup = series[series.duplicated(keep=False)]
    return dup.str.lower().unique().tolist()

def generate_report(df1, df2):
    village_col_2 = df2.columns[1]

    df1['Matched Village'] = df1['Matched Village'].astype(str).str.strip()
    df1 = df1[df1['Matched Village'].notnull() & (df1['Matched Village'] != '')]
    df1 = clean_total_rows(df1, 'Matched Village')

    df2[village_col_2] = df2[village_col_2].astype(str).str.strip()
    df2 = df2[df2[village_col_2].notnull() & (df2[village_col_2] != '')]

    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)

    duplicates = find_duplicates(df1['Matched Village'])

    total_villages = len(df1)
    villages_2_set = set(df2[village_col_2].str.lower())

    def match_status(row):
        village_name = row['Matched Village'].lower()
        if village_name == '' or pd.isna(village_name):
            return 'Missing Village Name'
        elif village_name in villages_2_set:
            return 'Matched'
        else:
            return 'Not Matched'

    df1['Match Status'] = df1.apply(match_status, axis=1)

    matched_df = df1[df1['Match Status'] == 'Matched'].copy()
    unmatched_df = df1[df1['Match Status'] == 'Not Matched'].copy()
    missing_df = df1[df1['Match Status'] == 'Missing Village Name'].copy()

    numeric_cols = [
        'Population of Gram Panchayat (as per Dept. of Panchayati Raj)',
        'No. of Presumptive TB Tests done in  the GP from Jan 2025 to September 2025 (as per Lab Register and NAAT Register)',
        'Number of TB Patient diagnosed in the GP from Jan 2025 to September 2025  as per Nikshay/TB Register'
    ]

    for col in numeric_cols:
        for df in [matched_df, unmatched_df, missing_df]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

    population_matched = matched_df['Population of Gram Panchayat (as per Dept. of Panchayati Raj)'].sum()
    population_unmatched = unmatched_df['Population of Gram Panchayat (as per Dept. of Panchayati Raj)'].sum()

    presumptive_tb_tests_matched = matched_df['No. of Presumptive TB Tests done in  the GP from Jan 2025 to September 2025 (as per Lab Register and NAAT Register)'].sum()
    presumptive_tb_tests_unmatched = unmatched_df['No. of Presumptive TB Tests done in  the GP from Jan 2025 to September 2025 (as per Lab Register and NAAT Register)'].sum()

    patient_diagnosed_matched = matched_df['Number of TB Patient diagnosed in the GP from Jan 2025 to September 2025  as per Nikshay/TB Register'].sum()
    patient_diagnosed_unmatched = unmatched_df['Number of TB Patient diagnosed in the GP from Jan 2025 to September 2025  as per Nikshay/TB Register'].sum()

    total_tb_units = df1['Name of TB Unit'].astype(str).str.strip().nunique()

    matched_village_tb_unit_wise = matched_df.groupby('Name of TB Unit')['Matched Village'].count().to_dict()

    report = {
        'Total Village in TB Unit': total_villages,
        'Total Number of TB Units': total_tb_units,
        'Total Matched Village': len(matched_df),
        'Total Not Matched Village': len(unmatched_df),
        'Total Missing Village Name': len(missing_df),
        'Total Population of Matched Village': population_matched,
        'Total Population of Not Matched Village': population_unmatched,
        'Total Presumptive TB Tests Done in Matched Village': presumptive_tb_tests_matched,
        'Total Presumptive TB Tests Done in Not Matched Village': presumptive_tb_tests_unmatched,
        'Total Patient Diagnosed in Matched Village': patient_diagnosed_matched,
        'Total Patient Diagnosed in Not Matched Village': patient_diagnosed_unmatched,
        'Matched Village TB Unit wise': matched_village_tb_unit_wise,
        'Duplicate Villages in Detailed TB Data': duplicates
    }

    return report, matched_df, unmatched_df, missing_df

def plot_pie(labels, sizes, title, colors=None):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=140, shadow=True, textprops={'fontsize': 12})
    ax.axis('equal')
    st.pyplot(fig)

def main():
    st.set_page_config(layout="wide")
    st.title("VMTB Data Analytics Dashboard")

    uploaded_file1 = st.file_uploader("Upload First Excel File (Detailed TB Data)", type=['xlsx'])
    uploaded_file2 = st.file_uploader("Upload Second Excel File (Village List)", type=['xlsx'])

    if uploaded_file1 and uploaded_file2:
        df1 = pd.read_excel(uploaded_file1, header=0)
        df2 = pd.read_excel(uploaded_file2, header=0)

        df1['Name of TB Unit'] = df1['Name of TB Unit'].astype(str).str.strip()
        df1 = df1[df1['Name of TB Unit'].notnull() & (df1['Name of TB Unit'] != '')]

        report, matched_df, unmatched_df, missing_df = generate_report(df1, df2)

        st.sidebar.header("Filter by TB Unit")
        all_tb_units = ['All'] + sorted(df1['Name of TB Unit'].unique())
        selected_tb_unit = st.sidebar.selectbox("Select TB Unit", all_tb_units)

        st.header("Summary Statistics Pie Charts")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Village Counts")
            labels = ['Matched', 'Not Matched', 'Missing']
            sizes = [report['Total Matched Village'], report['Total Not Matched Village'], report['Total Missing Village Name']]
            colors = ['#4CAF50', '#F44336', '#FFC107']
            plot_pie(labels, sizes, "Village Counts Distribution", colors)

            st.subheader("Population Summary")
            labels = ['Matched', 'Not Matched']
            sizes = [report['Total Population of Matched Village'], report['Total Population of Not Matched Village']]
            colors = ['#2196F3', '#FF9800']
            plot_pie(labels, sizes, "Population Distribution", colors)

        with col2:
            st.subheader("Presumptive TB Tests Summary")
            labels = ['Matched', 'Not Matched']
            sizes = [report['Total Presumptive TB Tests Done in Matched Village'], report['Total Presumptive TB Tests Done in Not Matched Village']]
            colors = ['#9C27B0', '#E91E63']
            plot_pie(labels, sizes, "Presumptive TB Tests", colors)

            st.subheader("Patients Diagnosed Summary")
            labels = ['Matched', 'Not Matched']
            sizes = [report['Total Patient Diagnosed in Matched Village'], report['Total Patient Diagnosed in Not Matched Village']]
            colors = ['#3F51B5', '#00BCD4']
            plot_pie(labels, sizes, "Patients Diagnosed", colors)

        if selected_tb_unit == 'All':
            filtered_matched = matched_df
            filtered_unmatched = unmatched_df
            filtered_missing = missing_df
        else:
            filtered_matched = matched_df[matched_df['Name of TB Unit'] == selected_tb_unit]
            filtered_unmatched = unmatched_df[unmatched_df['Name of TB Unit'] == selected_tb_unit]
            filtered_missing = missing_df[missing_df['Name of TB Unit'] == selected_tb_unit]

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
            full_detail_df = pd.concat([
                filtered_matched.assign(Match_Status='Matched'),
                filtered_unmatched.assign(Match_Status='Not Matched'),
                filtered_missing.assign(Match_Status='Missing'),
            ])
            excel_bytes = BytesIO()
            with pd.ExcelWriter(excel_bytes, engine='xlsxwriter') as writer:
                full_detail_df.to_excel(writer, index=False, sheet_name='Details')
            excel_bytes.seek(0)
            st.download_button("Download Excel File", data=excel_bytes,
                               file_name="TB_Village_Detail_Report.xlsx",
                               mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        if st.button("Export Summary Report to PDF"):
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, "TB Unit Village Matching Report", ln=True, align='C')
            pdf.ln(5)
            for title, content in report.items():
                if isinstance(content, dict):
                    pdf.cell(0, 10, f"{title}:", ln=True)
                    for key, val in content.items():
                        pdf.cell(0, 10, f"  {key}: {val}", ln=True)
                elif isinstance(content, list):
                    pdf.cell(0, 10, f"{title} (Count: {len(content)}):", ln=True)
                    for item in content[:15]:
                        pdf.cell(0, 10, f"  {item}", ln=True)
                    if len(content) > 15:
                        pdf.cell(0, 10, f"  ... {len(content)-15} more not shown", ln=True)
                else:
                    pdf.cell(0, 10, f"{title}: {content}", ln=True)
            pdf_output = pdf.output(dest='S').encode('latin1')
            st.download_button("Download Summary PDF", data=pdf_output,
                               file_name="TB_Village_Summary_Report.pdf",
                               mime="application/pdf")

if __name__ == "__main__":
    main()