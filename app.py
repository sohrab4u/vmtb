import streamlit as st
import pandas as pd
from io import BytesIO
import matplotlib
matplotlib.use('Agg')  # Non-GUI backend for matplotlib on Streamlit Cloud
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

def to_excel(summary, unknowns_df=None):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    summary.to_excel(writer, index=False, sheet_name='Summary')
    if unknowns_df is not None and not unknowns_df.empty:
        unknowns_df.to_excel(writer, index=False, sheet_name='Unknown_Villages')
    writer.close()
    processed_data = output.getvalue()
    return processed_data

def to_pdf(summary, report_data, unknown_info):
    pdf_bytes = BytesIO()
    with PdfPages(pdf_bytes) as pdf:
        # Single Page: Text Summary on left, Pie Chart on right
        fig = plt.figure(figsize=(12, 8))
        
        # Left subplot for text
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.axis('off')
        full_text = '\n'.join(report_data) + '\n\n' + unknown_info
        ax1.text(0.05, 0.95, full_text, fontsize=10, va='top', wrap=True, transform=ax1.transAxes)
        ax1.set_title('Summary Report', fontsize=12, pad=20)
        
        # Right subplot for pie chart (Villages distribution)
        ax2 = fig.add_subplot(1, 2, 2)
        if 'Total_Villages' in summary.columns:
            risk_status = summary.iloc[:, 0]  # First column is risk_col
            villages = summary['Total_Villages']
            wedges, texts, autotexts = ax2.pie(villages, labels=risk_status, autopct='%1.1f%%', startangle=90)
            ax2.set_title('Distribution of Villages by Risk Status', fontsize=12, pad=20)
        else:
            ax2.text(0.5, 0.5, 'No Villages Data Available', ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Distribution of Villages by Risk Status', fontsize=12, pad=20)
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)
    pdf_bytes.seek(0)
    return pdf_bytes.read()

def main():
    st.title("VMTB Data Analysis Dashboard - Uttar Pradesh")

    st.markdown("Upload the two Excel files:")
    file1 = st.file_uploader("Upload VMTB Village List Excel", type=["xlsx"])
    file2 = st.file_uploader("Upload Matched Panchayat Data Excel", type=["xlsx"])

    if file1 and file2:
        df_villages = pd.read_excel(file1)
        df_panchayat = pd.read_excel(file2)

        # Strip column names
        df_villages.columns = df_villages.columns.str.strip()
        df_panchayat.columns = df_panchayat.columns.str.strip()

        # Display available columns for selection
        st.header("Column Mapping")
        st.write("Please map the required columns from your uploaded files. If a column is not found, select the appropriate one from the dropdown.")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("VMTB Village List Columns")
            village_options = df_villages.columns.tolist() + ['Not Available']
            default_village_idx = df_villages.columns.tolist().index('Village Name (VMTB_List)') if 'Village Name (VMTB_List)' in df_villages.columns else len(df_villages.columns)
            selected_village_col = st.selectbox("Village Name Column:", options=village_options, index=default_village_idx)

            risk_options = df_villages.columns.tolist() + ['Not Available']
            default_risk_idx = df_villages.columns.tolist().index('Risk_Status') if 'Risk_Status' in df_villages.columns else len(df_villages.columns)
            selected_risk_col = st.selectbox("Risk Status Column:", options=risk_options, index=default_risk_idx)

            lat_options = df_villages.columns.tolist() + ['Not Available']
            default_lat_idx = df_villages.columns.tolist().index('Latitude') if 'Latitude' in df_villages.columns else len(df_villages.columns)
            selected_lat_col = st.selectbox("Latitude Column:", options=lat_options, index=default_lat_idx)

            lon_options = df_villages.columns.tolist() + ['Not Available']
            default_lon_idx = df_villages.columns.tolist().index('Longitude') if 'Longitude' in df_villages.columns else len(df_villages.columns)
            selected_lon_col = st.selectbox("Longitude Column:", options=lon_options, index=default_lon_idx)

        with col2:
            st.subheader("Matched Panchayat Data Columns")
            matched_options = df_panchayat.columns.tolist() + ['Not Available']
            default_matched_idx = df_panchayat.columns.tolist().index('Matched Village') if 'Matched Village' in df_panchayat.columns else len(df_panchayat.columns)
            selected_matched_col = st.selectbox("Matched Village Column:", options=matched_options, index=default_matched_idx)

            pop_options = df_panchayat.columns.tolist() + ['Not Available']
            default_pop_idx = df_panchayat.columns.tolist().index('Population of Gram Panchayat (as per Dept. of Panchayati Raj)') if 'Population of Gram Panchayat (as per Dept. of Panchayati Raj)' in df_panchayat.columns else len(df_panchayat.columns)
            selected_pop_col = st.selectbox("Population Column:", options=pop_options, index=default_pop_idx)

            tests_options = df_panchayat.columns.tolist() + ['Not Available']
            default_tests_idx = df_panchayat.columns.tolist().index('No. of Presumptive TB Tests done in  the GP from Jan 2025 to September 2025 (as per Lab Register and NAAT Register)') if 'No. of Presumptive TB Tests done in  the GP from Jan 2025 to September 2025 (as per Lab Register and NAAT Register)' in df_panchayat.columns else len(df_panchayat.columns)
            selected_tests_col = st.selectbox("Presumptive Tests Column:", options=tests_options, index=default_tests_idx)

            patients_options = df_panchayat.columns.tolist() + ['Not Available']
            default_patients_idx = df_panchayat.columns.tolist().index('Number of TB Patient diagnosed in the GP from Jan 2025 to September 2025  as per Nikshay/TB Register') if 'Number of TB Patient diagnosed in the GP from Jan 2025 to September 2025  as per Nikshay/TB Register' in df_panchayat.columns else len(df_panchayat.columns)
            selected_patients_col = st.selectbox("Diagnosed Patients Column:", options=patients_options, index=default_patients_idx)

        # Validation for required columns
        if selected_village_col == 'Not Available' or selected_matched_col == 'Not Available' or selected_risk_col == 'Not Available':
            st.error("Required columns (Village Name, Matched Village, Risk Status) must be selected from the dropdowns.")
            st.stop()

        # Availability flags for optional columns
        pop_available = selected_pop_col != 'Not Available'
        tests_available = selected_tests_col != 'Not Available'
        patients_available = selected_patients_col != 'Not Available'
        lat_available = selected_lat_col != 'Not Available'
        lon_available = selected_lon_col != 'Not Available'

        if not (lat_available and lon_available):
            st.warning("Latitude and Longitude columns not selected. Map visualization will not be displayed.")

        # Assign variables
        village_col = selected_village_col
        risk_col = selected_risk_col
        matched_village_col = selected_matched_col
        population_col = selected_pop_col
        presumptive_tests_col = selected_tests_col
        diagnosed_patients_col = selected_patients_col
        lat_col = selected_lat_col
        lon_col = selected_lon_col

        # Convert numeric columns in panchayat before grouping
        if pop_available:
            df_panchayat[population_col] = pd.to_numeric(df_panchayat[population_col], errors='coerce').fillna(0)
        if tests_available:
            df_panchayat[presumptive_tests_col] = pd.to_numeric(df_panchayat[presumptive_tests_col], errors='coerce').fillna(0)
        if patients_available:
            df_panchayat[diagnosed_patients_col] = pd.to_numeric(df_panchayat[diagnosed_patients_col], errors='coerce').fillna(0)

        # Group panchayat data by matched_village_col to sum numerics per unique GP
        agg_dict_p = {}
        if pop_available:
            agg_dict_p[population_col] = 'sum'
        if tests_available:
            agg_dict_p[presumptive_tests_col] = 'sum'
        if patients_available:
            agg_dict_p[diagnosed_patients_col] = 'sum'

        grouped_panchayat = df_panchayat.groupby(matched_village_col).agg(agg_dict_p).reset_index()

        # Now merge: all villages left join with grouped panchayat
        merged = pd.merge(
            df_villages, grouped_panchayat,
            left_on=village_col,
            right_on=matched_village_col,
            how='left'
        )

        # Fill NaN in numerics with 0
        if pop_available:
            merged[population_col] = merged[population_col].fillna(0)
        if tests_available:
            merged[presumptive_tests_col] = merged[presumptive_tests_col].fillna(0)
        if patients_available:
            merged[diagnosed_patients_col] = merged[diagnosed_patients_col].fillna(0)

        # Fill risk with 'Unknown' if NaN
        merged[risk_col] = merged[risk_col].fillna('Unknown')

        risk_options = merged[risk_col].unique().tolist()
        selected_risk = st.multiselect("Select Risk Status to filter report", risk_options, default=risk_options)

        filtered = merged[merged[risk_col].isin(selected_risk)].copy()

        # Compute Unknowns for non-matching info (from full merged, independent of filter)
        unknown_filter = merged[merged[risk_col] == 'Unknown']
        num_unknown = len(unknown_filter)
        unknown_cols = [village_col]
        if lat_available:
            unknown_cols.append(lat_col)
        if lon_available:
            unknown_cols.append(lon_col)
        if pop_available:
            unknown_cols.append(population_col)
        if tests_available:
            unknown_cols.append(presumptive_tests_col)
        if patients_available:
            unknown_cols.append(diagnosed_patients_col)
        unknowns_df = unknown_filter[unknown_cols].copy()

        # Check if required columns exist
        if risk_col not in filtered.columns:
            st.error(f"Risk column '{risk_col}' not found after merge.")
            st.stop()
        if village_col not in filtered.columns:
            st.error(f"Village column '{village_col}' not found.")
            st.stop()

        # Data Overview - Compute totals from raw files for comparison
        st.header("Data Overview (Raw Totals)")
        st.write(f"Total Villages in VMTB List: {len(df_villages)}")

        unique_matched_gps = len(grouped_panchayat)
        st.write(f"Unique Matched GPs in Panchayat Data: {unique_matched_gps}")

        if pop_available:
            raw_pop_total = grouped_panchayat[population_col].sum()
            st.write(f"Total Population (sum per unique GP): {raw_pop_total:.0f}")

        if tests_available:
            raw_tests_total = grouped_panchayat[presumptive_tests_col].sum()
            st.write(f"Total Presumptive Tests (sum per unique GP): {raw_tests_total:.0f}")

        if patients_available:
            raw_patients_total = grouped_panchayat[diagnosed_patients_col].sum()
            st.write(f"Total Diagnosed Patients (sum per unique GP): {raw_patients_total:.0f}")

        # Build summary step by step
        summary = filtered.groupby(risk_col).agg(
            Total_Villages=(village_col, 'count')
        ).reset_index()

        if pop_available and population_col in filtered.columns:
            pop_sum = filtered.groupby(risk_col)[population_col].sum().reset_index(name='Total_Population')
            summary = summary.merge(pop_sum, on=risk_col, how='left')

        if tests_available and presumptive_tests_col in filtered.columns:
            tests_sum = filtered.groupby(risk_col)[presumptive_tests_col].sum().reset_index(name='Total_Presumptive_Tests')
            summary = summary.merge(tests_sum, on=risk_col, how='left')

        if patients_available and diagnosed_patients_col in filtered.columns:
            patients_sum = filtered.groupby(risk_col)[diagnosed_patients_col].sum().reset_index(name='Total_Diagnosed_Patients')
            summary = summary.merge(patients_sum, on=risk_col, how='left')

        st.header("Summary Dashboard by Risk Status")
        st.dataframe(summary)

        # Non-Matching Data Section
        st.subheader(f"Non-Matching Data: {num_unknown} Unknown Villages")
        if num_unknown > 0:
            st.dataframe(unknowns_df)
        else:
            st.success("All villages matched successfully!")

        # Grand Totals from Analysis for Comparison
        st.subheader("Grand Totals from Analysis")
        analysis_villages_total = summary['Total_Villages'].sum()
        st.write(f"Analysis Total Villages (across all risks): {analysis_villages_total}")

        if 'Total_Population' in summary.columns:
            analysis_pop_total = summary['Total_Population'].sum()
            st.write(f"Analysis Total Population (sum across all risks): {analysis_pop_total:.0f}")

        if 'Total_Presumptive_Tests' in summary.columns:
            analysis_tests_total = summary['Total_Presumptive_Tests'].sum()
            st.write(f"Analysis Total Presumptive Tests (sum across all risks): {analysis_tests_total:.0f}")

        if 'Total_Diagnosed_Patients' in summary.columns:
            analysis_patients_total = summary['Total_Diagnosed_Patients'].sum()
            st.write(f"Analysis Total Diagnosed Patients (sum across all risks): {analysis_patients_total:.0f}")

        # Interactive Charts
        st.header("Interactive Charts")

        # Bar chart for Total Population
        if 'Total_Population' in summary.columns:
            fig_pop = px.bar(summary, x=risk_col, y='Total_Population', 
                             title='Total Population by Risk Status',
                             color='Total_Population',
                             labels={risk_col: 'Risk Status', 'Total_Population': 'Population'})
            st.plotly_chart(fig_pop, use_container_width=True)

        # Bar chart for Total Presumptive Tests
        if 'Total_Presumptive_Tests' in summary.columns:
            fig_tests = px.bar(summary, x=risk_col, y='Total_Presumptive_Tests', 
                               title='Total Presumptive Tests by Risk Status',
                               color='Total_Presumptive_Tests',
                               labels={risk_col: 'Risk Status', 'Total_Presumptive_Tests': 'Presumptive Tests'})
            st.plotly_chart(fig_tests, use_container_width=True)

        # Bar chart for Total Diagnosed Patients
        if 'Total_Diagnosed_Patients' in summary.columns:
            fig_patients = px.bar(summary, x=risk_col, y='Total_Diagnosed_Patients', 
                                  title='Total Diagnosed Patients by Risk Status',
                                  color='Total_Diagnosed_Patients',
                                  labels={risk_col: 'Risk Status', 'Total_Diagnosed_Patients': 'Diagnosed Patients'})
            st.plotly_chart(fig_patients, use_container_width=True)

        # Pie chart for Total Villages distribution
        if 'Total_Villages' in summary.columns:
            fig_pie = px.pie(summary, values='Total_Villages', names=risk_col,
                             title='Distribution of Villages by Risk Status')
            st.plotly_chart(fig_pie, use_container_width=True)

        # Combined subplot for all metrics if all available
        if all(col in summary.columns for col in ['Total_Population', 'Total_Presumptive_Tests', 'Total_Diagnosed_Patients']):
            fig_sub = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Total Population', 'Total Presumptive Tests', 
                               'Total Diagnosed Patients', 'Total Villages'),
                specs=[[{"type": "bar"}, {"type": "bar"}],
                       [{"type": "bar"}, {"type": "pie"}]]
            )

            fig_sub.add_trace(
                go.Bar(x=summary[risk_col], y=summary['Total_Population'], name='Population'), row=1, col=1
            )
            fig_sub.add_trace(
                go.Bar(x=summary[risk_col], y=summary['Total_Presumptive_Tests'], name='Tests'), row=1, col=2
            )
            fig_sub.add_trace(
                go.Bar(x=summary[risk_col], y=summary['Total_Diagnosed_Patients'], name='Patients'), row=2, col=1
            )
            fig_sub.add_trace(
                go.Pie(labels=summary[risk_col], values=summary['Total_Villages'], name='Villages'), row=2, col=2
            )

            fig_sub.update_layout(height=800, showlegend=False, title_text="TB Metrics Dashboard")
            st.plotly_chart(fig_sub, use_container_width=True)

        # Interactive Map Visualization
        st.header("Interactive Map Visualization")
        if lat_available and lon_available and lat_col in filtered.columns and lon_col in filtered.columns and not filtered.empty:
            # Ensure numeric for lat/lon
            filtered[lat_col] = pd.to_numeric(filtered[lat_col], errors='coerce')
            filtered[lon_col] = pd.to_numeric(filtered[lon_col], errors='coerce')
            filtered = filtered.dropna(subset=[lat_col, lon_col])

            if not filtered.empty:
                hover_data = {}
                if pop_available and population_col in filtered.columns:
                    hover_data[population_col] = ':.0f'
                if tests_available and presumptive_tests_col in filtered.columns:
                    hover_data[presumptive_tests_col] = ':.0f'
                if patients_available and diagnosed_patients_col in filtered.columns:
                    hover_data[diagnosed_patients_col] = ':.0f'

                size_col = population_col if pop_available and population_col in filtered.columns else None

                fig_map = px.scatter_mapbox(filtered, 
                                            lat=lat_col, 
                                            lon=lon_col, 
                                            color=risk_col,
                                            size=size_col,
                                            hover_name=village_col,
                                            hover_data=hover_data,
                                            mapbox_style="open-street-map",
                                            zoom=5,
                                            height=600,
                                            title="Villages by Risk Status on Map")
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.warning("No valid latitude/longitude data available for mapping.")
        else:
            st.warning("Latitude and Longitude columns not available or no data after filtering. Cannot display map.")

        excel_data = to_excel(summary, unknowns_df)
        st.download_button(
            label="Download Report as Excel",
            data=excel_data,
            file_name='TB_Risk_Status_Report.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

        pdf_lines = []
        for _, row in summary.iterrows():
            line = f"Risk Status: {row[risk_col]}\n"
            line += f"Total Villages: {int(row['Total_Villages'])}\n"
            if 'Total_Population' in summary.columns:
                line += f"Total Population: {int(row['Total_Population'])}\n"
            if 'Total_Presumptive_Tests' in summary.columns:
                line += f"Total Presumptive Tests: {int(row['Total_Presumptive_Tests'])}\n"
            if 'Total_Diagnosed_Patients' in summary.columns:
                line += f"Total Diagnosed Patients: {int(row['Total_Diagnosed_Patients'])}\n"
            pdf_lines.append(line)

        # Unknown info for PDF
        unknown_info_lines = [""]
        unknown_info_lines.append(f"Non-Matching Data Summary:")
        unknown_info_lines.append(f"Number of Unknown Villages (no matching GP data): {num_unknown}")
        if num_unknown > 0:
            unknown_info_lines.append("List of Unknown Villages:")
            for _, row in unknowns_df.iterrows():
                unknown_info_lines.append(f"- {row[village_col]}")
        unknown_info = '\n'.join(unknown_info_lines)

        pdf_data = to_pdf(summary, pdf_lines, unknown_info)

        st.download_button(
            label="Download Report as PDF",
            data=pdf_data,
            file_name='TB_Risk_Status_Report.pdf',
            mime='application/pdf'
        )

if __name__ == "__main__":
    main()