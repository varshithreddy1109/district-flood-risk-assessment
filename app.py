import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import predict

# --- Page Configuration ---
st.set_page_config(
    page_title="District Flood Risk Assessment",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Caching ---
@st.cache_resource
def load_and_train_model():
    # Train the model once and cache the pipeline
    predict.train_model()
    return predict.get_model_metadata()

@st.cache_data
def get_districts_list():
    return predict.get_all_districts()

@st.cache_data
def load_shap_data():
    shap_path = os.path.join("outputs", "shap_feature_importance.csv")
    if os.path.exists(shap_path):
        return pd.read_csv(shap_path)
    return None

@st.cache_data
def load_rainfall_data():
    rain_path = os.path.join("data", "rainfall", "Rainfall_DistrictWise_Daily_IMD.csv")
    if os.path.exists(rain_path):
        try:
            return pd.read_csv(rain_path)
        except:
            return None
    return None

# --- Main App ---
def main():
    # Load resources
    try:
        model_meta = load_and_train_model()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    districts_data = get_districts_list()
    df_districts = pd.DataFrame(districts_data)

    # 1. HEADER
    st.title("District Flood Risk Assessment & Decision Support System")
    st.markdown("### AI-assisted classification of historical district flood-risk patterns across India")
    
    st.warning("**Historical DFSI-derived classification — not a future flood prediction system.**")
    st.markdown("---")

    # 2. SIDEBAR
    st.sidebar.header("Location Selection")
    
    # State selector
    states = sorted(df_districts["state"].unique())
    selected_state = st.sidebar.selectbox("Select State", states)

    # District selector based on state
    districts_in_state = sorted(df_districts[df_districts["state"] == selected_state]["district"].unique())
    selected_district = st.sidebar.selectbox("Select District", districts_in_state)
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("About Model")
    st.sidebar.info(
        f"**Model:** {model_meta['model_name']}\n\n"
        f"**Target:** {model_meta['target_definition']}\n\n"
        f"**Model development:** {model_meta['training_rows']} usable district records\n\n"
        f"**Classes:** {', '.join(model_meta['class_order'])}"
    )

    if not selected_district or not selected_state:
        st.info("Please select a state and district from the sidebar.")
        return

    # Fetch Prediction and Data
    try:
        pred_result = predict.predict_district(selected_district, selected_state)
        dist_data = predict.get_district_data(selected_district, selected_state)
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return

    # 3. RISK OVERVIEW
    st.header("Risk Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    predicted_class = pred_result["predicted_risk_class"]
    probs = pred_result["class_probabilities"]
    max_prob = probs.get(predicted_class, 0.0)
    
    # Format DFSI gracefully
    dfsi_val = dist_data.get("dfsi")
    dfsi_display = f"{dfsi_val:.2f}" if dfsi_val is not None else "N/A"

    # Color coding based on risk class
    color_map = {
        "Low": "green",
        "Moderate": "orange",
        "High": "darkorange",
        "Very High": "red"
    }
    color = color_map.get(predicted_class, "black")
    
    col1.metric("Selected Location", f"{selected_district}, {selected_state}")
    col2.markdown(f"**Predicted Risk Class**<br><h2 style='color:{color}; margin-top:0px;'>{predicted_class}</h2>", unsafe_allow_html=True)
    col3.metric("Confidence (Probability)", f"{max_prob*100:.1f}%")
    col4.metric("Reference DFSI", dfsi_display)
    
    st.caption("*DFSI is shown as the reference target used to derive the historical risk classes. It is NOT used as a model input.*")
    
    st.markdown("---")

    # 4. CLASS PROBABILITIES
    st.header("Class Probabilities")
    
    # Create a horizontal bar chart of probabilities
    prob_df = pd.DataFrame(
        {"Probability": list(probs.values())}, 
        index=list(probs.keys())
    )
    # Ensure standard ordering
    prob_df = prob_df.reindex(model_meta['class_order'])
    
    fig, ax = plt.subplots(figsize=(10, 3))
    colors = ['#2ca02c', '#ff7f0e', '#d62728', '#8c564b'] # colors for Low, Mod, High, VHigh
    bars = ax.barh(prob_df.index, prob_df['Probability'], color=colors, alpha=0.7)
    
    ax.set_xlim(0, 1.0)
    ax.set_xlabel('Probability')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.1%}', 
                va='center', ha='left', fontsize=10)
    
    st.pyplot(fig)
    
    st.markdown("---")

    # 5. KEY HISTORICAL FLOOD INDICATORS
    st.header("Key Historical Flood Indicators")
    st.markdown("Historical indicators for the selected district derived from the available flood inventory and physical-area datasets.")
    
    def format_val(val, fmt="{:.2f}"):
        if val is None or pd.isna(val):
            return "N/A"
        return fmt.format(val)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Flood Event Count", format_val(dist_data.get('flood_event_count'), "{:.0f}"))
    k2.metric("Avg Flood Duration", format_val(dist_data.get('avg_flood_duration')) + " days")
    k3.metric("Max Flood Duration", format_val(dist_data.get('max_flood_duration')) + " days")
    k4.metric("Median Flood Duration", format_val(dist_data.get('median_flood_duration')) + " days")
    
    k5, k6, k7, k8 = st.columns(4)
    k5.metric("Std Dev Duration", format_val(dist_data.get('std_flood_duration')))
    k6.metric("Percent Flooded Area", format_val(dist_data.get('percent_flooded_area')) + "%")
    k7.metric("Corrected Flooded Area", format_val(dist_data.get('corrected_percent_flooded_area')) + "%")
    k8.metric("Permanent Water", format_val(dist_data.get('permanent_water')) + "%")
    
    k9, k10, k11, k12 = st.columns(4)
    k9.metric("Human Displaced", format_val(dist_data.get('total_human_displaced'), "{:.0f}"))
    k10.metric("Animal Fatalities", format_val(dist_data.get('total_animal_fatalities'), "{:.0f}"))
    k11.metric("Human Fatalities", format_val(dist_data.get('impact_human_fatalities'), "{:.0f}")) # Derived metric shown for context
    k12.metric("Population", format_val(dist_data.get('population'), "{:.0f}"))
    
    st.markdown("---")

    # 6. WHY THIS CLASSIFICATION? / LOCAL EXPLANATION
    st.header("Why this classification?")

    @st.cache_resource
    def get_local_explainer():
        try:
            import shap
            pipe = predict._pipeline
            if pipe is None:
                return None
            rf_model = pipe.named_steps["clf"]
            explainer = shap.TreeExplainer(rf_model)
            return explainer
        except Exception:
            return None

    explainer = get_local_explainer()
    local_shap_success = False

    if explainer is not None:
        try:
            # 1. Transform the row
            X_row = pd.DataFrame([pred_result["feature_values"]])
            X_row = X_row[model_meta['features']]
            pipe = predict._pipeline
            imputer = pipe.named_steps["imputer"]
            X_row_t = imputer.transform(X_row)
            
            # 2. Compute SHAP
            shap_values = explainer.shap_values(X_row_t, check_additivity=False)
            
            # 3. Extract the SHAP values for the predicted class
            predicted_class = pred_result["predicted_risk_class"]
            class_idx = model_meta['class_order'].index(predicted_class)
            
            if isinstance(shap_values, list):
                # shap_values[class_idx] is array of shape (1, n_features)
                local_shap = shap_values[class_idx][0]
            else:
                # shap_values is (1, n_features, n_classes)
                local_shap = shap_values[0, :, class_idx]
                
            # 4. Create dataframe
            human_names = {
                "total_animal_fatalities": "Animal fatalities",
                "max_flood_duration": "Maximum flood duration",
                "std_flood_duration": "Flood duration variability",
                "median_flood_duration": "Median flood duration",
                "percent_flooded_area": "Percent flooded area",
                "corrected_percent_flooded_area": "Corrected flooded area",
                "permanent_water": "Permanent water",
                "total_human_displaced": "Human displaced"
            }
            
            local_df = pd.DataFrame({
                "feature": [human_names.get(f, f) for f in model_meta['features']],
                "contribution": local_shap
            })
            
            # Sort by absolute contribution and take top 5 if desired, or show all. Let's show all for context, ordered by magnitude.
            local_df['abs_contribution'] = local_df['contribution'].abs()
            local_df = local_df.sort_values('abs_contribution', ascending=True)
            
            st.markdown(f"### Local model explanation for **{selected_district}**")
            st.write(f"The chart below shows the feature contributions contributing to the model's classification of {selected_district} as **{predicted_class}**. These are predictive model contributions, not causal factors.")
            
            fig_local, ax_local = plt.subplots(figsize=(10, 5))
            colors = ['#d62728' if c < 0 else '#2ca02c' for c in local_df['contribution']]
            
            ax_local.barh(local_df['feature'], local_df['contribution'], color=colors)
            ax_local.set_xlabel(f'SHAP Value (Impact pushing towards {predicted_class})')
            ax_local.spines['top'].set_visible(False)
            ax_local.spines['right'].set_visible(False)
            
            # Add a vertical line at 0
            ax_local.axvline(x=0, color='black', linewidth=0.8, linestyle='--')
            
            # Add text labels
            for i, (val, name) in enumerate(zip(local_df['contribution'], local_df['feature'])):
                ha = 'left' if val >= 0 else 'right'
                offset = 0.005 if val >= 0 else -0.005
                ax_local.text(val + offset, i, f'{val:+.3f}', va='center', ha=ha, fontsize=9)
            
            # Expand x limits slightly so labels fit
            x_min, x_max = ax_local.get_xlim()
            ax_local.set_xlim(x_min - (x_max-x_min)*0.1, x_max + (x_max-x_min)*0.1)

            st.pyplot(fig_local)
            
            st.caption("🟢 Positive contribution: pushes prediction towards the predicted class.<br>🔴 Negative contribution: pushes away from the predicted class.", unsafe_allow_html=True)
            local_shap_success = True
            
        except Exception as e:
            st.warning(f"Could not compute local explanation: {e}")
            local_shap_success = False

    if not local_shap_success:
        st.markdown("### Global model feature importance")
        
        shap_df = load_shap_data()
        if shap_df is not None and not shap_df.empty:
            if shap_df['importance'].isna().all():
                 st.info("SHAP explainability data is currently unavailable (placeholder found). Showing model features used.")
                 st.write(model_meta['features'])
            else:
                 st.markdown("The chart below shows the **SHAP Global Importance** across the entire dataset.")
                 
                 fig_shap, ax_shap = plt.subplots(figsize=(10, 5))
                 plot_df = shap_df.sort_values('importance', ascending=True)
                 
                 # Map to human names
                 human_names = {
                    "total_animal_fatalities": "Animal fatalities",
                    "max_flood_duration": "Maximum flood duration",
                    "std_flood_duration": "Flood duration variability",
                    "median_flood_duration": "Median flood duration",
                    "percent_flooded_area": "Percent flooded area",
                    "corrected_percent_flooded_area": "Corrected flooded area",
                    "permanent_water": "Permanent water",
                    "total_human_displaced": "Human displaced"
                 }
                 labels = [human_names.get(f, f) for f in plot_df['feature']]
                 
                 ax_shap.barh(labels, plot_df['importance'], color='steelblue')
                 ax_shap.set_xlabel('Mean |SHAP Value| (Impact on Model Output)')
                 ax_shap.spines['top'].set_visible(False)
                 ax_shap.spines['right'].set_visible(False)
                 
                 st.pyplot(fig_shap)
        else:
            st.info("SHAP importance data not found in outputs/ directory.")

    st.markdown("---")

    # 11. CURRENT RAINFALL CONTEXT (Optional)
    rain_df = load_rainfall_data()
    if rain_df is not None and not rain_df.empty:
        st.header("Current Rainfall Context")
        st.warning("**Context only — current rainfall is NOT used by the ML model.**")
        
        # Try to find the district in rainfall data
        # Normalise for matching
        rain_districts = rain_df['District'].astype(str).str.strip().str.lower()
        search_d = selected_district.strip().lower()
        
        # Simple string matching for robustness
        match_idx = rain_districts == search_d
        
        if match_idx.any():
            dist_rain = rain_df[match_idx].iloc[-1:] # Take latest row if multiple
            st.dataframe(dist_rain, use_container_width=True)
        else:
            # Fallback to contains
            match_idx_contains = rain_districts.str.contains(search_d, na=False)
            if match_idx_contains.any():
                dist_rain = rain_df[match_idx_contains].iloc[-1:]
                st.dataframe(dist_rain, use_container_width=True)
            else:
                st.info(f"No recent rainfall data found for {selected_district}.")
                
        st.caption("Please note that absence of rainfall data does not mean there was no rainfall.")
        
        st.markdown("---")

    # 8. MODEL PERFORMANCE & 9. DATA / METHODOLOGY
    st.header("Methodology & Performance")
    
    col_perf, col_meth = st.columns(2)
    
    with col_perf:
        st.subheader("Model Performance")
        st.markdown(f"**Model:** {model_meta['model_name']}")
        st.markdown("**(5-fold Stratified Cross-Validation on Historical Data)**")
        st.markdown(f"- **Macro F1:** {model_meta['cv_macro_f1']:.4f} ± {model_meta['cv_macro_f1_std']:.4f}")
        st.markdown(f"- **Accuracy:** {model_meta['cv_accuracy']:.4f}")
        st.markdown(f"**Baseline (Physical Exposure Only):**")
        st.markdown(f"- **Macro F1:** {model_meta['baseline_m2_macro_f1']:.4f}")
        st.caption("The reported metrics measure performance on the historical district-classification task.")

    with col_meth:
        st.subheader("Data & Methodology")
        st.markdown(f"- **Districts analyzed:** {model_meta['training_rows']} usable districts")
        st.markdown(f"- **Risk classes:** {', '.join(model_meta['class_order'])}")
        st.markdown("- **Algorithm:** Random Forest Classifier")
        st.markdown("- **Explainability:** SHAP (Shapley Additive exPlanations) / Permutation importance")
        
        st.markdown("**Data Sources**")
        st.markdown("- HydroSense Lab / IIT Delhi — India Flood Inventory / DFSI-related datasets")
        st.markdown("- Government of India Local Government Directory (LGD) district data")
        st.markdown("- IMD rainfall data (for contextual rainfall section only)")

    st.markdown("---")

    # 10. RESPONSIBLE AI / LIMITATIONS
    st.header("Responsible AI & Limitations")
    st.info(
        """
        - **Historical classification, not future flood forecasting:** This system classifies historical risk based on past characteristics.
        - **DFSI is not a model input:** The model is benchmarked against the District Flood Severity Index, but does not use it as an input, avoiding target leakage.
        - **Predictive importance does not imply causation:** Feature importance indicates predictive utility, not causation.
        - **Historical reporting completeness varies:** Reporting completeness and data quality can vary across districts.
        - **Current rainfall is contextual only:** It is not a model feature.
        - **Expert support tool:** Model outputs should support, not replace, expert judgement.
        - **Not intended for emergency response or life-critical decision-making.**
        """
    )
    
    st.markdown("---")
    
    # 11. PROJECT CONTEXT & FOOTER
    st.subheader("Project Context")
    st.markdown("Developed as a sustainability-focused AI project during the 1M1B AI for Sustainability Virtual Internship.")
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align: center; color: gray; font-size: 0.8em;'>"
        "AI-assisted historical flood-risk classification prototype &bull; For educational and research purposes"
        "</div>", 
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
