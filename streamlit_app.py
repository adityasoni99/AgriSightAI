import streamlit as st
import subprocess
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

st.set_page_config(
    page_title="AgriSight",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 36px;
        font-weight: bold;
        color: #2E7D32;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 24px;
        font-weight: bold;
        color: #388E3C;
        margin-top: 30px;
        margin-bottom: 10px;
    }
    .section {
        background-color: #F1F8E9;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .highlight {
        color: #1B5E20;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Page Header
st.markdown("<div class='main-header'>🌱 AgriSight: Multi-Agent Decision Support System for Precision Agriculture</div>",
            unsafe_allow_html=True)

st.markdown("""
AgriSight is an innovative multi-agent system that revolutionizes precision agriculture by combining 
computer vision, environmental sensing, and powerful decision-making capabilities powered by NVIDIA's 
Agent Intelligence toolkit.
""")

# Sidebar
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1523348837708-15d4a09cfac2?q=80&w=2070", width=250)
    st.markdown("### Settings")

    api_key = st.text_input("NVIDIA API Key", type="password", value="sk-dummy-key")

    st.markdown("### About")
    st.markdown("""
    AgriSight integrates with existing farm equipment and IoT devices, processes multi-modal data 
    (images, sensor readings, historical records), and employs RAG workflows to incorporate domain 
    knowledge from agricultural research.
    """)

# Main content with tabs
tab1, tab2, tab3 = st.tabs(["Analysis", "Data Visualization", "System Architecture"])

with tab1:
    st.markdown("<div class='sub-header'>Agricultural Analysis</div>", unsafe_allow_html=True)

    # Input Section
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    query = st.text_area(
        "Describe your agricultural concern",
        "Analyze my corn field for potential disease issues. The leaves have yellow spots and the soil moisture has been high due to recent rainfall.",
        height=100
    )

    uploaded_file = st.file_uploader("Upload field image (optional)", type=["jpg", "jpeg", "png"])

    analyze_button = st.button("Analyze", type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    # Results Section
    if analyze_button:
        with st.spinner("Running analysis with AgriSight multi-agent system..."):
            # Set the API key to environment variable
            os.environ["NVIDIA_API_KEY"] = api_key

            # Set env vars
            os.environ["NVIDIA_API_KEY"] = api_key

            # Create the command to run the AgriSight analysis
            cmd = f'aiq run --config_file src/agriculsight/configs/config.yml --input "{query} Image: {uploaded_file}"'

            try:
                st.info(
                    "Executing AIQ command... (Note: This would normally call the NVIDIA API but may fail with dummy credentials)")

                # Run the command and capture the output
                result = subprocess.run(cmd, shell=True, capture_output=True, text=True)

                # Check if there were any errors
                if result.returncode != 0:
                    st.error(f"Error running AgriSight analysis: {result.stderr}")

                    # Provide sample output for demo purposes when there's an error
                    output = """
                    Workflow Result:
                    ['Based on the analysis of your corn field, there are signs of possible fungal infection due to the yellow spots on leaves combined with high soil moisture. This could indicate the early stages of Common Rust or Northern Leaf Blight. The recent rainfall has created favorable conditions for fungal growth. Recommendations: 1) Improve field drainage to reduce soil moisture, 2) Consider applying a fungicide treatment within the next 3-5 days, 3) Monitor the affected areas closely for spread of symptoms. The Vision Agent has identified the pattern as consistent with early-stage fungal infection, and the Environmental Agent confirms that current conditions (high humidity, moderate temperatures) support this diagnosis.']
                    """
                else:
                    # Extract the result from the output
                    output = result.stdout + "\n" + result.stderr
                    matches = re.findall(r"Workflow Result:\n\[([^\]]*)\]", output, re.DOTALL)

                    if matches:
                        # Select the last match, as it’s likely the final result
                        analysis_result = matches[-1].strip("'")

                        # Check if the result is a function call (e.g., <function>soil_analysis_tool...)
                        if analysis_result.startswith("<function>"):
                            # Try to find a non-function-call result
                            for match in reversed(matches):
                                if not match.startswith("<function>"):
                                    analysis_result = match.strip("'")
                                    break
                            else:
                                analysis_result = "Analysis completed, but only function call found. Please check input parameters."
                    else:
                        analysis_result = "Analysis completed, but couldn't extract results. Full output provided below."
                        output = result.stdout + "\n" + result.stderr

                    # Replace \n with actual newlines for proper rendering
                    formatted_result = analysis_result.replace("\\n", "\n")

                    # Display the results in a nice format
                    st.markdown("<div class='section'>", unsafe_allow_html=True)
                    st.markdown("<div class='sub-header'>Analysis Results</div>", unsafe_allow_html=True)

                    # If we have an image, show it alongside the results
                    if uploaded_file is not None:
                        col1, col2 = st.columns([1, 2])
                        with col1:
                            image = Image.open(uploaded_file)
                            st.image(image, caption="Uploaded Field Image", use_column_width=True)
                        with col2:
                            st.markdown(formatted_result)
                    else:
                        st.markdown(formatted_result)

                    st.markdown("</div>", unsafe_allow_html=True)

                    # Show the raw output in an expander for debugging
                    with st.expander("Raw Output"):
                        st.code(output)

            except Exception as e:
                st.error(f"Error: {str(e)}")

with tab2:
    st.markdown("<div class='sub-header'>Agricultural Data Visualization</div>", unsafe_allow_html=True)

    # Create some demo visualizations
    st.markdown("<div class='section'>", unsafe_allow_html=True)
    option = st.selectbox(
        "Select visualization type",
        ["Soil Moisture Map", "Crop Health Index", "Disease Detection Heatmap"]
    )

    # Generate dummy data based on selection
    if option == "Soil Moisture Map":
        # Create a sample soil moisture grid
        fig, ax = plt.subplots(figsize=(10, 8))

        # Generate a moisture map (random data with spatial correlation)
        np.random.seed(42)  # For reproducibility
        x, y = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
        z = 0.5 + 0.5 * np.sin(5 * np.pi * x) * np.sin(5 * np.pi * y)
        z += np.random.normal(0, 0.1, z.shape)
        z = np.clip(z, 0, 1)  # Clip to 0-1 range

        # Create heatmap
        c = ax.pcolormesh(x, y, z, cmap='Blues', vmin=0, vmax=1)
        fig.colorbar(c, ax=ax, label='Soil Moisture Content (%)')
        ax.set_title('Field Soil Moisture Map')
        ax.set_xlabel('Field Width (m)')
        ax.set_ylabel('Field Length (m)')

        st.pyplot(fig)

        # Add some insights
        st.markdown("""
        ### Soil Moisture Insights

        - **Average moisture level**: 0.52 (Optimal: 0.3-0.6)
        - **High moisture areas**: Northeast corner shows excessive moisture levels
        - **Dry areas**: None detected
        - **Recommendation**: Monitor the northeast section for potential drainage issues
        """)

    elif option == "Crop Health Index":
        # Create a sample crop health visualization
        fig, ax = plt.subplots(figsize=(10, 6))

        # Sample data
        categories = ['Chlorophyll', 'Biomass', 'Water Content', 'Nutrient Level', 'Stress Index']
        values = [0.82, 0.65, 0.79, 0.58, 0.25]
        optimal_min = [0.7, 0.6, 0.6, 0.6, 0.0]
        optimal_max = [0.9, 0.8, 0.8, 0.8, 0.3]

        # Create bar chart
        bar_positions = np.arange(len(categories))
        bars = ax.bar(bar_positions, values, width=0.5)

        # Add optimal range indicators
        for i, (min_val, max_val) in enumerate(zip(optimal_min, optimal_max)):
            ax.plot([i - 0.25, i + 0.25], [min_val, min_val], 'g--')
            ax.plot([i - 0.25, i + 0.25], [max_val, max_val], 'g--')

        # Color the bars based on whether they're in the optimal range
        for i, bar in enumerate(bars):
            if optimal_min[i] <= values[i] <= optimal_max[i]:
                bar.set_color('green')
            elif values[i] < optimal_min[i]:
                bar.set_color('orange')
            else:
                bar.set_color('red')

        ax.set_ylim(0, 1)
        ax.set_ylabel('Index Value (0-1)')
        ax.set_title('Crop Health Indices')
        ax.set_xticks(bar_positions)
        ax.set_xticklabels(categories, rotation=45, ha='right')

        st.pyplot(fig)

        # Add some insights
        st.markdown("""
        ### Crop Health Insights

        - **Overall health score**: 0.72 (Good)
        - **Strengths**: Excellent chlorophyll levels and good water content
        - **Areas of concern**: Nutrient levels are slightly below optimal
        - **Recommendation**: Consider additional fertilization to improve nutrient availability
        """)

    else:  # Disease Detection Heatmap
        # Create a sample disease detection heatmap
        fig, ax = plt.subplots(figsize=(10, 8))

        # Generate a disease probability map (random data with spatial correlation)
        np.random.seed(123)  # For reproducibility
        x, y = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
        z = 0.1 + 0.8 * np.exp(-30 * ((x - 0.7) ** 2 + (y - 0.3) ** 2))  # Create a hotspot
        z += np.random.normal(0, 0.05, z.shape)
        z = np.clip(z, 0, 1)  # Clip to 0-1 range

        # Create heatmap
        c = ax.pcolormesh(x, y, z, cmap='YlOrRd', vmin=0, vmax=1)
        fig.colorbar(c, ax=ax, label='Disease Probability')
        ax.set_title('Leaf Rust Detection Probability Map')
        ax.set_xlabel('Field Width (m)')
        ax.set_ylabel('Field Length (m)')

        st.pyplot(fig)

        # Add some insights
        st.markdown("""
        ### Disease Detection Insights

        - **Disease type**: Leaf Rust (Puccinia sorghi)
        - **Affected area**: Approximately 15% of the field
        - **Severity**: High in the southeast corner
        - **Recommendation**: Apply fungicide treatment to the affected area within 2-3 days
        """)

    st.markdown("</div>", unsafe_allow_html=True)

with tab3:
    st.markdown("<div class='sub-header'>System Architecture</div>", unsafe_allow_html=True)

    st.markdown("<div class='section'>", unsafe_allow_html=True)
    st.markdown("""
    ### AgriSight System Architecture

    The AgriSight system combines multiple AI agents that work together to analyze agricultural data and provide recommendations:

    1. **Vision Agent**: Analyzes crop imagery to detect health issues, diseases, and growth patterns
    2. **Environmental Agent**: Processes sensor data to understand soil conditions and environmental factors
    3. **Research Agent**: Retrieves relevant agricultural knowledge and best practices
    3. **Strategy Agent**: Synthesizes insights from multiple sources to provide actionable recommendations

    The system is built using NVIDIA's Agent Intelligence (AgentIQ) toolkit, which enables sophisticated multi-agent communication and decision-making.
    """)

    # Display architecture diagram
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                               AgriSight System                              │
    └─────────────────────────────────────────────────────────────────────────────┘
                                          │
              ┌───────────────────────────┼───────────────────────────┐
              │                           │                           │
    ┌─────────▼────────────┐    ┌─────────▼─────────────┐   ┌─────────▼─────────┐
    │   Data Acquisition   │    │  Agent Orchestrator   │   │  Knowledge Base   │
    │       Module         │    │  (NVIDIA AgentIQ)     │   │    (RAG)          │
    └─────────┬────────────┘    └─────────┬─────────────┘   └─────────┬─────────┘
              │                           │                           │
              │                           │                           │
    ┌─────────▼────────────┐    ┌─────────▼─────────────┐   ┌─────────▼─────────┐
    │   Sensor Interface   │    │                       │   │  Document Store   │
    │   - Camera Data      │    │                       │   │  - Research       │
    │   - IoT Devices      │───►│                       │◄──│  - Best Practices │
    │   - Weather APIs     │    │                       │   │  - Historical Data│
    └──────────────────────┘    │                       │   └───────────────────┘
                                │                       │
    ┌──────────────────────┐    │    Agent Network      │   ┌───────────────────┐
    │  Image Processing    │    │                       │   │  Vector Database  │
    │  - NVIDIA TensorRT   │───►│                       │◄──│  - Embeddings     │
    │  - Computer Vision   │    │                       │   │  - Semantic Search│
    └──────────────────────┘    │                       │   └───────────────────┘
                                │                       │
    ┌──────────────────────┐    │                       │   ┌───────────────────┐
    │  Data Preprocessing  │    │                       │   │  External APIs    │
    │  - NVIDIA RAPIDS     │───►│                       │◄──│  - Weather        │
    │  - Feature Extraction│    └─────────┬─────────────┘   │  - Market Prices  │
    └──────────────────────┘              │                 └───────────────────┘
                                          │
                                ┌─────────▼─────────────┐
                                │  Decision Support &   │
                                │    User Interface     │
                                └───────────────────────┘
                                          │
                    ┌───────────────────┐ │  ┌─────────────────────┐
                    │  Mobile Interface │◄┴─►│  Dashboard/Web App  │
                    └───────────────────┘    └─────────────────────┘
    ```
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# Run the Streamlit app
if __name__ == "__main__":
    pass