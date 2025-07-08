# AgriSightAI

<div align="center">
  <img src="https://images.unsplash.com/photo-1523348837708-15d4a09cfac2?q=80&w=2070" width="500" alt="AgriSight Banner">
  <h3>Multi-Agent Decision Support System for Precision Agriculture</h3>
  <p>Powered by NVIDIA Agent Intelligence Toolkit</p>
</div>

## 🌱 Overview

AgriSightAI is an innovative multi-agent system that revolutionizes precision agriculture by combining computer vision, environmental sensing, and powerful decision-making capabilities. The system creates a collaborative network of specialized AI agents that work together to:

- Monitor crops and analyze soil conditions
- Detect diseases and pest infestations 
- Optimize resource usage (water, fertilizers, pesticides)
- Provide actionable insights to farmers
- Promote sustainable agricultural practices

AgriSightAI integrates with existing farm equipment and IoT devices, processes multi-modal data (images, sensor readings, historical records), and employs RAG workflows to incorporate domain knowledge from agricultural research. The system's modular architecture allows for flexible deployment across different scales - from small farms to large agricultural operations.


## 🏗️ Architecture

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

## ✨ Key Features

### 1. Multi-agent Coordination
- Inter-agent communication protocol
- Task prioritization and assignment
- Consensus-based decision making

### 2. Computer Vision Analysis
- Crop health assessment from aerial/ground imagery
- Disease and pest detection with localizations
- Growth stage classification

### 3. Sensor Data Integration
- Soil moisture, temperature, and nutrient tracking
- Weather data incorporation
- Trend analysis and anomaly detection

### 4. Knowledge-enhanced Recommendations
- RAG-powered retrieval of relevant agricultural knowledge
- Customized recommendations based on crop type and conditions
- Explanation of reasoning for decisions

### 5. Interactive Dashboard (Streamlit UI)
- Field mapping with issue overlays
- Time-series data visualization
- Action prioritization interface

This project is built as a Minimum Viable Product (MVP) during a hackathon, showcasing the potential of multi-agent AI in agriculture. While the current implementation demonstrates core functionalities, several advanced capabilities needs to be developed using the technology stack mentioned for continued development.

https://www.youtube.com/watch?v=7a6EpbnzQqk

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- NVIDIA API key for NIM models
- AIQ Toolkit (`aiq`)

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/AgriSightAI.git
cd AgriSightAI

# Install the required packages
pip install -e .

# Set environment variables
export NVIDIA_API_KEY="your-api-key-here"
```

### Usage

#### Command Line Interface
```bash
# Run AgriSightAI analysis
aiq run --config_file src/agriculsight/configs/config.yml --input "Analyze my corn field for potential disease issues. The leaves have yellow spots and the soil moisture has been high due to recent rainfall."
```

#### Streamlit UI
```bash
# Start the Streamlit web interface
streamlit run streamlit_app.py
```

## 🧰 Technology Stack

### NVIDIA Components
- **NVIDIA Agent Intelligence (AgentIQ) Toolkit**: Core orchestration framework for multi-agent system
- **NVIDIA NIM Microservices**: Optimized inference for generative AI and computer vision models
- **NVIDIA NeMo Framework**: For building and customizing the LLMs that power the agents
- **NVIDIA TensorRT**: For optimized inference on image and sensor data analysis
- **NVIDIA RAPIDS**: For accelerated data processing and analytics
- **NVIDIA NeMo Guardrails**: For ensuring responsible AI recommendations

### Additional Technologies
- **Backend**: FastAPI, Redis, PostgreSQL, Milvus/FAISS
- **Frontend**: Streamlit, React.js with Material UI
- **ML/AI**: PyTorch, Agno, Hugging Face Transformers

## 📂 Project Structure

```
AgriSightAI/
├── pyproject.toml             # Project metadata and dependencies
├── README.md                  # Project documentation
├── streamlit_app.py           # Streamlit web interface
├── src/
│   └── agriculsight/
│       ├── __init__.py
│       ├── agriculsight_function.py  # Main orchestration function
│       ├── register.py               # Register tools with AIQ
│       ├── configs/
│       │   └── config.yml            # AIQ configuration
│       └── tools/
│           ├── __init__.py
│           ├── image_processing.py   # Vision analysis tool
│           └── soil_analysis.py      # Environmental analysis tool
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- This project was developed as part of a hackathon focused on NVIDIA's Agent Intelligence toolkit.

![AgriSightAI](https://github.com/user-attachments/assets/0f18160a-f6ee-4798-b0a7-54a02d02017a)

- Special thanks to the NVIDIA team for providing the tools and resources to build this system.
- Agricultural research resources that informed our knowledge base.
