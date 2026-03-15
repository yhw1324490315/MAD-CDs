<div align="center">

# CD-LPL Autonomous Discovery System

**An Advanced Multi-Agent Framework for the Autonomous Design of Carbon Dot-Based Long Persistent Luminescence Materials**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

[🇨🇳 中文版](./README_ZH.md) | [🇬🇧 English](./README.md)

</div>

---

## 📑 Table of Contents
- [📖 Introduction](#-introduction)
- [🌟 Key Features](#-key-features)
- [⚙️ System Architecture & Workflow](#-system-architecture--workflow)
- [🚀 Getting Started](#-getting-started)
- [📂 Project Structure](#-project-structure)
- [📜 License](#-license)

---

## 📖 Introduction
The **CD-LPL Autonomous Discovery System** is a cutting-edge artificial intelligence platform designed to revolutionize the materials science research workflow. By integrating **Large Language Models (LLMs)** with **Explainable AI (XAI)** techniques, this system moves beyond traditional "black box" optimization. It acts as a digital research partner capable of reasoning, hypothesizing, confirming mechanisms, and autonomously designing experimental recipes for Carbon Dot (CD) materials with Long Persistent Luminescence (LPL) properties.

Targeting the complex domain of organic room-temperature phosphorescence (RTP) and LPL, the system navigates a chemical space of over 120 million molecules. It employs a rigorous, closed-loop iterative process that mimics the scientific method: **Hypothesis $\rightarrow$ Analysis $\rightarrow$ Synthesis Design $\rightarrow$ Peer Review $\rightarrow$ Iteration**.

## 🌟 Key Features

### 1. Multi-Agent Cognitive Architecture
The system is composed of five specialized agents:
- **Planner Agent**: Breaks down user requests, queries experimental databases, and formulates high-level design strategies based on global feature importance (SHAP).
- **Deep Analysis Tool**: Performs quantitative validation using Partial Dependence Plots (PDP) and decodes abstract "Fingerprint Bits" into visual chemical substructures.
- **Summary Agent**: Synthesizes vast amounts of data, charts, and chemical structures into a coherent research briefing.
- **Architect Agent**: Converts theoretical strategies into actionable Standard Operating Procedures (SOPs), identifying specific precursor candidates and synthesis parameters.
- **Critic Agent**: A panel of LLM judges evaluates proposed recipes against physical laws and project goals, enforcing a strict quality control loop.

### 2. Explainable AI & Visual Reasoning
- **Mechanism Transparency**: Explains *why* a specific precursor is chosen.
- **Visual Evidence**: Generates and interprets chemical structure images (SVG/PNG) and PDP charts.

### 3. Closed-Loop Self-Correction
Features a **Cumulative Failure Memory**. If a design fails the Critic's review, the reasons are explicitly sent back to the Planner for the next iteration to prevent repetitive errors.

## 🚀 Getting Started

### Prerequisites
- **Python 3.8+**
- **LLM API Access**: OpenAI (GPT-4) or Google (Gemini) API Keys.
- **Storage**: ~10GB of free space for chemical databases.

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/YourRepo/CD-LPL-Autonomous-Discovery.git
   cd CD-LPL-Autonomous-Discovery
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Configure Environment:
   - Create and edit `config/secrets.env` based on the example:
     ```env
     OPENAI_API_KEY=your_key_here
     ```

4. Download Required Data:
   Due to GitHub size limits, you must manually download the core datasets (CID-SMILES library ~8.7GB, Experimental spreadsheets, and pre-trained `.pkl` models) and place them in the `data/` and `data/models/` directories.
   Please refer to [data/README.md](data/README.md) for detailed download instructions and structure.

### Running the System
To initiate the autonomous discovery loop:
```bash
python test_runner.py
```
*(Tip: Edit `test_runner.py` to change `MOLECULE_SEARCH_LIMIT` to 5000 for a quick test run before searching the full 120M library!)*

- The system will create a unique directory in `experiments/` (e.g., `Run_20231227_153022`).
- Logs, intermediate plots, and the final recipe will be saved there.

## 📂 Project Structure

```text
CD-LPL-Autonomous-Discovery/
├── config/
│   ├── secrets.env       # API Credentials (ignored in git)
│   ├── config.yaml       # LLM Model & Path Configs
│   └── prompts.yaml      # Agent System Prompts
├── data/                 # Raw Datasets & XGBoost Inputs (Download required!)
│   ├── README.md         # Instructions to download data
│   ├── models/           # Pre-trained XGBoost Models (*.pkl, *.json)
│   └── (large files)     # CID-SMILES, Excel datasets
├── src/
│   ├── llm_agents/       # Core Agent Logic (Planner, Architect, Critic...)
│   └── utils.py          # Helper Functions
├── tests/                # Unit test scripts
├── requirements.txt      # Python dependencies
└── test_runner.py        # Main Execution Loop
```

## 📜 License
This project is licensed under the MIT License.
