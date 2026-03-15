<div align="center">

# CD-LPL 自主发现系统 (CD-LPL Autonomous Discovery System)

**基于多智能体协作与可解释性 AI 的碳点长余辉材料自主设计框架**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

[🇨🇳 中文版](./README_ZH.md) | [🇬🇧 English](./README.md)

</div>

---

## 📑 目录
- [📖 项目简介](#-项目简介)
- [🌟 核心特性](#-核心特性)
- [⚙️ 系统架构与运行逻辑](#-系统架构与运行逻辑)
- [🚀 快速开始](#-快速开始)
- [📂 项目目录结构](#-项目目录结构)
- [📜 许可证](#-许可证)

---

## 📖 项目简介
**CD-LPL 自主发现系统** 是一个前沿的人工智能平台，旨在彻底改变材料科学的研究范式。通过深度融合 **大语言模型 (LLMs)** 与 **可解释性人工智能 (XAI)** 技术，本系统突破了传统“黑盒”优化的局限，能够自主进行逻辑推理、假设验证、机理分析以及实验方案设计，专注于**碳点基长余辉 (LPL) 材料**的探索与发现。

面对超过 1.2 亿种分子的庞大化学空间，本系统模拟人类科研方法，采用严谨的闭环迭代流程：**假设提出 $\rightarrow$ 定量分析 $\rightarrow$ 方案设计 $\rightarrow$ 同行评审 $\rightarrow$ 迭代优化**。

## 🌟 核心特性

### 1. 多智能体认知架构
系统由五个各司其职的专业智能体组成：
- **规划智能体 (Planner Agent)**：解析需求，依据特征重要性 (SHAP) 制定顶层设计战略。
- **深度分析工具 (Deep Analysis Tool)**：生成偏依赖图 (PDP)，并将“指纹位 (Bits)”解码为化学子结构图像进行定量验证。
- **总结智能体 (Summary Agent)**：将数据、图表和化学结构汇总成严密的“研究简报”。
- **架构师智能体 (Architect Agent)**：将理论战略转化为可落地的标准实验操作流程 (SOP)，搜索最佳合成参数。
- **评论家智能体 (Critic Agent)**：由多个大模型组成的“评委组”，严格审查方案把控设计质量。

### 2. 可解释性与视觉推理
- **机理透明化**：能够解释选择某个特定前驱体的理化依据。
- **视觉证据链**：生成并分析 PDP 趋势图和结构图像，确保 AI 推理逻辑可被直观验证。

### 3. 闭环自纠正机制
内置 **累计失败记忆 (Cumulative Failure Memory)**：若是评审未通过，详细的拒绝理由直接反馈给 Planner 重塑下一轮的约束条件，避免系统重复无效设计。

## 🚀 快速开始

### 环境依赖
- **Python 3.8+**
- **LLM API**: 需要 OpenAI (GPT-4) 或 Google (Gemini) API 访问权限。
- **磁盘空间**: 推荐留出 10GB 以上用于存放 1.2亿分子数据库。

### 安装步骤
1. 克隆代码仓库：
   ```bash
   git clone https://github.com/YourRepo/CD-LPL-Autonomous-Discovery.git
   cd CD-LPL-Autonomous-Discovery
   ```
2. 安装环境依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. 配置环境：
   - 复制并修改 `config/secrets.env`（根据参考文件进行配置）并填入您的 API Key：
     ```env
     OPENAI_API_KEY=您的密钥
     ```

4. 准备数据文件（必须）：
   受限于 GitHub 文件大小，大型数据库（如近 8.7GB 的 CID-SMILES 库及数百MB的 Excel 实验数据表、pkl 预训练模型）未上传至仓库。详细数据获取方式及存放路径请参考 [`data/README.md`](data/README.md)。

### 运行系统
在准备好所有环境变量及数据文件后，启动自主发现主循环：
```bash
python test_runner.py
```
*(提示：首次测试可以将 `test_runner.py` 中的 `MOLECULE_SEARCH_LIMIT` 修改为 5000 以便快速体验流程！)*

- 系统将在 `experiments/` 目录下创建单次运行文件夹（如 `Run_20231227_153022`）。
- 全流程日志、中间图表、评审记录和最终实验方案均保存在该文件夹中。

## 📂 项目目录结构

```text
CD-LPL-Autonomous-Discovery/
├── config/
│   ├── secrets.env       # API 密钥配置 (需手动创建配置)
│   ├── config.yaml       # 系统及路径配置
│   └── prompts.yaml      # 智能体 Prompts 模版
├── data/                 # 数据库目录 (核心数据文件需自行下载)
│   ├── README.md         # 数据获取及配置指南
│   ├── models/           # XGBoost 模型文件
│   └── ...               # SMILES 数据及实验统计文件
├── src/
│   ├── llm_agents/       # 核心多智能体基类定义
│   └── utils.py          # 工具函数
├── tests/                # 单元侧测试脚本
├── requirements.txt      # 依赖库列表
└── test_runner.py        # 全过程主程入口
```

## 📜 许可证
本项目采用 MIT 许可证。
