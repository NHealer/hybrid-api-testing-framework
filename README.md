# An Adaptive Hybrid Framework for Automated REST API Test Case Generation

An end-to-end software testing framework that bridges interactive static JSON schema constraint analysis, Large Language Model (LLM) payload synthesis, dual-tier validation interceptor guardrails, and Reinforcement Learning (RL) test suite prioritization.

---

## 📌 Executive Summary

Modern cloud microservices and autonomous LLM agents rely on REST APIs as their operational communication backbone. However, standard OpenAPI/Swagger specifications are chronically under-specified, lacking machine-readable constraints, regex formats, and inter-parameter execution rules. While probabilistic LLMs offer cognitive reasoning for dynamic input synthesis, they frequently suffer from hallucinations and invalid tool calls. 

This framework resolves this tension by integrating:
1. **Interactive Schema Analysis (X_1):** Speculates hidden constraints, constants, enumerations, and defaults from historical traces to compile an Enriched OpenAPI Specification (eOAS).
2. **Context Transformer & Prompt Compiler:** Algorithmically converts enriched schemas into structured context prompts.
3. **Dual-Tier Validation Interceptor Guardrail (M):** Intercepts raw LLM outputs (X2) prior to server execution, halting spec-violating payloads, classifying errors via an IBM error taxonomy, and driving an automated self-healing loop.
4. **Reinforcement Learning Prioritization Engine:** Deploys Listwise A2C and DQN sorter agents to optimize the Average Percentage of Faults Detected (APFD) score.

---

## 🛠️ System Architecture

                            +-----------------------------------+
                            |     Raw OpenAPI Specification     |
                            +-----------------------------------+
                                              |
                                              v
                            +-----------------------------------+
                            |  Interactive Schema Inferrer (X1) |
                            +-----------------------------------+
                                              |
                                              v
                            +-----------------------------------+
                            |    Enriched OpenAPI Spec (eOAS)   |
                            +-----------------------------------+
                                              |
                                              v
                            +-----------------------------------+
                            |    Context Prompt Transformer     |
                            +-----------------------------------+
                                              |
                                              v
                            +-----------------------------------+
                            |     LLM Generative Engine (X2)    |
                            +-----------------------------------+
                                              |
                                              v
                            +-----------------------------------+
                            | Dual-Tier Interceptor Guard (M)   | <---+ (Self-Healing Loop)
                            +-----------------------------------+     |
                                              | (Validated Payload)   | (Feedback Taxonomy)
                                              v                       |
                            +-----------------------------------+     |
                            |     Execution & RL Prioritizer    | ----+
                            +-----------------------------------+

---

## 🚀 Quickstart Guide

### Prerequisites
* **Python 3.8+**
* Standard Library Modules: `json`, `re`, `math`, `typing`, `dataclasses`, `random`

### Installation & Execution
1. **Clone the Repository:**
   ```bash
   git clone https://github.com/<your-github-username>/hybrid-api-testing-framework.git
   cd hybrid-api-testing-framework
