#  AI Excel Analyst

Analyze Excel and CSV files with LLMs **without sending raw data**.

The tool automatically:

- 📊 Generates a statistical summary
- 🎯 Suggests dataset-specific KPIs
- 📈 Calculates KPIs safely with pandas
- 💡 Produces a business insight report using LLMs
- 🔒 Preserves privacy by sending only summarized data

Supports:

- Ollama (Local)
- Google Gemini

## Why this project?

Unlike many AI data analysis tools, this project:

- Never sends raw datasets to the LLM
- Dynamically discovers KPIs instead of using hard-coded metrics
- Supports both local and cloud models
- Calculates KPIs with pandas instead of executing LLM-generated code
## Tech Stack

- Python
- Pandas
- OpenPyXL
- Ollama
- Google Gemini
- argparse


#Workflow
- Excel
   │
   ▼

Data Summary

   │

   ▼

LLM

   │

   ▼

Suggested KPIs

   │

   ▼

Safe KPI Calculator

   │

   ▼

Insight Report




ai-excel-analyst/
├── core.py           
├── dynamic_kpi.py     
├── llm_ollama.py       
├── llm_gemini.py         
├── cli.py             
├── requirements.txt
├── .env.example
└── README.md
