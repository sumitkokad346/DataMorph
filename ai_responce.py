# mainapp/logic/ai_response.py

import pandas as pd
from .llm_logic import LLMAgent

def analyze_uploaded_file(file_path: str):
    """
    Reads the uploaded dataset and triggers AI analysis via Mistral LLM Agent.
    Logs the entire process and suggestions in terminal.
    """
    print("\n" + "=" * 80)
    print("AI RESPONSE ANALYZER STARTED")
    print("=" * 80)

    try:
        # Load CSV file
        print(f" Loading dataset from: {file_path}")
        df = pd.read_csv(file_path)
        print(f" File loaded successfully! Shape: {df.shape}")
        print(f" Columns: {list(df.columns)}")

        # Initialize LLM agent
        print("\n Initializing LLMAgent (Mistral 7B)...")
        llm = LLMAgent()

        # Trigger AI analysis
        print("\n Sending dataset to AI for preprocessing suggestions...")
        ai_result = llm.analyze_dataset(df)

        # Display results
        print("\n" + "-" * 80)
        print(" AI ANALYSIS RESULT")
        print("-" * 80)
        print(f"Status: {ai_result.get('status')}")
        print(f"Connection Verified: {ai_result.get('connection_verified')}")
        print(f"Suggestions: {ai_result.get('suggestions', [])}")
        print(f"Reasoning: {ai_result.get('reasoning', 'N/A')}")
        print("\nRaw AI Response (first 500 chars):")
        print(ai_result.get('raw_response', '')[:500])
        print("-" * 80)

        print(" AI analysis completed successfully.")
        return ai_result

    except Exception as e:
        print(f" ERROR in AI response analyzer: {str(e)}")
        return {"status": "error", "message": str(e)}

