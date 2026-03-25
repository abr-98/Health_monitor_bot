from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import pandas as pd
import warnings

from dotenv import load_dotenv
import os

import asyncio 

warnings.filterwarnings("ignore")


load_dotenv()

async def main():
    # Initialize the MCP client
    mcp_client = MultiServerMCPClient(
        {
                "diabetes": {
                    "url": "http://localhost:8000/mcp/",
                    "transport": "streamable-http",
                },
                "heart_diseases": {
                    "url": "http://localhost:8010/mcp/",
                    "transport": "streamable-http",
                }
        }
    )
    
    df_diabetes = pd.read_csv("datasets/diabetes.csv")
    sample = df_diabetes.iloc[0]
    
    prompt = f"""Analyze a 50-year-old male patient with the following health metrics:
    
    Diabetes Prediction Input:
    - Pregnancies: {int(sample['Pregnancies'])}
    - Glucose: {sample['Glucose']:.1f}
    - BloodPressure: {sample['BloodPressure']:.1f}
    - SkinThickness: {sample['SkinThickness']:.1f}
    - Insulin: {sample['Insulin']:.1f}
    - BMI: {sample['BMI']:.1f}
    - DiabetesPedigreeFunction: {sample['DiabetesPedigreeFunction']:.4f}
    - Age: 50


    Provide predictions for diabetes."""


    os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_KEY")
    
    tools = await mcp_client.get_tools()
    model = ChatOpenAI(model="gpt-4-0613")

    # Create a React agent with the MCP client
    react_agent = create_react_agent(model=model, tools=tools)

    # Run the agent with a user query
    response = await react_agent.ainvoke({"messages": [{"role": "user", "content": prompt}]})
    print(response["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())