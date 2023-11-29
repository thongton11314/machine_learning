import os, requests
from langchain.agents import create_json_agent
from langchain.agents.agent_toolkits import JsonToolkit
from langchain.tools.json.tool import JsonSpec
from langchain.prompts import PromptTemplate
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI

AV_API_KEY = os.environ.get('AV_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

class CompanyAPI:

    def __init__(self, symbol):
        self.symbol = symbol
    
    def _get_data_from_api(self, function, additional_params=None):
        url = "https://www.alphavantage.co/query"
        params = {
            "function": function,
            "symbol": self.symbol,
            "apikey": AV_API_KEY
        }
        if additional_params:
            params.update(additional_params)
        
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            if not data:
                print(f"No data found for {self.symbol}")
                return None
            return data
        else:
            print(f"Error: {response.status_code} - {response.text}")
            return None

# Model
llm = OpenAI(model_name='gpt-3.5-turbo', openai_api_key=OPENAI_API_KEY, temperature=0, max_tokens=250)

# Company overview information agent
companyOverview = CompanyAPI("AAPL")._get_data_from_api("OVERVIEW")
json_spec = JsonSpec(dict_=companyOverview, max_value_length=4000)
json_toolkit = JsonToolkit(spec=json_spec)
json_agent_executor_company_overview = create_json_agent(
    llm=llm, toolkit=json_toolkit, verbose=True, handle_parse_error=True
)

# Company income statement agent
companyIncomeStatement = CompanyAPI("AAPL")._get_data_from_api("INCOME_STATEMENT")
json_spec = JsonSpec(dict_=companyIncomeStatement, max_value_length=4000)
json_toolkit = JsonToolkit(spec=json_spec)
json_agent_executor_company_income_statement = create_json_agent(
    llm=llm, toolkit=json_toolkit, verbose=True, handle_parse_error=True
)

# Company balance sheet agent
companyBalanceSheet = CompanyAPI("AAPL")._get_data_from_api("BALANCE_SHEET")
json_spec = JsonSpec(dict_=companyBalanceSheet, max_value_length=4000)
json_toolkit = JsonToolkit(spec=json_spec)
json_agent_executor_company_balance_sheet = create_json_agent(
    llm=llm, toolkit=json_toolkit, verbose=True, handle_parse_error=True
)

# Company cash flow information agent
companyCashFlow = CompanyAPI("AAPL")._get_data_from_api("CASH_FLOW")
json_spec = JsonSpec(dict_=companyCashFlow, max_value_length=4000)
json_toolkit = JsonToolkit(spec=json_spec)
json_agent_executor_company_cash_flow = create_json_agent(
    llm=llm, toolkit=json_toolkit, verbose=True, handle_parse_error=True
)

# All agents here
tools = [
    Tool(
        name="Company Overview",
        func=json_agent_executor_company_overview.run,
        description="Useful for when you need to answer questions about the most recent data about company overview and similar questions relate to overview. Could be utilize for company description as well.",
    ),
    Tool(
        name="Company Imcome Statement",
        func=json_agent_executor_company_income_statement.run,
        description="Useful for when you need to answer questions about the most recent data about company imcome statement and similar questions relate to overview. Could be utilize for company description as well.",
    ),
    Tool(
        name="Company Balance Sheet",
        func=json_agent_executor_company_balance_sheet.run,
        description="Useful for when you need to answer questions about the most recent data about company balance sheet and similar questions relate to overview. Could be utilize for company description as well.",
    ),
    Tool(
        name="Company Cash Flow",
        func=json_agent_executor_company_cash_flow.run,
        description="Useful for when you need to answer questions about the most recent data about company cashflow and similar questions relate to overview. Could be utilize for company description as well.",
    )
]

# Converstional model
conversationBot = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY, temperature=0, max_tokens=500)
promptTemplate = PromptTemplate(
    template="You are a financial chat bot, you will response in a manner of financial consultant, or analyst from user. All data is in USD, make sure you convert all number into dollars unit with $ at start, and all dates into MM/DD/YYYY format if possible",
    input_variables=['input']
    )

# Agent controller
agent_chain = initialize_agent(
    tools=tools,
    llm=conversationBot,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
    PromptTemplate = promptTemplate,
    handle_parse_error=True,
)
print(agent_chain.run('Tell me who are you? What can you do'))
print(agent_chain.run("Give me some insight about this company, what are some number of income statement, balance sheet, or cash flow"))
print(agent_chain.run("Tell me about company 200 moving average"))
print(agent_chain.run("Tell me about company cash flow"))
print(agent_chain.run("What do you think about this company based on cash flow"))
print(agent_chain.run("What did I ask you? list all questions I asked you. Then, summarize the conversation"))
print(agent_chain.run('Tell me about company cash flow between 2020-2022, list by years'))
