import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from salesgpt.agents import SalesGPT

load_dotenv()

print(os.getenv("OPENAI_API_KEY"))

llm = ChatOpenAI(
    temperature=0,
    openai_api_key = os.getenv("OPENAI_API_KEY"),
    base_url = os.getenv("OPENAI_BASE_URL")
)

sales_agent = SalesGPT.from_llm(
            llm,
            verbose=True,
            use_tools=False,
            salesperson_name="Ted Lasso",
            salesperson_role="Sales Representative",
            company_name="Sleep Haven",
            company_business="""Sleep Haven 
                                    is a premium mattress company that provides
                                    customers with the most comfortable and
                                    supportive sleeping experience possible. 
                                    We offer a range of high-quality mattresses,
                                    pillows, and bedding accessories 
                                    that are designed to meet the unique 
                                    needs of our customers.""",
        )

sales_agent.seed_agent()
sales_agent.determine_conversation_stage()

sales_agent.step()
agent_output = sales_agent.conversation_history[-1]
assert agent_output is not None, "Agent output cannot be None."
assert isinstance(agent_output, str), "Agent output needs to be of type str"
assert len(agent_output) > 0, "Length of output needs to be greater than 0."

human_input = input('say something')
sales_agent.human_step(human_input)

sales_agent.determine_conversation_stage()
sales_agent.step()
agent_output = sales_agent.conversation_history[-1]

human_input = input('say something')
sales_agent.human_step(human_input)

sales_agent.determine_conversation_stage()
sales_agent.step()
agent_output = sales_agent.conversation_history[-1]