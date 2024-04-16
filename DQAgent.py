import pandas as pd
import sqlite3
import json
import os
import openai
from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

DEPLOYMENT_NAME="gpt-3.5-turbo-0125"
DELIMITER="####"
TEMPERATURE=0
MAX_TOKENS=4000

PROMPT_1="""You are a data analyst. Your role is to create deterministic rules in order to test the quality of a dataset.
You will be provided sample data below that represents the dataset you need to create rules for. 
The sample data provided may not be representative of every value in the data.
You will generally follow this process:
1. Observe the sample data.
2. For each column in the dataset, create a rule based on the observation. Each rule should have a name and a description.
Take into account that the observation is on a sample of data.
[{"rule_name":"some name", "rule_description": "some description",}]
3. Repeat the process to create a rule for each column in the dataset.

Do not return any results until you have at least one rule for each column in the dataset. 
Group all of the rules together and output the result into a single JSON object.
"""

PROMPT_2="""
1. Create a SQL query for each rule. 
2. The output value should be true or false and should return a single aggregate value for the entire column. 
Make sure the query is an aggregate query.
3. Be sure to include the "rule name" in the sql query.
4. The database is SQLite. Generate SQL compatible with SQLlite.
5. The dataset name is "Orders".
6. The output should be in JSON format like this:
output:
"[{"rule_name":"some name", "sql_query": "SELECT 'rule name' as Rule, 'true or false' as value FROM TABLE..."},
{"rule_name":"some name", "sql_query": "SELECT 'rule name' as Rule, 'true or false' as value FROM TABLE..."},
{"rule_name":"some name", "sql_query": "SELECT 'rule name' as Rule, 'true or false' as value FROM TABLE..."},
...]"
7. Make sure the results are valid JSON. 
 """

PROMPT_3="""Summarize the results.
Include the rule definition for each column, the SQL query used, and the results of the SQL query. 

Return the results in a JSON object without any additional notation. Make sure the results are valid JSON.
"""

class DQAgent():

    def exec_sql(self,sql_query):
        conn = sqlite3.connect("demo.sqlite")
        df = pd.read_sql_query(sql_query,conn)
        return json.loads(df.to_json(orient="records", index=False)) 
    
    def run(self,sample_data, stop="") :

        self.msg=[]
        self.sample_data=sample_data
        
        #Step 1: Prompt with System Prompt and data       
        self.append_msg("system",PROMPT_1+f"{DELIMITER}{sample_data}{DELIMITER}")
        #Call the model and append the message with the results
        if stop=="rules":
            return self.call_model()
        self.append_msg("assistant", self.call_model())

        #Step 2: Append the message with the SQL instructions Prompt   
        self.append_msg("user",PROMPT_2)
        if stop=="sql":
            return self.call_model()
        #Get results and query the database
        response = self.call_model()
        df = pd.DataFrame(json.loads(response)['output'])
        df2 = df['sql_query'].apply(lambda x:self.exec_sql(x))

        #Step 3: Append the output of the SQL queries to the message
        self.append_msg("assistant",df2.to_json(orient="records", index=False))

        #Step 4: Append the message with the Summarization Prompt    
        self.append_msg("user",PROMPT_3)

        #Return Results
        return self.call_model()

   
    def call_model(self):
        #Call Open AI API
        response = openai.chat.completions.create(
            model=self.model,
            response_format={ "type": "json_object" },
            messages=self.msg,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        self.response = response
        self.usage = response.usage
        return response.choices[0].message.content
    

    def append_msg(self,role, message):
        self.msg.append({'role':role,'content':message})

    def __init__(self,model=DEPLOYMENT_NAME,delimiter=DELIMITER, temperature=TEMPERATURE, max_tokens=MAX_TOKENS):
        self.delimiter = delimiter
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens


