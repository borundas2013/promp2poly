from openai import OpenAI
import sys
import os

import json
import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from promp2poly.AfterFineTuning.Data_util.template import *
import random
API_KEY = ""
MODEL_ID = "gpt-4o-mini-2024-07-18"

# API_KEY = ""
# MODEL_ID = "ft:gpt-4o-mini-2024-07-18:personal:tsmp-monomers-v1:C1dbxURY"

client=OpenAI(api_key=API_KEY)

temperatures =[0.3]#[0.3,0.5,0.7,0.9,1.0]

messages=[]
# messages.append({"role":"system","content":thiol_ene_system_prompts[0]})
# messages.append({"role":"user","content":"Generate a thermoset shape memory polymer (TSMP)"})
# messages.append({"role":"assistant","content":"I can generate a TSMP based on chemical groups or target properties. which do you prefer?"})
# messages.append({"role":"user","content":"I want to work with specific chemical groups"})
# messages.append({"role":"assistant","content":"What specific chemical groups are you interested in?"})
# messages.append({"role":"user","content":"Please give me some TSMP with C=C and CCS"})
# messages.append({"role":"assistant","content":"Excellent! I'll design a TSMP with C=C and CCS functionalities for controlled crosslinking."})
# messages.append({"role":"assistant","content":"The following monomers feature your desired functional groups and are suitable for crosslinking:\nMonomer 1 (C=C): C=C(C)C(=O)OCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOC(=O)C(=C)C\nMonomer 2 (CCS): CCCC(=O)OCC(CC)(COC(=O)CCS)COC(=O)CCS"})

# messages.append({"role":"system","content":property_focused_system_prompts[0]})
# messages.append({"role":"user","content":"I need a TSMP"})
# messages.append({"role":"assistant","content":"I can generate a TSMP based on chemical groups or target properties. which do you prefer?"})
# messages.append({"role":"user","content":"I want to work with specific properties"})
# messages.append({"role":"assistant","content":"I can work with Tg and Er. Please give me the Tg and Er values"})
# messages.append({"role":"user","content":"Tg = 306C and Er= 91MPa"})
# messages.append({"role":"assistant","content":"consider this monomer combination:\nMonomer 1: CC(C)(c2ccc(OCCOCCOCCOCC1CO1)cc2)c4ccc(OCCOCCOCCOCC3CO3)cc4\nMonomer 2: COCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCC(C)OCC(C)OCC(C)OCC(C)OCC(C)OCC(C)OCC(C)OCC(C)OCC(C)OCC(C)N"})



user_message_1 =['I want to make a thermoset shape memory polymer','Please suggest me some TSMPs']
user_message_2 =['Please focus on property based monomer pairs','Please focus on group based monomer pairs', "both"]
proeprty_specific_message = ["Please give me some TSMP with Tg = 100C and Er= 150MPa","Please generate some TSMP with Tg = 50C and Er= 100Mpa"]
group_specific_message = ["Please give me some TSMP with epoxy(C1OC1) groups in monomer 1 and imine(NC) groups in monomer 2","Please generate some TSMP with vinyl(C=C) groups in monomer 1 and thiol(CCS) groups in monomer 2"]
mixed_specific_message = ["Please give me some TSMP with Tg = 100C and Er= 150MPa and vinyl(C=C) groups in monomer 1 and vinyl(C=C) groups in monomer 2","Please generate some TSMP with Tg = 50C and Er= 100Mpa and Thiol(CCS) groups in monomer 1 and vinyl(C=C) groups in monomer 2"]

# for temperature in temperatures:
#     completion = client.chat.completions.create(
#         model=MODEL_ID,
#         messages=messages,
#         temperature=temperature,
#         max_tokens=300,
#         n=1)

def generate_new_TSMP(role,prompt_content, isFinalQuery=False):
    propmt={"role":role, "content": prompt_content}
    messages.append(propmt)
    if not isFinalQuery:
        completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages
        )
        result = completion.choices[0].message.content
    else:
        completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            temperature=1.0,
            max_tokens=300,
            n=1
        )
        result= completion
       
        
   
    messages.append({"role":'assistant', "content": result})
    
    # Print the response from the assistant
    return result, messages


messages.append({"role":"system","content":property_focused_system_prompts[0]})
replies_0, messages_0 = generate_new_TSMP('user',user_message_1[0], isFinalQuery=False)
print("User: ",user_message_1[0])
print("Assistant: ",replies_0)
replies_1, messages_1 = generate_new_TSMP('user',user_message_2[0], isFinalQuery=False)
print("User: ",user_message_2[0])
print("Assistant: ",replies_1)
replies_2, messages_2 = generate_new_TSMP('user',proeprty_specific_message[0], isFinalQuery=True)
print("User: ",proeprty_specific_message[0])
print("Assistant: Two TSMPs for you ( as I told model to generate two samples per query) ")
for i in range(1):
    print("Sample : :",i+1, replies_2.choices[i].message.content)
    