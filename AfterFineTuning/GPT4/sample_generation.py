from openai import OpenAI
import sys
import os

import json
import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from promp2poly.AfterFineTuning.Data_util.template import *
import random


client=OpenAI(api_key=API_KEY)



def save_results(prompt_data, Tg, Er, Group1, Group2, temperature, output):
    """Save input prompt, groups, temperature, and output to a JSON file"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generation_results_gpt4o_mini_mix.json"
    if not os.path.exists(filename):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump([], f, indent=2, ensure_ascii=False)
    
    result = {
        "timestamp": timestamp,
        "prompt_data": prompt_data,
        "Tg": Tg,
        "Er": Er,
        "Group1": Group1,
        "Group2": Group2,
        "temperature": temperature,
        "output": output
    }
    
    try:
        # Load existing results if file exists
        existing_results = []
        if os.path.exists(filename):
            with open(filename, 'r', encoding='utf-8') as f:
                existing_results = json.load(f)
        
        # Add new result
        existing_results.append(result)
        
        # Save updated results
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(existing_results, f, indent=2, ensure_ascii=False)
            
        print(f"Results saved to {filename}")
    except Exception as e:
        print(f"Error saving results: {e}")

temperatures =[0.3,0.5,0.7,0.9,1.0]
for prompt in TEST_PROPERTIES:
    messages=[]
    group1 = prompt['Group1']
    group2 = prompt['Group2']
    Tg = prompt['Tg']
    Er = prompt['Er']
    # if group1 == "vinyl(C=C)" and group2 == "vinyl(C=C)":
    #     system_prompt = vinyl_system_prompts[0]
    # elif group1 == "epoxy(C1OC1)" and group2 == "imine(NC)":
    #     system_prompt = epoxy_imine_system_prompts[0]
    # elif group1 == "imine(NC)" and group2 == "epoxy(C1OC1)":
    #     system_prompt = epoxy_imine_system_prompts[0]
    # elif group1 == "vinyl(C=C)" and group2 == "thiol(CCS)":
    #     system_prompt = thiol_ene_system_prompts[0]
    # elif group1 == "thiol(CCS)" and group2 == "vinyl(C=C)":
    #     system_prompt = thiol_ene_system_prompts[0]
    # elif group1 == "vinyl(C=C)" and group2 == "hydroxyl(=O)":
    #     system_prompt = hydroxyl_system_prompts[0]
    # elif group1 == "hydroxyl(=O)" and group2 == "vinyl(C=C)":
    #     system_prompt = hydroxyl_system_prompts[0]
    # elif group1 == "acrylate(C=C(C=O))" and group2 == "vinyl(C=C)":
    #     system_prompt = acrylate_vinyl_system_prompts[0]
    # elif group1 == "vinyl(C=C)" and group2 == "acrylate(C=C(C=O))":
    #     system_prompt = acrylate_vinyl_system_prompts[0]

    system_prompt =mixed_functionality_system_prompts[0]
 
   
    messages.append({"role":"system","content":system_prompt})
    #prompt = random.choice(TEST_USER_GROUP_PROMPT).format(Group1=group1, Group2=group2)
    prompt = random.choice(TEST_MIX_PROMPT).format(Tg=Tg, Er=Er,Group1=group1,Group2=group2)
    messages.append({"role":"user","content":prompt})
   
    for temperature in temperatures:
        completion = client.chat.completions.create(
            model=MODEL_ID,
            messages=messages,
            temperature=temperature,
            max_tokens=300,
            n=5)
        
        for i in range(5):
            output_content = completion.choices[i].message.content
            save_results(prompt, Tg, Er, group1, group2, temperature, output_content)





    







