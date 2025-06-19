import sys
sys.path.insert(0,'/ddnB/work/borun22/.cache/borun_torch/')
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from template import *
import random
import re
from dual_smile_process import *
import json

monomer1,monomer2, er,tg = process_dual_monomer_data('data/unique_smiles_Er.csv','data/smiles.xlsx')
monomer1, monomer2 = np.array(monomer1), np.array(monomer2)
print(len(monomer1))


model_path = "/ddnB/work/borun22/Transfer_learning/NewCOT/LLM/MistraAI/mistra_qlora_finetuned/"

# ✅ Load tokenizer as usual
#tokenizer = AutoTokenizer.from_pretrained(model_path)

# ✅ Load model using FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    load_in_4bit = True,
    max_seq_length = 2048
) 
FastLanguageModel.for_inference(model)
 # returns (model, tokenizer) tuple



def build_prompt(conversation):
    prompt = ""
    for msg in conversation:
        #role = msg["from"]
        prompt += f"<|user|>\n{msg['value']}\n" if msg["from"] == "human" else f"<|assistant|>\n{msg['value']}\n"
    prompt += "<|assistant|>\n"  # Model will respond next
    return prompt

def generate_prompt(prompt, isFinal=False, temperature=0.6):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    num_samples = 2 if isFinal else 1
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        num_return_sequences=num_samples,
        pad_token_id=tokenizer.eos_token_id
    )

    # Decode all responses
    generated_responses = []
    for i in range(num_samples):
        decoded = tokenizer.decode(outputs[i], skip_special_tokens=True)
        response = decoded[len(prompt):].strip()
        generated_responses.append(response)

    return generated_responses



# # Example conversation
# conversation = []

# # Turn 1
# conversation.append({"from": "human", "value": "I need a thermoset shape memory polymer"})
# assistant_reply = generate_prompt(build_prompt(conversation), False)
# conversation.append({"from": "assistant", "value": assistant_reply})
# #print("Assistant:", assistant_reply)

# # Turn 2
# conversation.append({"from": "human", "value": "property based TSMPs I need"})
# assistant_reply = generate_prompt(build_prompt(conversation), False)
# conversation.append({"from": "assistant", "value": assistant_reply})
# #print("Assistant:", assistant_reply)

# # Turn 3
# conversation.append({"from": "human", "value": "Please design a TSMP with Tg=100C and stress recovery=50MPa"})
# assistant_reply = generate_prompt(build_prompt(conversation), True)
# conversation.append({"from": "assistant", "value": assistant_reply})
#print("Assistant:", assistant_reply)

def extract_monomers2(responses):
    generated_samples = [] 
    monomer2, monomer1=None,None
    
       
    response = responses
        
    match = re.search(r"Monomer 1(?:\s*\(.*?\))?:\s*([A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]+)\s*"
                          r"Monomer 2(?:\s*\(.*?\))?:\s*([A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]+)",
                          response,
                          re.MULTILINE
                          )
    if match:
      monomer1 = match.group(1)
      monomer2 = match.group(2)
      #generated_samples.append([monomer1, monomer2])
    return monomer1,monomer2
    
def check_unique_samples(generated_responses, prompt, temperature):
    unique_pairs = []
    duplicates = []
    for index,smiles in enumerate(generated_responses):
        temperature = temperature
        smile1 = smiles[0]
        smile2 = smiles[1]
        print("Monomer 1:" , smile1)
        print("Monomer 2:", smile2)
        if smile1 == None or smile2==None:
          return [],[]
          
        found=False
        combined_smiles = list(zip(monomer1,monomer2))
        input_smile_combined = (smile1,smile2)
        reversed_smile_combined = (smile2,smile1)

        if input_smile_combined in combined_smiles or reversed_smile_combined in combined_smiles:
            found = True
        
        if not found:
            mol1= Chem.MolFromSmiles(smile1)
            mol2=Chem.MolFromSmiles(smile2)
            if mol1 != None and mol2 != None and smile1 != smile2 and smile1 != '' and smile2 != '':
                reaction, groups= check_reaction_validity(smile1,smile2)
                print("New Smiles 1: ", smile1, "New Smiles 2: ", smile2, "Reaction: ",reaction, "Groups: ", groups)
                unique_pairs.append({
                        'smile1': smile1,
                        'smile2': smile2,
                        'reaction': reaction,
                        'groups': groups,
                        'prompt': prompt,
                        'temperature': temperature
                        })
        else:
            duplicates.append({
                'smile1': smile1,
                'smile2': smile2,
                'prompt': prompt,
                'temperature': temperature
            })
    return unique_pairs, duplicates


def create_properties_prompt():
    
    final_responses = []
    temperatures = [1.6]#[0.5,0.8,1.0,1.3,1.6,2.0]
    for temp in temperatures:
      for i in range(len(TEST_PROPERTIES)):
        conversation = []
        
        starter_prompt = random.choice(conversational_tsmp_templates)
        conversation.append({"from": "human", "value": starter_prompt})
        assistant_reply = generate_prompt(build_prompt(conversation), False,temperature=temp)[0]
        conversation.append({"from": "assistant", "value": assistant_reply})
        
        property_prompt = random.choice(property_preference_responses)
        conversation.append({"from": "human", "value": property_prompt})
        assistant_reply = generate_prompt(build_prompt(conversation), False,temperature=temp)[0]
        conversation.append({"from": "assistant", "value": assistant_reply})
        
        user_prompt = random.choice(USER_PROPERTY_PROMPT)
        user_prompt = user_prompt.format(Tg=TEST_PROPERTIES[i]['Tg'], Er=TEST_PROPERTIES[i]['Er'])
        conversation.append({"from": "human", "value": user_prompt})
        assistant_reply = generate_prompt(build_prompt(conversation), True,temperature=temp)
        conversation.append({"from": "assistant", "value": assistant_reply})

        for i,msg in enumerate(conversation):
          role = "User" if msg["from"] == "human" else "Assistant"
          #print(f"{role}: {msg['value']}\n")
          entry = {
            "role": role,
            "message": msg["value"]
          }
          
          if isinstance(msg['value'], list)==True:
            processed_samples = []
            for j in range(len(msg['value'])):
               #print(f"{ j }:-> {msg['value'][j]}\n")
               value=msg['value'][j]
               
               monomer1,monomer2 = extract_monomers2(msg['value'][j])
               print(monomer1,monomer2)
               
               unique_pairs, duplicates = check_unique_samples([[monomer1,monomer2]], user_prompt, temp)
               processed_samples.append( {'unique_pairs': unique_pairs, 'duplicates': duplicates})
            entry["processed_samples"] = processed_samples
            
               
          final_responses.append(entry)
          print(final_responses)
               
       
       
    return final_responses
    
def create_group_prompt():
    
    final_responses = []
    temperatures = [1.6]#[0.5,0.8,1.0,1.3,1.6,2.0]
    for temp in temperatures:
      for i in range(len(TEST_PROPERTIES)):
        conversation = []
        
        starter_prompt = random.choice(conversational_tsmp_templates)
        conversation.append({"from": "human", "value": starter_prompt})
        assistant_reply = generate_prompt(build_prompt(conversation), False,temperature=temp)[0]
        conversation.append({"from": "assistant", "value": assistant_reply})
        
        property_prompt = random.choice(group_preference_responses)
        conversation.append({"from": "human", "value": property_prompt})
        assistant_reply = generate_prompt(build_prompt(conversation), False,temperature=temp)[0]
        conversation.append({"from": "assistant", "value": assistant_reply})
        
        user_prompt = random.choice(USER_GROUP_PROMPT)
        user_prompt = user_prompt.format(Group1=TEST_PROPERTIES[i]['Group1'], Group2=TEST_PROPERTIES[i]['Group2'])
        conversation.append({"from": "human", "value": user_prompt})
        assistant_reply = generate_prompt(build_prompt(conversation), True,temperature=temp)
        conversation.append({"from": "assistant", "value": assistant_reply})

        for i,msg in enumerate(conversation):
          role = "User" if msg["from"] == "human" else "Assistant"
          #print(f"{role}: {msg['value']}\n")
          entry = {
            "role": role,
            "message": msg["value"]
          }
          
          if isinstance(msg['value'], list)==True:
            processed_samples = []
            for j in range(len(msg['value'])):
               #print(f"{ j }:-> {msg['value'][j]}\n")
               value=msg['value'][j]
               
               monomer1,monomer2 = extract_monomers2(msg['value'][j])
               print(monomer1,monomer2)
               
               unique_pairs, duplicates = check_unique_samples([[monomer1,monomer2]], user_prompt, temp)
               processed_samples.append( {'unique_pairs': unique_pairs, 'duplicates': duplicates})
            entry["processed_samples"] = processed_samples
            
               
          final_responses.append(entry)
          print(final_responses)
               
       
       
    return final_responses
    
def create_mix_prompt():
    
    final_responses = []
    temperatures = [1.6]# [0.5,0.8,1.0,1.3,1.6,2.0]
    for temp in temperatures:
      for i in range(len(TEST_PROPERTIES)):
        conversation = []
        
        starter_prompt = random.choice(conversational_tsmp_templates)
        conversation.append({"from": "human", "value": starter_prompt})
        assistant_reply = generate_prompt(build_prompt(conversation), False,temperature=temp)[0]
        conversation.append({"from": "assistant", "value": assistant_reply})
        
        property_prompt = random.choice(both_preference_responses)
        conversation.append({"from": "human", "value": property_prompt})
        assistant_reply = generate_prompt(build_prompt(conversation), False,temperature=temp)[0]
        conversation.append({"from": "assistant", "value": assistant_reply})
        
        user_prompt = random.choice(MIX_PROMPT)
        user_prompt = user_prompt.format(Group1=TEST_PROPERTIES[i]['Group1'], Group2=TEST_PROPERTIES[i]['Group2'],
                                         Tg=TEST_PROPERTIES[i]['Tg'], Er=TEST_PROPERTIES[i]['Er'])
        conversation.append({"from": "human", "value": user_prompt})
        assistant_reply = generate_prompt(build_prompt(conversation), True,temperature=temp)
        conversation.append({"from": "assistant", "value": assistant_reply})

        for i,msg in enumerate(conversation):
          role = "User" if msg["from"] == "human" else "Assistant"
          #print(f"{role}: {msg['value']}\n")
          entry = {
            "role": role,
            "message": msg["value"]
          }
          
          if isinstance(msg['value'], list)==True:
            processed_samples = []
            for j in range(len(msg['value'])):
               #print(f"{ j }:-> {msg['value'][j]}\n")
               value=msg['value'][j]
               
               monomer1,monomer2 = extract_monomers2(msg['value'][j])
               print(monomer1,monomer2)
               
               unique_pairs, duplicates = check_unique_samples([[monomer1,monomer2]], user_prompt, temp)
               processed_samples.append( {'unique_pairs': unique_pairs, 'duplicates': duplicates})
            entry["processed_samples"] = processed_samples
            
               
          final_responses.append(entry)
          print(final_responses)
       
    return final_responses
    
final_responses=create_properties_prompt()
if len(final_responses) > 0:
    with open('/ddnB/work/borun22/Transfer_learning/NewCOT/LLM/MistraAI/output_json/mistra_properties_generated_responses_test_m.json', 'w') as f:
        json.dump(final_responses, f, indent=4)
        
final_responses2=create_group_prompt()
if len(final_responses2) > 0:
    with open('/ddnB/work/borun22/Transfer_learning/NewCOT/LLM/MistraAI/output_json/mistra_group_generated_responses_test_m.json', 'w') as f:
        json.dump(final_responses2, f, indent=4)
final_responses3=create_mix_prompt()
if len(final_responses3) > 0:
    with open('/ddnB/work/borun22/Transfer_learning/NewCOT/LLM/MistraAI/output_json/mistra_mix_generated_responses_test_m.json', 'w') as f:
         json.dump(final_responses3, f, indent=4)


