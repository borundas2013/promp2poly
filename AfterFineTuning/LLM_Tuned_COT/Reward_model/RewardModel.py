from NoveltyRewardModel.novelyRewardModel import NoveltyRewardModel
from GroupRewardModel.GroupRewardModel import GroupRewardModel
from PropertyRewards.property_rewards import PropertyRewardCalculator
from dual_smile_process import process_dual_monomer_data
import re
import Constants

class RewardModel:
    def __init__(self, train_data_list):
        self.novelty_model = NoveltyRewardModel()
        self.novelty_model.set_training_pairs(train_data_list)
        self.group_reward_model = GroupRewardModel()
        self.property_reward_model = PropertyRewardCalculator()

        self.weight_novelty = 0.4   
        self.weight_group = 0.3
        self.weight_tg = 0.15
        self.weight_er = 0.15

    def clip_reward(self,reward, min_value=0.0, max_value=1.0):
        return max(min(reward, max_value), min_value)
    
    @staticmethod
    def extract_monomers(responses):
        monomer2, monomer1 = None, None
        
        response = responses
            
        match = re.search(r"Monomer 1(?:\s*\(.*?\))?:\s*([A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]+)\s*"
                         r"Monomer 2(?:\s*\(.*?\))?:\s*([A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]+)",
                         response,
                         re.MULTILINE
                         )
        if match:
            monomer1 = match.group(1)
            monomer2 = match.group(2)
            
        return monomer1, monomer2
    
    def get_property_from_prompt(self, prompt):
        text = prompt.lower()  # Convert to lowercase for case-insensitive matching

        # More flexible patterns to catch Tg
        # More flexible patterns to catch Tg
        # tg_patterns = [
        #     r"(?:tg|glass transition temperature|Tg:?)\s*(?:of|:|around|near|approximately)?\s*(\d+\.?\d*)\s*(?:°c|c|degrees)?",
        #     r"(\d+\.?\d*)\s*(?:°c|c|degrees)\s*(?:tg|glass transition temperature)"
        # ]

        # # More flexible patterns to catch Er
        # er_patterns = [
        #     r"(?:er|elastic recovery|stress recovery|Er:?)\s*(?:of|:|around|near|approximately)?\s*(\d+\.?\d*)\s*(?:mpa|mega pascal)?",
        #     r"(\d+\.?\d*)\s*(?:mpa|mega pascal)\s*(?:er|elastic recovery|stress recovery)"
        # ]

        tg_patterns = [
            # Standard format
            r"(?:tg|glass transition temperature|Tg:?)\s*(?:of|:|=|around|near|approximately|~)?\s*(\d+\.?\d*)\s*(?:°c|°C|C|c|degrees|deg|K|k)?",
            # Number first format
            r"(\d+\.?\d*)\s*(?:°c|°C|C|c|degrees|deg|K|k)?\s*(?:tg|glass transition temperature)",
            # Handle special encoding of degree symbol
            r"(?:tg|glass transition temperature|Tg:?)\s*(?:of|:|=|around|near|approximately|~)?\s*(\d+\.?\d*)\s*(?:Â°c|Â°C)",
            # Very flexible pattern
            r"(?:tg|Tg)[^\d]*(\d+\.?\d*)",
        ]

        # More flexible patterns for Er
        er_patterns = [
            # Standard format
            r"(?:er|elastic recovery|stress recovery|Er:?)\s*(?:of|:|=|around|near|approximately|~)?\s*(\d+\.?\d*)\s*(?:mpa|MPa|mega\s*pascal|MP|mega pascal)?",
            # Number first format
            r"(\d+\.?\d*)\s*(?:mpa|MPa|mega\s*pascal|MP|mega pascal)?\s*(?:er|elastic recovery|stress recovery)",
            # Very flexible pattern
            r"(?:er|Er)[^\d]*(\d+\.?\d*)",
            # Additional format for stress recovery
            r"(?:stress|elastic)\s*recovery[^\d]*(\d+\.?\d*)",
        ]


        # Search for Tg using multiple patterns
        tg_value = None
        for pattern in tg_patterns:
            tg_match = re.search(pattern, text, re.IGNORECASE)
            if tg_match:
                try:
                    tg_value = float(tg_match.group(1))
                    break
                except (ValueError, IndexError):
                    continue

        # Search for Er using multiple patterns
        er_value = None
        for pattern in er_patterns:
            er_match = re.search(pattern, text, re.IGNORECASE)
            if er_match:
                try:
                    er_value = float(er_match.group(1))
                    break
                except (ValueError, IndexError):
                    continue

        # Print the found values
        #print(f"Tg: {tg_value} °C" if tg_value is not None else "Tg: None °C")
        #print(f"Er: {er_value} MPa" if er_value is not None else "Er: None MPa")

        # Return the values
        return tg_value, er_value

    def get_group_from_prompt(self, prompt):
        text = prompt

        # More flexible pattern for extracting group name and group token
        group_patterns = [
            # Pattern 1: Standard format with optional prefix
            r"(?:rich in|containing|with|having|including)?\s*([a-zA-Z0-9\s\-]+)\s*\(\s*([A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]+)\s*\)",
            # Pattern 2: Group name in quotes
            r'"(?:rich in|containing|with|having|including)?\s*([a-zA-Z0-9\s\-]+)"\s*\(\s*([A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]+)\s*\)',
            # Pattern 3: Group name with optional description
            r"(?:rich in|containing|with|having|including)?\s*([a-zA-Z0-9\s\-]+(?:\s+group)?)\s*\(\s*([A-Za-z0-9@+\-\[\]\(\)=#$\\/%.]+)\s*\)"
        ]

        # Find all matches using all patterns
        matches = []
        for pattern in group_patterns:
            matches.extend(re.findall(pattern, text, re.IGNORECASE))
            if len(matches) >= 2:  # Stop if we found both groups
                break

        # Assign monomer1 group and monomer2 group
        group1 = matches[0] if len(matches) > 0 else (None,None)
        group2 = matches[1] if len(matches) > 1 else (None,None)

        # Print the found groups with better handling of None values
        #print(f"Monomer 1 group: {group1[0].strip()} ({group1[1]})" if group1[0] else "Monomer 1 group: None")
        #print(f"Monomer 2 group: {group2[0].strip()} ({group2[1]})" if group2[0] else "Monomer 2 group: None")

        # Return the groups
        if group1[1] not in Constants.GROUP_LIST:
            group1 = None
        else:
            group1 = group1[1]
        if group2[1] not in Constants.GROUP_LIST:
            group2 = None
        else:
            group2 = group2[1]
        
        return group1, group2
    

    def get_reward_from_prompt(self, prompt, responses):
        rewards = []
        
        
        

        for response in responses:
            group1, group2 = self.get_group_from_prompt(prompt)
            tg, er = self.get_property_from_prompt(prompt)
            
            
            monomer1, monomer2 = self.extract_monomers(response)
            print("Prompt: ", prompt)
            print("Response: ",response)
            print("--------------------")
            print("Monomer1: ",monomer1)
            print("Monomer2: ",monomer2)
            print("Group1: ",group1)
            print("Group2: ",group2)
            print("Tg: ",tg)
            print("Er: ",er)
            tg,er=self.get_property_from_prompt(prompt)
            group1,group2=self.get_group_from_prompt(prompt)
            
            reward = self.calculate_reward([monomer1,monomer2], group1, group2, tg, er)
            rewards.append(reward)
        return rewards
    
   
    

    def calculate_reward(self, gen_pair, group1, group2, asking_tg, asking_er):
        novelty_reward = self.novelty_model.calculate_novelty_and_diversity_score(gen_pair)
        group_reward = self.group_reward_model.calculate_reactivity_reward(gen_pair[0], gen_pair[1], group1, group2)
        tg_reward = 0.0
        er_reward = 0.0
        if asking_tg == None or  asking_er == None:
          tg_reward = 0.0
          er_reward = 0.0
        else:
          property_reward = self.property_reward_model.calculate_single_reward(gen_pair[0], gen_pair[1], asking_tg, asking_er)
          tg_reward = self.clip_reward(property_reward['tg_reward'])
          er_reward = self.clip_reward(property_reward['er_reward'])

        #print("Before clipping:")
        #print("Total reward:", total_reward)
        #print("Group reward:", group_reward)
        #print("TG reward:", property_reward['tg_reward'])
        #print("ER reward:", property_reward['er_reward'])

        total_reward = self.clip_reward(novelty_reward)
        group_reward = self.clip_reward(group_reward)
        

        # Calculate final reward using the weights defined in __init__
        final_reward = (
            self.weight_novelty * novelty_reward +
            self.weight_group * group_reward +
            self.weight_tg * tg_reward +
            self.weight_er * er_reward
        )

        print("After clipping:")
        print("Novelty reward:", novelty_reward)
        print("Group reward:", group_reward)
        print("TG reward:", tg_reward)
        print("ER reward:", er_reward)
        print("Final weighted reward:", final_reward)

        return final_reward
    
if __name__ == "__main__":
       
    #     # Example usage
    smiles1, smiles2, er_list, tg_list = process_dual_monomer_data('Data/unique_smiles_Er.csv', 'Data/smiles.xlsx')
    
    reward_model = RewardModel([smiles1, smiles2])
    prompt = "Design a TSMP with a Tg of 40°C and stress recovery of 80 MPa."
    tg,er=reward_model.get_property_from_prompt(prompt)
    print("Prompt:",prompt)
    print("Tg:",tg)
    print("Er:",er)
    prompt = "Provide a pair of monomers suitable for TSMPs, with  epoxy(C1OC1) in monomer1 and  imine(NC) in monomer2."
    group1,group2=reward_model.get_group_from_prompt(prompt)
    if group1 == None or group2 == None:
        group1 = None
        group2 = None
    print("Prompt:",prompt)
    print("Group1:",group1)
    print("Group2:",group2)
    mix_prompt = "Propose two monomers that form a TSMP with Tg around 40C, Er near 80 MPa, and containing epoxy(C1OC1) and imine(NC) groups."
    print("Mix prompt:")
    tg,er=reward_model.get_property_from_prompt(mix_prompt)
    group1,group2=reward_model.get_group_from_prompt(mix_prompt)
    print(tg,er)
    print(group1[1],group2[1])
    # gen_pair = [smiles1[0], smiles2[100]]
    # final_reward =   reward_model.calculate_reward(gen_pair,None, None, 150,150)
    # print(final_reward)
