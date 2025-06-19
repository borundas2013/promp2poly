from tools import remove_bond_by_smarts, add_group_by_smarts, get_all_properties
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from pydantic import BaseModel, Field

api_key = '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'

class RemoveBondBySmartsTool(BaseModel):
    smiles1: str = Field(..., description="First molecule (monomer 1)")
    smiles2: str = Field(..., description="Second molecule (monomer 2)")
    bond_smarts: str = Field(..., description="The group/bond pattern to remove (in SMARTS format)")
    target_monomer: str = Field(..., description="Which molecule to modify ('1' or '2')")

class AddGroupBySmartsTool(BaseModel):
    smiles1: str = Field(..., description="First molecule (monomer 1)")
    smiles2: str = Field(..., description="Second molecule (monomer 2)")
    group_smarts: str = Field(..., description="The group to add, defined in SMARTS (must contain [*] as attachment point)")
    target_monomer: str = Field("1", description="Which molecule to modify ('1' or '2')")
    attachment_atom_idx: int = Field(0, description="Atom index in target molecule where the group will attach (default = 0)")

class GetAllPropertiesTool(BaseModel):
    smiles1: str = Field(..., description="First molecule (monomer 1)")
    smiles2: str = Field(..., description="Second molecule (monomer 2)")
    ratio_1: float = Field(0.1, description="Ratio of monomer 1 in the final polymer (default = 0.5)")
    ratio_2: float = Field(0.9, description="Ratio of monomer 2 in the final polymer (default = 0.5)")
    property_type: str = Field("physical", description="Type of property to predict (default = 'physical') from the following options: all, physical, toxicity, solubility")

@tool
def remove_bond_by_smarts_tool(input: RemoveBondBySmartsTool) -> str:
    """
    Remove a specific group or bond from one of two molecules.

    Parameters:
    - smiles1 (str): First molecule (monomer 1)
    - smiles2 (str): Second molecule (monomer 2) 
    - bond_smarts (str): The group/bond pattern to remove (in SMARTS format)
    - target_monomer (str): Which molecule to modify ("1" or "2")

    Returns:
    - str: Result showing both molecules in the format:
           "Here is the revised output: monomer1 = [modified/unchanged] and monomer2 = [modified/unchanged]"
    """
    smiles1 = input.smiles1
    smiles2 = input.smiles2
    bond_smarts = input.bond_smarts
    target_monomer = input.target_monomer
    return remove_bond_by_smarts(smiles1, smiles2, bond_smarts, target_monomer)

@tool
def add_group_by_smarts_tool(input: AddGroupBySmartsTool) -> str:
    """
    Add a functional group or substructure defined by SMARTS to one of two molecules.

    Parameters:
    - smiles1 (str): First molecule (monomer 1)
    - smiles2 (str): Second molecule (monomer 2)
    - group_smarts (str): The group to add, defined in SMARTS (must contain [*] as attachment point)
    - target_monomer (str): Which molecule to modify ("1" or "2")
    - attachment_atom_idx (int): Atom index in target molecule where the group will attach (default = 0)

    Returns:
    - str: Result showing both molecules in the format:
           "Here is the revised output: monomer1 = [modified/unchanged] and monomer2 = [modified/unchanged]"
    """
    smiles1 = input.smiles1
    smiles2 = input.smiles2
    group_smarts = input.group_smarts
    target_monomer = input.target_monomer
    attachment_atom_idx = input.attachment_atom_idx 
    return add_group_by_smarts(smiles1, smiles2, group_smarts, target_monomer, attachment_atom_idx)

@tool
def get_all_properties_tool(input: GetAllPropertiesTool) -> str:
    """
    Get all properties of a given SMILES pair.

    Parameters:
    - smiles1 (str): First molecule (monomer 1)
    - smiles2 (str): Second molecule (monomer 2)
    - ratio_1 (float): Ratio of monomer 1 in the final polymer (default = 0.1)
    - ratio_2 (float): Ratio of monomer 2 in the final polymer (default = 0.9)
    - property_type (str): Type of property to predict (default = "physical")

    Returns:
    - dict: Result showing asked properties of the given SMILES pair
    """
    smiles1 = input.smiles1
    smiles2 = input.smiles2
    ratio_1 = input.ratio_1
    ratio_2 = input.ratio_2
    property_type = input.property_type
    #print("------------Parameters--------------------")
    #print(smiles1, smiles2, ratio_1, ratio_2)
    return get_all_properties(smiles1, smiles2, ratio_1, ratio_2, property_type)





def main():
    llm = ChatOpenAI(model="ft:gpt-4o-mini-2024-07-18:personal::BKodpSOI", api_key=api_key, 
                     temperature=0, max_tokens=1000)
    agent = create_react_agent(llm,
                               tools=[remove_bond_by_smarts_tool, add_group_by_smarts_tool, get_all_properties_tool])

    # query = "Here are two monomers: monomer1 = CCNC1OC1Cc1ccccc1CCCCBr and monomer2 = CCC2OC2COOCC. show the solubility properties of the given monomers."
    # print("User query:", query)
    # response = agent.invoke({"messages": [("human", query)]})
    # #print("Assistant response:", response["messages"][-1].content)
    # tool_names = [call["name"] for msg in response["messages"] 
    #           if hasattr(msg, "tool_calls") and msg.tool_calls 
    #           for call in msg.tool_calls]
    # print("Tools called by LLM:", tool_names)
    # print("Assistant response:", response["messages"][1].content)

    print("----------------Physical Properties----------------")
    query = "Here are two monomers: monomer1 = CCNC1OC1Cc1ccccc1CCCCBr and monomer2 = CCC2OC2COOCC. show the solubility properties of the given monomers."
    
    response = agent.invoke({"messages": [("human", query)]})
    #print(response['messages'][2].content)
    #print("Assistant response:", response["messages"][-1].content)
    tool_names = [call["name"] for msg in response["messages"] 
              if hasattr(msg, "tool_calls") and msg.tool_calls 
              for call in msg.tool_calls]
    print("User query:", query)
    print("Tools called by LLM:", tool_names)
    print("Assistant/Tool response:", response["messages"][2].content)

    # query1 = "Here are two monomers: monomer1 = CCNC1OC1Cc1ccccc1CCCCBr and monomer2 = CCCOOCC. remove CN group from monomer 1."
    # print("\nExample 1: Remove single atom")
    # print("User query:", query1)
    # response1 = agent.invoke({"messages": [("human", query1)]})
    # print("Assistant response:", response1["messages"][-1].content)


    # query2 = "Here are two monomers: monomer1 = O=C(OCC1CO1)C3CC2OC3CC2C(=O)OCC4CO4 and monomer2 = CCC2OC2COOCC. add [*]C(=O)O group to monomer 2."
    # print("\nExample 2:")
    # print("User query:", query2)
    # response2 = agent.invoke({"messages": [("human", query2)]})
    # tool_names = [call["name"] for msg in response2["messages"] 
    #           if hasattr(msg, "tool_calls") and msg.tool_calls 
    #           for call in msg.tool_calls]
    # print("Tools called by LLM:", tool_names)
    # print("Assistant response:", response2["messages"][2].content)

    #print("----------------Remove Group or Bond----------------")

    # # Example 1: Remove N from first molecule
    # query1 = "Here are two monomers: monomer1 = CCNC1OC1Cc1ccccc1CCCCBr and monomer2 = CCCOOCC. remove CN group from monomer 1."
    # print("\nExample 1: Remove single atom")
    # print("User query:", query1)
    # response1 = agent.invoke({"messages": [("human", query1)]})
    # print("Assistant response:", response1["messages"][-1].content)

    # #Example 2: Remove O[O] bonds from second molecule
    # query2 = "Here are two monomers: monomer1 = CCNC1OC1Cc1ccccc1CCCCBr and monomer2 = CCC2OC2COOCC. remove O[O] bonds from monomer 2."
    # print("\nExample 1: Remove specific bond")
    # print("User query:", query2)
    # response2 = agent.invoke({"messages": [("human", query2)]})
    # tool_names = [call["name"] for msg in response2["messages"] 
    #           if hasattr(msg, "tool_calls") and msg.tool_calls 
    #           for call in msg.tool_calls]
    # print("Tools called by LLM:", tool_names)
    # print("Assistant response:", response2["messages"][-1].content)

    # # Example 3: Remove Br from first molecule
    # query3 = "Here are two monomers: monomer1 = CCNC1OC1Cc1ccccc1CCCCBr and monomer2 = CCC2OC2COOCC. remove Br group from monomer 1."
    # print("\nExample 3: Remove halogen")
    # print("User query:", query3)
    # response3 = agent.invoke({"messages": [("human", query3)]})
    # print("Assistant response:", response3["messages"][-1].content)

    # #Example 4: Remove C1OC1 group from first molecule
    # query1 = "Here are two monomers: monomer1 = CCNC1OC1Cc1ccccc1CCCCBr and monomer2 = CCC2OC2COOCC. remove O-O bonds from monomer 2."
    # print("\nExample 1:")
    # print("User query:", query1)
    # response1 = agent.invoke({"messages": [("human", query1)]})
    # tool_names = [call["name"] for msg in response1["messages"] 
    #           if hasattr(msg, "tool_calls") and msg.tool_calls 
    #           for call in msg.tool_calls]
    # print("Tools called by LLM:", tool_names)
    # print("Assistant response:", response1["messages"][2].content)

    # query2 = "Here are two monomers: monomer1 = O=C(OCC1CO1)C3CC2OC3CC2C(=O)OCC4CO4 and monomer2 = CCC2OC2COOCC. add [*]C(=O)O group to monomer 2."
    # print("\nExample 2:")
    # print("User query:", query2)
    # response2 = agent.invoke({"messages": [("human", query2)]})
    # tool_names = [call["name"] for msg in response2["messages"] 
    #           if hasattr(msg, "tool_calls") and msg.tool_calls 
    #           for call in msg.tool_calls]
    # print("Tools called by LLM:", tool_names)
    # print("Assistant response:", response2["messages"][2].content)

    # # Example 5: Remove c1ccccc1 (benzene ring) from first molecule
    # query3 = "Here are two monomers: monomer1 = CCNC1OC1Cc1ccccc1CCCCBr and monomer2 = CCC2OC2COOCC. remove c1ccccc1 group from monomer 1."
    # print("\nExample 3:")
    # print("User query:", query3)
    # response3 = agent.invoke({"messages": [("human", query3)]})
    # tool_names = [call["name"] for msg in response3["messages"] 
    #           if hasattr(msg, "tool_calls") and msg.tool_calls 
    #           for call in msg.tool_calls]
    # print("Tools called by LLM:", tool_names)
    # print("Assistant response:", response3["messages"][2].content)

    # #print("----------------Add Group----------------")

    # # Example 1: Add OH group to first molecule
    # # query1 = "Here are two monomers: monomer1 = CCNC1OC1Cc1ccccc1CCCCBr and monomer2 = CCCOOCC. add [*]O group to monomer 1."
    # # print("\nExample 1: Add hydroxyl group")
    # # print("User query:", query1)
    # # response1 = agent.invoke({"messages": [("human", query1)]})
    # # print("Assistant response:", response1["messages"][-1].content)

    # # Example 2: Add COOH group to second molecule
    

    # # Example 3: Add benzene ring to first molecule
    # query4 = "Here are two monomers: monomer1=O=C(OCC1CO1)C3CC2OC3CC2C(=O)OCC4CO4 and monomer2=CCC2OC2COOCC. add [*]c1ccccc1 group to monomer2."
    # print("\nExample 4:")
    # print("User query:", query4)
    # response4 = agent.invoke({"messages": [("human", query4)]})
    # tool_names = [call["name"] for msg in response4["messages"] 
    #           if hasattr(msg, "tool_calls") and msg.tool_calls 
    #           for call in msg.tool_calls]
    # print("Tools called by LLM:", tool_names)
    # print("Assistant response:", response4["messages"][2].content)

    # # # Example 4: Add CH3 group to second molecule
    # # query4 = "Here are two monomers: monomer1 = CCNC1OC1Cc1ccccc1CCCCBr and monomer2 = CCC2OC2COOCC. add [*]C group to monomer 2."
    # # print("\nExample 4: Add methyl group")
    # # print("User query:", query4)
    # # response4 = agent.invoke({"messages": [("human", query4)]})
    # # print("Assistant response:", response4["messages"][-1].content)

    # # Example 5: Add Cl group to first molecule
    # query5 = "Here are two monomers: monomer1 = CCNC1OC1Cc1ccccc1CCCCBr and monomer2 = CCC2OC2COOCC. add [*]Cl group to monomer 1."
    # print("\nExample 5:")
    # print("User query:", query5)
    # response5 = agent.invoke({"messages": [("human", query5)]}) 
    # tool_names = [call["name"] for msg in response5["messages"] 
    #           if hasattr(msg, "tool_calls") and msg.tool_calls 
    #           for call in msg.tool_calls]
    # print("Tools called by LLM:", tool_names)
    # print("Assistant response:", response5["messages"][2].content)

    # # Example 6: Try to remove C=C bond (not present in monomers)
    # query6 = "Here are two monomers: monomer1 = CCNC1OC1Cc1ccccc1CCCCBr and monomer2 = CCC2OC2COOCC. remove N=N bonds from monomer 1."
    # print("\nExample 6:")
    # print("User query:", query6)
    # response6 = agent.invoke({"messages": [("human", query6)]}) 
    # tool_names = [call["name"] for msg in response6["messages"] 
    #           if hasattr(msg, "tool_calls") and msg.tool_calls 
    #           for call in msg.tool_calls]
    # print("Tools called by LLM:", tool_names)
    # print("Assistant response:", response6["messages"][2].content)

if __name__ == "__main__":
   main()
    # smiles1 = 'CCNC1OC1Cc1ccccc1CCCCBr'
    # smiles2 = 'CCC2OC2COOCC'
    # ratio_1 = 0.1
    # ratio_2 = 0.9
    # pred_result = get_all_properties(smiles1, smiles2, ratio_1, ratio_2, 'toxicity')
    # print(pred_result)
