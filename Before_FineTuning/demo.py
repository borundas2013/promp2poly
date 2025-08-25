p1=[{"role": "user", "content": "Generate a polymer formulation with Tg ≈ 94°C and Er ≈ 18Mpa, optimized for shape memory applications."}, {"role": "assistant", "content": "Based on your requirements, I suggest the following monomers:\nMonomer 1: C=C(C)C(=O)OCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOC(=O)C(=C)C\nMonomer 2: C=Cc1cccc(C=C)c1"}]
p2=[{"role": "user", "content": "I need a polymer system with Tg around 167°C and Er near 29Mpa. Suggest monomer candidates."}, {"role": "assistant", "content": "Here is a TSMP designed for Tg ≈ 167°C and Er ≈ 29Mpa:\nMonomer 1: C=CC(=O)OC(C)(C)C\nMonomer 2: C=C(C)C(=O)OCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOC(=O)C(=C)C"}]

import re



# Example usage
if __name__ == "__main__":
    # Extract from p1 user content
    p1_user_content = p1[0]["content"]
    p1_assistant_content = p1[1]["content"]

    print("P1 User Content:", p1_user_content)
    print("P1 Assistant Content:", p1_assistant_content)

    

