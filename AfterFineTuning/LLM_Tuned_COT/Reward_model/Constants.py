GROUP_VOCAB = {"C1OC1": 0, "NC": 1,"Nc":2, "C=C":3, "CCS":4,"C=C(C=O)":5, "c1ccccc1":6,"C1=CC=CC=C1":7}
GROUP_SIZE = len(GROUP_VOCAB)
EMBEDDING_DIM = 64
LATENT_DIM = 128
#VOCAB_SIZE = 21
EPOCHS = 10
BATCH_SIZE = 16

EPOXY_SMARTS = "C1OC1"#"[OX2]1[CX3][CX3]1"    # Epoxy group
IMINE_SMARTS = "NC"          # Imine group
VINYL_SMARTS = "C=C"                  # Vinyl group
THIOL_SMARTS = "CCS"                  # Thiol group
ACRYL_SMARTS = "C=C(C=O)"             # Acrylic group
BEZEN_SMARTS = "c1ccccc1"
EPOXY_2_SMARTS = "C2OC2"
EPOXY_3_SMARTS = "C3OC3"
EPOXY_4_SMARTS = "C4OC4"
EPOXY_5_SMARTS = "C5OC5"
EPOXY_6_SMARTS = "C6OC6"
EPOXY_7_SMARTS = "C7OC7"
EPOXY_8_SMARTS = "C8OC8"
EPOXY_9_SMARTS = "C9OC9"

GROUP_LIST = [EPOXY_SMARTS, IMINE_SMARTS, VINYL_SMARTS, THIOL_SMARTS, ACRYL_SMARTS, BEZEN_SMARTS, EPOXY_2_SMARTS, EPOXY_3_SMARTS, EPOXY_4_SMARTS, EPOXY_5_SMARTS, EPOXY_6_SMARTS, EPOXY_7_SMARTS, EPOXY_8_SMARTS, EPOXY_9_SMARTS]

PATIENCE = 10

NOISE_CONFIG = {
    'gaussian': {'enabled': True, 'level': 0.1},
    'dropout': {'enabled': False, 'rate': 0.1},
    'swap': {'enabled': True, 'rate': 0.3},
    'mask': {'enabled': False, 'rate': 0.05}
}

FEEDBACK_COLLECT_EPOCH = 20


VALID_PAIRS_FILE = "valid_pairs_during_training_noise_swap_trainable.json"

VOCAB_PATH = 'Two_Monomers_Group/tokenizers_updated/vocab_1000/vocab.txt'#'tokenizers/vocab_1000/vocab.txt'
TOKENIZER_PATH = 'Two_Monomers_Group/tokenizers_updated/vocab_1000/tokenizer'#tokenizers/vocab_1000/tokenizer'



