import os
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
import sys
sys.path.insert(0,'/ddnB/work/borun22/.cache/borun_torch/')
import re
import torch
from torch.optim import AdamW
from torch.nn.functional import log_softmax
from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from typing import List, Tuple, Dict, Optional
from RewardModel import RewardModel
from dual_smile_process import process_dual_monomer_data
import random
from template import *
import copy
from peft import LoraConfig,get_peft_model
import torch.nn.functional as F
import gc
from torch.optim.lr_scheduler import CosineAnnealingLR


class RLHFTrainer:
   
    
    def __init__(
        self, 
        model_path: str, 
        data_paths: Dict[str, str], 
        learning_rate: float = 1e-2,
        max_seq_length: int = 2048,
    ):
        self.device = 'cuda' 
        
        # Initialize data
        try:
            self.smiles1, self.smiles2, self.er_list, self.tg_list = process_dual_monomer_data(
                data_paths['er_path'], 
                data_paths['smiles_path']
            )
            
        except Exception as e:
            raise RuntimeError(f"Failed to load training data: {str(e)}")
            
        # Initialize model and tokenizer
        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=model_path,
                load_in_4bit=True,
                max_seq_length=max_seq_length,
                output_hidden_states=True,
                
            )

            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj",],
                lora_alpha = 16,
                lora_dropout = 0, # Supports any, but = 0 is optimized
                bias = "none",    # Supports any, but = "none" is optimized
                # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
                use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
                random_state = 3407,
                use_rslora = False,  # We support rank stabilized LoRA
                loftq_config = None, # And LoftQ
            )
            self.model = self.model.to(self.device)
            self.learning_rate =  1e-3

            # Initialize value head
            self.value_head = torch.nn.Linear(
                self.model.config.hidden_size,  # Use model's hidden size
                1,
                device=self.device,
                dtype=torch.float32
            )
            torch.nn.init.uniform_(self.value_head.weight, -0.01, 0.01)
            
            # Add value head parameters to optimizer
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if "lora" in n.lower()],
                    "weight_decay": 0.0,
                    "lr": learning_rate
                },
                {
                    "params": self.value_head.parameters(),
                    "weight_decay": 0.0,
                    "lr": learning_rate
                }
            ]
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model: {str(e)}")
            
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        # Initialize reward model
        self.reward_model = RewardModel([self.smiles1, self.smiles2])

        # Add learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=10,
            eta_min=1e-5
        )

   

    def sample_n_responses(self, prompt, n_samples=10, temperature=0.7, max_new_tokens=256):
        try:
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
            outputs = self.model.generate(
                **inputs,
                do_sample=True,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                num_return_sequences=n_samples,
                pad_token_id=self.tokenizer.eos_token_id,
                top_p=0.9,
                
                repetition_penalty=1.2
            )
            #top_k=50,
            
            generated_responses = []
            for i in range(n_samples):
                try:
                    decoded = self.tokenizer.decode(outputs[i], skip_special_tokens=True)
                    response = decoded[len(prompt):].strip()
                    if response and len(response) > 0:  # Ensure response is not empty
                        generated_responses.append(response)
                except Exception as e:
                    print(f"Error decoding response {i}: {str(e)}")
                    continue
                
            if not generated_responses:
                print("Warning: No valid responses generated")
                return [""]  # Return empty string instead of None
            
            return generated_responses
            
        except Exception as e:
            print(f"Error in sample_n_responses: {str(e)}")
            return [""]  # Return empty string instead of None

    def select_top_k_responses(self, prompt, responses, k=2):
       
        
        rewards = self.reward_model.get_reward_from_prompt(prompt, responses)
        rewards_tensor = torch.tensor(rewards, device=self.device)
        sorted_indices = torch.argsort(rewards_tensor, descending=True)
        sorted_indices = sorted_indices.cpu()
        top_responses = [responses[i] for i in sorted_indices[:k]]
        top_rewards = [rewards[i] for i in sorted_indices[:k]]
        return top_responses, top_rewards
    
    def build_prompt(self, conversation):
        prompt = ""
        for msg in conversation:
            #role = msg["from"]
            prompt += f"<|user|>\n{msg['value']}\n" if msg["from"] == "human" else f"<|assistant|>\n{msg['value']}\n"
        prompt += "<|assistant|>\n"  # Model will respond next
        return prompt

   

    def compute_a2c_loss(self, prompt_ids, response_ids, rewards):
        try:
            # Forward pass through model to get policy logits and value estimates
            outputs = self.model(
                input_ids=torch.cat([prompt_ids, response_ids], dim=1),
                attention_mask=torch.ones_like(torch.cat([prompt_ids, response_ids], dim=1)),
                output_hidden_states=True
            )
            
            # Convert all tensors to float32 for consistent computation
            logits = outputs.logits.to(torch.float32) / 50.0  # Increased scaling factor
            
            # Get logits for policy (actor)
            policy_logits = logits[:, prompt_ids.shape[1]-1:-1, :]
            log_probs = F.log_softmax(policy_logits, dim=-1)
            
            # Compute action log probabilities with stricter clipping
            action_log_probs = torch.gather(
                log_probs, 
                -1, 
                response_ids.unsqueeze(-1)
            ).squeeze(-1).sum(dim=-1)
            action_log_probs = torch.clamp(action_log_probs, min=-5.0, max=0.0)  # Tighter clipping
            
            # Get the last hidden state and convert to float32
            last_hidden = outputs.hidden_states[-1][:, -1, :].to(torch.float32)
            
            # Use the class's value head
            values = self.value_head(last_hidden)
            
            # More conservative reward normalization
            rewards_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float32)
            rewards_mean = rewards_tensor.mean()
            rewards_std = rewards_tensor.std().clamp(min=1.0)  # Prevent too small std
            rewards_normalized = (rewards_tensor - rewards_mean) / rewards_std
            rewards_normalized = torch.clamp(rewards_normalized, min=-2.0, max=2.0)  # Clip normalized rewards
            
            # Compute advantages with tighter clipping
            advantages = (rewards_normalized - values.squeeze()).detach()
            advantages = torch.clamp(advantages, min=-2.0, max=2.0)  # Tighter advantage clipping
            
            # Policy (Actor) loss with smaller coefficient
            actor_loss = -(action_log_probs * advantages).mean() * 0.05  # Reduced actor impact
            
            # Value (Critic) loss with stronger regularization
            critic_loss = F.mse_loss(values.squeeze(), rewards_normalized) + \
                          0.1 * (values ** 2).mean() + \
                          0.01 * torch.norm(self.value_head.weight)
            
            # Entropy loss with smaller coefficient
            probs = torch.exp(log_probs)
            entropy_loss = -(probs * log_probs).sum(dim=-1).mean() * 0.0005  # Reduced entropy impact
            
            # Combined loss with adjusted coefficients
            total_loss = (
                actor_loss +          # Policy improvement (scaled down)
                0.5 * critic_loss +   # Reduced value estimation impact
                entropy_loss          # Minimal exploration bonus
            )
            
            # Print diagnostics
            print(f"\nA2C Loss Components:")
            print(f"Actor Loss: {actor_loss.item():.4f}")
            print(f"Critic Loss: {critic_loss.item():.4f}")
            print(f"Entropy Loss: {entropy_loss.item():.4f}")
            print(f"Advantages Mean: {advantages.mean().item():.4f}")
            print(f"Advantages Std: {advantages.std().item():.4f}")
            
            return total_loss
            
        except Exception as e:
            print(f"Error in compute_a2c_loss: {str(e)}")
            print(f"Shapes - prompt_ids: {prompt_ids.shape}, response_ids: {response_ids.shape}")
            print(f"Rewards: {rewards}")
            raise

    def _train_step_batch(self, prompts: List[str], responses: List[str], rewards: List[float]) -> float:
        try:
            # Zero gradients
            self.optimizer.zero_grad()
            
            
            # Encode inputs
            prompt_encodings = self.tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=256
            )
            response_encodings = self.tokenizer(
                responses, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=256
            )
            
            prompt_ids = prompt_encodings['input_ids'].to(self.device)
            response_ids = response_encodings['input_ids'].to(self.device)
            
            # Use gradient scaling for mixed precision training
            # Compute A2C loss directly
            loss = self.compute_a2c_loss(prompt_ids, response_ids, rewards)
            
            
            # Standard backward pass
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]
            print(f"Current learning rate: {current_lr:.6f}")

            

            return loss.item()
            
        except Exception as e:
            print(f"Error in _train_step_batch: {str(e)}")
            raise

   

    
    
    def prepare_properties_conversation(self, i, temp):
        try:
            conversation = []
            starter_prompt = random.choice(conversational_tsmp_templates)
            conversation.append({"from": "human", "value": starter_prompt})
            
            # Get initial response
            responses = self.sample_n_responses(self.build_prompt(conversation), n_samples=1, temperature=temp)
            if not responses or not responses[0]:
                print("Warning: Empty initial response")
                return None, None
            conversation.append({"from": "assistant", "value": responses[0]})
            
            # Get property response
            property_prompt = random.choice(property_preference_responses)
            conversation.append({"from": "human", "value": property_prompt})
            responses = self.sample_n_responses(self.build_prompt(conversation), n_samples=1, temperature=temp)
            if not responses or not responses[0]:
                print("Warning: Empty property response")
                return None, None
            conversation.append({"from": "assistant", "value": responses[0]})
            
            # Get final responses
            user_prompt = random.choice(USER_PROPERTY_PROMPT)
            user_prompt = user_prompt.format(Tg=TEST_PROPERTIES[i]['Tg'], Er=TEST_PROPERTIES[i]['Er'])
            conversation.append({"from": "human", "value": user_prompt})
            responses = self.sample_n_responses(self.build_prompt(conversation), n_samples=20, temperature=temp)
            if not responses:
                print("Warning: No valid responses generated")
                return None, None
            conversation.append({"from": "assistant", "value": responses})
            
            return conversation, user_prompt
            
        except Exception as e:
            print(f"Error in prepare_properties_conversation: {str(e)}")
            return None, None
    def prepare_group_conversation(self, i, temp):
        try:
            conversation = []
            starter_prompt = random.choice(conversational_tsmp_templates)
            conversation.append({"from": "human", "value": starter_prompt})
            
            # Get initial response
            responses = self.sample_n_responses(self.build_prompt(conversation), n_samples=1, temperature=temp)
            if not responses or not responses[0]:
                print("Warning: Empty initial response in group conversation")
                return None, None
            conversation.append({"from": "assistant", "value": responses[0]})

            # Get property response
            property_prompt = random.choice(group_preference_responses)
            conversation.append({"from": "human", "value": property_prompt})
            responses = self.sample_n_responses(self.build_prompt(conversation), n_samples=1, temperature=temp)
            if not responses or not responses[0]:
                print("Warning: Empty property response in group conversation")
                return None, None
            conversation.append({"from": "assistant", "value": responses[0]})

            # Get final responses
            user_prompt = random.choice(USER_GROUP_PROMPT)
            user_prompt = user_prompt.format(Group1=TEST_PROPERTIES[i]['Group1'], Group2=TEST_PROPERTIES[i]['Group2'])
            conversation.append({"from": "human", "value": user_prompt})
            responses = self.sample_n_responses(self.build_prompt(conversation), n_samples=20, temperature=temp)
            if not responses:
                print("Warning: No valid responses generated in group conversation")
                return None, None
            conversation.append({"from": "assistant", "value": responses})
            
            return conversation, user_prompt
            
        except Exception as e:
            print(f"Error in prepare_group_conversation: {str(e)}")
            return None, None
    def prepare_mix_conversation(self, i, temp):
        try:
            conversation = []
            starter_prompt = random.choice(conversational_tsmp_templates)
            conversation.append({"from": "human", "value": starter_prompt})
            
            # Get initial response
            responses = self.sample_n_responses(self.build_prompt(conversation), n_samples=1, temperature=temp)
            if not responses or not responses[0]:
                print("Warning: Empty initial response in mix conversation")
                return None, None
            conversation.append({"from": "assistant", "value": responses[0]})

            # Get property response
            property_prompt = random.choice(both_preference_responses)
            conversation.append({"from": "human", "value": property_prompt})
            responses = self.sample_n_responses(self.build_prompt(conversation), n_samples=1, temperature=temp)
            if not responses or not responses[0]:
                print("Warning: Empty property response in mix conversation")
                return None, None
            conversation.append({"from": "assistant", "value": responses[0]})

            # Get final responses
            user_prompt = random.choice(MIX_PROMPT)
            user_prompt = user_prompt.format(
                Group1=TEST_PROPERTIES[i]['Group1'], 
                Group2=TEST_PROPERTIES[i]['Group2'],
                Tg=TEST_PROPERTIES[i]['Tg'], 
                Er=TEST_PROPERTIES[i]['Er']
            )
            conversation.append({"from": "human", "value": user_prompt})
            responses = self.sample_n_responses(self.build_prompt(conversation), n_samples=20, temperature=temp)
            if not responses:
                print("Warning: No valid responses generated in mix conversation")
                return None, None
            conversation.append({"from": "assistant", "value": responses})
            
            return conversation, user_prompt
            
        except Exception as e:
            print(f"Error in prepare_mix_conversation: {str(e)}")
            return None, None

    def train(
        self,
        num_epochs: int = 1, 
        k: int = 5,
        checkpoint_dir: Optional[str] = None
    ):
        self.configure_optimizer()
        
        # Add training progress tracking
        best_reward = float('-inf')
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_reward = 0.0
            num_batches = 0  # Add counter for number of batches
            temperatures = [0.3]
            
            for j in range(3):
                print(f"---------{j}------------")
                for temp in temperatures:
                    for i in range(len(TEST_PROPERTIES[:2])):
                        try:
                            # Get conversation
                            if j == 0:
                                conversation, user_prompt = self.prepare_properties_conversation(i, temp)
                            elif j == 1:    
                                conversation, user_prompt = self.prepare_group_conversation(i, temp)
                            else:
                                conversation, user_prompt = self.prepare_mix_conversation(i, temp)
                            
                            # Skip if conversation preparation failed
                            if conversation is None or user_prompt is None:
                                print(f"Skipping iteration due to failed conversation preparation")
                                continue
                            
                            # Process responses
                            responses = []
                            for msg in conversation:
                                if isinstance(msg['value'], list):
                                    responses.extend([r for r in msg['value'] if r])  # Filter out empty responses
                            
                            if not responses:
                                print("No valid responses to process")
                                continue
                            
                            # Get top responses
                            top_responses, top_rewards = self.select_top_k_responses(
                                user_prompt, responses, k=max(k, len(responses))
                            )
                            
                            if not top_responses:
                                print("No valid top responses selected")
                                continue
                            
                            print("TOTAL RESPONSE", len(top_responses))
                            
                            # Training step
                            if len(top_responses) > 0:
                                loss = self._train_step_batch([user_prompt] * len(top_responses), 
                                                            top_responses, 
                                                            top_rewards)
                                epoch_loss += loss
                                epoch_reward += sum(top_rewards) / len(top_rewards)
                                num_batches += 1
                                
                                print(f"Epoch {epoch + 1}, Temp {temp}, "
                                      f"Loss: {loss:.4f}, "
                                      f"Avg Reward: {sum(top_rewards) / len(top_rewards):.4f}")
                            
                            torch.cuda.empty_cache()
                            
                        except Exception as e:
                            print(f"Error in training iteration: {str(e)}")
                            continue

            # Compute proper averages using number of batches
            avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
            avg_epoch_reward = epoch_reward / num_batches if num_batches > 0 else 0.0
            
            print(f"Epoch {epoch + 1} completed. "
                  f"Average Loss: {avg_epoch_loss:.4f}, "
                  f"Average Reward: {avg_epoch_reward:.4f}, "
                  f"Total Batches: {num_batches}")
            
            if checkpoint_dir:
                self.model.save_pretrained(f"{checkpoint_dir}/checkpoint_epoch_{epoch}")
                self.tokenizer.save_pretrained(f"{checkpoint_dir}/checkpoint_epoch_{epoch}")

            if avg_epoch_reward > best_reward:
                best_reward = avg_epoch_reward
                self.save_model(f"{checkpoint_dir}/best_model")

            # Add periodic memory cleanup
            torch.cuda.empty_cache()
            if epoch % 5 == 0:  # Every 5 epochs
                gc.collect()

   
    def save_model(self, save_path):
        self.model.save_pretrained(f"{save_path}")
        self.tokenizer.save_pretrained(f"{save_path}")
        torch.save(self.value_head.state_dict(), f"{save_path}/value_head.pt")

   
    def configure_optimizer(self):
        """Configure the optimizer to update LoRA parameters."""
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if "lora" in n.lower()],
                "weight_decay": 0.0,
                "lr": self.learning_rate
            },
            {
                "params": self.value_head.parameters(),
                "weight_decay": 0.0,
                "lr": self.learning_rate
            }
        ]
        
        self.optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.0
        )

    def load_model(self, load_path):
        self.model.load_pretrained(load_path)
        self.tokenizer.load_pretrained(load_path)
        self.value_head.load_state_dict(torch.load(f"{load_path}/value_head.pt"))

# Example usage:
if __name__ == "__main__":
    model_path = "/ddnB/work/borun22/Transfer_learning/NewCOT/LLM/MistraAI/large_dataset/mistra_qlora_finetuned/"
    data_paths = {
        'er_path': 'Data/unique_smiles_Er.csv',
        'smiles_path': 'Data/smiles.xlsx'
    }
    
    trainer = RLHFTrainer(model_path, data_paths)
    trainer.train(num_epochs=1, k=10, checkpoint_dir="/ddnB/work/borun22/Transfer_learning/NewCOT/LLM/MistraAI/Reward_model/large_mistra_chk_point")  # Uncomment when training_data is defined
    trainer.save_model("/ddnB/work/borun22/Transfer_learning/NewCOT/LLM/MistraAI/Reward_model/large_saved_models")
