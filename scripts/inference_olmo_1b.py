from transformers import AutoModelForCausalLM, AutoTokenizer

# olmo = AutoModelForCausalLM.from_pretrained("./hf_checkpoint_dir/OLMo-1B_gtbs2048_dtms2_md1e_ddp_compile_all_save100_withoutload/step300")
# tokenizer = AutoTokenizer.from_pretrained("./hf_checkpoint_dir/OLMo-1B_gtbs2048_dtms2_md1e_ddp_compile_all_save100_withoutload/step300")

# olmo = AutoModelForCausalLM.from_pretrained("./hf_checkpoint_dir/official/OLMo-1B/step738020-unsharded")
# tokenizer = AutoTokenizer.from_pretrained("./hf_checkpoint_dir/official/OLMo-1B/step738020-unsharded")

# path = "./hf_checkpoint_dir/sft/OLMo-1B_sft-nh-v1/step452-unsharded/"
# path = "./hf_checkpoint_dir/sft/OLMo-1B_s21.5k_ddp_compile_nh/step452/"
path = "./hf_checkpoint_dir/sft/OLMo-1B_s21.5k_ddp_compile_nh/step452/"

olmo = AutoModelForCausalLM.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path)

message = ["Language modeling is "]
# message = ["愚公移山的成語出處，用這個成語造句" ]
# message = ["You are a robot assistant, please help me with the question and think step by step: if 2 * A +2 = 10, so what is A ?"]
# message = ["You are a robot assistant. Please help me write a Python code snippet that prints \"helloword\" "]
# message = ["You are a robot assistant. Please help me write a Python code snippet that prints \"you are my sweetheart\" "]
# message = ["You are a robot assistant. Please introduce yourself to others."]
# message = ["Who are you?"]

# message = ['What is python']
inputs = tokenizer(message, return_tensors='pt', return_token_type_ids=False)
response = olmo.generate(**inputs, max_new_tokens=256, do_sample=True, top_k=20, top_p=0.95)
print(tokenizer.batch_decode(response, skip_special_tokens=True)[0])