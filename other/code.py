from transformers import LlamaForCausalLM, AutoTokenizer
import torch

ckpt = 'BelleGroup/BELLE-LLAMA-7B-2M'
device = torch.device('cuda')
model = LlamaForCausalLM.from_pretrained(ckpt, device_map='auto', low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained(ckpt)

def infer(task):
  #  prompt = "Human: 写一首中文歌曲，赞美大自然 \n\nAssistant: "
  prompt = "Human: {} \n\nAssistant: ".format(task)
  input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
  generate_ids = model.generate(input_ids, max_new_tokens=500, do_sample = True, top_k = 30, top_p = 0.85, temperature = 0.5, repetition_penalty=1., eos_token_id=2, bos_token_id=1, pad_token_id=0)
  output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
  response = output[len(prompt):]
  print(response)





