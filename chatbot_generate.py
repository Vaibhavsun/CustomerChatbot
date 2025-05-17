from transformers import AutoTokenizer,TextIteratorStreamer
import torch
from threading import Thread
from peft import AutoPeftModelForCausalLM
from langchain_core.prompts import ChatPromptTemplate,FewShotChatMessagePromptTemplate
examples=[{"input":"What is the Company Name?","output":"Nova Company"},{"input":"When it was started?","output":"2014 december"},{"input":"What is the Company Address?","output":"123 Nova Street, Nova City, NC 12345"}]

example_prompt=ChatPromptTemplate.from_messages([('human',"Question: {input}\nAnswer: "),('ai',"{output}")])


few_shot_prompt = FewShotChatMessagePromptTemplate(
    examples=examples,
    # This is a prompt template used to format each individual example.
    example_prompt=example_prompt,
)


main_prompt=ChatPromptTemplate.from_messages([('system',"You are V.I.P. Concierge, an elite customer assistant for premium clients.\n"
        "Speak politely, professionally, and warmly.\n"
        "Highlight exclusivity, limited edition products, early access, and special perks.\n"
        "If you do not know the answer, say politely: 'Please contact our VIP support team at vip-support@example.com for assistance.'\n"
        "Never reveal confidential info.\n"),few_shot_prompt,('human',"Question: {input}\nAnswer: ")])



tokenizer=AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
# model=AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0",device_map="auto",cache_dir="./.cache/huggingface_hub", torch_dtype=torch.float16)


model = AutoPeftModelForCausalLM.from_pretrained("finetuned_model_llama",cache_dir="./.cache/huggingface_hub")
tokenizer.pad_token = tokenizer.eos_token
def tokenize_input(prompt):
    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt",truncation=False)
    return inputs
def generate_response(prompt):

    inputs = tokenize_input(main_prompt.format(input=prompt+"?"))


    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True)

    generation_kwargs = dict(inputs, streamer=streamer,max_length=800, do_sample=True, top_k=50, top_p=0.95, temperature=0.1)

    thread = Thread(target=model.generate, kwargs=generation_kwargs)

    thread.start()
    for new_text in streamer:
        print(new_text, end="")
    # Generate a response using the model
    # outputs = model.generate(**inputs,max_length=800, do_sample=True, top_k=50, top_p=0.95, temperature=0.1)
    
    # # Decode the generated response
    # response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # return response


generated_response = generate_response("Question:who are You?\nAnswer:")

print(generated_response)
