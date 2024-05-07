# GIX 510-Lab7
# RAG and Agents

This module explores the integration of retrieval mechanisms in natural language processing tasks using the RAG approach. It also delves into the role of agents in managing and orchestrating these processes.

## Get started
```
pip install -r requirements.txt

from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration

tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True)
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

input_ids = tokenizer("What is the capital of France?", return_tensors="pt").input_ids

# Generate answers
output = model.generate(input_ids)
print(tokenizer.decode(output[0], skip_special_tokens=True))

```

## Lessons Learned
- Retrieval Integration: Integrating retrieval into generative models can significantly improve the relevance and accuracy of the generated content.
- Agent Design: Designing agents that can effectively manage and utilize these models is crucial for building efficient NLP systems.
- Performance Considerations: While RAG introduces powerful capabilities, managing computational resources effectively remains a challenge due to the intensive nature of retrieval operations.


## TODO
1. In class exercise
2. Try to build a web app using RAG