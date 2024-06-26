# A Simple Implementation of the RAG Framework
[中文](README_CH.md)
Retrieval-Augmented Generation (RAG) is a technique that constrains the output of Large Language Models (LLMs) using an external knowledge base and reference texts. It not only effectively mitigates the "hallucination problem" of LLMs but also allows LLMs to acquire up-to-date knowledge in real-time as the external knowledge base can be updated. This project aims to implement a basic RAG framework from the fundamental principles of RAG, without using large model development tools like ollama and LangChain. The large language model used in this project is the open-source model [GLM4](https://github.com/THUDM/GLM-4.git) launched by Zhipu AI.

### Quick Start
1. Create an environment
   ```bash
   conda create -n RAG-GLM4 python=3.10 && conda activate RAG-GLM4
   Then install the necessary packages as per requirements.txt,
2. Place the reference texts in the RAG/dataset directory,
3. Download the weights for GLM4 from [here](https://huggingface.co/THUDM/glm-4-9b-chat) and place them in the RAG/GLM4CKPT directory,
4. Use the tokenizer and embedding weights of GLM4, now separated and placed in the RAG/tokenizer directory, the embedding weight can be downloaded [here](https://drive.google.com/file/d/1xhEqOz2mqyQd5AkbVrYoJvtSU4pgr00m/view?usp=sharing),
5. Now you can directly run RAG/RAG.ipynb[^1], for more details about this notebook please refer to the [tutorial_EN]((https://huggingface.co/THUDM/glm-4-9b-chat)).
[^1]: If running on a linux server, you may need to install the jupyter server first.

