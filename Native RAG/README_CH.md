# 一个RAG框架的简单实现
[English](README.md)
检索增强生成（Retrieval-Augmented Generation, RAG）是使用外部知识库和参考文本约束大语言模型（Large Language Models, LLMs）输出的一种技术。它不仅可以有效的缓解LLMs的“幻觉问题”，由于外部知识库可以实时更新，所以RAG也可以让LLMs实时获取最新的知识。本项目旨在从RAG的基本原理出发，不使用ollama和LangChain等大模型开发工具实现基本的RAG框架。本项目中使用的大语言模型是由智谱AI推出的开源模型[GLM4](https://github.com/THUDM/GLM-4.git)。

### 快速开始
1. 创建环境
   ```bash
   conda create -n RAG-GLM4 python=3.10 && conda activate RAG-GLM4
   ```
   然后参考requirements.txt安装必备的包；
2. 将待参考文本放置于RAG/dataset目录下；
3. 在[这里](https://huggingface.co/THUDM/glm-4-9b-chat)下载GLM4的权重，放置于RAG/GLM4CKPT目录下；
4. 使用GLM4的分词器和词嵌入权重，这里已剥离放置于RAG/tokenizer目录下，其中词嵌入权重可于[此处](https://drive.google.com/file/d/1xhEqOz2mqyQd5AkbVrYoJvtSU4pgr00m/view?usp=sharing)下载；
5. 现在你可以直接运行RAG/RAG.ipynb[^1]，关于此notebook的更多细节请参考[说明](Tutorial_CH.md)。
[^1]: 如果是在linux服务器上运行，你可能需要先安装jupyter服务器。