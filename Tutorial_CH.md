# RAG实现
[[English]](Tutorial_EN.md)
### Retrieval：构建查询Vector Data Base

1. 读取参考文档（PDF）

   选择pdfplumber读取参考文献，简单的使用案例如下：

   ```python
   with pdfplumber.open(path) as pdf: 
       content = ''
       for i in range(len(pdf.pages)):
           page = pdf.pages[i] 
           page_content = '\n'.join(page.extract_text().split('\n')[:-1])
           content = content + page_content
   ```
   
2. 过滤无效字符并分词

   如果有需要可以过滤掉无效字符：

   ```python
   def filter_invalid_characters(text, valid_chars):
       """Filter out characters that are not in the valid character set."""
       vilid_characters = ''.join([char for char in text if char in valid_chars])
       filtered_characters = ''.join([char for char in text if char not in vilid_characters])
       return vilid_characters, filtered_characters
   
   # valid characters set
   valid_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:()[]{}<>/-_ \n")
   
   clean_content, filtered_characters = filter_invalid_characters(content, valid_chars)
   print(filtered_characters)
   ```

   分词可以使用现有的大模型分词器，这里以GLM4为例：

   ```python
   from tokenizer.tokenization_chatglm import ChatGLM4Tokenizer
   
   tokenizer_glm4 = ChatGLM4Tokenizer('tokenizer/tokenizer.model')
   
   tokens = tokenizer_glm4._tokenize('This is a Sentence. 这是一个句子。')
   token_ids = [tokenizer_glm4._convert_token_to_id(token) for token in tokens]
   
   recover_tokens = [tokenizer_glm4._convert_id_to_token(token_id) for token_id in token_ids]
   recover_text = tokenizer_glm4.convert_tokens_to_string(recover_tokens)
   ```

3. 分块并计算词嵌入

   由于查询文本可能过长，一方面可能超过大模型输入长度限制，另一方面过长的文本可能会降低查询内容的相关度，所以需要对文本进行分块处理：

   ```python
   def Split_to_Chunks(content, chunk_size, overlap_size):
       character_index = 0
       chunks = []
       while character_index + chunk_size <= len(content):
           chunks.append(content[character_index : character_index + chunk_size])
           character_index += chunk_size - overlap_size
       chunks.append(content[character_index:])
       return chunks
   
   test_content = 'abcdefghigklmnopqrstuvwxyz'
   text_chunks = Split_to_Chunks(test_content, 10, 2)
   for chunk in text_chunks:
       print(chunk)
   ```

   关于词嵌入的计算，可以使用seq2seq训练，这里直接使用GLM4的嵌入矩阵权重。

4. Vector Data Base构建

   Vector Data Base主要计算每一个块的token的词嵌入矩阵，这里构建一个VectorDB类，该类提供四个方法。

   - VectorDB.create_vector_db()：创建VectorDB的chunks对应的词嵌入矩阵，返回一个列表，列表的每个元素对应每个chunk的词嵌入矩阵；
   - VectorDB._query_to_vectors(query, top_k)：根据用户查询query返回最相似的前top_k个chunk的词嵌入矩阵；
   - VectorDB._query_to_text(query, top_k)：根据用户查询query返回最相似的前top_k个chunk文本；
   - VectorDB.average_cosine_similarity(matrix_a, matrix_b)

   ```python
   class VectorDB:
       def __init__(self, tokenizer:any, embedding_weight_path:str, chunks):
           self.chunks = chunks
           self.tokenizer = tokenizer
           self.vectors = torch.load(embedding_weight_path)
           self.vector_db = self.create_vector_db()
           self.chunk_nums = len(self.vector_db)
           
       def create_vector_db(self):
           vector_db = []
           for i, chunk in enumerate(self.chunks):
               tokens = self.tokenizer._tokenize(chunk)
               token_ids = [self.tokenizer._convert_token_to_id(token) for token in tokens]
               vector_db.append(self.vectors[token_ids])
           return vector_db
       
       def _query_to_vectors(self, query, top_k):
           tokens = self.tokenizer._tokenize(query)
           token_ids = [self.tokenizer._convert_token_to_id(token) for token in tokens]
           query_vectors = self.vectors[token_ids]
           cosine_similarity_score = torch.ones(self.chunk_nums)
           for i in range(self.chunk_nums):
               cosine_similarity_score[i] = self.average_cosine_similarity(self.vector_db[i], query_vectors)
           similarity_score, chunk_indices = torch.topk(cosine_similarity_score, top_k)
           results = [self.vector_db[chunk_indice] for chunk_indice in chunk_indices]
           return results, chunk_indices, similarity_score
           
       def _query_to_text(self, query, top_k):
           _, chunk_indices, _ = self._query_to_vectors(query, top_k)
           results = [self.chunks[chunk_indice] for chunk_indice in chunk_indices]
           return results
           
       def average_cosine_similarity(self, chunk_vectors, query_vertors):
           chunk_vectors_norm = chunk_vectors / chunk_vectors.norm(dim=1, keepdim=True)
           query_vertors_norm = query_vertors / query_vertors.norm(dim=1, keepdim=True)
           cosine_sim = torch.mm(chunk_vectors_norm, query_vertors_norm.t())  
           return cosine_sim.mean()
   ```

   关于VectorDB的使用示例：

   ```python
   query_text = VectorDB._query_to_text('注意力', 3)
   for text in query_text:
       print(text)
       print('-'*50)
   ```

   至此，我们已经完成了查询向量的构建，实现了从提供的PDF文本中查询和用户输入最为相关的内容，即实现了Retrieval－Augmented Generation的第一步Retrieval。

### Augment and Genetarion：使用检索的内容增强生成

能够检索相关内容之后，后续需要使用检索的内容增强LLM的生成内容。

1. 加载LLM

   这里我们以ChatGLM4为例，首先加载模型：

   ```python
   from transformers import AutoTokenizer, AutoModelForCausalLM
   import torch
   model_dir = 'GLM4CKPT'
   glm4_tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
   model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True)
   device = torch.device("cuda:0")
   model.to(device)
   model.eval()
   ```

2. 构建RAG

   ```python
   class RAG_GLM4:
       def __init__(self, model, tokenizer, VectorDB):
           self.model = model
           self.VectorDB = VectorDB
           self.tokenizer = tokenizer
       def generate(self, query, top_k=1):
           response, _ = model.chat(self.tokenizer, self.augmented(query, top_k), history=[])
           return response
       def augmented(self, query, top_k=1):
           prompt = f"You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question: {query}. If you don't know the answer, say that you don't know. \n\n Retrieved context: {self.VectorDB._query_to_text(query, top_k)}."
           return prompt
   ```

3. 用例：

   ```
   rag_GLM4 = RAG_GLM4(model, glm4_tokenizer, VectorDB)
   query = 'a text about query'
   response_rag = rag_GLM4.generate(query)
   print(response_rag)
   print('-'*100)
   response, _ = model.chat(glm4_tokenizer, query, history=[])
   print('\n', response)sad 
   ```