# RAG Simple Implementation
[[中文]](Tutorial_CH.md)
### Retrieval：Build Queries Vector Data Base

1. Read Reference Document (PDF)

   Select pdfplumber to read PDF files, a simple use case is as follows:

   ```python
    path = '/your/pdf/file/path'

   with pdfplumber.open(path) as pdf: 
       content = ''
       for i in range(len(pdf.pages)):
           page = pdf.pages[i] 
           page_content = '\n'.join(page.extract_text().split('\n')[:-1])
           content = content + page_content
   ```
   
2. Filter invalid characters and make tokenizer

   Invalid characters can be filtered out if necessary:

   ```python
   def filter_invalid_characters(text, valid_chars):
       """Filter out characters that are not in the valid character set."""
       vilid_characters = ''.join([char for char in text if char in valid_chars])
       filtered_characters = ''.join([char for char in text if char not in vilid_characters])
       return vilid_characters, filtered_characters
   
   # valid characters set
   valid_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?,:()[]{}<>/-_ \n")
   
   clean_content, filtered_characters = filter_invalid_characters(content, valid_chars)
   print(filtered_characters)
   ```

   The tokenizer can be done using existing open source LLMs' tokenizer. Take GLM4 as an example:

   ```python
   from tokenizer.tokenization_chatglm import ChatGLM4Tokenizer
   
   tokenizer_glm4 = ChatGLM4Tokenizer('tokenizer/tokenizer.model')
   
   tokens = tokenizer_glm4._tokenize('This is a Sentence.')
   token_ids = [tokenizer_glm4._convert_token_to_id(token) for token in tokens]
   
   recover_tokens = [tokenizer_glm4._convert_id_to_token(token_id) for token_id in token_ids]
   recover_text = tokenizer_glm4.convert_tokens_to_string(recover_tokens)
   ```

3. Chunking and calculating word embeddings

   Since the query text may be too long, which may exceed the large model input length limit on the one hand, and on the other hand too long text may reduce the relevance of the query content, the text needs to be chunked:

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

   Regarding the computation of word embeddings, seq2seq training can be used, and here the embedding matrix weights of GLM4 are used directly.

4. Build Vector Data Base

   Vector Data Base mainly calculates the word embedding matrix for each block of token, here a VectorDB class is constructed which provides four methods.

   - VectorDB.create_vector_db()：Create the word embedding matrices corresponding to the chunks of VectorDB and returns a list where each element of the list corresponds to the word embedding matrix of each chunk,
   - VectorDB._query_to_vectors(query, top_k)：Return the word embedding matrix of the top_k most similar chunks based on the user query query,
   - VectorDB._query_to_text(query, top_k)：Return the top_k most similar chunks of text based on the user query query,
   - VectorDB.average_cosine_similarity(chunk_vectors, query_vertors): Calculate cosine similarity between block vectors and query vectors.

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

   For an example of the use of VectorDB:

   ```python
   query_text = VectorDB._query_to_text('your query text', 3)
   for text in query_text:
       print(text)
       print('-'*50)
   ```

   At this point, we have completed the construction of the query vector to achieve the most relevant content from the provided PDF text query and user input, that is, to achieve the first step of Retrieval-Augmented Generation Retrieval.

### Augment and Genetarion：Enhanced generation using retrieved content

After being able to retrieve relevant content, it is then necessary to use the retrieved content to enhance the generated content of the LLMs.

1. Load LLM

   Here we take ChatGLM4 as an example and start by loading the model:

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

2. Building the RAG class

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

3. Use Case：

   ``` python
   rag_GLM4 = RAG_GLM4(model, glm4_tokenizer, VectorDB)
   query = 'a text about query'
   response_rag = rag_GLM4.generate(query)
   print(response_rag)
   print('-'*100)
   response, _ = model.chat(glm4_tokenizer, query, history=[])
   print('\n', response)sad 
   ```