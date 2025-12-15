from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text = 'arvind'
print(len(embeddings.embed_query(text)))
# print('Embeddings  : ',embeddings.embed_query(text))

'''
384 : [I am happy] --> [-0.09]

3076 : 

'''
