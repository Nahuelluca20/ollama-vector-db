import ollama
import chromadb
from info_list import info

client = chromadb.Client()
collection = client.create_collection(name="tech_stacks_ollama")

for i, d in enumerate(info):
  response = ollama.embeddings(model="mxbai-embed-large", prompt=d)
  embedding = response["embedding"]
  collection.add(
    ids=[str(i)],
    embeddings=[embedding],
    documents=[d]
  )

while True:  
  prompt = input("Introduce un prompt (o escribe /exit para salir): ")  # {{ edit_2 }}
  if prompt == "/exit":  
    break  

  response = ollama.embeddings(
    prompt=prompt,
    model="mxbai-embed-large"
  )
  results = collection.query(
    query_embeddings=[response["embedding"]],
    n_results=1
  )
  data = results['documents'][0][0]
  print(data)