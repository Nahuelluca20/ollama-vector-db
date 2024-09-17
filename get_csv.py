import ollama
import pandas as pd
from info_list import info

embeddings = []
for i, d in enumerate(info): 
  response = techstack_ollama.embeddings(model="mxbai-embed-large", prompt=d)
  embedding = response["embedding"]
  embeddings.append(embedding)

embeddings_df = pd.DataFrame(embeddings)

embeddings_df.to_csv("embeddings.csv", index=False)

print(embeddings_df)
