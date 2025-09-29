import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset from URL
url = ""
df = pd.read_csv(url)

# Combine title and description for embedding
df["text"] = df["title"] + " " + df["description"]

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Compute embeddings for all courses
course_embeddings = model.encode(df["text"].tolist(), convert_to_tensor=True)

# Define recommendation function
def recommend_courses(user_query, top_k=5):
    query_embedding = model.encode([user_query], convert_to_tensor=True)
    similarities = cosine_similarity(query_embedding, course_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    recommendations = df.iloc[top_indices][["title", "description"]]
    scores = similarities[top_indices]
    return recommendations, scores

# Sample input queries
sample_queries = [
    "I’ve completed the ‘Python Programming for Data Science’ course and enjoy data visualization. What should I take next?",
    "I know Azure basics and want to manage containers and build CI/CD pipelines. Recommend courses.",
    "My background is in ML fundamentals; I’d like to specialize in neural networks and production workflows.",
    "I want to learn to build and deploy microservices with Kubernetes—what courses fit best?",
    "I’m interested in blockchain and smart contracts but have no prior experience. Which courses do you suggest?"
]

# Evaluate and print recommendations
for i, query in enumerate(sample_queries, 1):
    print(f"\nQuery {i}: {query}")
    recs, scores = recommend_courses(query)
    for idx, (title, desc) in enumerate(zip(recs["title"], recs["description"])):
        print(f"  {idx+1}. {title} (Score: {scores[idx]:.4f})")
        print(f"     {desc}")
