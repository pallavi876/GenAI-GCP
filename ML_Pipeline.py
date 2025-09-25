import mlflow
from langchain.prompts import PromptTemplate
from pydantic import BaseModel
from typing import List

# Step 1: Accept company name
company_name = "Google"

# Step 2: Stock code lookup (static dictionary for demo)
stock_lookup = {
    "Google": "GOOGL",
    "Apple Inc": "AAPL",
    "Microsoft": "MSFT",
    "Amazon": "AMZN"
}
stock_code = stock_lookup.get(company_name, "UNKNOWN")

# Step 3: Simulated news fetching (replace with LangChain news tool integration)
news_summaries = [
    "Google announces new AI features in its search engine.",
    "Alphabet's earnings beat expectations amid strong ad revenue.",
    "Google faces antitrust scrutiny in Europe over advertising practices."
]

# Step 4: Define output schema using Pydantic
class SentimentOutput(BaseModel):
    company_name: str
    stock_code: str
    newsdesc: str
    sentiment: str
    people_names: List[str]
    places_names: List[str]
    other_companies_referred: List[str]
    related_industries: List[str]
    market_implications: str
    confidence_score: float

# Step 4: Prompt template for Gemini model
prompt_template = PromptTemplate(
    input_variables=["company", "stock", "news"],
    template="""
You are a financial analyst LLM. Analyze the following news about {company} ({stock}) and provide a structured JSON with:
- Sentiment classification (Positive/Negative/Neutral)
- Named entities: people, places, other companies
- Related industries
- Market implications
- Confidence score (0.0 to 1.0)

News:
{news}

Respond in JSON format with the following fields:
{{
"company_name": "",
"stock_code": "",
"newsdesc": "",
"sentiment": "Positive/Negative/Neutral",
"people_names": [],
"places_names": [],
"other_companies_referred": [],
"related_industries": [],
"market_implications": "",
"confidence_score": 0.0
}}
"""
)

# Step 4: Simulated Gemini response (replace with actual LangChain + Vertex AI call)
def simulate_gemini_response(company, stock, news):
    return SentimentOutput(
        company_name=company,
        stock_code=stock,
        newsdesc=" ".join(news),
        sentiment="Positive",
        people_names=["Sundar Pichai"],
        places_names=["Europe"],
        other_companies_referred=["Meta", "Amazon"],
        related_industries=["Advertising", "Artificial Intelligence"],
        market_implications="Positive outlook for Google's ad revenue and AI leadership.",
        confidence_score=0.92
    )

# Step 5: mlflow logging
mlflow.set_experiment("MarketSentimentAnalyzer")

with mlflow.start_run():
    mlflow.log_param("company_name", company_name)
    mlflow.log_param("stock_code", stock_code)
    mlflow.log_param("news_count", len(news_summaries))
    mlflow.set_tag("stage", "sentiment_analysis")

    # Simulate Gemini response
    result = simulate_gemini_response(company_name, stock_code, news_summaries)

    # Log output
    mlflow.log_dict(result.dict(), "sentiment_output.json")

    # Print result
    print(result.json(indent=2))
