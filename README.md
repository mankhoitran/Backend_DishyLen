# Food Agent: Menu Information Retrieval API

Production-ready AI agent system for menu/dish information retrieval using:

- Gemini model `gemini-2.0-flash`
- Gemini Google Search tool grounding
- Tool-based agent architecture (function calling)
- FastAPI REST endpoint
- SQLite persistence

## Project Structure

```text
food_agent/
├── app.py
├── agent/
│   ├── agent.py
│   ├── tools.py
│   ├── prompts.py
│   ├── parser.py
├── services/
│   ├── gemini_client.py
│   ├── search.py
├── db/
│   ├── database.py
│   ├── models.py
│   ├── crud.py
├── schemas/
│   ├── request.py
│   ├── response.py
├── config.py
├── requirements.txt
└── README.md
```

## Setup

1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set environment variables:

```bash
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
# optional
export SQLITE_DB_URL="sqlite:///./food_agent.db"
```

You can also use a `.env` file in the `food_agent/` directory:

```env
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
SQLITE_DB_URL=sqlite:///./food_agent.db
```

## Run API

```bash
uvicorn app:app --reload --port 8000
```

## Example Request

```bash
curl -X POST "http://127.0.0.1:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "pad thai spicy level"
  }'
```

Example response:

```json
{
  "dish": "Pad Thai",
  "spicy_level": "medium",
  "macros": {
    "calories_kcal": 520,
    "protein_g": 19,
    "carbs_g": 58,
    "fat_g": 18
  },
  "summary": "Pad Thai is a Thai stir-fried noodle dish with tamarind sauce, tofu or shrimp, egg, and peanuts.",
  "image_url": "https://example.com/pad-thai.jpg",
  "source": "search"
}
```

## Agent Flow

1. Parse input (`text` or `ocr:` simulation)
2. Extract dish candidate
3. Tool call: `get_processed_dish`
4. If missing, call `search_dish` (+ gap-filling tools when required)
5. Persist structured result to SQLite
6. Return normalized response object

## Available Tools

- `get_processed_dish(dish_name)`
- `search_dish(dish_name)`
- `get_spicy_level(dish_name)`
- `get_dish_macro(dish_name)`
- `get_dish_summary(dish_name)`
- `get_dish_image_url(dish_name)`
- `translate(text, target_language)`

## Notes

- No secrets are hardcoded.
- The implementation uses real Gemini API structures, including function declarations and Google search tool usage.
- Agent actions are logged via Python logging for transparency.
