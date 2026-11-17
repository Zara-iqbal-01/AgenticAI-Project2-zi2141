from agents import MODEL_SMART, VERTEX_LOCATION  # same model, same project settings
import litellm

QUESTIONS = [
    "What if Alcaraz had been in Djokovic's spot at Wimbledon 2023?",
    "Which quarter was the group of death at the 2025 US Open men's?",
    "Who had the easiest draw at the 2026 US Open men's?",
]
for q in QUESTIONS:
    r = litellm.completion(model=MODEL_SMART, vertex_location=VERTEX_LOCATION,
                           messages=[{"role": "user", "content": q}])
    print("=" * 80)
    print("Q:", q)
    print(r.choices[0].message.content)