"""Golden example utterances for the Yelp review router.

Contains 10 seed utterances per category (Dietary, Service, General)
for use in contrastive learning. These are expanded via
augment.py (GLM 5.1 via Ollama) to generate hard negatives.
"""

SEED_UTTERANCES: list[dict[str, str | int]] = [
    # Dietary (label 0) — food allergies, celiac, FODMAP, ingredient safety
    {"text": "They were so careful with my celiac needs.", "label": 0},
    {"text": "I asked if the soup had gluten and the chef came out to tell me.", "label": 0},
    {"text": "My nut allergy was taken seriously here, they even cleaned the grill.", "label": 0},
    {"text": "The menu clearly marked all dairy-free options.", "label": 0},
    {"text": "I have a shellfish allergy and they prepared my dish separately.", "label": 0},
    {"text": "They substituted tamari for soy sauce for my soy allergy.", "label": 0},
    {"text": "The server double-checked with the kitchen about cross-contamination.", "label": 0},
    {"text": "As a vegan I felt completely safe eating here.", "label": 0},
    {"text": "They accidentally served me regular bread instead of gluten-free.", "label": 0},
    {"text": "The kitchen uses shared fryers, so it's not safe for celiacs.", "label": 0},
    # Service (label 1) — wait times, staff behavior, reservation issues
    {"text": "The waiter brought me the wrong order.", "label": 1},
    {"text": "We waited 45 minutes for a table even with a reservation.", "label": 1},
    {"text": "The host was incredibly rude when we arrived.", "label": 1},
    {"text": "Our server checked on us constantly and refilled our drinks.", "label": 1},
    {"text": "They forgot our appetizer and didn't apologize.", "label": 1},
    {"text": "The manager comped our meal after we complained about the wait.", "label": 1},
    {"text": "Service was incredibly slow even though the restaurant was empty.", "label": 1},
    {"text": "The bartender was attentive and remembered our names.", "label": 1},
    {"text": "They refused to seat us even though we had a confirmed booking.", "label": 1},
    {"text": "Our food came out cold and the server didn't offer to reheat it.", "label": 1},
    # General (label 2) — ambiance, price, general food quality
    {"text": "The garlic bread was way too salty.", "label": 2},
    {"text": "Great ambiance but the prices are steep for what you get.", "label": 2},
    {"text": "The decor is beautiful and the music is just right.", "label": 2},
    {"text": "Portions are huge but the quality is just okay.", "label": 2},
    {"text": "This place has an amazing view of the city skyline.", "label": 2},
    {"text": "The pasta was decent but nothing special.", "label": 2},
    {"text": "Best pizza I've had in this neighborhood.", "label": 2},
    {"text": "The restaurant is cozy but the food is overpriced.", "label": 2},
    {"text": "Loud music made it hard to have a conversation.", "label": 2},
    {"text": "The dessert menu is worth staying for.", "label": 2},
]
