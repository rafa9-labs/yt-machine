GEOPOLITICAL_FEEDS = [
    {
        "name": "Reuters World",
        "url": "https://feeds.reuters.com/reuters/worldNews",
        "category": "geopolitical",
        "priority": 1
    },
    {
        "name": "Al Jazeera English",
        "url": "https://www.aljazeera.com/xml/rss/all.xml",
        "category": "geopolitical",
        "priority": 1
    },
    {
        "name": "Foreign Policy",
        "url": "https://foreignpolicy.com/feed/",
        "category": "geopolitical",
        "priority": 1
    },
    {
        "name": "Stratfor",
        "url": "https://worldview.stratfor.com/rss.xml",
        "category": "geopolitical",
        "priority": 1
    },
    {
        "name": "BBC World",
        "url": "http://feeds.bbci.co.uk/news/world/rss.xml",
        "category": "geopolitical",
        "priority": 2
    },
    {
        "name": "Associated Press World",
        "url": "https://rsshub.app/apnews/topics/apf-intlnews",
        "category": "geopolitical",
        "priority": 2
    },
    {
        "name": "Middle East Eye",
        "url": "https://www.middleeasteye.net/rss",
        "category": "geopolitical",
        "priority": 2
    },
    {
        "name": "Defense One",
        "url": "https://www.defenseone.com/rss/all/",
        "category": "kinetic",
        "priority": 2
    }
]

# PHASE 3.3: Expanded keyword categories for topic diversification
GEOPOLITICAL_KEYWORDS = {
    "middle_east_conflict": [
        "iran", "israel", "hormuz", "hezbollah", "irgc", "idf",
        "gaza", "hamas", "syria", "yemen", "saudi", "uae",
        "dubai", "qatar", "kuwait", "bahrain", "oman", "iraq",
        "lebanon", "jordan", "egypt", "sinai", "red sea", "persian gulf"
    ],
    "technology_disruption": [
        "ai", "artificial intelligence", "quantum", "semiconductor", "chip",
        "5g", "6g", "cyber", "cybersecurity", "hack", "ransomware",
        "blockchain", "cryptocurrency", "bitcoin", "ethereum", "web3",
        "autonomous", "robotics", "drone tech", "space", "satellite",
        "neural network", "machine learning", "deepfake", "surveillance"
    ],
    "climate_geopolitics": [
        "climate", "carbon", "renewable", "solar", "wind", "nuclear",
        "drought", "water crisis", "food security", "famine", "crop failure",
        "extreme weather", "flooding", "wildfire", "hurricane", "typhoon",
        "sea level", "glacier", "arctic", "antarctica", "permafrost",
        "green energy", "battery", "lithium", "cobalt", "rare minerals"
    ],
    "health_security": [
        "pandemic", "epidemic", "outbreak", "virus", "vaccine",
        "bioweapon", "bioterror", "lab leak", "gain of function",
        "antibiotic resistance", "superbug", "bird flu", "h5n1",
        "pharmaceutical", "drug shortage", "medical supply", "hospital",
        "public health", "quarantine", "lockdown", "variant"
    ],
    "great_power_competition": [
        "china", "russia", "ukraine", "taiwan", "nato", "quad",
        "south china sea", "crimea", "donbas", "xi jinping", "putin",
        "biden", "pentagon", "kremlin", "beijing", "moscow", "kyiv",
        "aukus", "nord stream", "baltic", "black sea", "indopacific"
    ],
    "economic_warfare": [
        "sanctions", "tariff", "trade war", "embargo", "currency",
        "swift", "yuan", "petrodollar", "opec", "inflation", "recession",
        "debt", "default", "supply chain", "semiconductor", "chip", "rare earth",
        "frozen assets", "central bank", "imf", "world bank", "wto"
    ],
    "regional_flashpoints": [
        "africa", "sahel", "ethiopia", "sudan", "congo", "somalia",
        "latin america", "venezuela", "colombia", "cuba", "haiti", "mexico",
        "cartel", "favela", "amazon", "andes", "caribbean", "panama",
        "asia", "korea", "kashmir", "pakistan", "india", "myanmar", "thailand",
        "vietnam", "philippines", "indonesia", "japan", "north korea"
    ],
    "kinetic_operations": [
        "strike", "airstrike", "missile", "military", "war", "attack",
        "blockade", "naval", "troops", "offensive", "ceasefire", "escalation",
        "invasion", "bombing", "drone", "fighter jet", "warship", "submarine",
        "tank", "artillery", "special forces", "exercise", "deployment", "frontline"
    ],
    "diplomatic_pivot": [
        "negotiation", "deal", "summit", "alliance", "treaty",
        "envoy", "backchannel", "normalization", "pivot", "realignment",
        "transactional", "leverage", "washington", "state department",
        "foreign minister", "ambassador", "un", "security council", "g20"
    ]
}

# PHASE 3.3: Category weights for balanced content distribution
CATEGORY_WEIGHTS = {
    "middle_east_conflict": 4,  # Reduced from implicit 4 to balance with other topics
    "technology_disruption": 4,  # Equal weight for tech topics
    "climate_geopolitics": 4,   # Equal weight for climate topics
    "health_security": 4,        # Equal weight for health topics
    "great_power_competition": 4,
    "economic_warfare": 4,
    "regional_flashpoints": 3,
    "kinetic_operations": 4,
    "diplomatic_pivot": 4
}

VIRALITY_BOOST_KEYWORDS = [
    "breaking", "exclusive", "leaked", "revealed", "secret", "warns",
    "first time", "unprecedented", "imminent", "crisis", "collapse",
    "threatens", "ultimatum", "covert", "classified"
]

IMPACT_SCORE_THRESHOLD = 5
MAX_ARTICLE_AGE_HOURS = 24
TOP_N_CANDIDATES = 20
