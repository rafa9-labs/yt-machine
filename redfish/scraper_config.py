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

GEOPOLITICAL_KEYWORDS = {
    "kinetic_conflict": [
        "strike", "airstrike", "missile", "military", "war", "attack",
        "blockade", "naval", "troops", "offensive", "ceasefire", "escalation",
        "iran", "israel", "hormuz", "hezbollah", "irgc", "idf"
    ],
    "diplomatic_pivot": [
        "negotiation", "deal", "summit", "alliance", "sanctions", "treaty",
        "envoy", "backchannel", "normalization", "pivot", "realignment",
        "transactional", "leverage", "washington", "state department"
    ],
    "economic_leverage": [
        "oil", "energy", "shipping", "tanker", "strait", "embargo",
        "currency", "dollar", "yuan", "petrodollar", "opec", "inflation",
        "supply chain", "chokepoint", "reserve", "freeze"
    ]
}

VIRALITY_BOOST_KEYWORDS = [
    "breaking", "exclusive", "leaked", "revealed", "secret", "warns",
    "first time", "unprecedented", "imminent", "crisis", "collapse",
    "threatens", "ultimatum", "covert", "classified"
]

IMPACT_SCORE_THRESHOLD = 5
MAX_ARTICLE_AGE_HOURS = 24
TOP_N_CANDIDATES = 20
