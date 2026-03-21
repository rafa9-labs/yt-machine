# Video Selection Algorithm

## How the System Decides Which Article to Make a Video About

### 1. **RSS Feed Scraping** (`redfish/rss_scraper.py`)
The system scrapes 8 RSS feeds every 24 hours:
- **Priority 1 (High-value)**: Reuters, Al Jazeera, Foreign Policy, Stratfor
- **Priority 2 (Standard)**: BBC, AP, Middle East Eye, Defense One

### 2. **Virality Scoring Algorithm** (`filter_viral_potential()`)
Each article gets scored based on:

#### **Keyword Matching (Main Score)**
- **Kinetic Conflict** keywords in title: +4 points each
  - Examples: strike, missile, military, war, Iran, Israel, Hormuz
- **Diplomatic Pivot** keywords in title: +4 points each
  - Examples: sanctions, deal, summit, alliance, treaty
- **Economic Leverage** keywords in title: +4 points each
  - Examples: oil, energy, shipping, tanker, embargo, OPEC

#### **Virality Boost** keywords in title: +3 points each
- breaking, exclusive, leaked, revealed, secret, warns
- unprecedented, imminent, crisis, collapse, threatens

#### **Feed Priority Bonus**
- Priority 1 feeds (Reuters, Al Jazeera, etc.): +3 points
- Priority 2 feeds: +0 points

#### **Title Length Bonus**
- Titles between 40-100 characters: +1 point

### 3. **Selection Process**
1. Score all articles from the last 24 hours
2. Sort by virality_score (highest first)
3. Return top 5 candidates
4. **Select the #1 highest-scoring article** for video generation

### 4. **Why You Keep Getting Iran Videos**
The current keyword list is **heavily biased toward Middle East geopolitics**:
- "iran", "israel", "hormuz", "hezbollah", "irgc", "idf" are all high-value keywords
- Iran-related articles naturally score 12-20+ points
- Other topics (tech, climate, economics) score 0-5 points unless they have military/conflict angles

## How to Get Different Topics

### **Option 1: Manual Article Selection**
Create a script that lets you choose from the top 10 articles instead of auto-selecting #1.

### **Option 2: Expand Keyword Categories**
Add new keyword categories to `scraper_config.py`:
```python
"technology_disruption": ["ai", "quantum", "semiconductor", "chip", "5g", "cyber"],
"climate_geopolitics": ["climate", "carbon", "renewable", "drought", "water", "food security"],
"economic_warfare": ["tariff", "trade war", "debt", "default", "recession", "inflation"]
```

### **Option 3: Category Rotation**
Modify the selection algorithm to rotate between categories:
- Day 1: Kinetic conflict
- Day 2: Economic leverage
- Day 3: Technology disruption
- Day 4: Climate geopolitics

### **Option 4: Diversity Filter**
Track the last 5 videos generated and penalize articles with similar topics to avoid repetition.

## Current Behavior Summary
**The system automatically selects the highest-scoring article, which is almost always Iran/Israel/Middle East related due to keyword bias.**

To test other topics, you need to either:
1. Manually select a different article from the top 10
2. Expand the keyword list to include non-military topics
3. Implement a diversity filter
