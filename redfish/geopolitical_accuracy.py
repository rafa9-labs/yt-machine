"""
Geopolitical Visual Accuracy System
Ensures accurate country representation with zero tolerance for hallucination
"""

from typing import Dict, List, Tuple, Any

# Country-specific visual specifications with exact details
COUNTRY_VISUAL_SPECS = {
    'iran': {
        'flag_colors': 'green, white, red with emblem',
        'military_branches': {
            'army': 'Islamic Republic of Iran Army, green camouflage with Persian script',
            'navy': 'Islamic Republic of Iran Navy, white uniforms with naval ensign',
            'air_force': 'Islamic Republic of Iran Air Force, blue-grey uniforms with roundel',
            'irgc': 'Islamic Revolutionary Guard Corps, green uniforms with IRGC insignia'
        },
        'aircraft_markings': 'green and white roundel with red emblem, Persian script on tail',
        'naval_markings': 'white hull with green stripe, Iranian naval jack',
        'uniform_colors': 'olive drab, green camouflage, desert tan',
        'equipment_specific': {
            'F-14': 'F-14 Tomcat (Iranian variant with IRIAF markings)',
            'F-4': 'F-4 Phantom II (IAF variants with IRIAF markings)',
            'MiG-29': 'MiG-29 Fulcrum (IRIAF with Persian roundels)',
            'missiles': 'Shahab series ballistic missiles, Fateh-110 missiles',
            'naval': 'Jamaran-class frigates, Ghadir-class submarines'
        }
    },
    
    'israel': {
        'flag_colors': 'white with blue Star of David and stripes',
        'military_branches': {
            'army': 'Israel Defense Forces, olive drab with IDF insignia',
            'navy': 'Israeli Navy, white uniforms with naval anchor',
            'air_force': 'Israeli Air Force, blue uniforms with IAF roundel'
        },
        'aircraft_markings': 'blue Star of David roundel on wings and fuselage',
        'naval_markings': 'grey hull with Israeli naval ensign',
        'uniform_colors': 'olive drab, desert tan, navy blue',
        'equipment_specific': {
            'F-35': 'F-35I Adir with Israeli markings and conformal fuel tanks',
            'F-16': 'F-16I Fighting Falcon with IAF roundels',
            'Apache': 'AH-64 Apache with IAF markings',
            'missiles': 'Iron Dome, Arrow missile system, David\'s Sling',
            'naval': 'Sa\'ar-class corvettes, Dolphin-class submarines'
        }
    },
    
    'russia': {
        'flag_colors': 'white, blue, red tricolor',
        'military_branches': {
            'army': 'Russian Ground Forces, green uniforms with red star',
            'navy': 'Russian Navy, dark blue uniforms with naval ensign',
            'air_force': 'Russian Aerospace Forces, blue uniforms with red star'
        },
        'aircraft_markings': 'red star on wings and tail, Russian Federation insignia',
        'naval_markings': 'dark grey hull with red star and white numbers',
        'uniform_colors': 'green, navy blue, camouflage with Russian patterns',
        'equipment_specific': {
            'Su-35': 'Sukhoi Su-35 with Russian VVS markings',
            'Su-57': 'Sukhoi Su-57 Felon with Russian Aerospace Forces',
            'MiG-31': 'MiG-31 Foxhound with Russian markings',
            'missiles': 'S-400 Triumf, Iskander missiles, Kalibr cruise missiles',
            'naval': 'Admiral Kuznetsov carrier, Kirov-class battlecruisers'
        }
    },
    
    'china': {
        'flag_colors': 'red with five yellow stars',
        'military_branches': {
            'army': 'People\'s Liberation Army Ground Force, green uniforms with red star',
            'navy': 'People\'s Liberation Army Navy, white uniforms with PLA naval ensign',
            'air_force': 'People\'s Liberation Army Air Force, blue uniforms with red star'
        },
        'aircraft_markings': 'red star with blue border, Chinese characters on tail',
        'naval_markings': 'grey hull with PLA Navy pennant number',
        'uniform_colors': 'green PLA camouflage, navy blue, light blue',
        'equipment_specific': {
            'J-20': 'Chengdu J-20 with PLAAF markings',
            'J-16': 'Shenyang J-16 with PLAAF markings',
            'J-10': 'Chengdu J-10 with PLAAF roundels',
            'missiles': 'DF-21 missiles, HQ-9 air defense, YJ-18 anti-ship missiles',
            'naval': 'Type 055 destroyers, Type 052D destroyers, Liaoning carrier'
        }
    },
    
    'ukraine': {
        'flag_colors': 'blue and yellow bicolour',
        'military_branches': {
            'army': 'Armed Forces of Ukraine, digital camouflage with trident insignia',
            'navy': 'Ukrainian Navy, blue uniforms with naval trident',
            'air_force': 'Ukrainian Air Force, blue uniforms with trident roundel'
        },
        'aircraft_markings': 'blue and yellow roundel with trident, Ukrainian insignia',
        'naval_markings': 'grey hull with Ukrainian flag and trident',
        'uniform_colors': 'digital camouflage, blue and yellow accents',
        'equipment_specific': {
            'MiG-29': 'MiG-29 Fulcrum with Ukrainian Air Force markings',
            'Su-24': 'Su-24 Fencer with Ukrainian markings',
            'Su-27': 'Su-27 Flanker with Ukrainian roundels',
            'missiles': 'S-300 air defense, Neptune anti-ship missiles',
            'naval': 'Hetman-class frigates, Ukrainian patrol boats'
        }
    },
    
    'usa': {
        'flag_colors': 'red, white, blue with stars and stripes',
        'military_branches': {
            'army': 'US Army, ACU camouflage with US flag patch',
            'navy': 'US Navy, blue uniforms with Navy crest',
            'air_force': 'US Air Force, blue uniforms with USAF roundel',
            'marines': 'US Marine Corps, MARPAT with Eagle Globe Anchor'
        },
        'aircraft_markings': 'US Air Force roundel, US Navy star and bars',
        'naval_markings': 'grey hull with hull number and US Navy ensign',
        'uniform_colors': 'ACU camouflage, MARPAT, Navy blue, Air Force blue',
        'equipment_specific': {
            'F-35': 'F-35A/B/C Lightning II with USAF/USMC/USN markings',
            'F-22': 'F-22 Raptor with USAF markings',
            'F-16': 'F-16 Fighting Falcon with USAF roundels',
            'missiles': 'Patriot, THAAD, Tomahawk cruise missiles',
            'naval': 'Nimitz-class carriers, Arleigh Burke destroyers'
        }
    },
    
    'nato': {
        'flag_colors': 'blue and white NATO emblem',
        'military_branches': {
            'command': 'NATO Command Structure, multinational uniforms with NATO emblem',
            'multinational': 'Multinational forces, mixed national uniforms with NATO patches'
        },
        'aircraft_markings': 'NATO roundel with multinational variations',
        'naval_markings': 'multinational naval task force with NATO pennant',
        'uniform_colors': 'multinational camouflage patterns, NATO blue',
        'equipment_specific': {
            'multinational': 'Combined NATO forces, mixed equipment from member nations',
            'E-3': 'E-3 Sentry AWACS with NATO markings',
            'naval': 'Standing Naval Forces, multinational ship classes'
        }
    }
}

# Country-equipment mapping to prevent incorrect combinations
COUNTRY_EQUIPMENT_MAPPING = {
    'iran': ['F-14', 'F-4', 'MiG-29', 'Shahab', 'Jamaran', 'Ghadir', 'IRGC', 'F-14 Tomcat'],
    'israel': ['F-35I', 'F-16I', 'AH-64', 'Iron Dome', 'Sa\'ar', 'Dolphin', 'IDF', 'IAF'],
    'russia': ['Su-35', 'Su-57', 'MiG-31', 'S-400', 'Admiral Kuznetsov', 'Kirov', 'VVS'],
    'china': ['J-20', 'J-16', 'J-10', 'DF-21', 'Type 055', 'Type 052D', 'Liaoning', 'PLAAF', 'PLAN'],
    'ukraine': ['MiG-29', 'Su-24', 'Su-27', 'S-300', 'Neptune', 'Hetman', 'trident'],
    'usa': ['F-35', 'F-22', 'F-16', 'Patriot', 'THAAD', 'Nimitz', 'Arleigh Burke', 'USAF', 'USN', 'USMC', 'F-14'],
    'nato': ['multinational', 'E-3', 'AWACS', 'Standing Naval Forces']
}

# Hallucination prevention rules
HALLUCINATION_PREVENTION_RULES = {
    'no_mixing': [
        ('iran', 'F-35'),      # Iran doesn't use F-35
        ('iran', 'F-22'),      # Iran doesn't use F-22
        ('iran', 'Patriot'),   # Iran doesn't use Patriot
        ('israel', 'Su-57'),   # Israel doesn't use Russian aircraft
        ('israel', 'J-20'),    # Israel doesn't use Chinese aircraft
        ('usa', 'S-400'),      # USA doesn't use S-400
        ('usa', 'Su-57'),      # USA doesn't use Su-57
        ('ukraine', 'J-20'),   # Ukraine doesn't use Chinese aircraft
        ('ukraine', 'F-35'),   # Ukraine doesn't use F-35 (yet)
        ('russia', 'F-35'),    # Russia doesn't use Western aircraft
        ('china', 'F-35'),     # China doesn't use F-35
    ],
    'required_elements': {
        'iran': ['green', 'Persian script', 'IRGC insignia', 'Islamic Republic'],
        'israel': ['Star of David', 'IAF roundel', 'IDF insignia', 'Israeli'],
        'russia': ['red star', 'Russian Federation', 'Cyrillic', 'VVS'],
        'china': ['red star', 'Chinese characters', 'PLA insignia', 'PLAAF'],
        'ukraine': ['trident', 'blue-yellow', 'Ukrainian insignia', 'Armed Forces of Ukraine'],
        'usa': ['stars and stripes', 'US roundel', 'USA markings', 'United States'],
        'nato': ['NATO emblem', 'multinational', 'blue and white', 'Alliance']
    }
}

def validate_country_equipment_combination(country: str, equipment: str) -> bool:
    """Validate that equipment belongs to the specified country"""
    if country not in COUNTRY_EQUIPMENT_MAPPING:
        return False
    
    # Check if equipment is in country's approved list
    country_equipment = COUNTRY_EQUIPMENT_MAPPING[country]
    for approved in country_equipment:
        # Check for partial matches (e.g., "F-14 Tomcat" should match "F-14")
        if approved.lower() in equipment.lower() or equipment.lower() in approved.lower():
            return True
    
    return False

def get_country_visual_spec(country: str) -> dict:
    """Get detailed visual specifications for a country"""
    return COUNTRY_VISUAL_SPECS.get(country.lower(), {})

def get_required_country_elements(country: str) -> list:
    """Get required visual elements that must appear for country accuracy"""
    return HALLUCINATION_PREVENTION_RULES['required_elements'].get(country.lower(), [])

def check_hallucination_risks(country: str, equipment: str) -> list:
    """Check for potential hallucination risks"""
    risks = []
    
    # Check forbidden combinations
    for forbidden_country, forbidden_equipment in HALLUCINATION_PREVENTION_RULES['no_mixing']:
        if country.lower() == forbidden_country and forbidden_equipment.lower() in equipment.lower():
            risks.append(f"Hallucination risk: {forbidden_country} does not use {forbidden_equipment}")
    
    return risks

def extract_countries_from_text(text: str) -> List[str]:
    """Extract country mentions from text"""
    found_countries = []
    text_lower = text.lower()
    
    for country in COUNTRY_VISUAL_SPECS.keys():
        if country in text_lower:
            found_countries.append(country)
    
    return found_countries

def extract_equipment_from_text(text: str) -> List[str]:
    """Extract equipment mentions from text"""
    found_equipment = []
    text_lower = text.lower()
    
    # Check all equipment in country mappings
    for country, equipment_list in COUNTRY_EQUIPMENT_MAPPING.items():
        for equipment in equipment_list:
            if equipment.lower() in text_lower:
                found_equipment.append(equipment)
    
    return list(set(found_equipment))  # Remove duplicates

def validate_geopolitical_accuracy(text: str) -> Dict[str, Any]:
    """Comprehensive geopolitical accuracy validation"""
    countries = extract_countries_from_text(text)
    equipment = extract_equipment_from_text(text)
    
    validation_results = {
        'countries_found': countries,
        'equipment_found': equipment,
        'combinations_valid': True,
        'risks': [],
        'missing_required_elements': [],
        'accuracy_score': 100
    }
    
    # Check each country-equipment combination
    for country in countries:
        for equip in equipment:
            if not validate_country_equipment_combination(country, equip):
                validation_results['combinations_valid'] = False
                validation_results['risks'].append(f"Invalid combination: {country} + {equip}")
                validation_results['accuracy_score'] -= 25
            
            # Check hallucination risks
            risks = check_hallucination_risks(country, equip)
            validation_results['risks'].extend(risks)
            if risks:
                validation_results['accuracy_score'] -= 50
    
    # Check required elements for each country
    for country in countries:
        required_elements = get_required_country_elements(country)
        for element in required_elements:
            if element.lower() not in text.lower():
                validation_results['missing_required_elements'].append(f"Missing {element} for {country}")
                validation_results['accuracy_score'] -= 10
    
    # Ensure score doesn't go negative
    validation_results['accuracy_score'] = max(0, validation_results['accuracy_score'])
    
    return validation_results
