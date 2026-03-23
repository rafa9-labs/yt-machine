"""
Military Equipment Database for Visual Extraction
Maps generic terms to exact military nomenclature with country-specific details
Enhanced with geopolitical accuracy system
"""

from typing import List, Dict, Any

MILITARY_EQUIPMENT_DB = {
    'aircraft': {
        'F-35': {
            'full_name': 'F-35 Lightning II',
            'countries': ['usa', 'israel'],
            'variants': {
                'usa': 'F-35A/B/C Lightning II with USAF/USMC/USN markings',
                'israel': 'F-35I Adir with Israeli markings and conformal fuel tanks'
            },
            'markings': 'US Air Force roundel, Israeli Air Force Star of David'
        },
        'F-16': {
            'full_name': 'F-16 Fighting Falcon',
            'countries': ['usa', 'israel', 'ukraine'],
            'variants': {
                'usa': 'F-16 Fighting Falcon with USAF roundels',
                'israel': 'F-16I Fighting Falcon with IAF roundels',
                'ukraine': 'F-16 Fighting Falcon with Ukrainian Air Force markings'
            },
            'markings': 'USAF roundel, IAF roundel, Ukrainian trident roundel'
        },
        'F-22': {
            'full_name': 'F-22 Raptor',
            'countries': ['usa'],
            'variants': {
                'usa': 'F-22 Raptor with USAF markings'
            },
            'markings': 'US Air Force roundel'
        },
        'F-15': {
            'full_name': 'F-15E Strike Eagle',
            'countries': ['usa', 'israel'],
            'variants': {
                'usa': 'F-15E Strike Eagle with USAF markings',
                'israel': 'F-15I Thunder with IAF markings'
            },
            'markings': 'USAF roundel, IAF roundel'
        },
        'F/A-18': {
            'full_name': 'F/A-18 Super Hornet',
            'countries': ['usa'],
            'variants': {
                'usa': 'F/A-18E/F Super Hornet with US Navy markings'
            },
            'markings': 'US Navy star and bars'
        },
        'A-10': {
            'full_name': 'A-10 Thunderbolt II',
            'countries': ['usa'],
            'variants': {
                'usa': 'A-10 Thunderbolt II Warthog with USAF markings'
            },
            'markings': 'USAF roundel'
        },
        'B-2': {
            'full_name': 'B-2 Spirit stealth bomber',
            'countries': ['usa'],
            'variants': {
                'usa': 'B-2 Spirit with USAF markings'
            },
            'markings': 'USAF roundel'
        },
        'B-52': {
            'full_name': 'B-52 Stratofortress',
            'countries': ['usa'],
            'variants': {
                'usa': 'B-52 Stratofortress with USAF markings'
            },
            'markings': 'USAF roundel'
        },
        'Su-35': {
            'full_name': 'Sukhoi Su-35 Flanker-E',
            'countries': ['russia', 'china'],
            'variants': {
                'russia': 'Sukhoi Su-35 with Russian VVS markings',
                'china': 'Su-35 imported with PLAAF markings'
            },
            'markings': 'Russian red star, Chinese red star with blue border'
        },
        'Su-57': {
            'full_name': 'Sukhoi Su-57 Felon',
            'countries': ['russia'],
            'variants': {
                'russia': 'Sukhoi Su-57 Felon with Russian Aerospace Forces markings'
            },
            'markings': 'Russian red star'
        },
        'MiG-29': {
            'full_name': 'MiG-29 Fulcrum',
            'countries': ['russia', 'iran', 'ukraine'],
            'variants': {
                'russia': 'MiG-29 Fulcrum with Russian VVS markings',
                'iran': 'MiG-29 Fulcrum with IRIAF Persian roundels',
                'ukraine': 'MiG-29 Fulcrum with Ukrainian Air Force markings'
            },
            'markings': 'Russian red star, Iranian green-white roundel, Ukrainian trident roundel'
        },
        'MiG-31': {
            'full_name': 'MiG-31 Foxhound',
            'countries': ['russia'],
            'variants': {
                'russia': 'MiG-31 Foxhound with Russian VVS markings'
            },
            'markings': 'Russian red star'
        },
        'J-20': {
            'full_name': 'Chengdu J-20 Mighty Dragon',
            'countries': ['china'],
            'variants': {
                'china': 'Chengdu J-20 with PLAAF markings'
            },
            'markings': 'Chinese red star with blue border, Chinese characters'
        },
        'J-16': {
            'full_name': 'Shenyang J-16',
            'countries': ['china'],
            'variants': {
                'china': 'Shenyang J-16 with PLAAF markings'
            },
            'markings': 'Chinese red star with blue border'
        },
        'J-10': {
            'full_name': 'Chengdu J-10',
            'countries': ['china'],
            'variants': {
                'china': 'Chengdu J-10 with PLAAF roundels'
            },
            'markings': 'Chinese red star'
        },
        'F-14': {
            'full_name': 'F-14 Tomcat',
            'countries': ['iran', 'usa'],
            'variants': {
                'iran': 'F-14 Tomcat (Iranian variant with IRIAF markings)',
                'usa': 'F-14 Tomcat with US Navy markings (retired)'
            },
            'markings': 'Iranian green-white roundel, US Navy star and bars'
        },
        'F-4': {
            'full_name': 'F-4 Phantom II',
            'countries': ['iran', 'usa'],
            'variants': {
                'iran': 'F-4 Phantom II (IAF variants with IRIAF markings)',
                'usa': 'F-4 Phantom II with USAF markings (retired)'
            },
            'markings': 'Iranian green-white roundel, USAF roundel'
        },
        'Su-24': {
            'full_name': 'Su-24 Fencer',
            'countries': ['russia', 'ukraine'],
            'variants': {
                'russia': 'Su-24 Fencer with Russian VVS markings',
                'ukraine': 'Su-24 Fencer with Ukrainian markings'
            },
            'markings': 'Russian red star, Ukrainian trident roundel'
        },
        'Su-27': {
            'full_name': 'Su-27 Flanker',
            'countries': ['russia', 'ukraine', 'china'],
            'variants': {
                'russia': 'Su-27 Flanker with Russian VVS markings',
                'ukraine': 'Su-27 Flanker with Ukrainian Air Force markings',
                'china': 'Su-27 imported with PLAAF markings'
            },
            'markings': 'Russian red star, Ukrainian trident roundel, Chinese red star'
        },
        'Rafale': {
            'full_name': 'Dassault Rafale',
            'countries': ['france', 'india', 'egypt', 'greece'],
            'variants': {
                'france': 'Dassault Rafale with French Air Force markings',
                'india': 'Dassault Rafale with Indian Air Force markings'
            },
            'markings': 'French roundel, Indian Air Force roundel'
        },
        'Eurofighter': {
            'full_name': 'Eurofighter Typhoon',
            'countries': ['uk', 'germany', 'italy', 'spain'],
            'variants': {
                'uk': 'Eurofighter Typhoon with RAF markings',
                'germany': 'Eurofighter Typhoon with Luftwaffe markings'
            },
            'markings': 'RAF roundel, German Iron Cross'
        },
        'Apache': {
            'full_name': 'AH-64 Apache attack helicopter',
            'countries': ['usa', 'israel', 'uk'],
            'variants': {
                'usa': 'AH-64 Apache with US Army markings',
                'israel': 'AH-64 Apache with IAF markings',
                'uk': 'AH-64 Apache with Army Air Corps markings'
            },
            'markings': 'US Army star, IAF roundel, British Army roundel'
        },
        'Black Hawk': {
            'full_name': 'UH-60 Black Hawk helicopter',
            'countries': ['usa', 'israel'],
            'variants': {
                'usa': 'UH-60 Black Hawk with US Army markings',
                'israel': 'UH-60 Black Hawk with IAF markings'
            },
            'markings': 'US Army star, IAF roundel'
        },
        'Chinook': {
            'full_name': 'CH-47 Chinook heavy-lift helicopter',
            'countries': ['usa', 'uk'],
            'variants': {
                'usa': 'CH-47 Chinook with US Army markings',
                'uk': 'CH-47 Chinook with RAF markings'
            },
            'markings': 'US Army star, RAF roundel'
        },
        'Predator': {
            'full_name': 'MQ-1 Predator drone',
            'countries': ['usa', 'italy'],
            'variants': {
                'usa': 'MQ-1 Predator with USAF markings',
                'italy': 'MQ-1 Predator with Italian Air Force markings'
            },
            'markings': 'USAF roundel, Italian Air Force roundel'
        },
        'Reaper': {
            'full_name': 'MQ-9 Reaper drone',
            'countries': ['usa', 'uk', 'italy', 'france', 'netherlands'],
            'variants': {
                'usa': 'MQ-9 Reaper with USAF markings',
                'uk': 'MQ-9 Reaper with RAF markings'
            },
            'markings': 'USAF roundel, RAF roundel'
        },
        'Global Hawk': {
            'full_name': 'RQ-4 Global Hawk surveillance drone',
            'countries': ['usa', 'nato'],
            'variants': {
                'usa': 'RQ-4 Global Hawk with USAF markings',
                'nato': 'RQ-4 Global Hawk with NATO markings'
            },
            'markings': 'USAF roundel, NATO emblem'
        }
    },
    'naval': {
        'carrier': {
            'full_name': 'Aircraft Carrier',
            'countries': ['usa', 'russia', 'china', 'uk', 'france'],
            'variants': {
                'usa': 'Nimitz-class aircraft carrier with US Navy markings',
                'russia': 'Admiral Kuznetsov aircraft carrier with Russian Navy markings',
                'china': 'Liaoning aircraft carrier with PLA Navy markings',
                'uk': 'Queen Elizabeth-class aircraft carrier with Royal Navy markings',
                'france': 'Charles de Gaulle aircraft carrier with French Navy markings'
            },
            'markings': 'US Navy hull number, Russian red star, PLA Navy pennant, Royal Navy crest, French roundel'
        },
        'Ford carrier': {
            'full_name': 'Gerald R. Ford-class aircraft carrier',
            'countries': ['usa'],
            'variants': {
                'usa': 'USS Gerald R. Ford with US Navy markings'
            },
            'markings': 'US Navy hull number and crest'
        },
        'destroyer': {
            'full_name': 'Guided Missile Destroyer',
            'countries': ['usa', 'russia', 'china', 'uk', 'iran'],
            'variants': {
                'usa': 'Arleigh Burke-class destroyer with US Navy markings',
                'russia': 'Sovremennyy-class destroyer with Russian Navy markings',
                'china': 'Type 055 destroyer with PLA Navy markings',
                'uk': 'Type 45 destroyer with Royal Navy markings',
                'iran': 'Jamaran-class frigate with Iranian Navy markings'
            },
            'markings': 'US Navy hull number, Russian red star, PLA Navy pennant, Royal Navy crest, Iranian naval jack'
        },
        'cruiser': {
            'full_name': 'Guided Missile Cruiser',
            'countries': ['usa', 'russia'],
            'variants': {
                'usa': 'Ticonderoga-class guided missile cruiser with US Navy markings',
                'russia': 'Kirov-class battlecruiser with Russian Navy markings'
            },
            'markings': 'US Navy hull number, Russian red star'
        },
        'submarine': {
            'full_name': 'Attack Submarine',
            'countries': ['usa', 'russia', 'china', 'uk', 'iran'],
            'variants': {
                'usa': 'Virginia-class attack submarine with US Navy markings',
                'russia': 'Yasen-class submarine with Russian Navy markings',
                'china': 'Type 093 submarine with PLA Navy markings',
                'uk': 'Astute-class submarine with Royal Navy markings',
                'iran': 'Ghadir-class submarine with Iranian Navy markings'
            },
            'markings': 'US Navy hull number, Russian red star, PLA Navy pennant, Royal Navy crest, Iranian naval jack'
        },
        'nuclear submarine': {
            'full_name': 'Ballistic Missile Submarine',
            'countries': ['usa', 'russia', 'china', 'uk'],
            'variants': {
                'usa': 'Ohio-class ballistic missile submarine with US Navy markings',
                'russia': 'Borei-class submarine with Russian Navy markings',
                'china': 'Type 094 submarine with PLA Navy markings',
                'uk': 'Vanguard-class submarine with Royal Navy markings'
            },
            'markings': 'US Navy hull number, Russian red star, PLA Navy pennant, Royal Navy crest'
        },
        'frigate': {
            'full_name': 'Frigate',
            'countries': ['usa', 'uk', 'iran', 'ukraine'],
            'variants': {
                'usa': 'Constellation-class frigate with US Navy markings',
                'uk': 'Type 23 frigate with Royal Navy markings',
                'iran': 'Jamaran-class frigate with Iranian Navy markings',
                'ukraine': 'Hetman-class frigate with Ukrainian Navy markings'
            },
            'markings': 'US Navy hull number, Royal Navy crest, Iranian naval jack, Ukrainian trident'
        },
        'amphibious': {
            'full_name': 'Amphibious Assault Ship',
            'countries': ['usa', 'uk', 'france'],
            'variants': {
                'usa': 'Wasp-class amphibious assault ship with US Navy markings',
                'uk': 'Albion-class amphibious transport dock with Royal Navy markings',
                'france': 'Mistral-class amphibious assault ship with French Navy markings'
            },
            'markings': 'US Navy hull number, Royal Navy crest, French Navy jack'
        },
        'LCS': {
            'full_name': 'Littoral Combat Ship',
            'countries': ['usa'],
            'variants': {
                'usa': 'Freedom-class LCS with US Navy markings'
            },
            'markings': 'US Navy hull number'
        },
        'Type 055': {
            'full_name': 'Type 055 Renhai-class destroyer',
            'countries': ['china'],
            'variants': {
                'china': 'Type 055 Renhai-class destroyer with PLA Navy markings'
            },
            'markings': 'PLA Navy pennant number'
        },
        'Type 052D': {
            'full_name': 'Type 052D Luyang III-class destroyer',
            'countries': ['china'],
            'variants': {
                'china': 'Type 052D Luyang III-class destroyer with PLA Navy markings'
            },
            'markings': 'PLA Navy pennant number'
        },
        'Kirov': {
            'full_name': 'Kirov-class battlecruiser',
            'countries': ['russia'],
            'variants': {
                'russia': 'Kirov-class battlecruiser with Russian Navy markings'
            },
            'markings': 'Russian red star and hull number'
        },
        'Admiral Kuznetsov': {
            'full_name': 'Admiral Kuznetsov aircraft carrier',
            'countries': ['russia'],
            'variants': {
                'russia': 'Admiral Kuznetsov aircraft carrier with Russian Navy markings'
            },
            'markings': 'Russian red star and hull number'
        },
        'Jamaran': {
            'full_name': 'Jamaran-class frigate',
            'countries': ['iran'],
            'variants': {
                'iran': 'Jamaran-class frigate with Iranian Navy markings'
            },
            'markings': 'Iranian naval jack and hull number'
        },
        'Ghadir': {
            'full_name': 'Ghadir-class submarine',
            'countries': ['iran'],
            'variants': {
                'iran': 'Ghadir-class submarine with Iranian Navy markings'
            },
            'markings': 'Iranian naval jack and hull number'
        },
        'Hetman': {
            'full_name': 'Hetman-class frigate',
            'countries': ['ukraine'],
            'variants': {
                'ukraine': 'Hetman-class frigate with Ukrainian Navy markings'
            },
            'markings': 'Ukrainian trident and naval ensign'
        },
        'Sa\'ar': {
            'full_name': 'Sa\'ar-class corvette',
            'countries': ['israel'],
            'variants': {
                'israel': 'Sa\'ar-class corvette with Israeli Navy markings'
            },
            'markings': 'Israeli naval ensign and hull number'
        },
        'Dolphin': {
            'full_name': 'Dolphin-class submarine',
            'countries': ['israel'],
            'variants': {
                'israel': 'Dolphin-class submarine with Israeli Navy markings'
            },
            'markings': 'Israeli naval ensign and hull number'
        }
    },
    'missiles': {
        'Tomahawk': {
            'full_name': 'BGM-109 Tomahawk cruise missile',
            'countries': ['usa', 'uk'],
            'variants': {
                'usa': 'BGM-109 Tomahawk with US Navy markings',
                'uk': 'BGM-109 Tomahawk with Royal Navy markings'
            },
            'markings': 'US Navy insignia, Royal Navy crest'
        },
        'Patriot': {
            'full_name': 'MIM-104 Patriot surface-to-air missile',
            'countries': ['usa', 'israel', 'germany', 'japan', 'saudi_arabia', 'ukraine'],
            'variants': {
                'usa': 'MIM-104 Patriot with US Army markings',
                'israel': 'MIM-104 Patriot with IAF markings',
                'ukraine': 'MIM-104 Patriot with Ukrainian Air Force markings'
            },
            'markings': 'US Army star, IAF roundel, Ukrainian trident'
        },
        'THAAD': {
            'full_name': 'Terminal High Altitude Area Defense system',
            'countries': ['usa', 'uae', 'saudi_arabia'],
            'variants': {
                'usa': 'THAAD system with US Army markings'
            },
            'markings': 'US Army insignia'
        },
        'Iron Dome': {
            'full_name': 'Iron Dome missile defense system',
            'countries': ['israel', 'usa'],
            'variants': {
                'israel': 'Iron Dome with IDF markings',
                'usa': 'Iron Dome with US Army markings'
            },
            'markings': 'IDF insignia, US Army star'
        },
        'S-400': {
            'full_name': 'S-400 Triumf air defense system',
            'countries': ['russia', 'china', 'india', 'turkey'],
            'variants': {
                'russia': 'S-400 Triumf with Russian Aerospace Forces markings',
                'china': 'S-400 Triumf with PLAAF markings',
                'turkey': 'S-400 Triumf with Turkish Air Force markings'
            },
            'markings': 'Russian red star, Chinese red star, Turkish Air Force roundel'
        },
        'S-300': {
            'full_name': 'S-300 air defense system',
            'countries': ['russia', 'china', 'ukraine', 'iran'],
            'variants': {
                'russia': 'S-300 with Russian Aerospace Forces markings',
                'ukraine': 'S-300 with Ukrainian Air Force markings',
                'iran': 'S-300 with IRIAF markings'
            },
            'markings': 'Russian red star, Ukrainian trident, Iranian green-white roundel'
        },
        'HIMARS': {
            'full_name': 'M142 High Mobility Artillery Rocket System',
            'countries': ['usa', 'ukraine', 'poland', 'romania'],
            'variants': {
                'usa': 'M142 HIMARS with US Army markings',
                'ukraine': 'M142 HIMARS with Ukrainian Armed Forces markings'
            },
            'markings': 'US Army star, Ukrainian trident'
        },
        'Javelin': {
            'full_name': 'FGM-148 Javelin anti-tank missile',
            'countries': ['usa', 'ukraine', 'uk', 'australia'],
            'variants': {
                'usa': 'FGM-148 Javelin with US Army markings',
                'ukraine': 'FGM-148 Javelin with Ukrainian Armed Forces markings'
            },
            'markings': 'US Army star, Ukrainian trident'
        },
        'Hellfire': {
            'full_name': 'AGM-114 Hellfire missile',
            'countries': ['usa', 'uk', 'israel'],
            'variants': {
                'usa': 'AGM-114 Hellfire with US Army/USAF markings',
                'israel': 'AGM-114 Hellfire with IAF markings'
            },
            'markings': 'US Army star, IAF roundel'
        },
        'Stinger': {
            'full_name': 'FIM-92 Stinger missile',
            'countries': ['usa', 'ukraine'],
            'variants': {
                'usa': 'FIM-92 Stinger with US Army markings',
                'ukraine': 'FIM-92 Stinger with Ukrainian Armed Forces markings'
            },
            'markings': 'US Army star, Ukrainian trident'
        },
        'Shahab': {
            'full_name': 'Shahab series ballistic missiles',
            'countries': ['iran'],
            'variants': {
                'iran': 'Shahab-3 ballistic missile with IRGC markings'
            },
            'markings': 'IRGC insignia, Iranian flag colors'
        },
        'Fateh': {
            'full_name': 'Fateh-110 missile',
            'countries': ['iran'],
            'variants': {
                'iran': 'Fateh-110 missile with IRGC markings'
            },
            'markings': 'IRGC insignia'
        },
        'Iskander': {
            'full_name': 'Iskander missile',
            'countries': ['russia'],
            'variants': {
                'russia': 'Iskander missile with Russian Ground Forces markings'
            },
            'markings': 'Russian red star'
        },
        'Kalibr': {
            'full_name': 'Kalibr cruise missile',
            'countries': ['russia'],
            'variants': {
                'russia': 'Kalibr cruise missile with Russian Navy markings'
            },
            'markings': 'Russian red star'
        },
        'DF-21': {
            'full_name': 'DF-21 missile',
            'countries': ['china'],
            'variants': {
                'china': 'DF-21 missile with PLAAF markings'
            },
            'markings': 'Chinese red star'
        },
        'Neptune': {
            'full_name': 'Neptune anti-ship missile',
            'countries': ['ukraine'],
            'variants': {
                'ukraine': 'Neptune anti-ship missile with Ukrainian Armed Forces markings'
            },
            'markings': 'Ukrainian trident'
        },
        'Arrow': {
            'full_name': 'Arrow missile system',
            'countries': ['israel', 'usa'],
            'variants': {
                'israel': 'Arrow missile system with IDF markings'
            },
            'markings': 'IDF insignia'
        },
        'David\'s Sling': {
            'full_name': 'David\'s Sling missile system',
            'countries': ['israel', 'usa'],
            'variants': {
                'israel': 'David\'s Sling with IDF markings'
            },
            'markings': 'IDF insignia'
        },
        'Scalpel': {
            'full_name': 'Scalpel cruise missile',
            'countries': ['ukraine'],
            'variants': {
                'ukraine': 'Scalpel cruise missile with Ukrainian Armed Forces markings'
            },
            'markings': 'Ukrainian trident'
        }
    }
}

# Helper functions for equipment-country validation
def get_equipment_countries(equipment_key: str) -> List[str]:
    """Get list of countries that use specific equipment"""
    for category, equipment_dict in MILITARY_EQUIPMENT_DB.items():
        if equipment_key in equipment_dict:
            equipment_info = equipment_dict[equipment_key]
            if isinstance(equipment_info, dict) and 'countries' in equipment_info:
                return equipment_info['countries']
            # Backward compatibility for old string format
            return ['usa']  # Default to USA for old format
    return []

def get_country_specific_variant(equipment_key: str, country: str) -> str:
    """Get country-specific variant description of equipment"""
    for category, equipment_dict in MILITARY_EQUIPMENT_DB.items():
        if equipment_key in equipment_dict:
            equipment_info = equipment_dict[equipment_key]
            if isinstance(equipment_info, dict) and 'variants' in equipment_info:
                variants = equipment_info['variants']
                if country in variants:
                    return variants[country]
                # Fallback to first available variant
                return list(variants.values())[0]
            # Backward compatibility for old string format
            return equipment_info if isinstance(equipment_info, str) else equipment_info.get('full_name', equipment_key)
    return equipment_key

def validate_equipment_country_combination(equipment: str, country: str) -> bool:
    """Validate if equipment can be used by specified country"""
    equipment_countries = get_equipment_countries(equipment)
    return country.lower() in [c.lower() for c in equipment_countries]

def get_equipment_markings(equipment_key: str, country: str) -> str:
    """Get appropriate markings for equipment-country combination"""
    for category, equipment_dict in MILITARY_EQUIPMENT_DB.items():
        if equipment_key in equipment_dict:
            equipment_info = equipment_dict[equipment_key]
            if isinstance(equipment_info, dict) and 'markings' in equipment_info:
                markings = equipment_info['markings']
                # Extract country-specific markings if comma-separated
                if ',' in markings:
                    marking_parts = [m.strip() for m in markings.split(',')]
                    for marking in marking_parts:
                        if any(country_tag in marking.lower() for country_tag in [country, 'us', 'usa', 'russian', 'chinese', 'iranian', 'israeli', 'ukrainian']):
                            return marking
                return markings
    return ''

# Known geographic locations for validation
KNOWN_LOCATIONS = {
    'middle_east': [
        'Strait of Hormuz',
        'Persian Gulf',
        'Gulf of Oman',
        'Red Sea',
        'Suez Canal',
        'Bab el-Mandeb',
        'Tehran',
        'Tel Aviv',
        'Jerusalem',
        'Baghdad',
        'Damascus',
        'Beirut',
        'Riyadh',
        'Dubai',
        'Abu Dhabi',
        'Kuwait City',
        'Doha',
        'Manama',
        'Muscat',
        'Haifa',
        'Gaza Strip',
        'West Bank',
        'Golan Heights',
        'Sinai Peninsula',
        'Natanz',
        'Fordow',
        'Bushehr',
        'Bandar Abbas',
        'Kharg Island'
    ],
    'asia_pacific': [
        'South China Sea',
        'Taiwan Strait',
        'East China Sea',
        'Sea of Japan',
        'Korean Peninsula',
        'Malacca Strait',
        'Senkaku Islands',
        'Spratly Islands',
        'Paracel Islands',
        'Beijing',
        'Shanghai',
        'Hong Kong',
        'Taipei',
        'Seoul',
        'Pyongyang',
        'Tokyo',
        'Manila',
        'Singapore',
        'Hanoi'
    ],
    'europe': [
        'Baltic Sea',
        'Black Sea',
        'Mediterranean Sea',
        'English Channel',
        'North Sea',
        'Moscow',
        'Kyiv',
        'Warsaw',
        'Berlin',
        'Paris',
        'London',
        'Istanbul',
        'Crimea',
        'Donbas',
        'Kaliningrad',
        'Gibraltar'
    ],
    'americas': [
        'Caribbean Sea',
        'Panama Canal',
        'Gulf of Mexico',
        'Washington DC',
        'New York',
        'Los Angeles',
        'Miami',
        'Guantanamo Bay',
        'Caracas',
        'Havana'
    ],
    'africa': [
        'Suez Canal',
        'Gulf of Guinea',
        'Horn of Africa',
        'Djibouti',
        'Somalia',
        'Libya',
        'Cairo',
        'Tripoli'
    ]
}

def normalize_equipment(text: str) -> str:
    """
    Convert generic equipment mentions to specific nomenclature
    
    Args:
        text: Text potentially containing equipment mentions
        
    Returns:
        Text with normalized equipment names
    """
    normalized = text
    
    for category, equipment_dict in MILITARY_EQUIPMENT_DB.items():
        for short_name, full_name in equipment_dict.items():
            # Case-insensitive replacement
            import re
            pattern = re.compile(re.escape(short_name), re.IGNORECASE)
            normalized = pattern.sub(full_name, normalized)
    
    return normalized

def get_all_locations() -> list:
    """Get flat list of all known locations"""
    all_locs = []
    for region, locs in KNOWN_LOCATIONS.items():
        all_locs.extend(locs)
    return all_locs

def is_valid_location(location: str) -> bool:
    """Check if location is in known locations database"""
    all_locs = get_all_locations()
    return any(loc.lower() in location.lower() for loc in all_locs)
