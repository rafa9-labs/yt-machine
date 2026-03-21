"""
Military Equipment Database for Visual Extraction
Maps generic terms to exact military nomenclature
"""

MILITARY_EQUIPMENT_DB = {
    'aircraft': {
        'F-35': 'F-35 Lightning II',
        'F-16': 'F-16I Strike Eagle',
        'F-15': 'F-15E Strike Eagle',
        'F-22': 'F-22 Raptor',
        'F/A-18': 'F/A-18 Super Hornet',
        'A-10': 'A-10 Thunderbolt II',
        'B-2': 'B-2 Spirit stealth bomber',
        'B-52': 'B-52 Stratofortress',
        'Su-35': 'Sukhoi Su-35 Flanker-E',
        'Su-57': 'Sukhoi Su-57 Felon',
        'MiG-29': 'MiG-29 Fulcrum',
        'J-20': 'Chengdu J-20 Mighty Dragon',
        'Rafale': 'Dassault Rafale',
        'Eurofighter': 'Eurofighter Typhoon',
        'Apache': 'AH-64 Apache attack helicopter',
        'Black Hawk': 'UH-60 Black Hawk helicopter',
        'Chinook': 'CH-47 Chinook heavy-lift helicopter',
        'Predator': 'MQ-1 Predator drone',
        'Reaper': 'MQ-9 Reaper drone',
        'Global Hawk': 'RQ-4 Global Hawk surveillance drone'
    },
    'naval': {
        'carrier': 'Nimitz-class aircraft carrier',
        'Ford carrier': 'Gerald R. Ford-class aircraft carrier',
        'destroyer': 'Arleigh Burke-class destroyer',
        'cruiser': 'Ticonderoga-class guided missile cruiser',
        'submarine': 'Virginia-class attack submarine',
        'nuclear submarine': 'Ohio-class ballistic missile submarine',
        'frigate': 'Constellation-class frigate',
        'amphibious': 'Wasp-class amphibious assault ship',
        'LCS': 'Littoral Combat Ship',
        'Type 055': 'Type 055 Renhai-class destroyer',
        'Type 052D': 'Type 052D Luyang III-class destroyer',
        'Kirov': 'Kirov-class battlecruiser',
        'Admiral Kuznetsov': 'Admiral Kuznetsov aircraft carrier'
    },
    'missiles': {
        'Tomahawk': 'BGM-109 Tomahawk cruise missile',
        'Patriot': 'MIM-104 Patriot surface-to-air missile',
        'THAAD': 'Terminal High Altitude Area Defense system',
        'Iron Dome': 'Iron Dome missile defense system',
        'S-400': 'S-400 Triumf air defense system',
        'S-300': 'S-300 air defense system',
        'HIMARS': 'M142 High Mobility Artillery Rocket System',
        'Javelin': 'FGM-148 Javelin anti-tank missile',
        'Stinger': 'FIM-92 Stinger surface-to-air missile',
        'Harpoon': 'AGM-84 Harpoon anti-ship missile',
        'JDAM': 'Joint Direct Attack Munition',
        'Hellfire': 'AGM-114 Hellfire missile',
        'Kinzhal': 'Kh-47M2 Kinzhal hypersonic missile',
        'DF-21': 'DF-21D anti-ship ballistic missile',
        'Shahab': 'Shahab-3 medium-range ballistic missile'
    },
    'ground': {
        'Abrams': 'M1A2 Abrams main battle tank',
        'Bradley': 'M2 Bradley infantry fighting vehicle',
        'Stryker': 'M1126 Stryker armored vehicle',
        'MRAP': 'Mine-Resistant Ambush Protected vehicle',
        'Humvee': 'M1151 HMMWV tactical vehicle',
        'Merkava': 'Merkava Mk 4 main battle tank',
        'T-90': 'T-90M main battle tank',
        'T-72': 'T-72B3 main battle tank',
        'Leopard': 'Leopard 2A7 main battle tank',
        'Challenger': 'Challenger 2 main battle tank',
        'Type 99': 'Type 99A main battle tank',
        'Paladin': 'M109A7 Paladin self-propelled howitzer'
    },
    'personnel': {
        'US Army': 'US Army soldiers in operational camouflage pattern',
        'US Marines': 'US Marine Corps personnel in MARPAT camouflage',
        'Navy SEALs': 'US Navy SEAL special operations forces',
        'Delta Force': 'US Army Delta Force operators',
        'Green Berets': 'US Army Special Forces Green Berets',
        'IDF': 'Israeli Defense Forces soldiers',
        'IRGC': 'Iranian Revolutionary Guard Corps forces',
        'Spetsnaz': 'Russian Spetsnaz special forces',
        'SAS': 'British Special Air Service operators',
        'Commandos': 'Special operations commandos'
    }
}

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
