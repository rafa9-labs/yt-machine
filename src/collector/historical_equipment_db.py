"""
Historical Military Equipment Database
Era-specific equipment for accurate pixel art generation (1980s-2020s)
"""

HISTORICAL_EQUIPMENT_DB = {
    '1980s': {
        'aircraft': {
            'F-14': 'F-14 Tomcat',
            'F-15': 'F-15 Eagle',
            'F-16': 'F-16 Fighting Falcon',
            'F-4': 'F-4 Phantom II',
            'A-6': 'A-6 Intruder',
            'A-7': 'A-7 Corsair II',
            'F-111': 'F-111 Aardvark',
            'MiG-23': 'MiG-23 Flogger',
            'MiG-25': 'MiG-25 Foxbat',
            'Su-24': 'Sukhoi Su-24 Fencer',
            'AH-1': 'AH-1 Cobra attack helicopter',
            'UH-1': 'UH-1 Huey helicopter'
        },
        'naval': {
            'Iowa': 'Iowa-class battleship',
            'Nimitz': 'Nimitz-class aircraft carrier',
            'Spruance': 'Spruance-class destroyer',
            'Ticonderoga': 'Ticonderoga-class cruiser',
            'Oliver Hazard Perry': 'Oliver Hazard Perry-class frigate',
            'Los Angeles': 'Los Angeles-class submarine',
            'Kirov': 'Kirov-class battlecruiser'
        },
        'ground': {
            'M1': 'M1 Abrams tank',
            'M60': 'M60 Patton tank',
            'M2': 'M2 Bradley fighting vehicle',
            'M113': 'M113 armored personnel carrier',
            'T-72': 'T-72 main battle tank',
            'BMP-1': 'BMP-1 infantry fighting vehicle'
        },
        'missiles': {
            'Tomahawk': 'BGM-109 Tomahawk cruise missile',
            'Harpoon': 'AGM-84 Harpoon anti-ship missile',
            'Exocet': 'Exocet anti-ship missile',
            'Stinger': 'FIM-92 Stinger surface-to-air missile',
            'Scud': 'Scud ballistic missile'
        }
    },
    '1990s': {
        'aircraft': {
            'F-117': 'F-117 Nighthawk stealth fighter',
            'F-15E': 'F-15E Strike Eagle',
            'F-16C': 'F-16C Fighting Falcon',
            'F/A-18': 'F/A-18 Hornet',
            'B-2': 'B-2 Spirit stealth bomber',
            'B-52': 'B-52 Stratofortress',
            'A-10': 'A-10 Thunderbolt II',
            'AH-64': 'AH-64 Apache attack helicopter',
            'UH-60': 'UH-60 Black Hawk helicopter',
            'CH-47': 'CH-47 Chinook helicopter',
            'MiG-29': 'MiG-29 Fulcrum',
            'Su-27': 'Sukhoi Su-27 Flanker'
        },
        'naval': {
            'Nimitz': 'Nimitz-class aircraft carrier',
            'Arleigh Burke': 'Arleigh Burke-class destroyer',
            'Ticonderoga': 'Ticonderoga-class cruiser',
            'Los Angeles': 'Los Angeles-class submarine',
            'Wasp': 'Wasp-class amphibious assault ship'
        },
        'ground': {
            'M1A1': 'M1A1 Abrams tank',
            'M1A2': 'M1A2 Abrams tank',
            'M2A2': 'M2A2 Bradley fighting vehicle',
            'HMMWV': 'HMMWV Humvee',
            'T-72': 'T-72 main battle tank',
            'T-80': 'T-80 main battle tank'
        },
        'missiles': {
            'Tomahawk': 'BGM-109 Tomahawk cruise missile',
            'Patriot': 'MIM-104 Patriot surface-to-air missile',
            'HARM': 'AGM-88 HARM anti-radiation missile',
            'Hellfire': 'AGM-114 Hellfire missile',
            'JDAM': 'Joint Direct Attack Munition'
        }
    },
    '2000s': {
        'aircraft': {
            'F-22': 'F-22 Raptor',
            'F-15E': 'F-15E Strike Eagle',
            'F-16C': 'F-16C Fighting Falcon',
            'F/A-18E': 'F/A-18E/F Super Hornet',
            'B-2': 'B-2 Spirit stealth bomber',
            'B-1B': 'B-1B Lancer',
            'A-10C': 'A-10C Thunderbolt II',
            'AH-64D': 'AH-64D Apache Longbow',
            'Predator': 'MQ-1 Predator drone',
            'Reaper': 'MQ-9 Reaper drone',
            'Global Hawk': 'RQ-4 Global Hawk drone'
        },
        'naval': {
            'Nimitz': 'Nimitz-class aircraft carrier',
            'Arleigh Burke': 'Arleigh Burke-class destroyer',
            'Virginia': 'Virginia-class submarine',
            'San Antonio': 'San Antonio-class amphibious transport dock',
            'LCS': 'Littoral Combat Ship'
        },
        'ground': {
            'M1A2 SEP': 'M1A2 SEP Abrams tank',
            'M2A3': 'M2A3 Bradley fighting vehicle',
            'Stryker': 'Stryker armored vehicle',
            'MRAP': 'Mine-Resistant Ambush Protected vehicle',
            'M-ATV': 'M-ATV all-terrain vehicle'
        },
        'missiles': {
            'Tomahawk Block IV': 'Tomahawk Block IV cruise missile',
            'Patriot PAC-3': 'Patriot PAC-3 missile',
            'THAAD': 'Terminal High Altitude Area Defense',
            'Hellfire II': 'AGM-114 Hellfire II missile',
            'JASSM': 'AGM-158 JASSM cruise missile'
        }
    },
    '2010s': {
        'aircraft': {
            'F-35A': 'F-35A Lightning II',
            'F-35B': 'F-35B Lightning II STOVL',
            'F-35C': 'F-35C Lightning II carrier variant',
            'F-22': 'F-22 Raptor',
            'F-15EX': 'F-15EX Eagle II',
            'F/A-18E': 'F/A-18E/F Super Hornet',
            'B-2': 'B-2 Spirit',
            'AH-64E': 'AH-64E Apache Guardian',
            'MQ-9': 'MQ-9 Reaper drone',
            'RQ-4': 'RQ-4 Global Hawk',
            'MQ-1C': 'MQ-1C Gray Eagle drone'
        },
        'naval': {
            'Gerald R. Ford': 'Gerald R. Ford-class aircraft carrier',
            'Nimitz': 'Nimitz-class aircraft carrier',
            'Arleigh Burke Flight IIA': 'Arleigh Burke Flight IIA destroyer',
            'Zumwalt': 'Zumwalt-class destroyer',
            'Virginia Block III': 'Virginia Block III submarine',
            'America': 'America-class amphibious assault ship'
        },
        'ground': {
            'M1A2 SEPv2': 'M1A2 SEPv2 Abrams tank',
            'M1A2 SEPv3': 'M1A2 SEPv3 Abrams tank',
            'M2A4': 'M2A4 Bradley fighting vehicle',
            'Stryker DVH': 'Stryker Double V Hull',
            'JLTV': 'Joint Light Tactical Vehicle'
        },
        'missiles': {
            'Tomahawk Block IV': 'Tomahawk Block IV cruise missile',
            'SM-6': 'Standard Missile-6',
            'THAAD': 'THAAD missile defense',
            'Iron Dome': 'Iron Dome missile defense',
            'LRASM': 'AGM-158C LRASM anti-ship missile'
        }
    },
    '2020s': {
        'aircraft': {
            'F-35': 'F-35 Lightning II',
            'F-22': 'F-22 Raptor',
            'F-15EX': 'F-15EX Eagle II',
            'F/A-18E': 'F/A-18E/F Super Hornet',
            'B-21': 'B-21 Raider stealth bomber',
            'MQ-9': 'MQ-9 Reaper drone',
            'MQ-25': 'MQ-25 Stingray tanker drone',
            'AH-64E': 'AH-64E Apache Guardian',
            'V-22': 'V-22 Osprey tiltrotor'
        },
        'naval': {
            'Gerald R. Ford': 'Gerald R. Ford-class aircraft carrier',
            'Arleigh Burke Flight III': 'Arleigh Burke Flight III destroyer',
            'Constellation': 'Constellation-class frigate',
            'Virginia Block V': 'Virginia Block V submarine',
            'Columbia': 'Columbia-class ballistic missile submarine'
        },
        'ground': {
            'M1A2 SEPv3': 'M1A2 SEPv3 Abrams tank',
            'M1A2 SEPv4': 'M1A2 SEPv4 Abrams tank (planned)',
            'AMPV': 'Armored Multi-Purpose Vehicle',
            'JLTV': 'Joint Light Tactical Vehicle',
            'Stryker A1': 'Stryker A1'
        },
        'missiles': {
            'Tomahawk Block V': 'Tomahawk Block V cruise missile',
            'SM-6': 'Standard Missile-6',
            'Patriot PAC-3 MSE': 'Patriot PAC-3 MSE',
            'THAAD': 'THAAD missile defense',
            'LRASM': 'Long Range Anti-Ship Missile',
            'Hypersonic': 'AGM-183 ARRW hypersonic missile'
        }
    }
}


def get_equipment_for_era(era: str, category: str = None) -> dict:
    """
    Get equipment for a specific era
    
    Args:
        era: Era string (1980s, 1990s, 2000s, 2010s, 2020s)
        category: Optional category filter (aircraft, naval, ground, missiles)
        
    Returns:
        Dictionary of equipment for that era
    """
    if era not in HISTORICAL_EQUIPMENT_DB:
        # Default to 2020s if era not found
        era = '2020s'
    
    era_equipment = HISTORICAL_EQUIPMENT_DB[era]
    
    if category and category in era_equipment:
        return era_equipment[category]
    
    return era_equipment


def normalize_historical_equipment(equipment_name: str, era: str) -> str:
    """
    Normalize equipment name to full nomenclature for a specific era
    
    Args:
        equipment_name: Short equipment name (e.g., 'F-15')
        era: Era string (1980s, 1990s, etc.)
        
    Returns:
        Full equipment name with nomenclature
    """
    if era not in HISTORICAL_EQUIPMENT_DB:
        era = '2020s'
    
    era_equipment = HISTORICAL_EQUIPMENT_DB[era]
    
    # Search across all categories
    for category, equipment_dict in era_equipment.items():
        for short_name, full_name in equipment_dict.items():
            if short_name.lower() in equipment_name.lower():
                return full_name
    
    # Return original if not found
    return equipment_name


def get_era_from_year(year: int) -> str:
    """
    Convert year to era string
    
    Args:
        year: Year (e.g., 1991)
        
    Returns:
        Era string (e.g., '1990s')
    """
    if year < 1990:
        return '1980s'
    elif year < 2000:
        return '1990s'
    elif year < 2010:
        return '2000s'
    elif year < 2020:
        return '2010s'
    else:
        return '2020s'
