"""
Auto-Generate Diverse Training Set for LoRA Training

Generates 60+ diverse pixel art images across all visual categories
(military, economic, diplomatic, human_impact, geographic, historical)
to ensure the trained LoRA generalises across all geopolitical content types.

Usage:
    python tools/auto_generate_training_set.py
    python tools/auto_generate_training_set.py --count 80 --output training_data/
    python tools/auto_generate_training_set.py --dry-run

Output:
    training_data/
        image_001.png  +  image_001.txt  (caption pair)
        image_002.png  +  image_002.txt
        ...
    training_data/manifest.json  (metadata for each image)
"""

import os
import sys
import json
import time
import argparse
import requests
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
TRAINING_DATA_DIR = ROOT / "training_data"
CONFIG_PATH = ROOT / "config" / "image_style.json"

with open(CONFIG_PATH, "r", encoding="utf-8") as _f:
    _STYLE_CFG = json.load(_f)

STYLE_SUFFIX = _STYLE_CFG["style_suffix"]
COLOR_PALETTE = _STYLE_CFG["color_palette_prompt"]
NEGATIVE_PROMPT = _STYLE_CFG["negative_prompt"]
GEN_PARAMS = _STYLE_CFG["generation_params"]
LORA_DEFAULTS = _STYLE_CFG["lora_defaults"]

FAL_KEY = os.getenv("FAL_KEY")

# ── Diverse prompt bank — 8 categories, ~10 prompts each ──────────────────
# Caption (used as training label) + scene description (used as generation prompt)
# Each entry: {"caption": str, "prompt": str, "category": str}

TRAINING_PROMPTS: List[Dict[str, str]] = [

    # ── MILITARY / AIR ────────────────────────────────────────────────────
    {
        "category": "military_air",
        "caption": "F-35 stealth fighter jet flying over Persian Gulf at dusk, isometric pixel art",
        "prompt": "(F-35 stealth fighter:1.4), Persian Gulf, flying at dusk, contrail visible, dramatic orange sky, isometric aerial view",
    },
    {
        "category": "military_air",
        "caption": "Israeli F-16I Sufa jet conducting airstrike, IDF insignia visible, pixel art",
        "prompt": "(F-16I Sufa:1.4), (IDF insignia:1.3), airstrike, smoke plume rising, desert terrain below, dramatic angle",
    },
    {
        "category": "military_air",
        "caption": "Russian Su-35 intercepting drone over Black Sea, pixel art",
        "prompt": "(Su-35 Flanker-E:1.4), drone interception, Black Sea, dark overcast sky, tactical chase angle, (red star insignia:1.2)",
    },
    {
        "category": "military_air",
        "caption": "US B-52 bomber flying over Pacific Ocean, strategic bombing run, pixel art",
        "prompt": "(B-52 Stratofortress:1.4), Pacific Ocean, long-range bombing run, high altitude, formation flight, dusk lighting",
    },
    {
        "category": "military_air",
        "caption": "Chinese J-20 stealth fighter patrolling Taiwan Strait, pixel art",
        "prompt": "(J-20 stealth fighter:1.4), (PLA insignia:1.3), Taiwan Strait, low cloud cover, tense patrol angle, grey sky",
    },
    {
        "category": "military_air",
        "caption": "Turkish Bayraktar TB2 drone flying over conflict zone, pixel art",
        "prompt": "(Bayraktar TB2 drone:1.4), conflict zone, surveillance flight, hilly terrain, golden hour lighting, (Turkish roundel:1.2)",
    },
    {
        "category": "military_air",
        "caption": "US MQ-9 Reaper drone conducting surveillance over Middle East desert, pixel art",
        "prompt": "(MQ-9 Reaper drone:1.4), surveillance flight over desert, Middle East, (USAF roundel:1.2), high altitude, vast landscape below",
    },
    {
        "category": "military_air",
        "caption": "Houthi ballistic missile launching from Yemen toward Red Sea, pixel art",
        "prompt": "(ballistic missile launch:1.4), Yemen coastline, rocket exhaust trail rising, Red Sea visible, night launch, dramatic fire glow",
    },
    {
        "category": "military_air",
        "caption": "Israeli Iron Dome intercepting missiles over Tel Aviv at night, pixel art",
        "prompt": "(Iron Dome missile battery:1.4), (interception trails:1.3), Tel Aviv night skyline, explosions mid-air, searchlights",
    },
    {
        "category": "military_air",
        "caption": "Pakistan JF-17 Thunder fighter jet on combat patrol, pixel art",
        "prompt": "(JF-17 Thunder:1.4), (PAF roundel:1.2), combat patrol, mountainous terrain, dawn lighting, dramatic angle",
    },

    # ── MILITARY / NAVAL ──────────────────────────────────────────────────
    {
        "category": "military_naval",
        "caption": "US aircraft carrier USS Nimitz sailing through Strait of Hormuz, pixel art",
        "prompt": "(USS Nimitz carrier:1.4), (Stars and Stripes:1.2), Strait of Hormuz, naval escort formation, blue waters, afternoon light",
    },
    {
        "category": "military_naval",
        "caption": "IRGC speedboats confronting tanker in Persian Gulf, pixel art",
        "prompt": "(IRGC patrol speedboats:1.4), (Iranian Revolutionary Guard:1.3), tanker vessel, Persian Gulf, tense confrontation, choppy waters",
    },
    {
        "category": "military_naval",
        "caption": "Russian guided missile cruiser Moskva sailing Black Sea, pixel art",
        "prompt": "(guided missile cruiser:1.4), (red star insignia:1.2), Black Sea, grey sky, wartime patrol, smoke from stacks",
    },
    {
        "category": "military_naval",
        "caption": "Chinese Type 055 destroyer leading naval formation in South China Sea, pixel art",
        "prompt": "(Type 055 destroyer:1.4), (PLA Navy insignia:1.3), South China Sea, carrier battle group, blue water, tactical formation",
    },
    {
        "category": "military_naval",
        "caption": "Naval blockade of Red Sea shipping lane, Houthi-controlled waters, pixel art",
        "prompt": "(naval blockade:1.4), Red Sea shipping lane, commercial tankers halted, military vessels, tense standoff, Bab-el-Mandeb Strait",
    },
    {
        "category": "military_naval",
        "caption": "US Arleigh Burke destroyer firing Tomahawk missile at night, pixel art",
        "prompt": "(Arleigh Burke destroyer:1.4), (Tomahawk missile launch:1.3), night firing, sea spray, missile exhaust trail, (US Navy ensign:1.2)",
    },
    {
        "category": "military_naval",
        "caption": "Iranian submarine surfacing in Persian Gulf, pixel art",
        "prompt": "(Iranian submarine:1.4), (Persian script insignia:1.2), Persian Gulf, surfacing at dawn, foam and wake, strategic positioning",
    },

    # ── MILITARY / GROUND ─────────────────────────────────────────────────
    {
        "category": "military_ground",
        "caption": "Israeli Merkava tank advancing through Gaza urban terrain, pixel art",
        "prompt": "(Merkava IV tank:1.4), (IDF insignia:1.3), urban combat zone, rubble and dust, dawn lighting, armoured column",
    },
    {
        "category": "military_ground",
        "caption": "Russian T-72 tank column advancing through Ukrainian countryside, pixel art",
        "prompt": "(T-72 tank column:1.4), (red star insignia:1.2), Ukrainian farmland, snow and mud, overcast sky, armoured advance",
    },
    {
        "category": "military_ground",
        "caption": "US Army Abrams tank in desert formation, Middle East deployment, pixel art",
        "prompt": "(M1A2 Abrams tank:1.4), (Stars and Stripes:1.2), desert terrain, sand dunes, tactical formation, harsh midday sun",
    },
    {
        "category": "military_ground",
        "caption": "Iranian IRGC troops patrolling mountain border with Iraq, pixel art",
        "prompt": "(IRGC soldiers:1.4), (Iranian flag insignia:1.2), mountain border patrol, rocky terrain, Iran-Iraq border, dusk lighting",
    },
    {
        "category": "military_ground",
        "caption": "Saudi Arabian military convoy crossing Yemen border, pixel art",
        "prompt": "(Saudi military convoy:1.4), (Saudi flag insignia:1.2), Yemen border crossing, armoured vehicles, desert dust, afternoon heat",
    },
    {
        "category": "military_ground",
        "caption": "Ukrainian troops defending Kyiv with S-300 missile system, pixel art",
        "prompt": "(S-300 missile battery:1.4), (Ukrainian trident insignia:1.3), Kyiv defence, snow covered ground, winter sky, (blue and yellow:1.2)",
    },

    # ── ECONOMIC ──────────────────────────────────────────────────────────
    {
        "category": "economic",
        "caption": "Oil price surge display showing dollar per barrel, trading floor, pixel art",
        "prompt": "(oil price display board:1.4), $120 per barrel, trading floor, price spike upward, urgent red indicators, financial data screens",
    },
    {
        "category": "economic",
        "caption": "Long queue of cars at gas station during fuel shortage, pixel art",
        "prompt": "(long fuel queue:1.4), gas station, cars lined up stretching into distance, shortage signs, stressed civilians, ground-level view",
    },
    {
        "category": "economic",
        "caption": "Wall Street trading floor during market crash, red screens, pixel art",
        "prompt": "(trading floor:1.4), red market crash indicators, traders in panic, multiple screens showing falling prices, harsh fluorescent lighting",
    },
    {
        "category": "economic",
        "caption": "Empty supermarket shelves during sanctions, civilians searching, pixel art",
        "prompt": "(empty shelves:1.4), supermarket interior, sanctions impact, civilians looking for goods, sparse products, economic despair",
    },
    {
        "category": "economic",
        "caption": "Oil tanker passing through Strait of Hormuz, strategic chokepoint, pixel art",
        "prompt": "(oil tanker:1.4), Strait of Hormuz, strategic shipping lane, blue waters, sunset horizon, commercial vessel passage",
    },
    {
        "category": "economic",
        "caption": "Stock market chart showing inflation spike and interest rate rise, pixel art",
        "prompt": "(financial chart:1.4), inflation spike graph, interest rate indicators, dark screen glow, economic analysis display, abstract data",
    },
    {
        "category": "economic",
        "caption": "Sanctions impact on Iranian oil exports, tanker blocked at port, pixel art",
        "prompt": "(blocked oil tanker:1.4), Iranian port, sanctions enforcement, naval vessel nearby, sunset, economic isolation imagery",
    },
    {
        "category": "economic",
        "caption": "OPEC meeting room with oil production charts and ministers, pixel art",
        "prompt": "(OPEC meeting:1.4), ministerial conference room, oil production charts on wall, formal setting, diplomatic atmosphere",
    },

    # ── DIPLOMATIC ────────────────────────────────────────────────────────
    {
        "category": "diplomatic",
        "caption": "UN Security Council emergency meeting, world leaders at round table, pixel art",
        "prompt": "(UN Security Council chamber:1.4), round table, world leaders, (UN flag:1.2), emergency session, formal lighting, diplomatic tension",
    },
    {
        "category": "diplomatic",
        "caption": "US Secretary of State meeting Iranian foreign minister in Geneva, pixel art",
        "prompt": "(diplomatic meeting:1.4), Geneva conference room, two delegations facing each other, (US and Iranian flags:1.2), formal suits, tense atmosphere",
    },
    {
        "category": "diplomatic",
        "caption": "NATO summit with leaders signing defence agreement, pixel art",
        "prompt": "(NATO summit:1.4), (NATO emblem:1.3), world leaders signing agreement, formal hall, multiple national flags, press cameras",
    },
    {
        "category": "diplomatic",
        "caption": "Abraham Accords signing ceremony, Middle East peace deal, pixel art",
        "prompt": "(peace agreement signing:1.4), (Israeli and Arab flags:1.2), formal ceremony, White House backdrop, multiple leaders, historic moment",
    },
    {
        "category": "diplomatic",
        "caption": "G7 summit leaders meeting around conference table, global issues, pixel art",
        "prompt": "(G7 summit:1.4), conference hall, world leaders seated, national flags, formal attire, global policy discussion",
    },
    {
        "category": "diplomatic",
        "caption": "Saudi Crown Prince meeting Chinese President, strategic partnership, pixel art",
        "prompt": "(bilateral summit:1.4), (Saudi and Chinese flags:1.2), formal handshake, conference hall, strategic partnership signing",
    },
    {
        "category": "diplomatic",
        "caption": "Ceasefire negotiation between Israeli and Hamas representatives, pixel art",
        "prompt": "(ceasefire negotiation:1.4), neutral meeting room, two delegations, mediator at head of table, tense but formal atmosphere, documents",
    },

    # ── MAPS / GEOGRAPHIC ─────────────────────────────────────────────────
    {
        "category": "geographic",
        "caption": "Strategic map of Middle East with conflict zones highlighted, pixel art",
        "prompt": "(strategic map overlay:1.4), Middle East region, conflict zones marked, naval routes, color-coded territories, birds-eye map view",
    },
    {
        "category": "geographic",
        "caption": "Satellite view of Strait of Hormuz with tanker traffic, pixel art",
        "prompt": "(Strait of Hormuz aerial:1.4), oil tankers in shipping lane, Iran and Oman coastlines visible, strategic chokepoint, birds-eye view",
    },
    {
        "category": "geographic",
        "caption": "Taiwan Strait showing military positions across water, pixel art",
        "prompt": "(Taiwan Strait map:1.4), Taiwan island, mainland China coast, military positions marked, contested waters, strategic overview",
    },
    {
        "category": "geographic",
        "caption": "Red Sea and Bab-el-Mandeb Strait with Houthi-controlled areas marked, pixel art",
        "prompt": "(Red Sea strategic map:1.4), Bab-el-Mandeb Strait, Yemen coastline, Houthi territory marked, shipping disruption zones, naval routes",
    },
    {
        "category": "geographic",
        "caption": "Eastern Europe map showing Ukraine frontlines and NATO borders, pixel art",
        "prompt": "(Eastern Europe map:1.4), Ukraine frontlines marked, NATO territory, Russian advance arrows, conflict zone overlay, strategic depth view",
    },
    {
        "category": "geographic",
        "caption": "South China Sea disputed islands with competing territorial claims, pixel art",
        "prompt": "(South China Sea map:1.4), disputed islands, overlapping territorial claims, naval patrol zones, strategic island chains, military installations",
    },

    # ── HUMAN IMPACT ──────────────────────────────────────────────────────
    {
        "category": "human_impact",
        "caption": "Palestinian civilians evacuating Gaza on foot, column of refugees, pixel art",
        "prompt": "(civilian evacuation column:1.4), Gaza street, families carrying belongings, long line of people, dust and smoke, ground-level perspective",
    },
    {
        "category": "human_impact",
        "caption": "Anti-government protest in Tehran with crowds holding signs, pixel art",
        "prompt": "(mass protest crowd:1.4), Tehran city square, protesters, signs and banners, night demonstration, police presence, emotional atmosphere",
    },
    {
        "category": "human_impact",
        "caption": "Ukrainian family sheltering in Kyiv metro station during air raid, pixel art",
        "prompt": "(metro station shelter:1.4), Kyiv underground, families huddled, air raid lighting, (Ukrainian flag:1.2), children and elderly, fear and resilience",
    },
    {
        "category": "human_impact",
        "caption": "Humanitarian aid convoy delivering food to Gaza civilians, pixel art",
        "prompt": "(humanitarian convoy:1.4), aid trucks, UN markings, Gaza checkpoint, civilians waiting, (UN flag:1.2), relief distribution",
    },
    {
        "category": "human_impact",
        "caption": "Families fleeing conflict zone across border on foot at night, pixel art",
        "prompt": "(refugee column:1.4), border crossing at night, families with children, carrying luggage, searchlights, (UNHCR flag:1.2), dramatic night scene",
    },
    {
        "category": "human_impact",
        "caption": "Citizens queuing for water after infrastructure attack, pixel art",
        "prompt": "(civilian water queue:1.4), destroyed infrastructure, urban street, people with containers, aftermath of conflict, morning light, devastation",
    },
    {
        "category": "human_impact",
        "caption": "Anti-war protest in European capital city, large crowd, pixel art",
        "prompt": "(European anti-war protest:1.4), city square, thousands of protesters, peace signs, national flags, daytime, democratic expression",
    },

    # ── HISTORICAL ────────────────────────────────────────────────────────
    {
        "category": "historical",
        "caption": "1973 Yom Kippur War tank battle in Sinai Desert, historical pixel art",
        "prompt": "(1973 tank battle:1.4), Sinai Desert, Cold War era tanks, sepia-amber tone, vintage military, historical conflict, dust clouds, archival feel",
    },
    {
        "category": "historical",
        "caption": "1979 Iranian Revolution street scenes, historical pixel art",
        "prompt": "(1979 Iran Revolution:1.4), Tehran street scene, revolutionary crowds, (Iranian flag:1.2), historical sepia tone, vintage photography feel",
    },
    {
        "category": "historical",
        "caption": "1991 Gulf War coalition forces advancing into Kuwait, historical pixel art",
        "prompt": "(Gulf War 1991:1.4), coalition military advance, Kuwait desert, M1 Abrams tanks, historical warm palette, period-accurate equipment, archival feel",
    },
    {
        "category": "historical",
        "caption": "Cold War era US-Soviet nuclear standoff, missiles ready, historical pixel art",
        "prompt": "(Cold War nuclear standoff:1.4), US and Soviet missiles, tense strategic positioning, 1960s colour palette, historical drama, competing superpowers",
    },
    {
        "category": "historical",
        "caption": "2003 Iraq War US forces entering Baghdad, historical pixel art",
        "prompt": "(Iraq War 2003:1.4), US forces, Baghdad urban advance, (Stars and Stripes:1.2), early 2000s military equipment, historical accuracy, dust and smoke",
    },
    {
        "category": "historical",
        "caption": "2011 Arab Spring protests in Cairo's Tahrir Square, historical pixel art",
        "prompt": "(Arab Spring 2011:1.4), Tahrir Square Cairo, massive protest crowd, Egyptian flags, night scene, historical significance, emotional crowd",
    },

    # ── ENERGY / OIL INFRASTRUCTURE ───────────────────────────────────────
    {
        "category": "energy",
        "caption": "Oil refinery with flames and industrial complex, Persian Gulf, pixel art",
        "prompt": "(oil refinery complex:1.4), Persian Gulf coast, industrial towers, fire stacks burning, night industrial glow, strategic infrastructure",
    },
    {
        "category": "energy",
        "caption": "Natural gas pipeline network across European map, strategic energy, pixel art",
        "prompt": "(gas pipeline map:1.4), European energy network, pipeline routes marked, supply flow lines, strategic energy infrastructure, birds-eye overview",
    },
    {
        "category": "energy",
        "caption": "Saudi Aramco oil facility ablaze after drone strike, pixel art",
        "prompt": "(oil facility on fire:1.4), Saudi Arabia, drone strike aftermath, smoke plumes rising, night glow from fires, strategic energy attack",
    },
    {
        "category": "energy",
        "caption": "Nuclear power plant with cooling towers, Iran's Bushehr facility, pixel art",
        "prompt": "(nuclear power plant:1.4), Bushehr Iran, cooling towers, coastal location, Persian Gulf visible, strategic significance, dawn lighting",
    },
]


def _call_fal(prompt: str, seed: int) -> str:
    """Call fal.ai to generate one image. Returns image URL."""
    import fal_client
    os.environ["FAL_KEY"] = FAL_KEY

    lora_path = LORA_DEFAULTS.get("path", "prithivMLmods/Retro-Pixel-Flux-LoRA")
    trigger = LORA_DEFAULTS.get("trigger_word", "Retro Pixel")
    scale = LORA_DEFAULTS.get("scale", 0.85)

    full_prompt = f"{trigger}, {prompt}, {STYLE_SUFFIX}, {COLOR_PALETTE}"

    result = fal_client.run(
        "fal-ai/flux-lora",
        arguments={
            "prompt": full_prompt,
            "negative_prompt": NEGATIVE_PROMPT,
            "image_size": GEN_PARAMS.get("image_size", "portrait_4_3"),
            "num_images": 1,
            "num_inference_steps": GEN_PARAMS.get("num_inference_steps", 28),
            "guidance_scale": GEN_PARAMS.get("guidance_scale", 3.5),
            "enable_safety_checker": False,
            "output_format": "png",
            "seed": seed,
            "loras": [{"path": lora_path, "scale": scale}],
        },
    )
    return result["images"][0]["url"]


def _download(url: str, dest: Path) -> bool:
    """Download image from URL to local path."""
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        dest.write_bytes(r.content)
        return True
    except Exception as e:
        print(f"    Download failed: {e}")
        return False


def _score(prompt: str) -> int:
    """Quick specificity score — higher = more visually concrete."""
    pl = prompt.lower()
    score = 0
    if any(w in pl for w in ["strait", "gulf", "sea", "desert", "city", "mountain", "coast"]):
        score += 20
    if any(w in pl for w in ["iran", "israel", "ukraine", "russia", "china", "usa", "nato", "houthi"]):
        score += 15
    if any(w in pl for w in ["missile", "tank", "destroyer", "fighter", "drone", "convoy", "carrier"]):
        score += 20
    if any(w in pl for w in ["launch", "strike", "patrol", "advance", "blockade", "intercept", "deploy"]):
        score += 15
    if any(w in pl for w in ["dusk", "dawn", "night", "smoke", "explosion", "fire", "overcast"]):
        score += 10
    if any(w in pl for w in ["pixel art", "isometric", "retro", "16-bit"]):
        score += 10
    return min(score, 100)


def generate_training_set(
    target_count: int = 70,
    output_dir: Path = TRAINING_DATA_DIR,
    dry_run: bool = False,
    resume: bool = True,
) -> None:
    """
    Generate a diverse training set for LoRA fine-tuning.

    Args:
        target_count: Target number of images to generate (default 70)
        output_dir: Directory to save images + captions
        dry_run: Print what would be generated without calling fal.ai
        resume: Skip already-generated images (use for resuming interrupted runs)
    """
    if not FAL_KEY and not dry_run:
        print("ERROR: FAL_KEY not set in .env file")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"

    # Load existing manifest if resuming
    manifest: List[Dict] = []
    if resume and manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    already_done = {entry["index"] for entry in manifest}

    # Select prompts — cycle through the bank to reach target_count
    import random
    random.seed(42)
    all_prompts = TRAINING_PROMPTS.copy()
    while len(all_prompts) < target_count:
        # Duplicate with slight variation index to hit target
        all_prompts.extend(random.sample(TRAINING_PROMPTS, min(10, target_count - len(all_prompts))))
    prompts_to_use = all_prompts[:target_count]

    print("=" * 65)
    print(f"  TRAINING SET GENERATOR")
    print("=" * 65)
    print(f"  Target images  : {target_count}")
    print(f"  Output dir     : {output_dir}")
    print(f"  Already done   : {len(already_done)}")
    print(f"  Remaining      : {target_count - len(already_done)}")
    print()

    # Category breakdown
    cats: Dict[str, int] = {}
    for p in prompts_to_use:
        cats[p["category"]] = cats.get(p["category"], 0) + 1
    print("  Category distribution:")
    for cat, count in sorted(cats.items()):
        print(f"    {cat:<25} {count} images")
    print()

    if dry_run:
        print("  [DRY RUN] No images will be generated. Remove --dry-run to proceed.")
        for i, p in enumerate(prompts_to_use, 1):
            print(f"  [{i:03d}] {p['category']:<22} {p['caption'][:60]}")
        return

    # Generate
    generated = 0
    failed = 0
    start_time = time.time()

    for idx, entry in enumerate(prompts_to_use, 1):
        if idx in already_done:
            print(f"  [{idx:03d}/{target_count}] SKIP (already exists)")
            continue

        img_path = output_dir / f"image_{idx:03d}.png"
        txt_path = output_dir / f"image_{idx:03d}.txt"

        print(f"  [{idx:03d}/{target_count}] {entry['category']:<22} {entry['caption'][:50]}...")

        try:
            seed = 1000 + idx  # Deterministic seed per image
            url = _call_fal(entry["prompt"], seed=seed)
            ok = _download(url, img_path)

            if not ok:
                failed += 1
                continue

            # Write caption file (used as training label)
            txt_path.write_text(entry["caption"], encoding="utf-8")

            # Update manifest
            manifest.append({
                "index": idx,
                "filename": img_path.name,
                "caption": entry["caption"],
                "category": entry["category"],
                "prompt_used": entry["prompt"],
                "score": _score(entry["prompt"]),
                "generated_at": datetime.now().isoformat(),
            })
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            generated += 1
            elapsed = time.time() - start_time
            rate = generated / elapsed * 60 if elapsed > 0 else 0
            remaining = target_count - idx
            eta_min = remaining / (rate if rate > 0 else 1)
            print(f"        ✓ Saved  |  {generated} done  |  ETA ~{eta_min:.0f} min")

            # Polite delay to stay within rate limits
            time.sleep(0.5)

        except KeyboardInterrupt:
            print("\n  Interrupted by user. Progress saved to manifest.json")
            break
        except Exception as e:
            print(f"        ✗ FAILED: {e}")
            failed += 1
            time.sleep(2)  # Back off on error

    # Final summary
    total_time = time.time() - start_time
    print()
    print("=" * 65)
    print(f"  GENERATION COMPLETE")
    print("=" * 65)
    print(f"  Generated  : {generated}")
    print(f"  Failed     : {failed}")
    print(f"  Skipped    : {len(already_done)}")
    print(f"  Total time : {total_time / 60:.1f} minutes")
    print(f"  Output dir : {output_dir}")
    print()
    print(f"  Next step:")
    print(f"    python tools/train_lora_local.py {output_dir} --upload-to-hub")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Auto-generate diverse LoRA training set")
    parser.add_argument("--count", type=int, default=70,
                        help="Number of training images to generate (default: 70)")
    parser.add_argument("--output", type=str, default=str(TRAINING_DATA_DIR),
                        help="Output directory for images and captions")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview what would be generated without calling fal.ai")
    parser.add_argument("--no-resume", action="store_true",
                        help="Re-generate all images even if they already exist")
    args = parser.parse_args()

    generate_training_set(
        target_count=args.count,
        output_dir=Path(args.output),
        dry_run=args.dry_run,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
