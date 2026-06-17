import requests
import time
import pandas as pd
import os
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta


def fetch_smogon_data(url: str) -> Optional[str]:
    """
    Fetch data from Smogon stats URL with error handling.
    Includes retry logic and rate limiting.
    """
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                print(f"Failed to fetch data after {max_retries} attempts")
                return None
    return None


def parse_usage_stats(data: str, generation: int = 6) -> List[Dict[str, Any]]:
    """
    Parse usage statistics from the main stats file.
    Returns list of dicts with rank, name, usage_percent, raw_count, percent, generation.
    """
    lines = data.split('\n')
    stats = []
    parsing = False

    for line in lines:
        line = line.strip()
        if line.startswith('+ ---- +'):
            parsing = True
            continue
        if parsing and line and line.startswith('|'):
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) >= 7:
                try:
                    rank = int(parts[0])
                    name = parts[1]
                    usage_percent = float(parts[2].rstrip('%'))
                    raw_count = int(parts[3])
                    percent = float(parts[4].rstrip('%'))
                    real_count = int(parts[5])
                    real_percent = float(parts[6].rstrip('%'))
                    stats.append({
                        'rank': rank,
                        'name': name,
                        'usage_percent': usage_percent,
                        'raw_count': raw_count,
                        'percent': percent,
                        'real_count': real_count,
                        'real_percent': real_percent,
                        'generation': generation
                    })
                except (ValueError, IndexError) as e:
                    print(f"Error parsing line: {line} - {e}")
                    continue
        elif parsing and not line.startswith('|') and line:
            # End of table
            break

    return stats


def parse_moveset_data(data: str) -> Dict[str, Dict[str, Any]]:
    """
    Parse moveset data for all Pokemon.
    Returns dict with Pokemon names as keys and their data as values.
    """
    lines = data.split('\n')
    pokemon_data = {}
    current_pokemon = None
    current_section = None

    for line in lines:
        line = line.strip()
        if line.startswith('+----------------------------------------+'):
            # New Pokemon section
            current_pokemon = None
            current_section = None
            continue
        if line and not line.startswith('|') and '|' in line and not current_pokemon:
            # Pokemon name line
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) >= 1:
                name = parts[0]
                pokemon_data[name] = {
                    'name': name,
                    'raw_count': None,
                    'avg_weight': None,
                    'viability_ceiling': None,
                    'abilities': [],
                    'items': [],
                    'moves': [],
                    'tera_types': [],
                    'teammates': []
                }
                current_pokemon = name
        elif current_pokemon and line.startswith('|') and '|' in line:
            parts = [p.strip() for p in line.split('|') if p.strip()]
            if len(parts) >= 2:
                if 'Raw count:' in line:
                    try:
                        pokemon_data[current_pokemon]['raw_count'] = int(parts[1])
                    except (ValueError, IndexError):
                        pass
                elif 'Avg. weight:' in line:
                    try:
                        pokemon_data[current_pokemon]['avg_weight'] = float(parts[1])
                    except (ValueError, IndexError):
                        pass
                elif 'Viability Ceiling:' in line:
                    try:
                        pokemon_data[current_pokemon]['viability_ceiling'] = int(parts[1])
                    except (ValueError, IndexError):
                        pass
                elif 'Abilities' in line:
                    current_section = 'abilities'
                elif 'Items' in line:
                    current_section = 'items'
                elif 'Moves' in line:
                    current_section = 'moves'
                elif 'Tera Types' in line:
                    current_section = 'tera_types'
                elif 'Teammates' in line:
                    current_section = 'teammates'
                elif current_section and len(parts) >= 2:
                    try:
                        item_name = parts[0]
                        percent = float(parts[1].rstrip('%'))
                        pokemon_data[current_pokemon][current_section].append({
                            'name': item_name,
                            'percent': percent
                        })
                    except (ValueError, IndexError):
                        continue

    return pokemon_data


def get_latest_smogon_stats() -> Dict[str, Any]:
    """
    Fetch and parse the latest Gen 1-6 OU stats (closest available to AG).
    Returns dict with usage_stats and moveset_data.
    """
    base_url = "https://www.smogon.com/stats"
    # Note: In practice, you'd need to determine the latest month dynamically
    # For now, using 2024-12 as latest (adjust based on current date)
    latest_month = "2024-12"

    # Fetch data for each generation (Gen 1-6 OU)
    all_usage_stats = []
    all_moveset_data = {}
    
    for gen in range(1, 7):
        print(f"Fetching Gen {gen} OU data...")
        
        usage_url = f"{base_url}/{latest_month}/gen{gen}ou-0.txt"
        moveset_url = f"{base_url}/{latest_month}/moveset/gen{gen}ou-0.txt"
        
        usage_text = fetch_smogon_data(usage_url)
        if usage_text:
            gen_usage = parse_usage_stats(usage_text, gen)
            all_usage_stats.extend(gen_usage)
            print(f"Gen {gen}: {len(gen_usage)} Pokemon")
        else:
            print(f"Failed to fetch Gen {gen} usage data")
        
        time.sleep(1)  # Polite delay
        
        moveset_text = fetch_smogon_data(moveset_url)
        if moveset_text:
            gen_moveset = parse_moveset_data(moveset_text)
            all_moveset_data.update(gen_moveset)
        else:
            print(f"Failed to fetch Gen {gen} moveset data")
        
        time.sleep(1)  # Polite delay

    if not all_usage_stats:
        return {"error": "Failed to fetch any usage stats"}

    return {
        "usage_stats": all_usage_stats,
        "moveset_data": all_moveset_data,
        "total_battles": len(all_usage_stats),  # Approximate
        "fetched_at": datetime.now().isoformat(),
        "note": "Using Gen 1-6 OU data (AG format not available in Smogon stats)"
    }


def extract_total_battles(data: str) -> Optional[int]:
    """Extract total battles from usage stats."""
    for line in data.split('\n'):
        if line.startswith('Total battles:'):
            try:
                return int(line.split(':')[1].strip())
            except (ValueError, IndexError):
                return None
    return None


def create_synergy_dataframe(usage_stats: List[Dict], moveset_data: Dict) -> pd.DataFrame:
    """
    Combine usage stats and moveset data into a single DataFrame for easy lookup.
    """
    rows = []

    for pokemon in usage_stats:
        name = pokemon['name']
        row = {
            'name': name,
            'usage_percent': pokemon['usage_percent'],
            'raw_count': pokemon['raw_count'],
            'percent': pokemon['percent'],
            'real_count': pokemon['real_count'],
            'real_percent': pokemon['real_percent'],
            'rank': pokemon['rank']
        }

        # Add moveset data if available
        if name in moveset_data:
            pokemon_moveset = moveset_data[name]
            row['raw_count_moveset'] = pokemon_moveset['raw_count']
            row['avg_weight'] = pokemon_moveset['avg_weight']
            row['viability_ceiling'] = pokemon_moveset['viability_ceiling']

            # Convert lists to JSON strings for CSV storage
            row['common_moves'] = str([{m['name']: m['percent']} for m in pokemon_moveset['moves'][:4]])  # Top 4 moves
            row['common_items'] = str([{i['name']: i['percent']} for i in pokemon_moveset['items'][:3]])  # Top 3 items
            row['common_abilities'] = str([{a['name']: a['percent']} for a in pokemon_moveset['abilities'][:2]])  # Top 2 abilities
            row['common_teammates'] = str([{t['name']: t['percent']} for t in pokemon_moveset['teammates'][:10]])  # Top 10 teammates
            row['tera_types'] = str([{t['name']: t['percent']} for t in pokemon_moveset['tera_types'][:5]])  # Top 5 tera types
        else:
            row['raw_count_moveset'] = None
            row['avg_weight'] = None
            row['viability_ceiling'] = None
            row['common_moves'] = '[]'
            row['common_items'] = '[]'
            row['common_abilities'] = '[]'
            row['common_teammates'] = '[]'
            row['tera_types'] = '[]'

        rows.append(row)

    return pd.DataFrame(rows)


def save_synergy_data(df: pd.DataFrame, filepath: str):
    """Save the synergy DataFrame to CSV."""
    df.to_csv(filepath, index=False)
    print(f"Saved synergy data to {filepath}")


def load_synergy_data(filepath: str) -> Optional[pd.DataFrame]:
    """Load synergy data from CSV if it exists."""
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return None


def is_cache_stale(filepath: str, max_age_days: int = 7) -> bool:
    """Check if cache file is older than max_age_days."""
    if not os.path.exists(filepath):
        return True

    file_time = datetime.fromtimestamp(os.path.getmtime(filepath))
    return datetime.now() - file_time > timedelta(days=max_age_days)


def refresh_synergy_data(cache_path: str = "data/processed/ag_synergy.csv") -> pd.DataFrame:
    """
    Refresh synergy data from Smogon and save to cache.
    Returns the DataFrame.
    """
    print("Refreshing Smogon synergy data...")

    stats = get_latest_smogon_stats()
    if "error" in stats:
        raise Exception(f"Failed to fetch Smogon data: {stats['error']}")

    df = create_synergy_dataframe(stats['usage_stats'], stats['moveset_data'])
    save_synergy_data(df, cache_path)

    return df


def get_synergy_data(cache_path: str = "data/processed/ag_synergy.csv",
                    auto_refresh: bool = True) -> pd.DataFrame:
    """
    Get synergy data, refreshing from Smogon if cache is stale or missing.
    """
    if auto_refresh and is_cache_stale(cache_path):
        print("Cache is stale, refreshing...")
        return refresh_synergy_data(cache_path)

    df = load_synergy_data(cache_path)
    if df is None:
        print("No cache found, fetching fresh data...")
        return refresh_synergy_data(cache_path)

    print(f"Loaded synergy data from cache ({len(df)} Pokemon)")
    return df


if __name__ == "__main__":
    # Test the module
    df = get_synergy_data()
    print(f"Loaded {len(df)} Pokemon with AG synergy data")
    print("Top 5 by usage:")
    print(df[['name', 'usage_percent']].head())