"""
ARIA Structured Scraper — Pull structured product data from real sources.
Usage:
  python tools/scrape_structured.py                          # All sources
  python tools/scrape_structured.py --source github          # Single source
  python tools/scrape_structured.py --query "SaaS tools"     # Custom query
  python tools/scrape_structured.py --ph-key KEY --ph-secret SECRET  # + Product Hunt

Sources (no auth required):
  github    — GitHub search API (repos related to a topic)
  hackernews — Hacker News via Algolia API (community discussions)
  npm       — npm registry (JavaScript packages)

Sources (requires API key):
  producthunt — Product Hunt GraphQL API (needs client_id + client_secret)

Output: working/structured_data.json
Also updates state/session.json
"""

import sys
import os
import json
import re
import argparse
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent.parent
WORKING_DIR = ROOT / "working"
STATE_DIR = ROOT / "state"

# ── Source: GitHub Search API ──

def fetch_github(query, limit=10):
    """Search GitHub for repos related to a topic. Returns structured data."""
    import requests
    headers = {
        'User-Agent': 'ARIA-LAB/1.0',
        'Accept': 'application/vnd.github.v3+json',
    }
    # Check for GitHub token
    token = os.environ.get('GITHUB_TOKEN')
    if token:
        headers['Authorization'] = f'token {token}'

    # Build multiple search queries
    queries = [
        f'{query} stars:>10',
        f'{query} tool',
        f'{query} platform',
    ]

    all_items = []
    seen = set()

    for q in queries[:2]:  # Limit to avoid rate limits
        try:
            import time
            url = f'https://api.github.com/search/repositories?q={q.replace(" ", "+")}&sort=stars&order=desc&per_page={limit}'
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 200:
                data = r.json()
                for repo in data.get('items', []):
                    name = repo.get('full_name', '')
                    if name not in seen:
                        seen.add(name)
                        desc = repo.get('description') or ''
                        topics = repo.get('topics', [])
                        all_items.append({
                            'name': name,
                            'description': desc[:200],
                            'url': repo.get('html_url', ''),
                            'stars': repo.get('stargazers_count', 0),
                            'language': repo.get('language', ''),
                            'topics': topics,
                            'source': 'github',
                            'created': repo.get('created_at', ''),
                            'updated': repo.get('updated_at', ''),
                        })
            elif r.status_code == 403:
                print(f"  GitHub rate limited. Set GITHUB_TOKEN env var for higher limits.")
                break
            time.sleep(1)
        except Exception as e:
            print(f"  GitHub search error: {e}")

    return all_items[:limit]


# ── Source: Hacker News (Algolia API) ──

def fetch_hackernews(query, limit=10):
    """Search Hacker News for discussions about products/tools."""
    import requests
    headers = {'User-Agent': 'ARIA-LAB/1.0'}

    queries = [
        f'best {query}',
        f'{query} tools',
        f'{query} alternatives',
    ]

    all_items = []
    seen = set()

    for q in queries[:2]:
        try:
            url = f'https://hn.algolia.com/api/v1/search?query={q.replace(" ", "+")}&tags=story&hitsPerPage={limit}'
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 200:
                data = r.json()
                for hit in data.get('hits', []):
                    title = hit.get('title', '')
                    obj_id = hit.get('objectID', '')
                    if obj_id not in seen and title:
                        seen.add(obj_id)
                        all_items.append({
                            'name': title,
                            'description': '',
                            'url': hit.get('url') or f'https://news.ycombinator.com/item?id={obj_id}',
                            'points': hit.get('points', 0),
                            'comments': hit.get('num_comments', 0),
                            'source': 'hackernews',
                            'created': hit.get('created_at', ''),
                        })
            import time
            time.sleep(0.5)
        except Exception as e:
            print(f"  HN search error: {e}")

    return all_items[:limit]


# ── Source: npm Registry ──

def fetch_npm(query, limit=10):
    """Search npm for packages related to a topic."""
    import requests
    headers = {'User-Agent': 'ARIA-LAB/1.0'}

    try:
        url = f'https://registry.npmjs.org/-/v1/search?text={query.replace(" ", "+")}&size={limit}'
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            data = r.json()
            items = []
            for obj in data.get('objects', []):
                pkg = obj.get('package', {})
                items.append({
                    'name': pkg.get('name', ''),
                    'description': (pkg.get('description') or '')[:200],
                    'url': f'https://www.npmjs.com/package/{pkg.get("name", "")}',
                    'version': pkg.get('version', ''),
                    'source': 'npm',
                    'score': round(obj.get('score', {}).get('final', 0) * 100),
                    'publisher': pkg.get('publisher', {}).get('username', ''),
                })
            return items
    except Exception as e:
        print(f"  npm search error: {e}")
    return []


# ── Source: Product Hunt GraphQL (requires API key) ──

def fetch_producthunt(query, limit=10, client_id=None, client_secret=None):
    """Fetch trending products from Product Hunt GraphQL API."""
    if not client_id or not client_secret:
        print("  Product Hunt: no credentials (set --ph-key and --ph-secret or PH_CLIENT_ID/PH_CLIENT_SECRET env vars)")
        return []

    import requests

    # Step 1: Get access token
    try:
        token_resp = requests.post('https://api.producthunt.com/v2/oauth/token',
            json={
                'client_id': client_id,
                'client_secret': client_secret,
                'grant_type': 'client_credentials',
            },
            headers={'User-Agent': 'ARIA-LAB/1.0', 'Content-Type': 'application/json'},
            timeout=15)
        if token_resp.status_code != 200:
            print(f"  PH auth failed: {token_resp.status_code} {token_resp.text[:100]}")
            return []
        token = token_resp.json().get('access_token')
    except Exception as e:
        print(f"  PH auth error: {e}")
        return []

    # Step 2: Query posts
    try:
        gql_query = '''
        query($first: Int!, $topic: String) {
          posts(order: VOTES, first: $first, topic: $topic) {
            edges {
              node {
                name
                tagline
                votesCount
                website
                createdAt
                url
                topics {
                  edges { node { name slug } }
                }
              }
            }
          }
        }
        '''
        variables = {'first': limit}
        # Map query to PH topic if possible
        topic_map = {
            'saas': 'saas', 'software': 'saas', 'design': 'design-tools',
            'developer': 'developer-tools', 'ai': 'artificial-intelligence',
            'marketing': 'marketing', 'analytics': 'analytics',
        }
        for key, topic in topic_map.items():
            if key in query.lower():
                variables['topic'] = topic
                break

        resp = requests.post('https://api.producthunt.com/v2/api/graphql',
            json={'query': gql_query, 'variables': variables},
            headers={
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json',
                'User-Agent': 'ARIA-LAB/1.0',
            },
            timeout=15)
        if resp.status_code != 200:
            print(f"  PH query failed: {resp.status_code}")
            return []

        data = resp.json()
        items = []
        for edge in data.get('data', {}).get('posts', {}).get('edges', []):
            node = edge.get('node', {})
            topics = [t['node']['name'] for t in node.get('topics', {}).get('edges', [])]
            items.append({
                'name': node.get('name', ''),
                'description': node.get('tagline', ''),
                'url': node.get('website') or node.get('url', ''),
                'votes': node.get('votesCount', 0),
                'topics': topics,
                'source': 'producthunt',
                'created': node.get('createdAt', ''),
            })
        return items[:limit]
    except Exception as e:
        print(f"  PH query error: {e}")
        return []


# ── Unified output ──

def extract_categories(items):
    """Add category classification to items based on content."""
    CATEGORY_SIGNALS = {
        'SaaS': ['saas', 'subscription', 'cloud', 'web app', 'platform', 'b2b', 'crm', 'erp'],
        'Developer Tool': ['developer', 'api', 'sdk', 'cli', 'library', 'framework', 'github', 'npm', 'open source', 'code', 'devtool'],
        'AI/ML': ['ai', 'artificial intelligence', 'machine learning', 'llm', 'gpt', 'chatbot', 'neural', 'ml'],
        'Design': ['design', 'ui', 'ux', 'figma', 'prototype', 'wireframe', 'mockup'],
        'Marketing': ['marketing', 'seo', 'analytics', 'email', 'social media', 'growth', 'ads'],
        'E-commerce': ['e-commerce', 'ecommerce', 'shop', 'store', 'sell', 'checkout', 'payment', 'digital product'],
        'Productivity': ['productivity', 'task', 'project management', 'notes', 'calendar', 'time'],
        'Education': ['course', 'learn', 'education', 'training', 'tutorial', 'teach'],
        'Mobile App': ['mobile', 'ios', 'android', 'app'],
        'Content': ['content', 'blog', 'cms', 'newsletter', 'writing', 'publish'],
    }

    for item in items:
        text = (item.get('name', '') + ' ' + item.get('description', '')).lower()
        topics_text = ' '.join(item.get('topics', [])).lower() if item.get('topics') else ''
        combined = text + ' ' + topics_text

        best_cat = 'Other'
        best_score = 0
        for cat, signals in CATEGORY_SIGNALS.items():
            score = sum(1 for s in signals if s in combined)
            if score > best_score:
                best_score = score
                best_cat = cat

        item['category'] = best_cat

    return items


def score_items(items, goal):
    """Score items by relevance to the goal."""
    goal_words = set(re.findall(r'\w+', goal.lower()))
    filler = {'the', 'a', 'an', 'of', 'for', 'and', 'or', 'in', 'to', 'is', 'are', 'with', 'this', 'that', 'from', 'by', 'on', 'at', 'it', 'be', 'as'}
    goal_keywords = goal_words - filler

    for item in items:
        text = (item.get('name', '') + ' ' + item.get('description', '')).lower()
        matches = sum(1 for kw in goal_keywords if kw in text)
        score = min(10, int((matches / max(len(goal_keywords), 1)) * 10))

        # Boost for social proof
        if item.get('stars', 0) > 1000:
            score = min(10, score + 2)
        elif item.get('stars', 0) > 100:
            score = min(10, score + 1)
        if item.get('votes', 0) > 100:
            score = min(10, score + 2)
        if item.get('score', 0) > 80:
            score = min(10, score + 1)
        if item.get('points', 0) > 50:
            score = min(10, score + 1)

        item['relevance_score'] = score

    return items


def run_all(query, goal=None, limit=10, ph_key=None, ph_secret=None):
    """Run all sources and combine results."""
    if not goal:
        goal = query

    print(f"  Query: {query}")
    print(f"  Goal:  {goal}")
    print()

    all_items = []

    # GitHub
    print("  [1/4] GitHub...")
    github_items = fetch_github(query, limit)
    print(f"    Found: {len(github_items)} repos")
    all_items.extend(github_items)

    # Hacker News
    print("  [2/4] Hacker News...")
    hn_items = fetch_hackernews(query, limit)
    print(f"    Found: {len(hn_items)} discussions")
    all_items.extend(hn_items)

    # npm
    print("  [3/4] npm registry...")
    npm_items = fetch_npm(query, limit)
    print(f"    Found: {len(npm_items)} packages")
    all_items.extend(npm_items)

    # Product Hunt (if credentials)
    print("  [4/4] Product Hunt...")
    ph_items = fetch_producthunt(query, limit, ph_key, ph_secret)
    print(f"    Found: {len(ph_items)} products")
    all_items.extend(ph_items)

    # Post-process
    print(f"\n  Total raw items: {len(all_items)}")
    all_items = extract_categories(all_items)
    all_items = score_items(all_items, goal)
    all_items.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)

    # Deduplicate by name
    seen = set()
    unique = []
    for item in all_items:
        key = item.get('name', '').lower().strip()
        if key and key not in seen:
            seen.add(key)
            unique.append(item)

    return unique


def save_results(items, query, goal):
    """Save to working/ and update state."""
    WORKING_DIR.mkdir(exist_ok=True)
    result = {
        'query': query,
        'goal': goal,
        'generated': datetime.now().isoformat(),
        'count': len(items),
        'items': items,
    }
    with open(WORKING_DIR / 'structured_data.json', 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # Update session state
    state_file = STATE_DIR / 'session.json'
    if state_file.exists():
        with open(state_file, 'r') as f:
            state = json.load(f)
        state['data']['structured_sources'] = items
        state['data']['structured_source_count'] = len(items)
        state['updated'] = datetime.now().isoformat()
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)

    return str(WORKING_DIR / 'structured_data.json')


def safe_print(text):
    """Print text with non-ASCII characters replaced for Windows console compatibility."""
    try:
        print(text)
    except UnicodeEncodeError:
        print(text.encode('ascii', 'replace').decode('ascii'))


def print_summary(items):
    """Print a summary of results."""
    safe_print("\n" + "=" * 60)
    safe_print("  STRUCTURED DATA RESULTS")
    safe_print("=" * 60)

    # Group by source
    by_source = {}
    for item in items:
        src = item.get('source', 'unknown')
        by_source.setdefault(src, []).append(item)

    for src, src_items in sorted(by_source.items()):
        safe_print(f"\n  {src.upper()} ({len(src_items)} items)")
        safe_print(f"  {'-' * 50}")
        for item in src_items[:5]:
            name = item.get('name', '?')[:50]
            desc = item.get('description', '')[:60]
            score = item.get('relevance_score', 0)
            cat = item.get('category', '?')

            details = []
            if item.get('stars'):
                details.append(f"{item['stars']} stars")
            if item.get('votes'):
                details.append(f"{item['votes']} votes")
            if item.get('score'):
                details.append(f"score:{item['score']}")

            detail_str = f" ({', '.join(details)})" if details else ""
            safe_print(f"    {name}")
            if desc:
                safe_print(f"      {desc}")
            safe_print(f"      [{cat}] relevance: {score}/10{detail_str}")

    # Category breakdown
    cats = {}
    for item in items:
        c = item.get('category', 'Other')
        cats[c] = cats.get(c, 0) + 1
    safe_print(f"\n  Categories: {', '.join(f'{k}({v})' for k, v in sorted(cats.items(), key=lambda x: -x[1]))}")
    safe_print(f"  Total: {len(items)} items")
    safe_print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", default=None, help="Search query (defaults to task goal)")
    parser.add_argument("--goal", default=None, help="Goal for relevance scoring (defaults to query)")
    parser.add_argument("--source", choices=["github", "hackernews", "npm", "producthunt", "all"], default="all")
    parser.add_argument("--limit", type=int, default=10, help="Results per source")
    parser.add_argument("--ph-key", default=None, help="Product Hunt client_id")
    parser.add_argument("--ph-secret", default=None, help="Product Hunt client_secret")
    args = parser.parse_args()

    # Get query/goal from session if not provided
    query = args.query
    goal = args.goal
    if not query:
        state_file = STATE_DIR / 'session.json'
        if state_file.exists():
            with open(state_file, 'r') as f:
                state = json.load(f)
            query = state.get('data', {}).get('goal', 'digital products')
            goal = goal or state.get('data', {}).get('goal', query)
        else:
            query = 'digital products'
            goal = query

    # PH credentials from env or args
    ph_key = args.ph_key or os.environ.get('PH_CLIENT_ID')
    ph_secret = args.ph_secret or os.environ.get('PH_CLIENT_SECRET')

    # Run
    if args.source == 'all':
        items = run_all(query, goal, args.limit, ph_key, ph_secret)
    elif args.source == 'github':
        items = fetch_github(query, args.limit)
        items = extract_categories(items)
        items = score_items(items, goal)
    elif args.source == 'hackernews':
        items = fetch_hackernews(query, args.limit)
        items = extract_categories(items)
        items = score_items(items, goal)
    elif args.source == 'npm':
        items = fetch_npm(query, args.limit)
        items = extract_categories(items)
        items = score_items(items, goal)
    elif args.source == 'producthunt':
        items = fetch_producthunt(query, args.limit, ph_key, ph_secret)
        items = extract_categories(items)
        items = score_items(items, goal)

    filepath = save_results(items, query, goal)
    print_summary(items)
    print(f"\n  Output: {filepath}")
