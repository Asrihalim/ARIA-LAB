"""
ARIA Web Scraper — Pull content from any URL.
Usage: python tools/scrape.py <url> [--format text|markdown|json] [--save filename]

Fetches page content, extracts text, saves to working/
"""

import sys
import os
import json
import argparse
from pathlib import Path
from urllib.parse import urlparse

def scrape_url(url, output_format="text"):
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        return {"error": "Install dependencies: pip install requests beautifulsoup4"}

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    try:
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
    except Exception:
        # Fallback: disable SSL verification for sites with cert issues
        try:
            resp = requests.get(url, headers=headers, timeout=30, verify=False)
            resp.raise_for_status()
        except Exception as e:
            return {"error": str(e), "url": url}

    soup = BeautifulSoup(resp.text, 'html.parser')

    # Remove script/style
    for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    # Extract links
    links = []
    for a in soup.find_all('a', href=True):
        href = a['href']
        text = a.get_text(strip=True)
        if text and href.startswith('http'):
            links.append({"text": text[:100], "url": href})

    # Extract images
    images = []
    for img in soup.find_all('img', src=True):
        images.append({"src": img['src'], "alt": img.get('alt', '')})

    # Get text
    text = soup.get_text(separator='\n', strip=True)
    # Clean up excessive whitespace
    lines = [l.strip() for l in text.split('\n') if l.strip()]
    clean_text = '\n'.join(lines)

    result = {
        "url": url,
        "title": title,
        "text_length": len(clean_text),
        "word_count": len(clean_text.split()),
        "links_count": len(links),
        "images_count": len(images),
        "text": clean_text[:50000],  # Cap at 50k chars
        "top_links": links[:20],
    }

    if output_format == "json":
        pass  # Already structured
    elif output_format == "markdown":
        md = f"# {title}\n\nSource: {url}\n\n---\n\n{clean_text[:50000]}"
        result["markdown"] = md

    return result

def save_result(result, filename=None):
    working_dir = Path(__file__).parent.parent / "working"
    working_dir.mkdir(exist_ok=True)
    if not filename:
        domain = urlparse(result.get("url", "unknown")).netloc or "unknown"
        filename = f"scrape_{domain.replace('.', '_')}.json"
    filepath = working_dir / filename
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return str(filepath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("url", help="URL to scrape")
    parser.add_argument("--format", default="text", choices=["text", "markdown", "json"])
    parser.add_argument("--save", default=None, help="Output filename")
    args = parser.parse_args()

    result = scrape_url(args.url, args.format)

    if "error" in result:
        print(f"ERROR: {result['error']}")
        sys.exit(1)

    saved = save_result(result, args.save)
    print(f"Scraped: {result['title']}")
    print(f"Words: {result['word_count']} | Links: {result['links_count']} | Images: {result['images_count']}")
    print(f"Saved to: {saved}")
    print(f"\n--- Content Preview (first 500 chars) ---\n{result['text'][:500]}")
