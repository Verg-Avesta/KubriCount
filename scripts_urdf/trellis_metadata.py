#!/usr/bin/env python3
"""Generate manifest.json for generated assets dataset"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any
from collections import defaultdict
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_prompt_from_filename(filename: str) -> str:
    """Extract prompt from filename (part before 'seed')"""
    name = Path(filename).stem
    match = re.match(r'^(.+?)_seed\d+$', name)
    if match:
        return match.group(1)
    match = re.match(r'^(.+?)seed\d+$', name)
    if match:
        return match.group(1).rstrip('_')
    return name


def collect_assets_from_category(category_work_dir: Path, category_archive_dir: Path, category_name: str) -> Dict[str, Any]:
    """Collect metadata from processed assets in a specific category folder"""
    assets: Dict[str, Any] = {}

    if not category_work_dir.exists():
        logger.warning(f"Category work directory does not exist: {category_work_dir}")
        return assets

    for asset_dir in category_work_dir.iterdir():
        if not asset_dir.is_dir():
            continue

        if asset_dir.name in ['logs', 'locks', '.DS_Store']:
            continue

        asset_id = asset_dir.name
        data_json = asset_dir / 'kubric' / 'data.json'

        if not data_json.exists():
            logger.debug(f"No data.json found for {category_name}/{asset_id}")
            continue

        tar_gz_path = category_archive_dir / f'{asset_id}.tar.gz'
        if not tar_gz_path.exists():
            logger.debug(f"No archive found for {category_name}/{asset_id}")
            continue

        try:
            with open(data_json, 'r') as f:
                asset_data = json.load(f)

            if 'metadata' not in asset_data or not isinstance(asset_data['metadata'], dict):
                asset_data['metadata'] = {}

            # keep provenance if present; else mark unknown
            asset_data['metadata'].setdefault('source', 'unknown')

            asset_data['metadata']['category'] = category_name
            asset_data['metadata']['category_id'] = ""  # Keep empty.

            if 'prompt' not in asset_data['metadata']:
                asset_data['metadata']['prompt'] = extract_prompt_from_filename(asset_id)

            full_key = f"{category_name}/{asset_id}"
            asset_data['id'] = full_key

            assets[full_key] = asset_data

        except Exception as e:
            logger.error(f"Failed to process {category_name}/{asset_id}: {e}")
            continue

    return assets


def collect_all_assets(work_dir: Path, archive_dir: Path) -> Dict[str, Any]:
    """Collect metadata from all processed assets"""
    all_assets: Dict[str, Any] = {}

    for category_dir in sorted(work_dir.iterdir()):
        if not category_dir.is_dir():
            continue

        if category_dir.name in ['logs', 'locks', '.DS_Store', 'metadata']:
            continue

        category_name = category_dir.name
        category_archive_dir = archive_dir / category_name

        logger.info(f"Scanning category: {category_name}")

        assets = collect_assets_from_category(category_dir, category_archive_dir, category_name)

        all_assets.update(assets)
        logger.info(f"  Found {len(assets)} assets in {category_name}")

    return all_assets


def generate_statistics(assets: Dict[str, Any]) -> Dict[str, Any]:
    """Generate statistics from assets"""
    categories = defaultdict(int)
    prompts = defaultdict(int)
    sources = defaultdict(int)

    total_volume = 0
    total_surface_area = 0
    total_mass = 0
    total_faces = 0
    total_vertices = 0

    for _, asset_data in assets.items():
        metadata = asset_data.get('metadata', {})

        cat = metadata.get('category', 'unknown')
        categories[cat] += 1

        prompt = metadata.get('prompt', 'unknown')
        prompts[prompt] += 1

        src = metadata.get('source', 'unknown')
        sources[src] += 1

        total_volume += metadata.get('volume', 0)
        total_surface_area += metadata.get('surface_area', 0)
        total_mass += asset_data.get('kwargs', {}).get('mass', 0)
        total_faces += metadata.get('nr_faces', 0)
        total_vertices += metadata.get('nr_vertices', 0)

    n_assets = len(assets) if assets else 1

    return {
        "total_assets": len(assets),
        "categories": dict(categories),
        "sources": dict(sources),
        "prompts": dict(sorted(prompts.items(), key=lambda x: -x[1])),
        "averages": {
            "volume": total_volume / n_assets,
            "surface_area": total_surface_area / n_assets,
            "mass": total_mass / n_assets,
            "faces": total_faces / n_assets,
            "vertices": total_vertices / n_assets,
        }
    }


def generate_combined_manifest(work_dir: Path, archive_dir: Path, output_path: Path):
    """Generate combined manifest for all categories"""
    logger.info(f"Generating combined manifest from {work_dir}")

    all_assets = collect_all_assets(work_dir, archive_dir)

    if not all_assets:
        logger.warning("No assets found!")

    manifest = {
        "name": "Generated Assets",
        "version": "1.0",
        "description": "Generated 3D assets processed for Kubric. Keys are in format 'category/asset_id'",
        "data_dir": str(archive_dir),
        "total_assets": len(all_assets),
        "assets": all_assets
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    logger.info(f"Manifest saved: {output_path} ({len(all_assets)} total assets)")

    stats = generate_statistics(all_assets)
    stats_path = output_path.parent / 'statistics.json'
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Statistics saved: {stats_path}")

    category_index = defaultdict(list)
    for full_key, asset_data in all_assets.items():
        cat = asset_data.get('metadata', {}).get('category', 'unknown')
        category_index[cat].append(full_key)

    category_index_path = output_path.parent / 'category_index.json'
    with open(category_index_path, 'w') as f:
        json.dump(dict(category_index), f, indent=2)
    logger.info(f"Category index saved: {category_index_path}")

    prompt_index = defaultdict(list)
    for full_key, asset_data in all_assets.items():
        prompt = asset_data.get('metadata', {}).get('prompt', 'unknown')
        prompt_index[prompt].append(full_key)

    prompt_index_path = output_path.parent / 'prompt_index.json'
    with open(prompt_index_path, 'w') as f:
        json.dump(dict(prompt_index), f, indent=2)
    logger.info(f"Prompt index saved: {prompt_index_path}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("Summary:")
    logger.info(f"Total assets: {len(all_assets)}")
    logger.info(f"  Categories: {len(stats['categories'])}")
    logger.info(f"  Sources: {stats.get('sources', {})}")
    logger.info(f"  Unique prompts: {len(stats['prompts'])}")
    logger.info("")
    logger.info("Per-category counts:")
    for cat, count in sorted(stats['categories'].items()):
        logger.info(f"{cat}: {count}")
    logger.info("=" * 60)

    return all_assets


def generate_single_category_manifest(work_dir: Path, archive_dir: Path, category_name: str, output_path: Path):
    """Generate manifest for a single category"""
    logger.info(f"Generating manifest for category: {category_name}")

    category_work_dir = work_dir / category_name
    category_archive_dir = archive_dir / category_name

    assets = collect_assets_from_category(category_work_dir, category_archive_dir, category_name)

    manifest = {
        "name": f"Generated Assets - {category_name}",
        "version": "1.0",
        "category": category_name,
        "total_assets": len(assets),
        "assets": assets
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)

    logger.info(f"Category manifest saved: {output_path} ({len(assets)} assets)")

    return assets


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate manifest.json for generated assets')
    parser.add_argument('--work_dir', required=True, help='Directory containing processed asset folders (work/)')
    parser.add_argument('--archive_dir', required=True, help='Directory containing tar.gz archives (archives/)')
    parser.add_argument('--output', required=True, help='Output path for manifest.json')
    parser.add_argument('--category', help='Specific category to process (e.g., geese)')
    parser.add_argument('--combined', action='store_true', help='Generate combined manifest for all categories')
    args = parser.parse_args()

    work_dir = Path(args.work_dir)
    archive_dir = Path(args.archive_dir)
    output_path = Path(args.output)

    if not work_dir.exists():
        logger.error(f"Work directory does not exist: {work_dir}")
        exit(1)

    if args.category:
        generate_single_category_manifest(work_dir, archive_dir, args.category, output_path)
    else:
        generate_combined_manifest(work_dir, archive_dir, output_path)
