# Copyright 2024
# Adapted for Objaverse Dataset
# Based on Kubric's ShapeNet processing code

import argparse
import json
import logging
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
from typing import Tuple, Optional, Dict, Any
import re

import trimesh

from shapenet2kubric.trimesh_utils import get_object_properties
import shapenet2kubric.trimesh_utils
from shapenet2kubric.urdf_template import URDF_TEMPLATE

_DEFAULT_LOGGER = logging.getLogger(__name__)


# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

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


def get_asset_id_and_category(object_folder: Path, category: str = None, metadata: dict = None) -> Tuple[str, str, str, str]:
    """Extract asset ID, category, and prompt from folder structure or metadata
    Returns:
        Tuple of (asset_id, category_id, category_name, prompt)
    """
    asset_id = object_folder.name

    prompt = extract_prompt_from_filename(asset_id)

    if category:
        category_name = category
        category_id = ""  # Keep empty.
    elif metadata:
        categories = metadata.get('categories', [])
        if categories and len(categories) > 0:
            category_name = categories[0].get('name', 'unknown')
        else:
            category_name = 'unknown'
        category_id = ""
    else:
        category_id = ""
        category_name = 'unknown'

    return asset_id, category_id, category_name, prompt


def get_object_volume(obj_path: Path, logger=_DEFAULT_LOGGER, density=1.0):
    """Calculate object volume and related properties"""
    trimesh.util.log = logger
    tmesh = shapenet2kubric.trimesh_utils.get_tmesh(str(obj_path))

    properties = {
        "volume": tmesh.volume,
        "surface_area": tmesh.area,
        "mass": tmesh.volume * density,
    }
    return properties


def get_visual_properties(obj_path: Path, logger=_DEFAULT_LOGGER):
    """Get visual mesh properties"""
    trimesh.util.log = logger

    if str(obj_path).endswith('.glb'):
        tmesh = trimesh.load(str(obj_path), force='mesh')
    else:
        tmesh = shapenet2kubric.trimesh_utils.get_tmesh(str(obj_path))

    properties = {
        "nr_vertices": len(tmesh.vertices),
        "nr_faces": len(tmesh.faces),
    }
    return properties


def load_objaverse_metadata(metadata_path: Path, asset_id: str, logger=_DEFAULT_LOGGER) -> Dict[str, Any]:
    """Load metadata for Objaverse object if available"""
    try:
        if metadata_path and metadata_path.is_file():
            with open(metadata_path, 'r') as f:
                all_metadata = json.load(f)
                return all_metadata.get(asset_id, {})
    except Exception as e:
        logger.warning(f'Could not load metadata: {e}')
    return {}


def get_license_from_metadata(metadata: dict) -> str:
    """Extract license information from Objaverse metadata"""
    license_code = metadata.get('license', 'unknown')
    license_map = {
        'by': 'CC-BY 4.0',
        'by-sa': 'CC-BY-SA 4.0',
        'by-nc': 'CC-BY-NC 4.0',
        'by-nc-sa': 'CC-BY-NC-SA 4.0',
        'by-nd': 'CC-BY-ND 4.0',
        'by-nc-nd': 'CC-BY-NC-ND 4.0',
        'cc0': 'CC0 1.0',
    }
    return license_map.get(license_code, f'Unknown ({license_code})')


# ------------------------------------------------------------------------------
# Stage 0: Extract and prepare GLB file
# ------------------------------------------------------------------------------

def stage0(object_folder: Path, glb_file: Path, logger=_DEFAULT_LOGGER):
    """Copy source GLB to kubric folder as visual_geometry_pre.glb"""
    target_path = object_folder / 'kubric' / 'visual_geometry_pre.glb'

    if target_path.is_file():
        logger.debug(f'skipping stage0 on "{object_folder}"')
        return

    if not glb_file.is_file():
        logger.error(f'stage0 pre-condition failed, file does not exist "{glb_file}"')
        return

    target_path.parent.mkdir(parents=True, exist_ok=True)

    logger.debug(f'stage0 running on "{object_folder}"')

    shutil.copy(str(glb_file), str(target_path))

    if not target_path.is_file():
        logger.error(f'stage0 post-condition failed, file does not exist "{target_path}"')


# ------------------------------------------------------------------------------
# Stage 1: Convert GLB to OBJ and create watertight mesh
# ------------------------------------------------------------------------------

def stage1(object_folder: Path, logger=_DEFAULT_LOGGER):
    """Convert GLB to OBJ and create watertight mesh using manifold"""
    source_glb = object_folder / 'kubric' / 'visual_geometry_pre.glb'
    intermediate_obj = object_folder / 'kubric' / 'model_temp.obj'
    target_path = object_folder / 'kubric' / 'model_watertight.obj'

    if target_path.is_file():
        logger.debug(f'skipping stage1 on "{object_folder}"')
        return

    if not source_glb.is_file():
        logger.error(f'stage1 pre-condition failed, file does not exist "{source_glb}"')
        return

    logger.debug(f'stage1 running on "{object_folder}"')

    try:
        mesh = trimesh.load(str(source_glb), force='mesh')
        mesh.export(str(intermediate_obj))
        logger.debug(f'Converted GLB to OBJ: {intermediate_obj}')
    except Exception as e:
        logger.error(f'Failed to convert GLB to OBJ: {e}')
        return

    cmd = f'manifold --input {intermediate_obj} --output {target_path}'
    retobj = subprocess.run(cmd, capture_output=True, shell=True, text=True)

    if retobj.returncode != 0:
        logger.error(f'manifold failed on "{object_folder}"')
        if retobj.stdout != '':
            logger.error(f'{retobj.stdout}')
        if retobj.stderr != '':
            logger.error(f'{retobj.stderr}')
        return

    if intermediate_obj.exists():
        intermediate_obj.unlink()

    if not target_path.is_file():
        logger.error(f'stage1 post-condition failed, file does not exist "{target_path}"')


# ------------------------------------------------------------------------------
# Stage 2: Generate Collision Geometry
# ------------------------------------------------------------------------------

def stage2(object_folder: Path, logger=_DEFAULT_LOGGER):
    """Generate collision geometry using V-HACD"""
    source_path = object_folder / 'kubric' / 'model_watertight.obj'
    target_path = object_folder / 'kubric' / 'collision_geometry.obj'
    log_path = object_folder / 'kubric' / 'stage2_logs.txt'
    stdout_path = str(object_folder / 'kubric' / 'stage2_stdout.txt')

    if target_path.is_file():
        logger.debug(f'skipping stage2 on "{object_folder}"')
        return

    if not source_path.is_file():
        logger.error(f'stage2 pre-condition failed, file does not exist "{source_path}"')
        return

    logger.debug(f'stage2 running on "{object_folder}"')
    command_string = (
        f"python shapenet2kubric/pybullet_vhacd.py "
        f"--source_path={source_path} --target_path={target_path} "
        f"--stdout_path={stdout_path} > {log_path}"
    )

    retobj = subprocess.run(command_string, shell=True, check=False,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if retobj.returncode != 0:
        logger.error(f'stage2 failed with return code {retobj.returncode}')

    if not target_path.is_file():
        logger.error(f'stage2 post-condition failed, file does not exist "{target_path}"')


# ------------------------------------------------------------------------------
# Stage 3: Clean Mesh with Blender
# ------------------------------------------------------------------------------

def stage3(object_folder: Path, logger=_DEFAULT_LOGGER):
    """Clean mesh using Blender"""
    source_path = object_folder / 'kubric' / 'visual_geometry_pre.glb'
    log_path = object_folder / 'kubric' / 'stage3_logs.txt'
    target_path = object_folder / 'kubric' / 'visual_geometry.glb'

    if target_path.is_file():
        logger.debug(f'skipping stage3 on "{object_folder}"')
        return

    logger.debug(f'stage3 running on "{object_folder}"')

    asset_id = object_folder.name

    command_string = (
        f"python shapenet2kubric/bpy_clean_mesh.py "
        f"--source_path={source_path} --target_path={target_path} "
        f"--asset_id={asset_id} > {log_path}"
    )

    retobj = subprocess.run(command_string, shell=True, check=False)
    if retobj.returncode != 0:
        logger.error(f'stage3 failed with return code {retobj.returncode}')

    if not target_path.is_file():
        logger.error(f'stage3 post-condition failed, file does not exist "{target_path}"')


# ------------------------------------------------------------------------------
# Stage 4: Generate URDF and Metadata (with trellis provenance)
# ------------------------------------------------------------------------------

def stage4(
    object_folder: Path,
    category: str = None,
    metadata: dict = None,
    source_name: str = "trellis1",
    logger=_DEFAULT_LOGGER
) -> Optional[Dict[str, Any]]:
    """Generate URDF and JSON metadata"""
    source_path = object_folder / 'kubric' / 'collision_geometry.obj'
    watertight_mesh_path = object_folder / 'kubric' / 'model_watertight.obj'
    vis_mesh_path = object_folder / 'kubric' / 'visual_geometry.glb'
    target_urdf_path = object_folder / 'kubric' / 'object.urdf'
    target_json_path = object_folder / 'kubric' / 'data.json'

    if target_urdf_path.is_file() and target_json_path.is_file():
        logger.debug(f'skipping stage4 on "{object_folder}"')
        with open(target_json_path, 'r') as f:
            return json.load(f)

    if not source_path.is_file():
        logger.error(f'stage4 pre-condition failed, file does not exist "{source_path}"')
        return None
    if not watertight_mesh_path.is_file():
        logger.error(f'stage4 pre-condition failed, file does not exist "{watertight_mesh_path}"')
        return None
    if not vis_mesh_path.is_file():
        logger.error(f'stage4 pre-condition failed, file does not exist "{vis_mesh_path}"')
        return None

    logger.debug(f'stage4 running on "{object_folder}"')

    properties = get_object_properties(source_path, logger)
    properties.update(get_object_volume(watertight_mesh_path))
    properties.update(get_visual_properties(vis_mesh_path))

    asset_id, category_id, category_name, prompt = get_asset_id_and_category(
        object_folder, category, metadata
    )
    properties["id"] = asset_id

    urdf_str = URDF_TEMPLATE.format(**properties)
    with open(target_urdf_path, 'w') as fd:
        fd.write(urdf_str)

    license_str = get_license_from_metadata(metadata) if metadata else 'Generated Asset'

    asset_entry = {
        "asset_type": "FileBasedObject",
        "id": f"{category_name}/{asset_id}" if category_name else asset_id,
        "kwargs": {
            "bounds": properties["bounds"],
            "mass": properties["mass"],
            "render_filename": "{asset_dir}/visual_geometry.glb",
            "simulation_filename": "{asset_dir}/object.urdf",
        },
        "license": license_str,
        "metadata": {
            "source": source_name,  # provenance: trellis1 / trellis2
            "category": category_name,
            "category_id": category_id,
            "prompt": prompt,
            "nr_faces": properties["nr_faces"],
            "nr_vertices": properties["nr_vertices"],
            "surface_area": properties["surface_area"],
            "volume": properties["volume"],
            "watertight_mesh_filename": "{asset_dir}/model_watertight.obj",
        }
    }

    if metadata:
        objaverse_meta = {
            "source": "objaverse",
            "name": metadata.get("name", ""),
            "uid": metadata.get("uid", asset_id),
        }
        asset_entry["metadata"]["objaverse"] = objaverse_meta

    with open(target_json_path, "w") as fd:
        json.dump(asset_entry, fd, indent=4, sort_keys=True)

    if not target_urdf_path.is_file():
        logger.error(f'stage4 post-condition failed, file does not exist "{target_urdf_path}"')

    return asset_entry


# ------------------------------------------------------------------------------
# Stage 5: Create Archive
# ------------------------------------------------------------------------------

def stage5(object_folder: Path, logger=_DEFAULT_LOGGER):
    """Package all processed files into tar.gz"""
    target_path = object_folder / 'kubric.tar.gz'

    if target_path.is_file():
        logger.debug(f'skipping stage5 on "{object_folder}"')
        return

    logger.debug(f'stage5 running on "{object_folder}"')

    try:
        with tarfile.open(target_path, 'w:gz') as tar:
            tar.add(object_folder / 'kubric' / 'visual_geometry.glb', arcname='visual_geometry.glb')
            tar.add(object_folder / 'kubric' / 'collision_geometry.obj', arcname='collision_geometry.obj')
            tar.add(object_folder / 'kubric' / 'model_watertight.obj', arcname='model_watertight.obj')
            tar.add(object_folder / 'kubric' / 'object.urdf', arcname='object.urdf')
            tar.add(object_folder / 'kubric' / 'data.json', arcname='data.json')
    except Exception as e:
        logger.error(f'stage5 failed with error: {e}')
        return

    if not target_path.is_file():
        logger.error(f'stage5 post-condition failed, file does not exist "{target_path}"')


# ------------------------------------------------------------------------------
# Stage 6: Move to Final Location
# ------------------------------------------------------------------------------

def stage6(object_folder: Path, archive_dir: Path, logger=_DEFAULT_LOGGER):
    """Move processed archive to final location"""
    asset_id = object_folder.name
    source_path = object_folder / 'kubric.tar.gz'
    target_path = archive_dir / f'{asset_id}.tar.gz'

    if target_path.is_file():
        logger.debug(f'skipping stage6 on "{object_folder}"')
        return

    if not source_path.is_file():
        logger.error(f'stage6 pre-condition failed, file does not exist "{source_path}"')
        return

    logger.debug(f'stage6 running on "{object_folder}"')

    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(source_path), str(target_path))
    source_path.unlink()

    if not target_path.is_file():
        logger.error(f'stage6 post-condition failed, file does not exist "{target_path}"')


# ------------------------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process Trellis GLB files for Kubric')
    parser.add_argument('--glb_file', required=True, help='Path to the GLB file to process')
    parser.add_argument('--asset_id', help='Asset ID (if not provided, will use filename without extension)')
    parser.add_argument('--category', help='Category name (folder name)')
    parser.add_argument('--metadata_file', help='Path to JSON file containing Objaverse metadata')
    parser.add_argument('--source_name', default='trellis1',
                        help='Provenance tag stored into data.json metadata["source"], e.g. trellis1/trellis2')
    parser.add_argument('--stages', nargs='+', type=int, default=[0, 1, 2, 3, 4, 5, 6],
                        help='Stages to run (0-6)')
    parser.add_argument('--output_dir', default="objaverse",
                        help='Output directory for processed files')
    parser.add_argument('--work_dir', help='Working directory for intermediate files')
    parser.add_argument('--archive_dir', help='Directory to store tar.gz archives')
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    glb_file = Path(args.glb_file)
    if not glb_file.exists():
        logger.error(f'GLB file does not exist: {glb_file}')
        sys.exit(1)

    asset_id = args.asset_id if args.asset_id else glb_file.stem

    output_dir = Path(args.output_dir)
    work_dir = Path(args.work_dir) if args.work_dir else output_dir / 'work'
    archive_dir = Path(args.archive_dir) if args.archive_dir else output_dir / 'archives'

    object_folder = work_dir / asset_id
    object_folder.mkdir(parents=True, exist_ok=True)

    metadata = {}
    if args.metadata_file:
        metadata_path = Path(args.metadata_file)
        metadata = load_objaverse_metadata(metadata_path, asset_id, logger)
        if metadata:
            logger.info(f'Loaded metadata for asset {asset_id}: {metadata.get("name", "N/A")}')

    logger.info(f'Processing GLB file: {glb_file}')
    logger.info(f'Asset ID: {asset_id}')
    logger.info(f'Category: {args.category or "N/A"}')
    logger.info(f'Source name: {args.source_name}')
    logger.info(f'Working directory: {object_folder}')
    logger.info(f'Archive directory: {archive_dir}')
    logger.info(f'Running stages: {args.stages}')

    try:
        asset_entry = None
        if 0 in args.stages:
            logger.info('Running stage 0: Copying GLB file')
            stage0(object_folder, glb_file, logger)
        if 1 in args.stages:
            logger.info('Running stage 1: GLB to OBJ conversion and watertight mesh generation')
            stage1(object_folder, logger)
        if 2 in args.stages:
            logger.info('Running stage 2: Collision geometry generation')
            stage2(object_folder, logger)
        if 3 in args.stages:
            logger.info('Running stage 3: Mesh cleaning')
            stage3(object_folder, logger)
        if 4 in args.stages:
            logger.info('Running stage 4: URDF and metadata generation')
            asset_entry = stage4(object_folder, args.category, metadata, args.source_name, logger)
            if asset_entry:
                logger.info(f'Generated asset entry for {asset_id}')
        if 5 in args.stages:
            logger.info('Running stage 5: Archive creation')
            stage5(object_folder, logger)
        if 6 in args.stages:
            logger.info('Running stage 6: Moving to final location')
            stage6(object_folder, archive_dir, logger)

        logger.info('Processing completed successfully!')

    except Exception as e:
        logger.error(f'Processing failed with error: {e}', exc_info=True)
        sys.exit(1)
