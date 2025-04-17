# -*- mode: python ; coding: utf-8 -*-
"""
AUTO-TAG PyInstaller Spec File

This file configures PyInstaller to create a distributable package of the AUTO-TAG application.
It handles the inclusion of all necessary dependencies, data files, and resources.
"""

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Application version
app_version = '1.0.0'

# Application metadata
app_name = 'AUTO-TAG'
app_description = 'AI-powered image tagging system'
app_author = 'AUTO-TAG Team'
app_copyright = '2025 AUTO-TAG Team'

# Collect all necessary data files
datas = [
    ('config', 'config'),
    ('models', 'models'),
    ('resources', 'resources'),
    ('vendor', 'vendor'),
    ('README.md', '.'),
]

# Collect all necessary hidden imports
hiddenimports = [
    'torch',
    'torchvision',
    'open_clip',
    'ultralytics',
    'PIL',
    'yaml',
    'tqdm',
    'numpy',
    'tagging',
]

# Add PyTorch-specific hidden imports
hiddenimports.extend(collect_submodules('torch'))
hiddenimports.extend(collect_submodules('torchvision'))

# Add CLIP-specific hidden imports
hiddenimports.extend(collect_submodules('open_clip'))

# Add YOLOv8-specific hidden imports
hiddenimports.extend(collect_submodules('ultralytics'))

# Create the Analysis object
a = Analysis(
    ['start.py'],  # Main script to execute
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Create the PYZ object (compressed Python modules)
pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher,
)

# Create the EXE object (executable file)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=app_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='resources/icon.ico' if os.path.exists('resources/icon.ico') else None,
    version='file_version_info.txt',
)

# Create the COLLECT object (directory with all files)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=app_name,
)

# Create a single-file executable (optional)
exe_onefile = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name=f'{app_name}-onefile',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='resources/icon.ico' if os.path.exists('resources/icon.ico') else None,
    version='file_version_info.txt',
)