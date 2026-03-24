# -*- mode: python ; coding: utf-8 -*-
import os

block_cipher = None
BASE = r'c:\Users\konst\OneDrive\Dokumente\Projekte\Vokabltrainer'

# customtkinter-Pfad ermitteln (braucht seine eigenen Assets)
import customtkinter
CTK_PATH = os.path.dirname(customtkinter.__file__)

a = Analysis(
    [os.path.join(BASE, 'vokabeltrainer.py')],
    pathex=[BASE],
    binaries=[],
    datas=[
        # Statische Ressourcen → kommen MIT ins Bundle
        (os.path.join(BASE, 'icons'), 'icons'),
        (os.path.join(BASE, 'ocr'), 'ocr'),
        (os.path.join(BASE, '.env'), '.'),
        (os.path.join(BASE, 'firebase_credentials.json'), '.'),
        # customtkinter braucht seine Assets (Themes, JSON-Dateien)
        (CTK_PATH, 'customtkinter'),
    ],
    hiddenimports=[
        'firebase_admin',
        'firebase_admin._utils',
        'firebase_admin.credentials',
        'firebase_admin.firestore',
        'google.cloud.firestore',
        'google.cloud.firestore_v1',
        'google.cloud.firestore_v1.base_query',
        'google.cloud.firestore_v1.query',
        'google.cloud.firestore_v1._helpers',
        'google.cloud.firestore_v1.transforms',
        'google.cloud.firestore_v1.async_client',
        'google.cloud.firestore_v1.async_collection',
        'google.cloud.firestore_v1.async_document',
        'google.cloud.firestore_v1.async_query',
        'google.cloud.firestore_v1.async_batch',
        'google.cloud.firestore_v1.async_transaction',
        'google.auth',
        'google.auth.transport',
        'google.auth.transport.requests',
        'google.auth.transport.grpc',
        'google.oauth2',
        'google.oauth2.service_account',
        'grpc',
        'grpc._cython',
        'grpc._cython.cygrpc',
        'customtkinter',
        'PIL',
        'PIL._tkinter_finder',
        'PIL.ImageOps',
        'PIL.ImageEnhance',
        'qrcode',
        'qrcode.image.pil',
        'openai',
        'pytesseract',
        'dotenv',
        'flask',
        'flask.json',
        'werkzeug',
        'werkzeug.utils',
        'zoneinfo',
        'hashlib',
        'csv',
        'threading',
        'json',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Vokabeltrainer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,   # Erst True zum Debuggen! Wenn alles klappt auf False aendern
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Vokabeltrainer',
)
