# ============================== Imports ======================================
import os
import sys
import json
import csv
import random
import re
import socket
import threading
from urllib.parse import urljoin
from flask import Flask, request, render_template_string
from werkzeug.utils import secure_filename
import qrcode
import openai
from openai import OpenAI  # <-- Import für den modernen Client
import pytesseract
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageEnhance
from datetime import date, datetime, timedelta
import time
import hashlib
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo
from dotenv import load_dotenv

# --- Vorab-Definitionen zur Vermeidung statischer NameError-Warnungen ---
# Farben / Theme
BTN_COLOR       = "#6366f1"  # Indigo
BTN_HOVER_COLOR = "#4f46e5"  # Dunkleres Indigo
SPRACH_COLOR    = "#10b981"  # Emerald (für Sprachanzeige)
SUCCESS_COLOR   = "#059669"  # Dunkleres Emerald
WARNING_COLOR   = "#f59e0b"  # Amber
ERROR_COLOR     = "#dc2626"  # Rot
TEXT_COLOR      = "#374151"  # Grau für Text
LIGHT_TEXT      = "#6b7280"  # Helleres Grau
DISABLED_COLOR  = "#9ca3af"  # Grau (disabled)
DISABLED_HOVER  = "#9ca3af"  # gleich lassen

# Plattform-/Emoji-Einstellungen (unter Linux/Tk ggf. problematisch)
IS_LINUX = sys.platform.startswith('linux')
USE_EMOJI = not IS_LINUX
EMOJI_OK    = "✅ " if USE_EMOJI else ""
EMOJI_BAD   = "❌ " if USE_EMOJI else ""
EMOJI_PART  = "✅ " if USE_EMOJI else ""
EMOJI_PARTY = "🎉 " if USE_EMOJI else ""

# Globale Platzhalter
app: ctk.CTk | None = None
haus_icon  = None
birne_icon = None
tipp_icon  = None
_resize_job = None

# Frames-Registry
frames = {}
current_visible_frame = None

def update_fenstertitel():
    try:
        if not app:
            return
        if aktuelle_sprache:
            app.title(f"Vokabeltrainer - {aktuelle_sprache.capitalize()}")
        else:
            app.title("Vokabeltrainer")
    except Exception:
        pass

def update_sprachanzeige():
    text = aktuelle_sprache.capitalize() if aktuelle_sprache else "Keine Sprache"

    for lbl in (sprach_anzeige_label, trainer_sprach_label, end_sprach_label):
        try:
            if lbl is not None and lbl.winfo_exists():
                lbl.configure(text=text)
        except Exception:
            pass

    try:
        if statistik_titel_label is not None and statistik_titel_label.winfo_exists():
            if aktuelle_sprache:
                statistik_titel_label.configure(text=f"Statistiken - {aktuelle_sprache.capitalize()}")
            else:
                statistik_titel_label.configure(text="Statistiken")
    except Exception:
        pass

def update_font_sizes(event=None):
    """Passt Schriftgrößen und Widget-Breiten dynamisch an die aktuelle Fenstergröße an."""
    global _resize_job
    _resize_job = None
    if not app:
        return
    try:
        w = app.winfo_width()
        h = app.winfo_height()
    except Exception:
        return
    if w < 100 or h < 100:
        return

    # Skalierungsfaktor relativ zur Basis-Auflösung 1200×800
    BASE_W, BASE_H = 1200, 800
    scale = min(w / BASE_W, h / BASE_H)
    scale = max(0.5, min(scale, 2.0))  # auf sinnvollen Bereich begrenzen

    def fs(base):
        """Skalierte Schriftgröße (Minimum 8)."""
        return max(8, int(base * scale))

    def safe_cfg(widget_name, **kwargs):
        """Konfiguriert ein Widget sicher per globalem Namen."""
        widget = globals().get(widget_name)
        try:
            if widget is not None and widget.winfo_exists():
                widget.configure(**kwargs)
        except Exception:
            pass

    # ── Trainer-Bildschirm ────────────────────────────────────────────────
    safe_cfg('frage_label',     font=('Arial', fs(36)))
    safe_cfg('eingabe',         font=('Arial', fs(22)),
                                width=max(300, int(900 * scale)))
    safe_cfg('feedback_label',  font=('Arial', fs(22)))
    safe_cfg('punktzahl_label', font=('Arial', fs(22)))
    safe_cfg('fortschritt',     width=max(300, int(900 * scale)))
    safe_cfg('btn_pruefen',
             font=('Segoe UI', fs(18), 'bold'),
             width=max(180, int(360 * scale)),
             height=max(40, int(60 * scale)))

    # Choice-Buttons (Liste)
    try:
        btn_w = max(300, int(720 * scale))
        btn_h = max(40,  int(72  * scale))
        for btn in choice_buttons:
            if btn is not None and btn.winfo_exists():
                btn.configure(
                    font=('Segoe UI', fs(24), 'bold'),
                    width=btn_w,
                    height=btn_h,
                )
    except Exception:
        pass

    # ── Sprachanzeige-Labels (Start- & Trainer-Screen) ────────────────────
    safe_cfg('sprach_anzeige_label', font=('Segoe UI', fs(24), 'bold'))
    safe_cfg('trainer_sprach_label', font=('Segoe UI', fs(24), 'bold'))

    # ── Start-Bildschirm ──────────────────────────────────────────────────
    safe_cfg('start_input_button',  font=('Segoe UI', fs(30), 'bold'))
    safe_cfg('start_choice_button', font=('Segoe UI', fs(30), 'bold'))
    safe_cfg('xp_start_label',      font=('Segoe UI', fs(20), 'bold'))

    # ── Statistik-Screen ─────────────────────────────────────────────────
    safe_cfg('statistik_titel_label', font=('Arial', fs(30)))
    safe_cfg('stat_feedback_label',   font=('Arial', fs(16)))

def debounced_update_font_sizes(event=None):
    """Debounced Wrapper für update_font_sizes, nur auf App-Resize reagieren."""
    global _resize_job
    try:
        if event is not None and hasattr(event, 'widget') and event.widget is not app:
            return
    except Exception:
        pass
    try:
        if _resize_job and app:
            app.after_cancel(_resize_job)
    except Exception:
        pass
    if app:
        _resize_job = app.after(80, update_font_sizes)

# Tippfehler-Erkennung mit Tastatur-Layout
def get_keyboard_distance(char1, char2):
    """Berechnet die Tastatur-Distanz zwischen zwei Buchstaben"""
    
    # Deutsche QWERTZ-Tastatur-Layout
    keyboard_layout = [
        ['q', 'w', 'e', 'r', 't', 'z', 'u', 'i', 'o', 'p', 'ü', '+'],
        
        ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'ö', 'ä', '#'],
        ['<', 'y', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '-']
    ]

    # Finde Positionen der Buchstaben
    pos1 = None
    pos2 = None
    
    for row_idx, row in enumerate(keyboard_layout):
        for col_idx, char in enumerate(row):
            if char == char1.lower():
                pos1 = (row_idx, col_idx)
            if char == char2.lower():
                pos2 = (row_idx, col_idx)
    
    # Wenn Buchstaben nicht gefunden, verwende große Distanz
    if pos1 is None or pos2 is None:
        return 3
    
    # Berechne Manhattan-Distanz
    distance = abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    return distance

def calculate_typo_probability(user_answer, correct_answer):
    """
    Berechnet die Wahrscheinlichkeit, dass es sich um einen Tippfehler handelt
    basierend auf Tastatur-Layout und Ähnlichkeit
    """
    if not user_answer or not correct_answer:
        return 0.0
    
    user_clean = user_answer.strip().lower()
    correct_clean = correct_answer.strip().lower()
    
    # Wenn identisch, keine Tippfehler-Wahrscheinlichkeit
    if user_clean == correct_clean:
        return 0.0
    
    # Wenn Längen zu unterschiedlich, unwahrscheinlich Tippfehler
    if abs(len(user_clean) - len(correct_clean)) > 2:
        return 0.0
    
    # Verwende dynamische Programmierung für die beste Übereinstimmung
    def min_edit_distance(s1, s2):
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    # Berechne Tastatur-Distanz für Substitution
                    keyboard_cost = get_keyboard_distance(s1[i-1], s2[j-1])
                    substitution_cost = min(keyboard_cost, 2)  # Max 2 für Substitution
                    
                    dp[i][j] = min(
                        dp[i-1][j] + 1,      # Deletion
                        dp[i][j-1] + 1,      # Insertion
                        dp[i-1][j-1] + substitution_cost  # Substitution
                    )
        
        return dp[m][n]
    
    # Berechne gewichtete Edit-Distanz
    weighted_distance = min_edit_distance(user_clean, correct_clean)
    
    # Normalisiere basierend auf Wortlänge
    max_length = max(len(user_clean), len(correct_clean))
    normalized_distance = weighted_distance / max_length
    
    # Konvertiere zu Wahrscheinlichkeit (0 = sicher kein Tippfehler, 1 = sicher Tippfehler)
    # Schwellenwerte basierend auf Erfahrung
    if normalized_distance <= 0.2:
        probability = 0.9  # Sehr wahrscheinlich Tippfehler
    elif normalized_distance <= 0.4:
        probability = 0.7  # Wahrscheinlich Tippfehler
    elif normalized_distance <= 0.6:
        probability = 0.3  # Möglicherweise Tippfehler
    else:
        probability = 0.0  # Unwahrscheinlich Tippfehler
    
    return probability

def is_typo(user_answer, correct_answer, threshold=None):
    # Threshold basierend auf Tippfehler-Toleranz
    if threshold is None:
        thresholds = [0.6, 0.75, 1.0]  # Leicht, Mittel, Schwer
        try:
            threshold = thresholds[training_settings['typo_tolerance']]
        except IndexError:
            threshold = thresholds[1] # Fallback auf Mittel

    
    probability = calculate_typo_probability(user_answer, correct_answer)
    return probability >= threshold

# ======================= Ermittlung des App-Ordners ==========================
if getattr(sys, "frozen", False):
    APP_DIR = os.path.dirname(sys.argv[0])
else:
    APP_DIR = os.path.dirname(os.path.abspath(__file__))

# ======================= Unterordner im APP_DIR ==============================
ICON_DIR  = os.path.join(APP_DIR, 'icons')
OCR_DIR   = os.path.join(APP_DIR, 'ocr')
VOCAB_DIR = os.path.join(APP_DIR, 'vocabularies')
STAT_DIR  = os.path.join(APP_DIR, 'stats')

# Verzeichnisse einmalig sicherstellen
os.makedirs(VOCAB_DIR, exist_ok=True)
os.makedirs(STAT_DIR, exist_ok=True)

# ======================= OpenAI API Key =======================================
load_dotenv(override=True)  # <--- FORCE Reload from file (ignores old memory)

# Client explizit initialisieren (verhindert Probleme mit alten global configs)
api_key = os.getenv("OPENAI_API_KEY")
client = None

if api_key:
    # Optional: Debug-Print (nur die ersten 10 Zeichen)
    # print(f"API Key geladen: {api_key[:10]}...")
    try:
        client = OpenAI(api_key=api_key)
    except Exception as e:
        print(f"Fehler beim Initialisieren des OpenAI Clients: {e}")
else:
    print("Warnung: OPENAI_API_KEY ist nicht gesetzt! Einige Funktionen sind möglicherweise deaktiviert.")

# ======================= Lade-Icon-Funktion ===================================
def lade_icon(dateiname, size=(60,60)):
    pfad = os.path.join(ICON_DIR, dateiname)
    if not os.path.exists(pfad):
        print(f"[lade_icon] Datei nicht gefunden: {pfad}")
        return None
    try:
        img = Image.open(pfad)
        return ctk.CTkImage(light_image=img, dark_image=img, size=size)
    except Exception as e:
        print(f"[lade_icon] Fehler beim Laden von {pfad}: {e}")
        return None

# ======================= Globale Variablen & Config ===========================
alle_vokabeln       = []
vokabeln_zu_lernen  = []
learning_queue      = []  # Warteschlange für den aktuellen Durchlauf
initial_queue_len   = 0   # Startlänge der Queue für Fortschrittsanzeige
xp_session          = 0   # XP in der aktuellen Übungseinheit
xp_woche            = 0   # XP in der aktuellen Woche (reset jeden Sonntag 20:00)
xp_gesamt           = 0   # Gesamt-XP aller Zeiten
combo_streak        = 0   # Anzahl aufeinanderfolgender richtiger Antworten (nur Eingabe-Modus)
firebase_db         = None  # Firestore-Client (None = Offline)
aktueller_nutzer    = None  # Benutzername
aktuelle_vokabel    = None
runde_status        = {}
vokabel_statistik   = {}

editor_entries      = []  # für den Editor

# Widgets (in Screens gesetzt)
frage_label         = None
eingabe             = None
feedback_label      = None
punktzahl_label     = None
fortschritt         = None
btn_pruefen         = None
start_button        = None
start_choice_button = None
start_choice_lock_label = None
start_choice_command = None
start_choice_container = None
_shake_in_progress = False
START_CHOICE_PADX_BASE = 6
start_choice_inner = None
START_CHOICE_SHAKE_BASE_X = 0
START_CHOICE_CONTAINER_WIDTH = 320
START_CHOICE_SHAKE_AMPL = 12
start_input_button = None
start_input_container = None
start_input_inner = None

statistik_frame     = None
stat_feedback_label = None
statistik_titel_label = None
editor_frame        = None
editor_feedback     = None
neu_de              = None
neu_en              = None
aktuelle_sprache = None

# Sprachanzeige-Widgets
sprach_anzeige_label = None
trainer_sprach_label = None
end_sprach_label = None
xp_start_label       = None  # XP-Anzeige im Startbildschirm
rangliste_scroll_frame = None  # Scroll-Container der Rangliste

# Training-Einstellungen
training_settings = {
    'typo_tolerance': 1,  # 0=Leicht, 1=Mittel, 2=Schwer
    'repetitions': 2,     # 1-5 Wiederholungen
    'mode': 'input',      # 'input' (Eingabefeld) oder 'choice' (3 Auswahl-Buttons)
    'direction': 'de_to_foreign', # oder 'foreign_to_de'
}

# Vokabel-Wiederholungs-Tracking
vokabel_repetitions = {}  # Speichert wie oft jede Vokabel richtig beantwortet wurde

# Feedback-System
feedback_active = False
weiter_button = None

# UI-Elemente für Modus 'choice'
answer_frame = None
input_frame = None
choice_frame = None
choice_buttons = []
timer_label = None
timer_bar = None
current_question_direction = None
ACTION_MIN_INTERVAL_SEC = 0.5
_action_next_time = 0.0
_action_pending_job = None
_action_pending = None

# ======================= Hilfsfunktion: Alle Sprachen aus tessdata ============
def get_all_tesseract_langs():
    tessdata_dir = os.path.join(OCR_DIR, 'tessdata')
    if not os.path.isdir(tessdata_dir):
        return ''
    codes = []
    for fname in os.listdir(tessdata_dir):
        if fname.endswith('.traineddata'):
            codes.append(fname[:-len('.traineddata')])
    codes.sort()
    return '+'.join(codes)

# ======================= OCR-Konfiguration (tesseract.exe) ====================
def configure_tesseract_path():
    lokal_tesseract = os.path.join(OCR_DIR, 'tesseract.exe')
    if os.path.exists(lokal_tesseract):
        pytesseract.pytesseract.tesseract_cmd = lokal_tesseract
    else:
        pytesseract.pytesseract.tesseract_cmd = 'tesseract'

configure_tesseract_path()


# ======================= Sprach- und Statistik-Funktionen ====================
def get_csv_datei():
    if aktuelle_sprache is None:
        raise ValueError("Keine Sprache ausgewählt (aktuelle_sprache ist None)")
    return os.path.join(VOCAB_DIR, f'vokabeln_{aktuelle_sprache}.csv')

def get_statistik_datei():
    if aktuelle_sprache is None:
        raise ValueError("Keine Sprache ausgewählt (aktuelle_sprache ist None)")
    return os.path.join(STAT_DIR, f'statistik_{aktuelle_sprache}.json')

def vorhandene_sprachen():
    files = [f for f in os.listdir(VOCAB_DIR) if f.startswith("vokabeln_") and f.endswith(".csv")]
    return [f[len("vokabeln_"):-len(".csv")] for f in files]

def initialisiere_sprache(sprache):
    global aktuelle_sprache, vokabel_statistik, alle_vokabeln, vokabeln_zu_lernen
    aktuelle_sprache = sprache.lower()

    # CSV-Datei anlegen falls nicht vorhanden:
    csv_path = get_csv_datei()
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Deutsch', 'Englisch'], delimiter=';')
            writer.writeheader()

    # Statistik-Datei anlegen falls nicht vorhanden:
    stat_path = get_statistik_datei()
    if not os.path.exists(stat_path):
        with open(stat_path, 'w', encoding='utf-8') as f:
            json.dump({}, f)

    vokabel_statistik = statistik_laden()
    lade_vokabeln()
    statistik_bereinigen()
    # Startscreen-Button-Zugriff aktualisieren (falls Screen bereits existiert)
    try:
        update_start_choice_access()
    except Exception:
        pass
    
    # Fenstertitel aktualisieren
    update_fenstertitel()
    
    # Sprachanzeige aktualisieren falls vorhanden
    update_sprachanzeige()

# ========================= Datei-I/O & Statistik =============================
def statistik_laden():
    stat_datei = get_statistik_datei()
    if not os.path.exists(stat_datei):
        return {}
    with open(stat_datei, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return {tuple(k.split('|')): v for k, v in data.items()}

def statistik_speichern():
    export = {f"{de}|{en}": werte for (de,en), werte in vokabel_statistik.items()}
    with open(get_statistik_datei(), 'w', encoding='utf-8') as f:
        json.dump(export, f, indent=4, ensure_ascii=False)

def statistik_bereinigen():
    gueltig = {(v['Deutsch'], v['Englisch']) for v in alle_vokabeln}
    for k in list(vokabel_statistik):
        if k not in gueltig:
            del vokabel_statistik[k]
    statistik_speichern()

# ======================= XP-System & Firebase ================================

XP_DATEI   = os.path.join(STAT_DIR, 'xp.json')
_xp_lock   = threading.RLock()
USER_DATEI = os.path.join(STAT_DIR, 'user.json')
CRED_DATEI = os.path.join(APP_DIR,  'firebase_credentials.json')


def init_firebase():
    """Initialisiert Firebase Admin SDK. Bei Fehler läuft die App im Offline-Modus weiter."""
    global firebase_db
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore
        if not os.path.exists(CRED_DATEI):
            print("[Firebase] Keine Credentials-Datei gefunden → Offline-Modus.")
            return
        cred = credentials.Certificate(CRED_DATEI)
        if not firebase_admin._apps:
            firebase_admin.initialize_app(cred)
        firebase_db = firestore.client()
        print("[Firebase] Verbunden.")
    except Exception as e:
        print(f"[Firebase] Fehler: {e} → Offline-Modus.")
        firebase_db = None


def _lade_user_data() -> dict:
    """Liest stats/user.json als Dict. Format: {active, users:[]}."""
    try:
        if os.path.exists(USER_DATEI):
            with open(USER_DATEI, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # Altes Format {username: ...} migrieren
            if 'username' in data and 'active' not in data:
                name = data['username']
                return {'active': name, 'users': [name]}
            return data
    except Exception:
        pass
    return {'active': None, 'users': []}


def _speichere_user_data(data: dict):
    """Schreibt stats/user.json."""
    try:
        with open(USER_DATEI, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[User] Speichern fehlgeschlagen: {e}")


def lade_benutzernamen() -> str | None:
    """Gibt den zuletzt aktiven Benutzernamen zurück."""
    return _lade_user_data().get('active')


def speichere_benutzernamen(name: str):
    """Setzt name als aktiven Benutzer und fügt ihn der Known-Users-Liste hinzu."""
    data = _lade_user_data()
    data['active'] = name
    if name not in data.get('users', []):
        data.setdefault('users', []).append(name)
    _speichere_user_data(data)


# ── PIN-System ─────────────────────────────────────────────────────────────

def _hash_pin(pin: str) -> str:
    """Gibt SHA-256-Hash des PINs zurück."""
    return hashlib.sha256(pin.encode('utf-8')).hexdigest()


def _hat_pin(name: str) -> bool | None:
    """Prüft in Firebase, ob der Nutzer einen PIN gesetzt hat.
    Gibt True/False zurück, oder None wenn Firebase nicht erreichbar ist."""
    if not firebase_db:
        return None
    try:
        doc = firebase_db.collection('users').document(name).get()
        if doc.exists:
            return bool(doc.to_dict().get('pin_hash'))
        return False
    except Exception as e:
        print(f"[PIN] Prüfung fehlgeschlagen: {e}")
        return None


def _verify_pin(name: str, pin: str) -> bool | None:
    """Prüft, ob der PIN korrekt ist. Gibt True/False oder None (Firebase-Fehler) zurück."""
    if not firebase_db:
        return None
    try:
        doc = firebase_db.collection('users').document(name).get()
        if doc.exists:
            gespeicherter_hash = doc.to_dict().get('pin_hash', '')
            return gespeicherter_hash == _hash_pin(pin)
        return False
    except Exception as e:
        print(f"[PIN] Verifizierung fehlgeschlagen: {e}")
        return None


def _set_pin_firebase(name: str, pin: str) -> bool:
    """Speichert den PIN-Hash in Firebase. Gibt True bei Erfolg zurück."""
    if not firebase_db:
        return False
    try:
        firebase_db.collection('users').document(name).set(
            {'pin_hash': _hash_pin(pin)}, merge=True
        )
        return True
    except Exception as e:
        print(f"[PIN] Speichern fehlgeschlagen: {e}")
        return False


def pin_erstellen_dialog(name: str, callback, closable=False):
    """Zeigt Dialog zum Erstellen eines 4-stelligen PINs.
    callback() wird nach erfolgreichem Setzen aufgerufen.
    Wenn closable=False, kann der Dialog nicht geschlossen werden."""
    win = ctk.CTkToplevel(app)
    win.title("PIN festlegen")
    win.geometry("400x380")
    win.resizable(False, False)

    if not closable:
        win.protocol("WM_DELETE_WINDOW", lambda: None)

    # ── Titel ──────────────────────────────────────────────────────────────
    ctk.CTkLabel(
        win, text="🔒 PIN festlegen",
        font=('Segoe UI', 22, 'bold')
    ).pack(pady=(28, 6))
    ctk.CTkLabel(
        win,
        text=f"Lege einen 4-stelligen PIN für\n\"{name}\" fest.\nDer PIN kann nicht zurückgesetzt werden!",
        font=('Segoe UI', 12), text_color=LIGHT_TEXT, justify='center'
    ).pack(pady=(0, 14))

    # ── Eingabefelder ─────────────────────────────────────────────────────
    fields_frame = ctk.CTkFrame(win, fg_color="transparent")
    fields_frame.pack(fill='x', padx=28)

    pin_entry1 = ctk.CTkEntry(
        fields_frame, placeholder_text="PIN eingeben (4 Ziffern)",
        height=42, font=('Segoe UI', 15), corner_radius=10,
        show="●"
    )
    pin_entry1.pack(fill='x', pady=(0, 8))

    pin_entry2 = ctk.CTkEntry(
        fields_frame, placeholder_text="PIN bestätigen",
        height=42, font=('Segoe UI', 15), corner_radius=10,
        show="●"
    )
    pin_entry2.pack(fill='x')

    # Eingabe auf Ziffern + max 4 beschränken
    def _limit_input(entry):
        val = entry.get()
        filtered = ''.join(c for c in val if c.isdigit())[:4]
        if val != filtered:
            entry.delete(0, 'end')
            entry.insert(0, filtered)
    pin_entry1.bind('<KeyRelease>', lambda e: _limit_input(pin_entry1))
    pin_entry2.bind('<KeyRelease>', lambda e: _limit_input(pin_entry2))

    # ── Fehlermeldung ─────────────────────────────────────────────────────
    fehler_label = ctk.CTkLabel(
        win, text="", font=('Segoe UI', 12),
        text_color=ERROR_COLOR
    )
    fehler_label.pack(pady=(6, 0))

    # ── Buttons ───────────────────────────────────────────────────────────
    btn_frame = ctk.CTkFrame(win, fg_color="transparent")
    btn_frame.pack(fill='x', padx=28, pady=(10, 20))

    def _bestaetigen():
        pin1 = pin_entry1.get().strip()
        pin2 = pin_entry2.get().strip()
        if len(pin1) != 4 or not pin1.isdigit():
            fehler_label.configure(text="Der PIN muss genau 4 Ziffern haben.")
            return
        if pin1 != pin2:
            fehler_label.configure(text="Die PINs stimmen nicht überein.")
            pin_entry2.delete(0, 'end')
            pin_entry2.focus_set()
            return
        if not firebase_db:
            fehler_label.configure(text="Internetverbindung erforderlich.")
            return
        # PIN speichern
        erfolg = _set_pin_firebase(name, pin1)
        if erfolg:
            win.destroy()
            callback()
        else:
            fehler_label.configure(text="Fehler beim Speichern. Prüfe deine\nInternetverbindung.")

    ctk.CTkButton(
        btn_frame, text="PIN festlegen",
        fg_color=BTN_COLOR, hover_color=BTN_HOVER_COLOR,
        height=42, corner_radius=10, font=('Segoe UI', 14, 'bold'),
        command=_bestaetigen
    ).pack(fill='x', pady=(0, 6))

    if closable:
        ctk.CTkButton(
            btn_frame, text="Abbrechen",
            fg_color="transparent", hover_color="#f3f4f6",
            height=36, corner_radius=10, font=('Segoe UI', 13),
            border_width=1, border_color=DISABLED_COLOR,
            text_color=TEXT_COLOR,
            command=win.destroy
        ).pack(fill='x')

    win.bind('<Return>', lambda e: _bestaetigen())

    # Focus verzögert setzen, damit das Fenster vollständig gerendert ist
    def _delayed_focus():
        try:
            win.grab_set()
            win.focus_set()
            pin_entry1.focus_set()
        except Exception:
            pass
    win.after(150, _delayed_focus)


def pin_eingabe_dialog(name: str, callback):
    """Zeigt Dialog zur Eingabe des 4-stelligen PINs.
    callback() wird bei korrektem PIN aufgerufen."""
    win = ctk.CTkToplevel(app)
    win.title("PIN eingeben")
    win.geometry("400x300")
    win.resizable(False, False)

    # ── Titel ──────────────────────────────────────────────────────────────
    ctk.CTkLabel(
        win, text="🔒 PIN eingeben",
        font=('Segoe UI', 22, 'bold')
    ).pack(pady=(28, 6))
    ctk.CTkLabel(
        win,
        text=f"Gib den PIN für \"{name}\" ein:",
        font=('Segoe UI', 12), text_color=LIGHT_TEXT, justify='center'
    ).pack(pady=(0, 14))

    # ── Eingabefeld ───────────────────────────────────────────────────────
    entry_frame = ctk.CTkFrame(win, fg_color="transparent")
    entry_frame.pack(fill='x', padx=28)

    pin_entry = ctk.CTkEntry(
        entry_frame, placeholder_text="4-stelliger PIN",
        height=42, font=('Segoe UI', 15), corner_radius=10,
        show="●"
    )
    pin_entry.pack(fill='x')

    # Eingabe auf Ziffern + max 4 beschränken
    def _limit_input(event=None):
        val = pin_entry.get()
        filtered = ''.join(c for c in val if c.isdigit())[:4]
        if val != filtered:
            pin_entry.delete(0, 'end')
            pin_entry.insert(0, filtered)
    pin_entry.bind('<KeyRelease>', _limit_input)

    # ── Fehlermeldung ─────────────────────────────────────────────────────
    fehler_label = ctk.CTkLabel(
        win, text="", font=('Segoe UI', 12),
        text_color=ERROR_COLOR
    )
    fehler_label.pack(pady=(6, 0))

    # ── Buttons ───────────────────────────────────────────────────────────
    btn_frame = ctk.CTkFrame(win, fg_color="transparent")
    btn_frame.pack(fill='x', padx=28, pady=(10, 20))

    def _pruefen():
        pin = pin_entry.get().strip()
        if len(pin) != 4 or not pin.isdigit():
            fehler_label.configure(text="Bitte 4 Ziffern eingeben.")
            return
        if not firebase_db:
            fehler_label.configure(text="Internetverbindung erforderlich.")
            return
        ergebnis = _verify_pin(name, pin)
        if ergebnis is True:
            win.destroy()
            callback()
        elif ergebnis is False:
            fehler_label.configure(text="Falscher PIN.")
            pin_entry.delete(0, 'end')
            pin_entry.focus_set()
        else:
            fehler_label.configure(text="Verbindungsfehler. Versuche es erneut.")

    ctk.CTkButton(
        btn_frame, text="Bestätigen",
        fg_color=BTN_COLOR, hover_color=BTN_HOVER_COLOR,
        height=42, corner_radius=10, font=('Segoe UI', 14, 'bold'),
        command=_pruefen
    ).pack(fill='x', pady=(0, 6))

    ctk.CTkButton(
        btn_frame, text="Abbrechen",
        fg_color="transparent", hover_color="#f3f4f6",
        height=36, corner_radius=10, font=('Segoe UI', 13),
        border_width=1, border_color=DISABLED_COLOR,
        text_color=TEXT_COLOR,
        command=win.destroy
    ).pack(fill='x')

    win.bind('<Return>', lambda e: _pruefen())
    # Focus verzögert setzen, damit das Fenster vollständig gerendert ist
    def _delayed_focus():
        try:
            win.grab_set()
            win.focus_set()
            pin_entry.focus_set()
        except Exception:
            pass
    win.after(150, _delayed_focus)


# ── Ende PIN-System ───────────────────────────────────────────────────────


def frage_benutzernamen():
    """Zeigt Eingabe-Dialog für den Benutzernamen beim ersten Start."""
    global aktueller_nutzer
    name = "Anonym"
    try:
        dialog = ctk.CTkInputDialog(
            text="Bitte gib deinen Benutzernamen ein.\nEr erscheint später in der Rangliste:",
            title="Benutzername"
        )
        eingabe_name = dialog.get_input()
        name = (eingabe_name or "").strip() or "Anonym"
        # Validierung: Firebase-problematische Zeichen entfernen
        name = re.sub(r'[/\\.\[\]#$]', '', name).strip() or "Anonym"
        name = name[:30]
    except Exception as e:
        print(f"[User] Dialog fehlgeschlagen: {e}")
    aktueller_nutzer = name
    speichere_benutzernamen(name)
    # XP-Eintrag für diesen Nutzer anlegen falls noch nicht vorhanden
    _init_xp_fuer_nutzer(name)


def _xp_current_week() -> str:
    """Gibt die aktuelle Wochen-ID zurück (ISO Jahr-Woche).
    Ab Sonntag 20:00 Uhr (Europe/Berlin) zählt bereits die neue Woche.
    """
    try:
        now = datetime.now(ZoneInfo('Europe/Berlin'))
    except Exception:
        now = datetime.now()
    # Sonntag (weekday==6) ab 20:00 → zählt als neue Woche
    if now.weekday() == 6 and now.hour >= 20:
        now = now + timedelta(hours=4)  # Kippt in den Montag → neue ISO-Woche
    iso = now.isocalendar()
    return f"{iso[0]}-{iso[1]:02d}"


def _previous_week_id() -> str:
    """Gibt die Wochen-ID der vorherigen Woche zurück (vor dem aktuellen _xp_current_week)."""
    try:
        now = datetime.now(ZoneInfo('Europe/Berlin'))
    except Exception:
        now = datetime.now()
    if now.weekday() == 6 and now.hour >= 20:
        now = now + timedelta(hours=4)
    prev = now - timedelta(days=7)
    iso = prev.isocalendar()
    return f"{iso[0]}-{iso[1]:02d}"


def _lade_alle_xp() -> dict:
    """Liest xp.json als Multi-User-Dict. Format: {username: {week_xp, total_xp, last_reset}}."""
    with _xp_lock:
        try:
            if os.path.exists(XP_DATEI):
                with open(XP_DATEI, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                # Altes Einzelnutzer-Format migrieren
                if 'username' in data and isinstance(data.get('week_xp'), int):
                    name = data.get('username', 'Anonym')
                    return {name: {
                        'week_xp':    data.get('week_xp', 0),
                        'total_xp':   data.get('total_xp', 0),
                        'last_reset': data.get('last_reset', ''),
                    }}
                return data
        except Exception:
            pass
        return {}


def _speichere_alle_xp(alle: dict):
    """Schreibt Multi-User-Dict in xp.json."""
    with _xp_lock:
        try:
            with open(XP_DATEI, 'w', encoding='utf-8') as f:
                json.dump(alle, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[XP] Speichern fehlgeschlagen: {e}")


def _init_xp_fuer_nutzer(name: str):
    """Legt einen leeren XP-Eintrag für name an, falls noch nicht vorhanden."""
    alle = _lade_alle_xp()
    if name not in alle:
        alle[name] = {'week_xp': 0, 'total_xp': 0, 'last_reset': _xp_current_week()}
        _speichere_alle_xp(alle)


def _fuehre_globalen_weekly_reset_durch():
    """Erstellt einen Snapshot der Rangliste der letzten Woche in Firebase.
    Wird einmalig vom ersten User nach Sonntag 20:00 getriggert (Lazy Reset).
    Schreibt Ergebnisse nach weekly_results/latest und setzt meta/last_reset.
    """
    if not firebase_db:
        return
    current_week = _xp_current_week()
    prev_week = _previous_week_id()
    try:
        # Prüfen ob der globale Reset schon durchgeführt wurde
        meta_ref = firebase_db.collection('meta').document('last_reset')
        meta_doc = meta_ref.get()
        if meta_doc.exists and meta_doc.to_dict().get('week_id', '') == current_week:
            return  # Bereits durch einen anderen User erledigt

        # Alle User aus Firebase laden
        docs = firebase_db.collection('users').stream()
        alle_spieler = []
        for doc in docs:
            d = doc.to_dict()
            name = doc.id
            wk_xp = d.get('week_xp', 0)
            # Nur User die in der vorherigen Woche aktiv waren (>0 XP)
            # und deren last_reset noch der alten Woche entspricht
            if wk_xp > 0:
                alle_spieler.append({
                    'username':  name,
                    'final_xp':  wk_xp,
                    'total_xp':  d.get('total_xp', 0),
                })

        if not alle_spieler:
            # Keine Aktivität → nur Meta-Flag setzen
            meta_ref.set({'week_id': current_week})
            return

        # Sortieren: primär week_xp desc, sekundär total_xp desc
        alle_spieler.sort(key=lambda x: (x['final_xp'], x['total_xp']), reverse=True)

        # Ränge vergeben
        ergebnisse = []
        for rang, sp in enumerate(alle_spieler, start=1):
            ergebnisse.append({
                'username':   sp['username'],
                'final_xp':   sp['final_xp'],
                'final_rank':  rang,
            })

        # Ergebnisse in weekly_results/latest schreiben (überschreibt alte Daten)
        firebase_db.collection('weekly_results').document('latest').set({
            'week_id':            prev_week,
            'results':            ergebnisse,
            'total_participants': len(ergebnisse),
            'created_at':         datetime.now().isoformat(),
        })

        # Meta-Flag setzen → verhindert doppelte Ausführung
        meta_ref.set({'week_id': current_week})
        print(f"[Weekly Reset] Globaler Snapshot für Woche {prev_week} erstellt "
              f"({len(ergebnisse)} Teilnehmer).")

    except Exception as e:
        print(f"[Weekly Reset] Globaler Reset fehlgeschlagen: {e}")


def pruefe_weekly_reset():
    """Setzt week_xp des aktiven Nutzers zurück, wenn die Woche gewechselt hat.
    Führt vorher den globalen Ranglisten-Snapshot durch (Lazy Reset).
    """
    global xp_woche
    if not aktueller_nutzer:
        return
    current_week = _xp_current_week()
    try:
        alle = _lade_alle_xp()
        eintrag = alle.get(aktueller_nutzer, {})
        if eintrag.get('last_reset', '') != current_week:
            # Globalen Snapshot erstellen bevor der eigene XP-Wert resettet wird
            _fuehre_globalen_weekly_reset_durch()
            xp_woche = 0
            speichere_xp()
    except Exception as e:
        print(f"[XP] Weekly-Reset-Prüfung fehlgeschlagen: {e}")


def pruefe_siegerehrung() -> dict | None:
    """Prüft ob eine ungesehene Siegerehrung für den aktuellen Nutzer vorliegt.
    Gibt die Ergebnisdaten zurück oder None.
    """
    if not firebase_db or not aktueller_nutzer:
        return None
    try:
        # Eigenes User-Dokument lesen: ceremony_seen_week
        user_doc = firebase_db.collection('users').document(aktueller_nutzer).get()
        seen_week = ''
        if user_doc.exists:
            seen_week = user_doc.to_dict().get('ceremony_seen_week', '')

        # Letzte Wochen-Ergebnisse laden
        result_doc = firebase_db.collection('weekly_results').document('latest').get()
        if not result_doc.exists:
            return None
        data = result_doc.to_dict()
        result_week = data.get('week_id', '')

        # Bereits gesehen?
        if seen_week == result_week:
            return None

        # Eigene Platzierung finden
        results = data.get('results', [])
        eigene_platzierung = None
        for entry in results:
            if entry.get('username') == aktueller_nutzer:
                eigene_platzierung = entry
                break

        if eigene_platzierung is None:
            # User war nicht aktiv in der Woche → keine Siegerehrung,
            # aber als gesehen markieren damit nicht jedes Mal geprüft wird
            try:
                firebase_db.collection('users').document(aktueller_nutzer).set(
                    {'ceremony_seen_week': result_week}, merge=True
                )
            except Exception:
                pass
            return None

        # Top 3 extrahieren
        top3 = [e for e in results if e.get('final_rank', 99) <= 3]

        return {
            'week_id':            result_week,
            'your_rank':          eigene_platzierung['final_rank'],
            'your_xp':            eigene_platzierung['final_xp'],
            'total_participants': data.get('total_participants', len(results)),
            'top3':               top3,
        }
    except Exception as e:
        print(f"[Siegerehrung] Prüfung fehlgeschlagen: {e}")
        return None


def _markiere_siegerehrung_gesehen(week_id: str):
    """Markiert die Siegerehrung als gesehen (async in Firebase)."""
    if not firebase_db or not aktueller_nutzer:
        return
    def _save():
        try:
            firebase_db.collection('users').document(aktueller_nutzer).set(
                {'ceremony_seen_week': week_id}, merge=True
            )
        except Exception as e:
            print(f"[Siegerehrung] Gesehen-Markierung fehlgeschlagen: {e}")
    threading.Thread(target=_save, daemon=True).start()


def siegerehrung_anzeigen(daten: dict, callback=None):
    """Zeigt einen modalen Siegerehrungs-Dialog mit Podium und eigener Platzierung.

    Args:
        daten: Dict von pruefe_siegerehrung() mit week_id, your_rank, your_xp,
               total_participants, top3.
        callback: Optionale Funktion die nach dem Schließen aufgerufen wird.
    """
    if not app:
        return

    GOLD   = "#FFD700"
    SILVER = "#C0C0C0"
    BRONZE = "#CD7F32"
    MEDALS = {1: "🥇", 2: "🥈", 3: "🥉"}
    RANK_COLORS = {1: GOLD, 2: SILVER, 3: BRONZE}
    PODIUM_HEIGHTS = {1: 130, 2: 95, 3: 70}

    win = ctk.CTkToplevel(app)
    win.title("Wochenergebnis")
    win.geometry("480x580")
    win.resizable(False, False)
    win.grab_set()
    win.focus_set()
    # Zentrieren
    try:
        win.after(10, lambda: win.geometry(
            f"+{app.winfo_x() + app.winfo_width() // 2 - 240}"
            f"+{app.winfo_y() + app.winfo_height() // 2 - 290}"
        ))
    except Exception:
        pass

    def _schliessen():
        _markiere_siegerehrung_gesehen(daten['week_id'])
        try:
            win.destroy()
        except Exception:
            pass
        if callback:
            callback()

    win.protocol("WM_DELETE_WINDOW", _schliessen)

    # ── Header ──────────────────────────────────────────────────────────
    ctk.CTkLabel(
        win, text="🏆  Wochenergebnis",
        font=('Segoe UI', 28, 'bold'), text_color=GOLD
    ).pack(pady=(28, 2))
    ctk.CTkLabel(
        win, text="Die Woche ist vorbei!",
        font=('Segoe UI', 14), text_color=LIGHT_TEXT
    ).pack(pady=(0, 18))

    # ── Podium (Top 3) ─────────────────────────────────────────────────
    top3 = sorted(daten.get('top3', []), key=lambda x: x.get('final_rank', 99))

    if top3:
        podium_frame = ctk.CTkFrame(win, fg_color="transparent")
        podium_frame.pack(fill='x', padx=30, pady=(0, 10))

        # Reihenfolge: Platz 2 links, Platz 1 mitte, Platz 3 rechts
        order = []
        platz2 = next((e for e in top3 if e['final_rank'] == 2), None)
        platz1 = next((e for e in top3 if e['final_rank'] == 1), None)
        platz3 = next((e for e in top3 if e['final_rank'] == 3), None)
        for entry in [platz2, platz1, platz3]:
            if entry:
                order.append(entry)

        podium_frame.grid_columnconfigure(tuple(range(len(order))), weight=1)

        for col, entry in enumerate(order):
            rank = entry['final_rank']
            cell = ctk.CTkFrame(podium_frame, fg_color="transparent")
            cell.grid(row=0, column=col, padx=6, sticky='s')

            # Username (gekürzt)
            display_name = entry['username']
            if len(display_name) > 12:
                display_name = display_name[:11] + "…"
            is_me = (entry['username'] == aktueller_nutzer)
            name_text = f"{display_name} (Du)" if is_me else display_name
            ctk.CTkLabel(
                cell, text=name_text,
                font=('Segoe UI', 12, 'bold') if is_me else ('Segoe UI', 12),
                text_color=BTN_COLOR if is_me else None,
            ).pack(pady=(0, 2))

            # XP
            ctk.CTkLabel(
                cell, text=f"{entry['final_xp']} XP",
                font=('Segoe UI', 11), text_color=LIGHT_TEXT
            ).pack(pady=(0, 4))

            # Podest-Block
            h = PODIUM_HEIGHTS.get(rank, 50)
            color = RANK_COLORS.get(rank, LIGHT_TEXT)
            podest = ctk.CTkFrame(cell, height=h, width=100, fg_color=color, corner_radius=10)
            podest.pack_propagate(False)
            podest.pack()
            ctk.CTkLabel(
                podest, text=MEDALS.get(rank, str(rank)),
                font=('Segoe UI', 30)
            ).pack(expand=True)

    # ── Eigene Platzierung ──────────────────────────────────────────────
    your_rank = daten['your_rank']
    your_xp   = daten['your_xp']
    total     = daten['total_participants']

    result_frame = ctk.CTkFrame(win, corner_radius=16, fg_color=("#dbeafe", "#1e3a5f"))
    result_frame.pack(fill='x', padx=40, pady=(18, 8))

    ctk.CTkLabel(
        result_frame, text="Dein Ergebnis",
        font=('Segoe UI', 13), text_color=LIGHT_TEXT
    ).pack(pady=(14, 2))

    emoji = MEDALS.get(your_rank, "🏅")
    rank_color = RANK_COLORS.get(your_rank, "#87CEEB")
    ctk.CTkLabel(
        result_frame, text=f"{emoji}  Platz {your_rank}",
        font=('Segoe UI', 36, 'bold'), text_color=rank_color
    ).pack(pady=(2, 2))

    ctk.CTkLabel(
        result_frame, text=f"{your_xp} XP diese Woche",
        font=('Segoe UI', 15), text_color=None
    ).pack(pady=(0, 2))

    ctk.CTkLabel(
        result_frame, text=f"von {total} Teilnehmern",
        font=('Segoe UI', 12), text_color=LIGHT_TEXT
    ).pack(pady=(0, 14))

    # ── Weiter-Button ───────────────────────────────────────────────────
    ctk.CTkButton(
        win, text="Weiter  🚀", height=48, corner_radius=14,
        fg_color=GOLD, hover_color="#e6c200", text_color="#1a1a2e",
        font=('Segoe UI', 16, 'bold'),
        command=_schliessen
    ).pack(fill='x', padx=60, pady=(16, 24))


def lade_xp():
    """Lädt XP des aktiven Nutzers; synchronisiert mit Firebase (nimmt höheren Wert)."""
    global xp_woche, xp_gesamt
    if not aktueller_nutzer:
        xp_woche = xp_gesamt = 0
        return
    current_week = _xp_current_week()
    # Lokal lesen
    try:
        alle = _lade_alle_xp()
        eintrag = alle.get(aktueller_nutzer, {})
        if eintrag.get('last_reset', '') == current_week:
            xp_woche = eintrag.get('week_xp', 0)
        else:
            xp_woche = 0
        xp_gesamt = eintrag.get('total_xp', 0)
    except Exception:
        xp_woche = 0
        xp_gesamt = 0
    # Firebase synchronisieren (höheren Wert nehmen)
    if firebase_db and aktueller_nutzer:
        try:
            doc = firebase_db.collection('users').document(aktueller_nutzer).get()
            if doc.exists:
                fb = doc.to_dict()
                if fb.get('last_reset', '') == current_week:
                    xp_woche = max(xp_woche, fb.get('week_xp', 0))
                xp_gesamt = max(xp_gesamt, fb.get('total_xp', 0))
        except Exception as e:
            print(f"[Firebase] Sync fehlgeschlagen: {e}")


def speichere_xp():
    """Speichert XP des aktiven Nutzers lokal (Multi-User) und asynchron in Firebase."""
    if not aktueller_nutzer:
        return
    current_week = _xp_current_week()
    # Lokal: nur den eigenen Eintrag aktualisieren, andere unberührt lassen
    try:
        alle = _lade_alle_xp()
        alle[aktueller_nutzer] = {
            'week_xp':    xp_woche,
            'total_xp':   xp_gesamt,
            'last_reset': current_week,
        }
        _speichere_alle_xp(alle)
    except Exception as e:
        print(f"[XP] Lokales Speichern fehlgeschlagen: {e}")
    # Firebase in Hintergrund-Thread (blockiert UI nicht)
    if firebase_db and aktueller_nutzer:
        def _save():
            try:
                firebase_db.collection('users').document(aktueller_nutzer).set({
                    'week_xp':    xp_woche,
                    'total_xp':   xp_gesamt,
                    'last_reset': current_week,
                    'updated_at': datetime.now().isoformat(),
                }, merge=True)
            except Exception as e:
                print(f"[Firebase] Speichern fehlgeschlagen: {e}")
        threading.Thread(target=_save, daemon=True).start()


def benutzer_wechseln():
    """Zeigt Dialog mit Eingabefeld zum Anmelden oder Erstellen eines Benutzers."""
    global aktueller_nutzer, xp_woche, xp_gesamt, xp_session

    # Dialog-Fenster
    win = ctk.CTkToplevel(app)
    win.title("Benutzer wechseln")
    win.geometry("420x340")
    win.resizable(False, False)
    win.grab_set()
    win.focus_set()

    # ── Titel ──────────────────────────────────────────────────────────────
    ctk.CTkLabel(
        win, text="Benutzer wechseln",
        font=('Segoe UI', 22, 'bold')
    ).pack(pady=(28, 6))
    ctk.CTkLabel(
        win,
        text="Gib den Benutzernamen ein, mit dem du dich\nanmelden oder den du neu erstellen möchtest:",
        font=('Segoe UI', 12), text_color=LIGHT_TEXT, justify='center'
    ).pack(pady=(0, 14))

    # ── Eingabefeld ────────────────────────────────────────────────────────
    entry_frame = ctk.CTkFrame(win, fg_color="transparent")
    entry_frame.pack(fill='x', padx=28)

    name_entry = ctk.CTkEntry(
        entry_frame, placeholder_text="Benutzername...",
        height=42, font=('Segoe UI', 15), corner_radius=10
    )
    name_entry.pack(fill='x')

    # ── Fehlermeldung (zunächst unsichtbar) ────────────────────────────────
    fehler_label = ctk.CTkLabel(
        win, text="", font=('Segoe UI', 12),
        text_color=ERROR_COLOR
    )
    fehler_label.pack(pady=(4, 0))

    # ── Buttons ────────────────────────────────────────────────────────────
    btn_frame = ctk.CTkFrame(win, fg_color="transparent")
    btn_frame.pack(fill='x', padx=28, pady=(10, 20))

    def _wechsle_zu(name: str):
        """Nutzer aktivieren, Daten laden, Dialog schließen."""
        global aktueller_nutzer, xp_session
        aktueller_nutzer = name
        speichere_benutzernamen(name)
        xp_session = 0
        pruefe_weekly_reset()
        lade_xp()
        update_xp_label()
        win.destroy()
        # Siegerehrung prüfen
        ceremony = pruefe_siegerehrung()
        if ceremony:
            siegerehrung_anzeigen(ceremony, callback=startbildschirm)
        else:
            startbildschirm()

    def _wechsle_zu_nach_pin(name: str):
        """Nutzer aktivieren nach PIN-Prüfung (Wechsel-Dialog ist bereits geschlossen)."""
        global aktueller_nutzer, xp_session
        aktueller_nutzer = name
        speichere_benutzernamen(name)
        xp_session = 0
        pruefe_weekly_reset()
        lade_xp()
        update_xp_label()
        # Siegerehrung prüfen
        ceremony = pruefe_siegerehrung()
        if ceremony:
            siegerehrung_anzeigen(ceremony, callback=startbildschirm)
        else:
            startbildschirm()

    def _nutzer_existiert_firebase(name: str) -> bool:
        """Prüft synchron, ob der Nutzer in Firebase existiert."""
        if not firebase_db:
            return False
        try:
            doc = firebase_db.collection('users').document(name).get()
            return doc.exists
        except Exception:
            return False

    def _weiter():
        name = name_entry.get().strip()
        if not name:
            fehler_label.configure(text="Bitte einen Benutzernamen eingeben.")
            return
        # Validierung: Firebase-problematische Zeichen entfernen
        name = re.sub(r'[/\\.\[\]#$]', '', name).strip()
        if not name:
            fehler_label.configure(text="Ungültiger Name (enthält nur Sonderzeichen).")
            return
        name = name[:30]
        fehler_label.configure(text="")

        # Bekannt lokal?
        bekannte_nutzer = _lade_user_data().get('users', [])
        lokal_bekannt = name in bekannte_nutzer

        # Bekannt in Firebase?
        firebase_bekannt = _nutzer_existiert_firebase(name)

        if lokal_bekannt or firebase_bekannt:
            # Nutzer existiert → PIN prüfen
            if not firebase_db:
                fehler_label.configure(text="Internetverbindung erforderlich,\num den Account zu wechseln.")
                return
            hat_pin_status = _hat_pin(name)
            if hat_pin_status is None:
                fehler_label.configure(text="Verbindungsfehler. Versuche es erneut.")
                return
            if hat_pin_status:
                # PIN eingeben
                win.destroy()
                pin_eingabe_dialog(name, callback=lambda: _wechsle_zu_nach_pin(name))
            else:
                # Kein PIN → erst erstellen, dann einloggen
                win.destroy()
                pin_erstellen_dialog(name, callback=lambda: _wechsle_zu_nach_pin(name), closable=False)
        else:
            # Nutzer unbekannt → Rückfrage
            _zeige_erstellen_frage(name)

    def _zeige_erstellen_frage(name: str):
        """Blendet Eingabe-UI aus und zeigt Ja/Nein-Rückfrage."""
        # Eingabe sperren
        name_entry.configure(state='disabled')
        weiter_btn.configure(state='disabled')
        # Fenster etwas größer machen für die 2-zeilige Meldung + Buttons
        win.geometry("420x370")

        fehler_label.configure(
            text=f'"{name}" existiert noch nicht.\nMöchtest du diesen Benutzer erstellen?',
            text_color=WARNING_COLOR
        )
        # Buttons tauschen
        weiter_btn.pack_forget()
        abbr_btn.pack_forget()

        ja_btn = ctk.CTkButton(
            btn_frame, text="Ja, erstellen",
            fg_color=SPRACH_COLOR, hover_color=SUCCESS_COLOR,
            height=42, corner_radius=10, font=('Segoe UI', 14, 'bold'),
            command=lambda: _erstelle_und_wechsle(name)
        )
        ja_btn.pack(fill='x', pady=(0, 6))

        nein_btn = ctk.CTkButton(
            btn_frame, text="Nein, zurück",
            fg_color="transparent", hover_color="#f3f4f6",
            height=38, corner_radius=10, font=('Segoe UI', 13),
            border_width=1, border_color=DISABLED_COLOR,
            text_color=TEXT_COLOR,
            command=lambda: _zurueck(ja_btn, nein_btn)
        )
        nein_btn.pack(fill='x')

    def _erstelle_und_wechsle(name: str):
        _init_xp_fuer_nutzer(name)
        # Neuen Nutzer sofort in Firebase anlegen (ohne PIN – der kommt gleich)
        if firebase_db:
            try:
                firebase_db.collection('users').document(name).set({
                    'week_xp':    0,
                    'total_xp':   0,
                    'last_reset': _xp_current_week(),
                    'created_at': datetime.now().isoformat(),
                    'updated_at': datetime.now().isoformat(),
                }, merge=True)
            except Exception as e:
                print(f"[Firebase] Neuer Nutzer konnte nicht gespeichert werden: {e}")
        # Dialog schließen, dann PIN erstellen lassen
        win.destroy()
        if firebase_db:
            pin_erstellen_dialog(name, callback=lambda: _wechsle_zu_nach_pin(name), closable=False)
        else:
            # Offline: direkt einloggen, PIN wird beim nächsten Online-Start erzwungen
            _wechsle_zu_nach_pin(name)

    def _zurueck(ja_btn, nein_btn):
        """Zurück zur Eingabe."""
        ja_btn.destroy()
        nein_btn.destroy()
        win.geometry("420x340")
        name_entry.configure(state='normal')
        name_entry.delete(0, 'end')
        fehler_label.configure(text="", text_color=ERROR_COLOR)
        weiter_btn.configure(state='normal')
        weiter_btn.pack(fill='x', pady=(0, 6))
        abbr_btn.pack(fill='x')
        name_entry.focus_set()

    weiter_btn = ctk.CTkButton(
        btn_frame, text="Weiter",
        fg_color=BTN_COLOR, hover_color=BTN_HOVER_COLOR,
        height=42, corner_radius=10, font=('Segoe UI', 14, 'bold'),
        command=_weiter
    )
    weiter_btn.pack(fill='x', pady=(0, 6))

    abbr_btn = ctk.CTkButton(
        btn_frame, text="Abbrechen",
        fg_color="transparent", hover_color="#f3f4f6",
        height=36, corner_radius=10, font=('Segoe UI', 13),
        border_width=1, border_color=DISABLED_COLOR,
        text_color=TEXT_COLOR,
        command=win.destroy
    )
    abbr_btn.pack(fill='x')

    # Enter-Taste bestätigt direkt
    win.bind('<Return>', lambda e: _weiter())
    name_entry.focus_set()


def update_xp_label():
    """Aktualisiert den XP-Label im Startbildschirm."""
    try:
        if xp_start_label is not None and xp_start_label.winfo_exists():
            xp_start_label.configure(text=f"XP diese Woche:  {xp_woche}")
    except Exception:
        pass


def lade_vokabeln():
    global alle_vokabeln, vokabeln_zu_lernen
    alle_vokabeln.clear()

    if not os.path.exists(get_csv_datei()):

        messagebox.showerror(
            "Datei nicht gefunden",
            f"Die Datei {get_csv_datei()} existiert nicht."
        )
        return

    with open(get_csv_datei(), 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Defensive Prüfung: Datei leer oder nur Header?
    if len(lines) <= 1:
        # CSV existiert, aber keine Datenzeilen => einfach leer initialisieren
        vokabeln_zu_lernen = []
        alle_vokabeln = []
        return

    # Versuche CSV-Dialekt zu erkennen
    sample = ''.join(lines[:5])
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=';,')
    except csv.Error:
        # Notfall: Fallback auf Standard-Dialekt (; als Trennzeichen)
        dialect = csv.excel
        dialect.delimiter = ';'

    reader = csv.DictReader(lines, dialect=dialect)
    for row in reader:
        try:
            d = row['Deutsch'].strip()
            e = row['Englisch'].strip()
        except KeyError:
            messagebox.showerror(
                "CSV-Fehler",
                f"In der CSV-Datei fehlen die Spalten 'Deutsch' und 'Englisch' "
                f"(gefunden: {reader.fieldnames})."
            )
            return
        alle_vokabeln.append({'Deutsch': d, 'Englisch': e})
        vokabel_statistik.setdefault((d, e), {'richtig': 0, 'falsch': 0})

    vokabeln_zu_lernen = alle_vokabeln.copy()
    random.shuffle(vokabeln_zu_lernen)
    # Nach dem Laden: Startscreen-Button-Zugriff aktualisieren
    try:
        update_start_choice_access()
    except Exception:
        pass


def save_vokabeln_csv():
    with open(get_csv_datei(), 'w', newline='', encoding='utf-8') as f:

        writer = csv.DictWriter(f, fieldnames=['Deutsch', 'Englisch'], delimiter=';')
        writer.writeheader()
        for v in alle_vokabeln:
            writer.writerow({'Deutsch': v['Deutsch'], 'Englisch': v['Englisch']})
    lade_vokabeln()
    statistik_bereinigen()
    try:
        update_start_choice_access()
    except Exception:
        pass

# Initial load - wird nach der Definition der UI-Funktionen aufgerufen


# ====================== OCR + GPT-Filter ======================================
def extract_pairs_with_gpt(raw_text: str) -> list[dict]:
    system = ("""
    Du bist ein Extraktions- und Normalisierungsmodul für einen schulischen Vokabeltrainer.

Der Eingabetext stammt aus einer OCR von Vokabellisten und enthält:
- Überschriften
- Beispielsätze
- Lautschrift
- Wortarten
- Klammern, Schrägstriche, Zusätze wie (r/s), (of), (Kindes-)
- Kommentare und sonstigen Müll

DEINE EINZIGE AUFGABE:
Extrahiere ausschließlich saubere, einfache und eindeutige Vokabelpaare.

VERBINDLICHE REGELN (kein Interpretationsspielraum):

1. Ausgabeformat:
   Deutsch;Fremdsprache
   – genau ein Semikolon
   – genau eine Zeile pro Paar
   – keine Überschriften, kein zusätzlicher Text

2. Reihenfolge ist ZWINGEND:
   Deutsch links, Fremdsprache rechts.
   Wenn die OCR etwas anderes nahelegt, korrigiere es.

3. Pro Wort genau EINE Übersetzung:
   - Wähle die gebräuchlichste und einfachste Bedeutung.
   - KEINE Synonyme
   - KEINE Schrägstriche
   - KEINE Kommas
   - KEINE Alternativen

4. Entferne konsequent ALLE Zusatzinformationen:
   - Alle Klammern: (), [], {}. Auch wenn sie wichtig erscheinen wie "fall in love (with sb.)" lass die klamemrn weg also nur "fall in love"
   - Geschlechtsmarker wie (r), (s), (e)
   - Wortartangaben
   - Zusätze wie „of“, „to“, „no pl“, Bindestrich-Erklärungen
   - Lautschrift
   - Präfix-Erklärungen wie „Ex-“ → **nur „ehemalig“**

   Beispiele:
   - „ehemalige(r, s)“ → „ehemalig“
   - „(Kindes-)Erziehung“ → „Erziehung“
   - „amount (of)“ → „Menge“
   - „to trust“ → „vertrauen“

5. Glätte die Sprache:
   - Bevorzuge Grundform und Alltagssprache
   - Keine Fach- oder Metaformen
   - Keine Dopplungen

6. OCR-Fehler korrigieren:
   - falsche Buchstaben
   - fehlende Leerzeichen
   - kaputte Sonderzeichen
   Aber: **Nichts erfinden**, nur eindeutig rekonstruierbare Wörter.

7. Wenn ein Eintrag nicht eindeutig ein Vokabelpaar ist:
   → komplett ignorieren.

BEISPIELE (verbindlich):

Eingabe: "Ex-; ehemalige(r, s)"
Ausgabe: "ehemalig;ex"

Eingabe: "amount (of); Betrag, Menge, Höhe"
Ausgabe: "Menge;amount"

"Eingabe: "to trust; trauen, vertrauen""
"Ausgabe: "vertrauen;to trust""
"
halte dich strikt an diese Regeln."""
)

    if not client:
        messagebox.showerror("Fehler", "Kein OpenAI Client initialisiert (API Key fehlt).")
        return []

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": raw_text}
            ]
        )
    except Exception as e:
        error_msg = f"Fehler bei der Anfrage an OpenAI:\n{e}"
        messagebox.showerror("API Fehler", error_msg)
        return []

    lines = resp.choices[0].message.content.strip().splitlines()
    pairs = []
    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split(';' , 1)
        if len(parts) != 2:
            print(f"⚠️ Ungültige Zeile übersprungen: {line!r}")
            continue

        deu, foreign = parts[0].strip(), parts[1].strip()

        # Hier wird direkt „umgedreht“:
        pairs.append({"Deutsch": deu, "Englisch": foreign})

    return pairs



def extract_pairs_from_image(path: str) -> list[dict]:
    # Bild öffnen
    img_raw = Image.open(path)
    
    # 0. EXIF-Daten (Rotation) anwenden und erzwingen als reines RGB
    # Verhindert Format-Bugs (wie versteckte Alphakanäle in PNGs oder komische JPEGs vom Handy)
    try:
        from PIL import ImageOps
        img_raw = ImageOps.exif_transpose(img_raw)
    except Exception:
        pass
    
    img = img_raw.convert('RGB')
    
    # --- PREPROCESSING FÜR HANDYBILDER ---
    # 1. Konvertierung in Graustufen
    img = img.convert('L')
    
    # 2. Skalierung bei extrem großen Handyfotos verhindern
    # (Tesseract tut sich oft schwer bei > 4000px Bildern)
    max_size = 2500
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # 3. Kontrast leicht erhöhen (hilft bei schwacher Beleuchtung)
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(1.5)
    # ---------------------------------------

    configure_tesseract_path()
    alle_sprachen = get_all_tesseract_langs()
    if not alle_sprachen:
        alle_sprachen = 'deu'
    raw = pytesseract.image_to_string(
        img,
        lang=alle_sprachen,
        config="--oem 1 --psm 6"
    )
    
    if not raw.strip():
        return []
        
    return extract_pairs_with_gpt(raw)


# ====================== Handy-Upload via Flask (QR-Code) =====================
qr_popup_window = None
temp_upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_uploads")
os.makedirs(temp_upload_dir, exist_ok=True)

flask_app = Flask(__name__)
flask_app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 10 MB max upload
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10 MB
# Reduziere Flask Logging output in der Konsole
import logging
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

def get_local_ip():
    """ Ermittelt die lokale IP-Adresse zuverlässig über einen UDP-Verbindungsversuch. """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Pingt keinen Inhalt, baut nur die Route auf
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

@flask_app.route('/')
def upload_form():
    html = '''
    <!doctype html>
    <html lang="de">
    <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=0">
      <title>Vokabel-Upload</title>
      <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; text-align: center; padding: 20px; background-color: #f9fafb; color: #1f2937; }
        h1 { font-size: 1.5rem; margin-bottom: 20px; }
        .upload-btn-wrapper { position: relative; overflow: hidden; display: inline-block; cursor: pointer; }
        .btn { border: none; background-color: #6366f1; color: white; padding: 15px 30px; font-size: 1.2rem; border-radius: 8px; font-weight: bold; cursor: pointer; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }
        .upload-btn-wrapper input[type=file] { padding: 40px; font-size: 100px; position: absolute; left: 0; top: 0; opacity: 0; cursor: pointer; height: 100%; width: 100%; }
        #loading { display: none; margin-top: 20px; font-weight: bold; color: #6366f1; }
      </style>
      <script>
        function showLoading() {
           document.getElementById('upload-area').style.display = 'none';
           document.getElementById('loading').style.display = 'block';
        }
      </script>
    </head>
    <body>
      <h1>📸 Buchseite abfotografieren</h1>
      <div id="upload-area">
          <form method="POST" action="/upload" enctype="multipart/form-data" onsubmit="showLoading()">
            <div class="upload-btn-wrapper">
              <button class="btn">Kamera oder Galerie</button>
              <input type="file" name="file" accept="image/*" required onchange="this.form.submit(); showLoading();" />
            </div>
            <p style="margin-top:10px; color:#6b7280; font-size:0.9rem;">Wähle, ob du ein neues<br>oder ein bestehendes Foto nutzen möchtest</p>
          </form>
      </div>
      <div id="loading">Bild wird gesendet...<br>⏳ Bitte kurz warten</div>
    </body>
    </html>
    '''
    return render_template_string(html)

@flask_app.route('/upload', methods=['POST'])
def handle_upload():
    if 'file' not in request.files:
        return "Kein Bild gesendet.", 400
    file = request.files['file']
    if file.filename == '':
        return "Keine Datei ausgewählt.", 400
    if file:
        # Dateigrößen-Check
        file.seek(0, 2)
        size = file.tell()
        file.seek(0)
        if size > MAX_UPLOAD_SIZE:
            return "Datei zu groß (max. 10 MB).", 400

        filename = secure_filename(file.filename)
        if not filename:
            filename = f"image_{int(time.time())}.jpg"

        # Dateityp-Check
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return "Ungültiger Dateityp. Nur Bilder erlaubt.", 400

        filepath = os.path.join(temp_upload_dir, filename)
        file.save(filepath)
        
        # UI-Update und OCR sicher im Main-Thread starten!
        app.after(100, lambda: process_mobile_upload(filepath))
        
        success_html = '''
        <!doctype html>
        <html lang="de">
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>Erfolg</title>
          <style>
              body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; text-align: center; padding: 40px 20px; background-color: #f0fdf4; color: #166534; }
              h1 { font-size: 2rem; margin-bottom: 20px; }
              p { font-size: 1.1rem; color: #15803d; }
          </style>
        </head>
        <body>
          <h1>Das hat geklappt!</h1>
          <p>Du kannst das Handy jetzt weglegen</p>
        </body>
        </html>
        '''
        return render_template_string(success_html)
    
    return "Fehler beim Verarbeiten des Bildes.", 400

def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

FLASK_PORT = get_free_port()

def start_flask_server():
    # Läuft im Hintergrund auf einem dynamischen Port (vermeidet 'Address already in use' auf macOS)
    flask_app.run(host='0.0.0.0', port=FLASK_PORT, debug=False, use_reloader=False)

def show_qr_popup():
    """ Zeigt den QR-Code als Toplevel Fenster an """
    global qr_popup_window
    if qr_popup_window is not None and qr_popup_window.winfo_exists():
        qr_popup_window.lift()
        return

    ip = get_local_ip()
    url = f"http://{ip}:{FLASK_PORT}"

    # Neues Toplevel Fenster
    qr_popup_window = ctk.CTkToplevel(app)
    qr_popup_window.title("Handy Upload")
    qr_popup_window.geometry("400x450")
    qr_popup_window.attributes("-topmost", True)
    qr_popup_window.focus()
    qr_popup_window.grab_set()  # Verhindert Klicks im Hauptfenster

    lbl_info = ctk.CTkLabel(qr_popup_window, text="Mit dem Handy scannen!", font=('Arial', 18, 'bold'))
    lbl_info.pack(pady=(20, 5))

    lbl_sub = ctk.CTkLabel(qr_popup_window, text=f"Stelle sicher,\ndass Mac und Handy\nim selben WLAN sind.", text_color=LIGHT_TEXT)
    lbl_sub.pack(pady=(0, 15))

    # QR Code generieren
    qr = qrcode.QRCode(version=1, box_size=8, border=2)
    qr.add_data(url)
    qr.make(fit=True)
    # Extrahiere das eigentliche PIL-Bild-Objekt
    pil_img_wrapper = qr.make_image(fill_color="black", back_color="white")
    pil_img = pil_img_wrapper.get_image()
    
    ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=(250, 250))
    img_lbl = ctk.CTkLabel(qr_popup_window, image=ctk_img, text="")
    img_lbl.pack(pady=10)

    lbl_url = ctk.CTkLabel(qr_popup_window, text=url, font=('Courier New', 12), text_color=SUCCESS_COLOR)
    lbl_url.pack(pady=(5,20))

def process_mobile_upload(filepath):
    """ Wird vom Flask-Thread (via app.after) ausgelöst, wenn ein Bild da ist. """
    global qr_popup_window
    if qr_popup_window is not None and qr_popup_window.winfo_exists():
        qr_popup_window.destroy()  # Popup schließen

    # Falls Editor noch nicht initialisiert, jetzt vorbereiten
    if editor_feedback is None:
        editor()

    try:
        editor_feedback.configure(text="Handy-Bild empfangen. Import läuft…", text_color=SUCCESS_COLOR)
        app.update_idletasks()

        try:
            neue = extract_pairs_from_image(filepath)
        except Exception as e:
            editor_feedback.configure(text=f"Fehler: {e}", text_color=ERROR_COLOR)
            return

        if not neue:
            editor_feedback.configure(text="⚠️ Kein lesbarer Text gefunden (Foto unscharf?)", text_color=WARNING_COLOR)
            app.update_idletasks()
            return

        imported = _import_pairs(neue)
        save_vokabeln_csv()
        show_editor()
        editor_feedback.configure(text=f"✅ {imported} Vokabel(n) vom Handy importiert!", text_color=SUCCESS_COLOR)
    finally:
        # Temp-Datei aufräumen
        try:
            os.remove(filepath)
        except OSError:
            pass


def _import_pairs(neue: list[dict]) -> int:
    """Fügt neue Vokabelpaare hinzu (Duplikate werden übersprungen). Gibt Anzahl importierter zurück."""
    bestehend = {(v['Deutsch'], v['Englisch']) for v in alle_vokabeln}
    imported = 0
    for v in neue:
        key = (v['Deutsch'], v['Englisch'])
        if key not in bestehend:
            alle_vokabeln.append(v)
            bestehend.add(key)
            imported += 1
    return imported


# ====================== Upload- und Import-Funktion ==========================
def upload_and_import():
    paths = filedialog.askopenfilenames(
        title="Bilder wählen",
        filetypes=[("Bilder", "*.png;*.jpg;*.jpeg;*.bmp")]
    )
    if not paths:
        return
    editor_feedback.configure(text="Import läuft…")
    app.update_idletasks()

    imported = 0
    for p in paths:
        try:
            neue = extract_pairs_from_image(p)
            imported += _import_pairs(neue)
        except Exception as e:
            editor_feedback.configure(text=f"Import fehlgeschlagen: {e}")
            return

    save_vokabeln_csv()
    show_editor()
    editor_feedback.configure(text=f"{imported} neue Vokabeln importiert {EMOJI_OK}")

# ======================== Light/Dark Mode ====================================
def wechsel_mode():
    ctk.set_appearance_mode("Light" if ctk.get_appearance_mode()=="Dark" else "Dark")

# ========================== Startbildschirm ==================================
def startbildschirm():
    if 'start' in frames:
        try:
            frames['start'].destroy()
        except Exception:
            pass
    frame = ctk.CTkFrame(app); frames['start'] = frame

    # Vollflächiger Outer-Container mit separater Bottom-Bar
    outer = ctk.CTkFrame(frame)
    outer.pack(expand=True, fill='both')

    # Grid: Top-Spacer, Content, Bottom-Spacer, Bottom-Bar
    outer.grid_rowconfigure(0, weight=1)
    outer.grid_rowconfigure(1, weight=0)
    outer.grid_rowconfigure(2, weight=1)
    outer.grid_rowconfigure(3, weight=0)
    outer.grid_columnconfigure(0, weight=1)

    # Inhalt mittig (Row 1)
    content = ctk.CTkFrame(outer, fg_color="transparent")
    content.grid(row=1, column=0)

    ctk.CTkLabel(content, text="Vokabeltrainer", font=('Segoe UI', 48, 'bold')).pack(pady=40)

    # Zwei Start-Buttons für die Modi
    btn_row = ctk.CTkFrame(content, fg_color="transparent")
    btn_row.pack(pady=(10, 32))

    def start_input_mode():
        training_settings['mode'] = 'input'
        starte_neu()

    def start_choice_mode():
        # Sicherheitscheck: Nur mit >=5 Vokabeln Auswahlmodus starten
        try:
            count = len(alle_vokabeln)
        except Exception:
            count = 0
        if count < 5:
            return  # still und leise nichts tun
        training_settings['mode'] = 'choice'
        starte_neu()

    # Command global verfügbar machen, damit wir ihn bei gesperrtem Button entfernen können
    global start_choice_command
    start_choice_command = start_choice_mode

    # Linker Container + innerer Frame (Inhalt am unteren Rand)
    input_container = ctk.CTkFrame(btn_row, fg_color="transparent")
    input_container.configure(width=START_CHOICE_CONTAINER_WIDTH)
    try:
        input_container.pack_propagate(False)
    except Exception:
        pass
    input_container.pack(side='left', padx=12)
    input_inner = ctk.CTkFrame(input_container, fg_color="transparent")
    try:
        input_inner.place(relx=0.5, rely=1.0, relheight=1.0, anchor='s', x=0, y=0)
    except Exception:
        pass
    btn_input = ctk.CTkButton(
        input_inner,
        text="Start (Eingabe)",
        fg_color=BTN_COLOR,
        hover_color=BTN_HOVER_COLOR,
        command=start_input_mode,
        width=280,
        height=100,
        corner_radius=30,
        font=('Segoe UI', 30, 'bold')
    )
    btn_input.pack(side='bottom')
    # globale Referenzen
    global start_input_button, start_input_container, start_input_inner
    start_input_button = btn_input
    start_input_container = input_container
    start_input_inner = input_inner

    # Container für den Auswahl-Button mit Schloss/Schlüssel-Emoji darüber
    choice_container = ctk.CTkFrame(btn_row, fg_color="transparent")
    choice_container.configure(width=START_CHOICE_CONTAINER_WIDTH)  # Fixbreite, damit äußeres Layout stabil bleibt
    try:
        choice_container.pack_propagate(False)
    except Exception:
        pass
    choice_container.pack(side='left', padx=START_CHOICE_PADX_BASE)
    # global referenzieren, damit die Shake-Animation darauf zugreifen kann
    global start_choice_container
    start_choice_container = choice_container

    # Innerer Frame, der den Inhalt trägt und per place() verschoben wird
    inner = ctk.CTkFrame(choice_container, fg_color="transparent")
    try:
        # Inhalt volle Höhe, unten verankert
        inner.place(relx=0.5, rely=1.0, relheight=1.0, anchor='s', x=START_CHOICE_SHAKE_BASE_X, y=0)
    except Exception:
        pass
    global start_choice_inner
    start_choice_inner = inner

    global start_choice_lock_label, start_choice_button
    # Schloss-Icon entfernt
    start_choice_lock_label = None

    start_choice_button = ctk.CTkButton(
        inner,
        text="Start (Auswahl)",
        fg_color=BTN_COLOR,
        hover_color=BTN_HOVER_COLOR,
        command=start_choice_mode,
        width=280,
        height=100,
        corner_radius=30,
        font=('Segoe UI', 30, 'bold')
    )
    start_choice_button.pack(side='bottom')

    # Nach Aufbau: Containerbreite dynamisch an inneren Inhalt + Shake-Puffer anpassen
    def _adjust_choice_width():
        try:
            if not start_choice_inner or not start_choice_container:
                return
            start_choice_inner.update_idletasks()
            req = start_choice_inner.winfo_reqwidth()
            need = req + 2 * START_CHOICE_SHAKE_AMPL + 12
            start_choice_container.configure(width=need)
            if start_input_container:
                start_input_container.configure(width=need)
        except Exception:
            pass
    try:
        app.after(0, _adjust_choice_width)
    except Exception:
        pass
    # Klicks im gesamten Container abfangen (auch wenn Button disabled ist)
    try:
        # Button-Klick löst ggf. Wackeln aus, wenn gesperrt
        start_choice_button.bind("<Button-1>", on_locked_choice_click)
    except Exception:
        pass

    # Nach Aufbau: Buttons vertikal ausrichten (gleich hohe Oberkante)
    try:
        app.after(0, update_start_buttons_alignment)
    except Exception:
        pass

    # Optional: kleine Sprachanzeige unter den Start-Buttons
    global sprach_anzeige_label
    sprach_anzeige_label = ctk.CTkLabel(
        content,
        text=aktuelle_sprache.capitalize() if aktuelle_sprache else "Keine Sprache",
        font=('Segoe UI', 24, 'bold'),
        text_color=SPRACH_COLOR
    )
    setattr(sprach_anzeige_label, '_is_sprach_label', True)
    sprach_anzeige_label.pack(pady=(0, 6))

    # Nutzer-Zeile: Name + Wechseln-Button
    nutzer_row = ctk.CTkFrame(content, fg_color="transparent")
    nutzer_row.pack(pady=(0, 4))
    ctk.CTkLabel(
        nutzer_row,
        text=f"👤  {aktueller_nutzer or 'Kein Benutzer'}",
        font=('Segoe UI', 15),
        text_color=LIGHT_TEXT
    ).pack(side='left', padx=(0, 10))
    ctk.CTkButton(
        nutzer_row, text="Wechseln",
        width=90, height=28, corner_radius=10,
        fg_color=BTN_COLOR, hover_color=BTN_HOVER_COLOR,
        font=('Segoe UI', 12),
        command=benutzer_wechseln
    ).pack(side='left')

    # XP-Anzeige
    global xp_start_label
    xp_start_label = ctk.CTkLabel(
        content,
        text=f"XP diese Woche:  {xp_woche}",
        font=('Segoe UI', 20, 'bold'),
        text_color="#f59e0b"
    )
    xp_start_label.pack(pady=(0, 18))

    # Vier kleinere Buttons in einer horizontalen Reihe
    actions_row = ctk.CTkFrame(content, fg_color="transparent")
    actions_row.pack(pady=6)
    for i in range(4):
        actions_row.grid_columnconfigure(i, weight=1, uniform="actions")

    btn_stats = ctk.CTkButton(
        actions_row, text="Statistiken", fg_color=BTN_COLOR,
        hover_color=BTN_HOVER_COLOR, width=200, height=60, corner_radius=20,
        font=('Segoe UI', 18, 'bold'),
        command=lambda: [zeige_frame('statistik'), zeige_statistik()]
    )
    btn_stats.grid(row=0, column=0, padx=8, pady=6)

    btn_edit = ctk.CTkButton(
        actions_row, text="Vokabeln bearbeiten", fg_color=BTN_COLOR,
        hover_color=BTN_HOVER_COLOR, width=200, height=60, corner_radius=20,
        font=('Segoe UI', 18, 'bold'),
        command=show_editor
    )
    btn_edit.grid(row=0, column=1, padx=8, pady=6)

    btn_settings = ctk.CTkButton(
        actions_row, text="Einstellungen", fg_color=BTN_COLOR,
        hover_color=BTN_HOVER_COLOR, width=200, height=60, corner_radius=20,
        font=('Segoe UI', 18, 'bold'),
        command=lambda: zeige_frame('einstellungen')
    )
    btn_settings.grid(row=0, column=2, padx=8, pady=6)

    btn_ranking = ctk.CTkButton(
        actions_row, text="Rangliste", fg_color=BTN_COLOR,
        hover_color=BTN_HOVER_COLOR, width=200, height=60, corner_radius=20,
        font=('Segoe UI', 18, 'bold'),
        command=lambda: [rangliste_laden(), zeige_frame('rangliste')]
    )
    btn_ranking.grid(row=0, column=3, padx=8, pady=6)

    # Bottom-Bar (Row 3): links Dark/Light, rechts Sprache wechseln
    bottom_bar = ctk.CTkFrame(outer, fg_color="transparent")
    bottom_bar.grid(row=3, column=0, sticky='ew', padx=16, pady=12)
    # Links ausrichten
    left_wrap = ctk.CTkFrame(bottom_bar, fg_color="transparent")
    left_wrap.pack(side='left', fill='x', expand=True)
    ctk.CTkButton(
        left_wrap, image=birne_icon, text="",
        width=48, height=48, corner_radius=24,
        fg_color="transparent", hover_color="#f3f4f6",
        command=wechsel_mode
    ).pack(side='left')

    # Rechts ausrichten
    right_wrap = ctk.CTkFrame(bottom_bar, fg_color="transparent")
    right_wrap.pack(side='right', fill='x', expand=True)
    ctk.CTkButton(
        right_wrap, text="Sprachen",
        width=100, height=36, corner_radius=18,
        fg_color=BTN_COLOR, hover_color=BTN_HOVER_COLOR,
        command=sprache_verwalten_screen
    ).pack(side='right')

    zeige_frame('start')
    # Nach Aufbau: Zugriffsstatus für Auswahl-Modus prüfen
    try:
        update_start_choice_access()
    except Exception:
        pass

# ========================== Rangliste-Screen =================================

def rangliste_screen():
    """Ranglisten-Screen: Zeigt alle Spieler sortiert nach Wochen-XP."""
    frame = ctk.CTkFrame(app)
    frames['rangliste'] = frame

    # Header-Leiste
    header = ctk.CTkFrame(frame, height=64, fg_color=BTN_COLOR, corner_radius=0)
    header.pack(fill='x')
    header.pack_propagate(False)
    header_inner = ctk.CTkFrame(header, fg_color="transparent")
    header_inner.pack(expand=True, fill='both', padx=16)
    ctk.CTkButton(
        header_inner, text="←", width=44, height=44, corner_radius=22,
        fg_color="transparent", hover_color=BTN_HOVER_COLOR, font=('Segoe UI', 22, 'bold'),
        command=lambda: zeige_frame('start')
    ).pack(side='left', pady=10)
    ctk.CTkLabel(
        header_inner, text="🏆  Rangliste",
        font=('Segoe UI', 24, 'bold'), text_color="white"
    ).pack(side='left', padx=16, pady=10)

    # Wochen-Hinweis
    ctk.CTkLabel(
        frame,
        text="Wöchentliches Ranking  •  Reset jeden Sonntag um 20:00 Uhr",
        font=('Segoe UI', 13), text_color=LIGHT_TEXT
    ).pack(pady=(12, 4))

    # Scrollbarer Listenbereich
    global rangliste_scroll_frame
    rangliste_scroll_frame = ctk.CTkScrollableFrame(frame, corner_radius=12)
    rangliste_scroll_frame.pack(expand=True, fill='both', padx=40, pady=(4, 20))

    # Platzhalter-Text (wird beim Laden ersetzt)
    ctk.CTkLabel(
        rangliste_scroll_frame,
        text="Lade Rangliste...",
        font=('Segoe UI', 18), text_color=LIGHT_TEXT
    ).pack(pady=40)


def rangliste_laden():
    """Lädt Spielerdaten aus Firestore und füllt den Ranglisten-Screen."""
    if rangliste_scroll_frame is None:
        return
    scroll = rangliste_scroll_frame

    # Bestehende Einträge löschen
    for widget in scroll.winfo_children():
        try:
            widget.destroy()
        except Exception:
            pass

    # Daten laden
    spieler = []

    # Eigene Daten immer einbeziehen (auch offline)
    if aktueller_nutzer:
        spieler.append({
            'username': aktueller_nutzer,
            'week_xp':  xp_woche,
            'total_xp': xp_gesamt,
            'is_self':  True,
        })

    if firebase_db:
        def _fetch():
            try:
                docs = firebase_db.collection('users').stream()
                remote = []
                for doc in docs:
                    d = doc.to_dict()
                    name = doc.id
                    # Eigenen Eintrag aus Firebase nehmen (aktuellster Stand)
                    if name == aktueller_nutzer:
                        for s in spieler:
                            if s['username'] == aktueller_nutzer:
                                s['week_xp']  = max(s['week_xp'],  d.get('week_xp', 0))
                                s['total_xp'] = max(s['total_xp'], d.get('total_xp', 0))
                    else:
                        remote.append({
                            'username': name,
                            'week_xp':  d.get('week_xp', 0),
                            'total_xp': d.get('total_xp', 0),
                            'is_self':  False,
                        })
                spieler.extend(remote)
            except Exception as e:
                print(f"[Rangliste] Ladefehler: {e}")
            finally:
                # UI-Update im Hauptthread
                if app:
                    app.after(0, lambda: _render_rangliste(spieler))
        threading.Thread(target=_fetch, daemon=True).start()

        # Sofort Ladeanzeige rendern während Firebase lädt
        ctk.CTkLabel(
            scroll,
            text="Verbinde mit Firebase...",
            font=('Segoe UI', 16), text_color=LIGHT_TEXT
        ).pack(pady=40)
    else:
        # Offline: nur eigene Daten
        _render_rangliste(spieler)


def _render_rangliste(spieler: list):
    """Rendert die Ranglisten-Einträge in den scroll_frame."""
    if rangliste_scroll_frame is None:
        return
    scroll = rangliste_scroll_frame

    # Alte Widgets entfernen
    for widget in scroll.winfo_children():
        try:
            widget.destroy()
        except Exception:
            pass

    # Nur Spieler mit Wöchentlichen XP > 0 anzeigen (verwaiste Accounts ausblenden)
    spieler_aktiv = [s for s in spieler if s.get('week_xp', 0) > 0]

    # Sortieren nach week_xp (absteigende Reihenfolge), Gleichstand → gesamt-XP
    sorted_spieler = sorted(spieler_aktiv, key=lambda x: (x['week_xp'], x['total_xp']), reverse=True)

    MEDALS = {1: "🥇", 2: "🥈", 3: "🥉"}
    MEDAL_COLORS = {1: "#f59e0b", 2: "#9ca3af", 3: "#b45309"}
    ROW_BG_SELF   = ("#dbeafe", "#1e3a5f")  # Hervorhebung eigene Zeile (hell/dunkel)
    ROW_BG_NORMAL = "transparent"

    if not sorted_spieler:
        ctk.CTkLabel(
            scroll, text="Diese Woche noch keine Aktivität.",
            font=('Segoe UI', 16), text_color=LIGHT_TEXT
        ).pack(pady=40)
        return

    for platz, s in enumerate(sorted_spieler, start=1):
        is_self  = s.get('is_self', False)
        row_bg   = ROW_BG_SELF if is_self else ROW_BG_NORMAL

        row = ctk.CTkFrame(scroll, corner_radius=14, fg_color=row_bg)
        row.pack(fill='x', padx=8, pady=5)
        row.grid_columnconfigure(1, weight=1)

        # Platz-Symbol
        if platz in MEDALS:
            platz_text  = MEDALS[platz]
            platz_color = MEDAL_COLORS[platz]
            platz_font  = ('Segoe UI', 28)
        else:
            platz_text  = f"{platz}."
            platz_color = LIGHT_TEXT
            platz_font  = ('Segoe UI', 20, 'bold')

        ctk.CTkLabel(
            row, text=platz_text, font=platz_font,
            text_color=platz_color, width=52
        ).grid(row=0, column=0, padx=(16, 8), pady=14, sticky='w')

        # Name ("Du" wenn eigene Zeile)
        name_display = f"{s['username']}  👉 Du" if is_self else s['username']
        name_color   = BTN_COLOR if is_self else None
        name_font    = ('Segoe UI', 18, 'bold') if is_self else ('Segoe UI', 17)
        lbl_kwargs = {'text_color': name_color} if name_color else {}
        ctk.CTkLabel(
            row, text=name_display, font=name_font, **lbl_kwargs
        ).grid(row=0, column=1, padx=8, pady=14, sticky='w')

        # XP rechts
        xp_text = f"{s['week_xp']} XP"
        ctk.CTkLabel(
            row, text=xp_text,
            font=('Segoe UI', 17, 'bold'), text_color="#f59e0b"
        ).grid(row=0, column=2, padx=(8, 20), pady=14, sticky='e')


def einstellungen_screen():
    """Einstellungsseite mit Schiebereglern für Training-Parameter"""
    frame = ctk.CTkFrame(app); frames['einstellungen'] = frame
    
    # Scrollbarer Container statt fixer Frame
    container = ctk.CTkScrollableFrame(frame, corner_radius=0)
    container.pack(expand=True, fill='both', padx=50, pady=50)
    
    # Header
    ctk.CTkLabel(
        container, 
        text="Training-Einstellungen", 
        font=('Arial', 32, 'bold')
    ).pack(pady=(0, 40))
    
    # Tippfehler-Toleranz
    typo_frame = ctk.CTkFrame(container, fg_color="transparent")
    typo_frame.pack(fill='x', pady=20)
    
    ctk.CTkLabel(
        typo_frame, 
        text="Tippfehler-Toleranz", 
        font=('Segoe UI', 18, 'bold')
    ).pack(pady=(20, 10))
    
    typo_labels = ["Leicht", "Mittel", "Schwer"]
    typo_colors = ["#eab308", "#22c55e", "#059669"]
    
    def _on_typo_change(value):
        v = int(value)
        training_settings['typo_tolerance'] = v
        update_typo_label(v, typo_value_label, typo_labels, typo_colors)
    typo_slider = ctk.CTkSlider(
        typo_frame,
        from_=0,
        to=2,
        number_of_steps=2,
        command=_on_typo_change
    )
    typo_slider.set(training_settings['typo_tolerance'])
    typo_slider.pack(pady=10)
    
    typo_value_label = ctk.CTkLabel(
        typo_frame,
        text=typo_labels[training_settings['typo_tolerance']],
        font=('Segoe UI', 16),
        text_color=typo_colors[training_settings['typo_tolerance']]
    )
    typo_value_label.pack(pady=(0, 20))

    # Richtung umschalten (Slider: 0=DE→Fremd, 1=Fremd→DE, 2=Gemischt)
    dir_frame = ctk.CTkFrame(container, fg_color="transparent")
    dir_frame.pack(fill='x', pady=10)
    ctk.CTkLabel(dir_frame, text="Abfrage-Richtung", font=('Segoe UI', 18, 'bold')).pack(pady=(10,6))
    dir_labels = ["Deutsch → Fremdsprache", "Fremdsprache → Deutsch", "Gemischt"]
    dir_values = ['de_to_foreign', 'foreign_to_de', 'mixed']
    # Map aktuelle Einstellung auf Slider-Position
    current_dir = training_settings.get('direction','de_to_foreign')
    try:
        current_idx = dir_values.index(current_dir)
    except ValueError:
        current_idx = 0
    dir_slider = ctk.CTkSlider(dir_frame, from_=0, to=2, number_of_steps=2)
    dir_slider.set(current_idx)
    def _on_dir_change(val):
        idx = int(round(float(val)))
        dir_slider.set(idx)
        try:
            dir_value_lbl.configure(text=dir_labels[idx])
        except Exception:
            pass
        try:
            training_settings['direction'] = dir_values[idx]
        except Exception:
            pass
    dir_slider.configure(command=_on_dir_change)
    dir_slider.pack(pady=6)
    # Wert-Anzeige zentriert unter dem Slider
    dir_value_lbl = ctk.CTkLabel(dir_frame, text=dir_labels[current_idx])
    dir_value_lbl.pack(pady=(6, 10))

    # Buttons
    button_frame = ctk.CTkFrame(container, fg_color="transparent")
    button_frame.pack(fill='x', pady=40)
    
    def start_training():
        # Einstellungen speichern
        training_settings['typo_tolerance'] = int(typo_slider.get())
        # Richtung aus Slider übernehmen
        try:
            dir_idx = int(round(float(dir_slider.get())))
        except Exception:
            dir_idx = 0
        dir_values = ['de_to_foreign', 'foreign_to_de', 'mixed']
        training_settings['direction'] = dir_values[dir_idx]
        # Training starten
        reset_for_new_attempt()
        try:
            punktzahl_label.configure(text=f"XP: +{xp_session} XP")
        except Exception:
            pass
        naechste_vokabel()
        zeige_frame('trainer')
    
    ctk.CTkButton(
        button_frame,
        text="Training starten",
        command=start_training,
        fg_color=SUCCESS_COLOR,
        hover_color="#047857",
        width=200,
        height=50,
        font=('Segoe UI', 16, 'bold')
    ).pack(side='left', padx=10)
    
    ctk.CTkButton(
        button_frame,
        text="Zurück",
        command=lambda: zeige_frame('start'),
        fg_color=BTN_COLOR,
        width=150,
        height=50,
        font=('Segoe UI', 16)
    ).pack(side='right', padx=10)

def update_typo_label(value, label, labels, colors):
    """Aktualisiert das Label für die Tippfehler-Toleranz"""
    label.configure(text=labels[value], text_color=colors[value])



#=============================Sprache-Verwaltung-Screen ==========================
def sprache_verwalten_screen():
    # Frame wiederverwenden oder neu anlegen
    if 'sprache_verwalten' in frames:
        frame = frames['sprache_verwalten']
        # verstecke es kurz und lösche alten Inhalt
        frame.pack_forget()
        for w in frame.winfo_children():
            w.destroy()
    else:
        frame = ctk.CTkFrame(app)
        frames['sprache_verwalten'] = frame

    # Header
    ctk.CTkLabel(frame, text="Sprache verwalten", font=('Segoe UI', 32, 'bold')).pack(pady=(20,15))

    # Für jede vorhandene Sprache: Wechsel-Button und Delete-Button
    for sprache in vorhandene_sprachen():
        row = ctk.CTkFrame(frame, fg_color="transparent")
        row.pack(fill='x', padx=20, pady=5)

        # Button zum Wechseln - mit Markierung der aktuellen Sprache
        button_color = SUCCESS_COLOR if sprache.lower() == aktuelle_sprache else BTN_COLOR
        button_text = f"✓ {sprache.capitalize()}" if sprache.lower() == aktuelle_sprache else sprache.capitalize()
        
        ctk.CTkButton(
            row,
            text=button_text,
            fg_color=button_color,
            hover_color=SPRACH_COLOR if sprache.lower() == aktuelle_sprache else BTN_HOVER_COLOR,
            width=150, height=35,
            font=('Segoe UI', 12),
            command=lambda s=sprache: sprache_wechseln(s)
        ).pack(side='left', fill='x', expand=True)

        # Button zum Löschen
        ctk.CTkButton(
            row,
            text="Löschen",
            fg_color=ERROR_COLOR,
            hover_color="#b91c1c",
            width=80, height=35,
            font=('Segoe UI', 11),
            command=lambda s=sprache: delete_language(s)
        ).pack(side='right')

    # Eingabe für neue Sprache (breites Feld)
    neu_frame = ctk.CTkFrame(frame, fg_color="transparent")
    neu_frame.pack(pady=(20,10), padx=20, fill='x')

    neu_eingabe = ctk.CTkEntry(
        neu_frame,
        placeholder_text="Neue Sprache",
        width=200
    )
    neu_eingabe.pack(side='left', fill='x', expand=True, padx=(0,10))

    ctk.CTkButton(
        neu_frame,
        text="Hinzufügen",
        fg_color=BTN_COLOR,
        hover_color=BTN_HOVER_COLOR,
        width=80, height=35,
        font=('Segoe UI', 12),
        command=lambda: (
            neue_sprache_hinzufuegen(neu_eingabe.get().strip()),
            neu_eingabe.delete(0, tk.END)
        )
    ).pack(side='left')

    # Zurück-Button
    ctk.CTkButton(
        frame,
        text="Zurück",
        fg_color=BTN_COLOR,
        width=150, height=45,
        corner_radius=22,
        font=('Segoe UI', 14, 'bold'),
        command=lambda: zeige_frame('start')
    ).pack(pady=(30,20))

    zeige_frame('sprache_verwalten')


def delete_language(sprache):
    """Entfernt CSV- und Statistikdatei, aktualisiert UI."""
    csv_path  = os.path.join(VOCAB_DIR, f'vokabeln_{sprache}.csv')
    stat_path = os.path.join(STAT_DIR,    f'statistik_{sprache}.json')

    for path in (csv_path, stat_path):
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            messagebox.showerror("Fehler", f"Konnte {path} nicht löschen:\n{e}")
            return

    # Falls gerade aktive Sprache gelöscht, zurücksetzen
    global aktuelle_sprache
    if aktuelle_sprache == sprache:
        aktuelle_sprache = None
        alle_vokabeln.clear()
        vokabel_statistik.clear()
        vokabeln_zu_lernen.clear()
        learning_queue.clear()

    # UI neu aufbauen
    sprache_verwalten_screen()



def sprache_wechseln(sprache):
    initialisiere_sprache(sprache)
    zeige_frame('start')
def neue_sprache_hinzufuegen(name):
    if name.strip():
        initialisiere_sprache(name.strip().lower())
        zeige_frame('start')

# ============================ Trainer-Screen =================================
def trainer_zurueck():
    """Zurück zum Startbildschirm mit Warnung, wenn Session-XP verloren gehen würden."""
    if xp_session > 0:
        win = ctk.CTkToplevel(app)
        win.title("Training abbrechen?")
        win.geometry("380x210")
        win.resizable(False, False)
        win.grab_set()
        win.focus_set()
        ctk.CTkLabel(
            win, text="Training abbrechen?",
            font=('Segoe UI', 20, 'bold')
        ).pack(pady=(24, 8))
        ctk.CTkLabel(
            win,
            text=f"Du verlierst {xp_session} XP!",
            font=('Segoe UI', 15), text_color=ERROR_COLOR
        ).pack(pady=(0, 20))
        btn_frame = ctk.CTkFrame(win, fg_color="transparent")
        btn_frame.pack(fill='x', padx=30)
        ctk.CTkButton(
            btn_frame, text="Abbrechen & verlassen",
            fg_color=ERROR_COLOR, hover_color="#b91c1c",
            height=42, corner_radius=10, font=('Segoe UI', 13, 'bold'),
            command=lambda: (win.destroy(), zeige_frame('start'))
        ).pack(fill='x', pady=(0, 8))
        ctk.CTkButton(
            btn_frame, text="Weiterlernen",
            fg_color=BTN_COLOR, hover_color=BTN_HOVER_COLOR,
            height=42, corner_radius=10, font=('Segoe UI', 13, 'bold'),
            command=win.destroy
        ).pack(fill='x')
    else:
        zeige_frame('start')


def trainer():
    frame = ctk.CTkFrame(app); frames['trainer'] = frame

    outer = ctk.CTkFrame(frame)
    outer.pack(expand=True, fill='both')

    # Topbar oben, volle Breite
    top = ctk.CTkFrame(outer, fg_color="transparent")
    top.pack(side='top', fill='x')

    ctk.CTkButton(top, image=haus_icon, text="", fg_color=BTN_COLOR,
                  width=40, height=40, corner_radius=20,
                  command=trainer_zurueck).pack(side='left', padx=10, pady=10)

    if aktuelle_sprache:
        global trainer_sprach_label
        trainer_sprach_label = ctk.CTkLabel(
            top,
            text=aktuelle_sprache.capitalize(),
            font=('Segoe UI', 24, 'bold'),
            text_color=SPRACH_COLOR
        )
        setattr(trainer_sprach_label, '_is_sprach_label', True)
        trainer_sprach_label.pack(side='left', padx=20, pady=10)

    # Tipp-Button
    global tipp_button
    tipp_button = ctk.CTkButton(top, image=tipp_icon, text="", fg_color=BTN_COLOR,
                  width=40, height=40, corner_radius=20,
                  command=zeige_tipp)
    tipp_button.pack(side='right', padx=10, pady=10)

    ctk.CTkButton(top, image=birne_icon, text="", fg_color=BTN_COLOR,
                  width=40, height=40, corner_radius=20,
                  command=wechsel_mode).pack(side='right', padx=10, pady=10)

    # Mittelteil: nimmt gesamte Fläche ein, Layout mit Grid
    content = ctk.CTkFrame(outer)
    content.pack(expand=True, fill='both', padx=40, pady=24)
    for r, wt in enumerate([1, 0, 0, 0, 0, 0, 0, 1]):
        content.grid_rowconfigure(r, weight=wt)
    content.grid_columnconfigure(0, weight=1)

    global frage_label, eingabe, feedback_label, punktzahl_label, fortschritt, btn_pruefen

    # Oben-Spacer (row 0)
    ctk.CTkLabel(content, text="").grid(row=0, column=0)

    # Frage (row 1)
    frage_label = ctk.CTkLabel(content, text="", font=('Arial', 36))
    frage_label.grid(row=1, column=0, pady=(40, 28), sticky='n')

    # Antwortbereich (row 2): enthält entweder Eingabe oder Auswahl-Buttons
    global answer_frame, input_frame, choice_frame, choice_buttons
    answer_frame = ctk.CTkFrame(content, fg_color="transparent")
    answer_frame.grid(row=2, column=0, pady=28, sticky='n')

    # Input-Frame (Eingabefeld)
    input_frame = ctk.CTkFrame(answer_frame, fg_color="transparent")
    input_frame.grid(row=0, column=0)
    eingabe = ctk.CTkEntry(input_frame, font=('Arial', 22), width=900)
    eingabe.pack()

    # Choice-Frame (3 Auswahl-Buttons)
    choice_frame = ctk.CTkFrame(answer_frame, fg_color="transparent")
    # zunächst nicht gridden, wird per Modus ein-/ausgeblendet
    choice_buttons = []
    for i in range(3):
        btn = ctk.CTkButton(choice_frame, text=f"Option {i+1}", fg_color=BTN_COLOR,
                             width=720, height=72, corner_radius=18,
                             font=('Segoe UI', 24, 'bold'),
                             command=lambda idx=i: rate_limited_call(choice_selected, idx))
        btn.pack(pady=8, fill='x')
        choice_buttons.append(btn)

    # Button (row 3)
    btn_pruefen = ctk.CTkButton(content, text="Antwort prüfen", fg_color=BTN_COLOR,
                  command=lambda: rate_limited_call(pruefe_antwort), width=360, height=60, corner_radius=20)
    btn_pruefen.grid(row=3, column=0, pady=24, sticky='n')

    # Feedback (row 4)
    feedback_label = ctk.CTkLabel(content, text="", font=('Arial', 22))
    feedback_label.grid(row=4, column=0, pady=20, sticky='n')

    # XP Session-Anzeige (row 5)
    punktzahl_label = ctk.CTkLabel(content, text=f"XP: +{xp_session} XP", font=('Arial', 22))
    punktzahl_label.grid(row=5, column=0, pady=20, sticky='n')

    # Progress unten (row 6)
    fortschritt = ctk.CTkProgressBar(content, width=900)
    fortschritt.set(0)
    fortschritt.grid(row=6, column=0, pady=(28, 40), sticky='s')

    # Unten-Spacer (row 7)
    ctk.CTkLabel(content, text="").grid(row=7, column=0)

    # Enter im Eingabefeld: erst prüfen, dann beim nächsten Enter weiter
    def on_entry_return(event):
        if feedback_active:
            rate_limited_call(naechste_vokabel)
        else:
            rate_limited_call(pruefe_antwort)
        return "break"  # Event nicht weiterreichen
    eingabe.bind('<Return>', on_entry_return)
    eingabe.bind('<KP_Enter>', on_entry_return)

    # Direkt nach Erstellen Größe anpassen (damit Sprachindikator sofort groß ist)
    try:
        update_trainer_mode_ui()
        update_font_sizes()
    except Exception:
        pass

    # Nach dem Aufbau: vertikale Abstände im Trainer lokal an Fenstergröße anpassen
    try:
        w, h = app.winfo_width(), app.winfo_height()
        scale = min(w/1200, h/800) if (w and h) else 1.0
        vgap_large = max(int(32 * scale), 16)
        vgap_med   = max(int(24 * scale), 12)
        if frage_label and frage_label.winfo_ismapped():
            frage_label.grid_configure(pady=(vgap_large, vgap_med))
        if answer_frame and answer_frame.winfo_ismapped():
            answer_frame.grid_configure(pady=vgap_large)
        if btn_pruefen and btn_pruefen.winfo_ismapped():
            btn_pruefen.grid_configure(pady=vgap_med)
        if feedback_label and feedback_label.winfo_ismapped():
            feedback_label.grid_configure(pady=vgap_med)
        if punktzahl_label and punktzahl_label.winfo_ismapped():
            punktzahl_label.grid_configure(pady=vgap_med)
        if fortschritt and fortschritt.winfo_ismapped():
            fortschritt.grid_configure(pady=(vgap_med, vgap_large))
    except Exception:
        pass

# =========================== Statistik-Screen =================================
def statistik():
    global stat_feedback_label, statistik_frame, statistik_titel_label
    frame = ctk.CTkFrame(app); frames['statistik'] = frame

    outer = ctk.CTkFrame(frame)
    outer.pack(expand=True, fill='both')
    outer.grid_rowconfigure(0, weight=0)  # Top
    outer.grid_rowconfigure(1, weight=1)  # Scroll
    outer.grid_rowconfigure(2, weight=0)  # Footer
    outer.grid_columnconfigure(0, weight=1)

    # Topbar/Head
    top = ctk.CTkFrame(outer, fg_color="transparent")
    top.grid(row=0, column=0, sticky='ew')

    ctk.CTkButton(top, image=haus_icon, text="", fg_color=BTN_COLOR,
                  width=40, height=40, corner_radius=20,
                  command=lambda: zeige_frame('start')).pack(side='left', padx=10, pady=10)

    # Sprachanzeige/Überschrift
    statistik_titel_label = ctk.CTkLabel(top, text="Statistiken", font=('Arial', 30))
    statistik_titel_label.pack(side='left', padx=20, pady=10)
    update_sprachanzeige()
    stat_feedback_label = ctk.CTkLabel(top, text="", font=('Arial', 16), text_color="green")
    stat_feedback_label.pack(side='right', padx=20, pady=10)

    # Scrollbarer Mittelteil
    statistik_frame = ctk.CTkScrollableFrame(outer, corner_radius=0)
    statistik_frame.grid(row=1, column=0, sticky='nsew', padx=20, pady=10)

    # Spaltenbreiten: erste Spalte breit, rest flexibel
    statistik_frame.grid_columnconfigure(0, weight=3)
    statistik_frame.grid_columnconfigure(1, weight=1)
    statistik_frame.grid_columnconfigure(2, weight=1)
    statistik_frame.grid_columnconfigure(3, weight=1)

    # Footer unten
    footer = ctk.CTkFrame(outer, fg_color="transparent")
    footer.grid(row=2, column=0, sticky='ew', pady=10)

    ctk.CTkButton(footer, text="Zurück", command=lambda: zeige_frame('start'), fg_color=BTN_COLOR,
                  width=150, height=40, corner_radius=20).pack(side='left', padx=20)

    ctk.CTkButton(footer, text="Statistik zurücksetzen", command=statistik_zuruecksetzen, fg_color=ERROR_COLOR,
                  width=200, height=40, corner_radius=20).pack(side='right', padx=20)



def zeige_statistik():
    global stat_feedback_label, statistik_frame
    stat_feedback_label.configure(text="")
    for w in statistik_frame.winfo_children():
        w.destroy()

    w, h = app.winfo_width(), app.winfo_height()
    scale = min(w/1200, h/800)
    hdr_font = ("Arial", max(18, int(18 * scale)), "bold")
    dt_font  = ("Arial", max(14, int(14 * scale)))

    headers  = ["Deutsch – Fremdsprache", "Richtig", "Falsch", "% richtig"]
    for col, txt in enumerate(headers):
        ctk.CTkLabel(statistik_frame, text=txt, font=hdr_font).grid(row=0, column=col, padx=10, pady=(10,5), sticky='w')

    r_tot = f_tot = 0
    for r, v in enumerate(alle_vokabeln, start=1):
        key = (v['Deutsch'], v['Englisch'])
        st  = vokabel_statistik.get(key, {'richtig': 0, 'falsch': 0})
        tot = st['richtig'] + st['falsch']
        pct = (st['richtig'] / tot * 100) if tot > 0 else 0
        fg  = "green" if pct >= 80 else "orange" if pct >= 50 else "red"

        # Spalte 0: Umbruchfähiger Text (Textbox ohne Rahmen)
        cell0 = ctk.CTkTextbox(statistik_frame, width=1, height=1)  # Größe wird durch grid gestreckt
        cell0.insert("1.0", f"{v['Deutsch']} – {v['Englisch']}")
        cell0.configure(state="disabled")
        cell0.grid(row=r, column=0, padx=10, pady=5, sticky='nsew')

        ctk.CTkLabel(statistik_frame, text=str(st['richtig']), font=dt_font
            ).grid(row=r, column=1, padx=10, pady=5, sticky='w')
        ctk.CTkLabel(statistik_frame, text=str(st['falsch']), font=dt_font
            ).grid(row=r, column=2, padx=10, pady=5, sticky='w')
        ctk.CTkLabel(statistik_frame, text=f"{pct:.2f}%", font=dt_font,
                     fg_color=fg, corner_radius=5
            ).grid(row=r, column=3, padx=10, pady=5, sticky='w')

        r_tot += st['richtig']; f_tot += st['falsch']

    row = len(alle_vokabeln) + 1
    ctk.CTkLabel(statistik_frame, text="GESAMT", font=hdr_font).grid(row=row, column=0, padx=10, pady=(10,5), sticky='w')
    ctk.CTkLabel(statistik_frame, text=str(r_tot), font=hdr_font).grid(row=row, column=1, padx=10, pady=(10,5), sticky='w')
    ctk.CTkLabel(statistik_frame, text=str(f_tot), font=hdr_font).grid(row=row, column=2, padx=10, pady=(10,5), sticky='w')
    pct_tot = (r_tot / (r_tot + f_tot) * 100) if (r_tot + f_tot) > 0 else 0
    ctk.CTkLabel(statistik_frame, text=f"Prozent richtig: {pct_tot:.2f}%", font=dt_font
    ).grid(row=row+1, column=0, columnspan=4, padx=10, pady=(0,10), sticky='w')

def statistik_zuruecksetzen():
    if not messagebox.askyesno("Bestätigung", "Möchten Sie wirklich alle Statistiken zurücksetzen?"):
        return
    vokabel_statistik.clear()
    vokabel_repetitions.clear()
    runde_status.clear()
    statistik_speichern()
    stat_feedback_label.configure(text=f"{EMOJI_OK}Statistik zurückgesetzt")
    zeige_statistik()

# =========================== Vokabel-Editor ==================================
def editor():
    frame = ctk.CTkFrame(app); frames['editor'] = frame
    ctk.CTkButton(frame, image=haus_icon, text="", width=40, height=40, fg_color=BTN_COLOR,
                  corner_radius=20, command=lambda: zeige_frame('start')).pack(anchor='nw', padx=10, pady=10)
    
    # Sprachanzeige im Editor
    ctk.CTkLabel(frame, text="Vokabel-Editor", font=('Arial',30)).pack(pady=(20,10))
    global editor_feedback
    editor_feedback = ctk.CTkLabel(frame, text="", font=('Arial',8), text_color="green"); editor_feedback.pack(pady=(0,10))
    
    # Neues Frame, um die beiden Import-Buttons nebeneinander zu setzen
    btn_frame = ctk.CTkFrame(frame, fg_color="transparent")
    btn_frame.pack(pady=(0,10))
    
    ctk.CTkButton(btn_frame, text="💻 Bild von PC…", command=upload_and_import, fg_color=BTN_COLOR,
                  width=180, height=40, corner_radius=20).pack(side='left', padx=10)
                  
    ctk.CTkButton(btn_frame, text="📱 Vom Handy knipsen (QR)", command=show_qr_popup, fg_color=SPRACH_COLOR, text_color="white",
                  width=220, height=40, corner_radius=20).pack(side='left', padx=10)

    global editor_frame
    editor_frame = ctk.CTkScrollableFrame(frame, corner_radius=0); editor_frame.pack(fill='both', expand=True, padx=20, pady=10)
    for col in range(3):
        editor_frame.grid_columnconfigure(col, weight=1, uniform="ed")
    footer = ctk.CTkFrame(frame, fg_color="transparent"); footer.pack(fill='x', side='bottom', pady=10)

    ctk.CTkButton(footer, text="Zurück", command=lambda: zeige_frame('start'), fg_color=BTN_COLOR,
              width=150, height=40, corner_radius=20).pack(side='left', padx=20)

    ctk.CTkButton(footer, text="Alle löschen", command=alle_vokabeln_loeschen, fg_color=ERROR_COLOR,
              width=150, height=40, corner_radius=20).pack(side='right', padx=20)

    ctk.CTkButton(footer, text="Speichern", command=save_editor, fg_color=BTN_COLOR,
              width=150, height=40, corner_radius=20).pack(side='right', padx=20)


def show_editor():

    global neu_de, neu_en

    try:
        lade_vokabeln()
    except Exception as ex:
        messagebox.showerror("Fehler", f"Fehler beim Laden der Vokabeln:\n{ex}")
        return

    editor_feedback.configure(text="")
    for w in editor_frame.winfo_children():
        w.destroy()
    editor_entries.clear()

    # 1) Bestehende Vokabeln eintragen
    for i, v in enumerate(alle_vokabeln):
        de_e = ctk.CTkEntry(editor_frame)
        de_e.insert(0, v['Deutsch'])
        en_e = ctk.CTkEntry(editor_frame)
        en_e.insert(0, v['Englisch'])
        # Rot gefärbter Löschen-Button
        del_btn = ctk.CTkButton(
            editor_frame, text="X", width=30,
            fg_color=ERROR_COLOR, hover_color="#b91c1c",
            font=('Segoe UI', 10),
            command=lambda idx=i: (
                entferne_vokabel(idx),
                editor_feedback.configure(text=f"{EMOJI_OK}Vokabel gelöscht")
            )
        )

        # Grid-Layout
        de_e.grid(row=i, column=0, padx=5, pady=2, sticky='ew')
        en_e.grid(row=i, column=1, padx=5, pady=2, sticky='ew')
        del_btn.grid(row=i, column=2, padx=5, pady=2)

        # Bind Enter in Deutsch → fokus auf Englisch
        de_e.bind("<Return>", lambda e, target=en_e: target.focus())

        # Bind Enter in Englisch
        def on_enter_en(event, idx=i, de_widget=de_e, en_widget=en_e):
            # wenn beide Felder nicht leer sind
            if de_widget.get().strip() and en_widget.get().strip():
                # speichern und refresh
                save_editor()
                editor_feedback.configure(text=f"{EMOJI_OK}Änderung gespeichert")
                # focus aufs nächste Deutsch-Feld (oder neu_de)
                next_idx = idx + 1
                if next_idx < len(editor_entries):
                    editor_entries[next_idx][0].focus()
                else:
                    neu_de.focus()

        en_e.bind("<Return>", on_enter_en)

        editor_entries.append((de_e, en_e))

    # 2) Neue Zeile zum Hinzufügen
    row = len(alle_vokabeln)
    neu_de = ctk.CTkEntry(editor_frame, placeholder_text="Deutsch")
    neu_en = ctk.CTkEntry(
    editor_frame,
    placeholder_text="Fremdsprache"
)
    # Grün gefärbter Hinzufügen-Button
    add_btn = ctk.CTkButton(
        editor_frame, text="+", width=30,
        fg_color=SUCCESS_COLOR, hover_color=SPRACH_COLOR,
        font=('Segoe UI', 12, 'bold'),
        command=lambda: (
            hinzufügen_vokabel(),
            editor_feedback.configure(text=f"{EMOJI_OK}Vokabel hinzugefügt"),
            save_editor()
        )
    )

    neu_de.grid(row=row, column=0, padx=5, pady=10, sticky='ew')
    neu_en.grid(row=row, column=1, padx=5, pady=10, sticky='ew')
    add_btn.grid(row=row, column=2, padx=5, pady=10)

    # Enter in neu_de → fokus neu_en
    neu_de.bind("<Return>", lambda e: neu_en.focus())
    # Enter in neu_en → hinzufügen, speichern, neu focus
    def on_enter_new(event):
        if neu_de.get().strip() and neu_en.get().strip():
            hinzufügen_vokabel()
            editor_feedback.configure(
                text=f"{EMOJI_OK}Vokabel hinzugefügt",
                font=("Arial", 12),
                text_color="green"
            )
            save_editor()
            show_editor()
            # Statt editor_entries[-1][0].focus():
            neu_de.focus()

    neu_en.bind("<Return>", on_enter_new)

    # Nach dem Editor-Refresh: Buttons reaktivieren, um Blinken zu vermeiden
    app.after(50, lambda: reenable_all_buttons(app))

    zeige_frame('editor')

# Bugfix-Funktion: Buttons nach Editor-Refresh reaktivieren
def reenable_all_buttons(widget):
    try:
        import customtkinter as ctk
        if isinstance(widget, ctk.CTkButton):
            # Absichtlich deaktivierte Buttons nicht reaktivieren
            if not getattr(widget, '_keep_disabled', False):
                widget.configure(state="normal")
        for child in widget.winfo_children():
            reenable_all_buttons(child)
    except Exception:
        pass


def save_editor():
    global alle_vokabeln, editor_entries
    neue_liste = []
    for de_e, en_e in editor_entries:
        d = de_e.get().strip()
        e = en_e.get().strip()
        if d and e:
            neue_liste.append({'Deutsch': d, 'Englisch': e})
    alle_vokabeln = neue_liste
    save_vokabeln_csv()
    editor_feedback.configure(text=f"{EMOJI_OK}Änderungen gespeichert")
    show_editor()

def alle_vokabeln_loeschen():
    global alle_vokabeln
    alle_vokabeln = []
    save_vokabeln_csv()
    editor_feedback.configure(text=f"{EMOJI_OK}Alle Vokabeln gelöscht")
    show_editor()

def entferne_vokabel(idx):
    alle_vokabeln.pop(idx)
    save_vokabeln_csv()
    show_editor()

def hinzufügen_vokabel():
    d = neu_de.get().strip()
    e = neu_en.get().strip()
    if not d or not e:
        return

    neu_key = (d.strip(), e.strip())
    vorhandene_keys = {(v['Deutsch'], v['Englisch']) for v in alle_vokabeln}

    if neu_key in vorhandene_keys:
        editor_feedback.configure(text="Diese Vokabel existiert bereits!")
    else:
        alle_vokabeln.append({'Deutsch': d, 'Englisch': e})
        save_vokabeln_csv()
        editor_feedback.configure(text=f"{EMOJI_OK}Vokabel hinzugefügt")

    show_editor()

# ============================ Tipp-Funktion ==================================
def zeige_tipp():
    if not aktuelle_vokabel:
        return

    direction = current_question_direction or training_settings.get('direction', 'de_to_foreign')
    zielwort = (aktuelle_vokabel['Englisch'] if direction != 'foreign_to_de' else aktuelle_vokabel['Deutsch']).strip().lower()

    artikel = [
        "to ", "a ", "an ", "the ",
        "le ", "la ", "les ", "l'",
        "un ", "une ",
        "des ", "du ",
        "de la ", "de l'",
        "der ", "die ", "das ", "ein ", "eine ", "den ", "dem ", "des "
    ]

    prefix = ""
    for art in artikel:
        if zielwort.startswith(art):
            prefix = art
            break

    core = zielwort[len(prefix):].strip()
    words = core.split()
    if len(words) <= 1:
        w = words[0] if words else ""
        if len(w) <= 2:
            hint_core = "*" * len(w)
        else:
            hint_core = w[:2] + "*" * (len(w) - 2)
    else:
        parts = [words[0]]
        for w in words[1:]:
            parts.append("*" * len(w))
        hint_core = " ".join(parts)

    hint = prefix + hint_core
    feedback_label.configure(text=f"Tipp: {hint}")

def handle_global_keys(event):
    """Globale Tastatur-Events für den Trainer-Modus"""
    if 'trainer' in frames and frames['trainer'].winfo_viewable():
        try:
            focus = app.focus_get()
        except Exception:
            focus = None

        # Enter: prüft Antwort bzw. geht weiter
        if event.keysym in ('Return', 'KP_Enter'):
            # Wenn Fokus im Eingabefeld liegt, überlassen wir es dem Entry-Binding
            if focus is not None and focus == eingabe:
                return None
            if feedback_active:
                naechste_vokabel()
                return "break"
            else:
                # Nur im Eingabe-Modus per Enter prüfen
                if training_settings.get('mode', 'input') == 'input':
                    pruefe_antwort()
                    return "break"
                return None

        # Leertaste: nur im Feedback-Modus weiterblättern, sonst normales Tippen erlauben
        if event.keysym == 'space':
            if feedback_active:
                naechste_vokabel()
                return "break"
            else:
                return None  # Space ins Eingabefeld durchlassen

        # F1 für Tipp
        if event.keysym == 'F1':
            zeige_tipp()
            return "break"

    return None

# ========================== Rate Limiting für Aktionen =======================
def rate_limited_call(fn, *args, **kwargs):
    """Erlaubt höchstens 2 Aktionen pro Sekunde. Bei schnellerem Klicken wird verzögert."""
    global _action_next_time, _action_pending_job, _action_pending
    now = time.time()
    delay = _action_next_time - now
    def _run():
        global _action_pending_job, _action_pending, _action_next_time
        _action_pending_job = None
        _action_pending = None
        try:
            fn(*args, **kwargs)
        finally:
            _action_next_time = time.time() + ACTION_MIN_INTERVAL_SEC
    # wenn noch gesperrt -> planen
    if delay > 0:
        _action_pending = (fn, args, kwargs)
        try:
            if _action_pending_job and app:
                app.after_cancel(_action_pending_job)
        except Exception:
            pass
        try:
            _action_pending_job = app.after(int(delay * 1000), _run)
        except Exception:
            _run()
        return False
    # sofort ausführen
    _run()
    return True
    
# ========================== Fortschritts-Anzeige =============================
def update_fortschritt():
    """Aktualisiert die Fortschrittsanzeige basierend auf der verbleibenden Queue"""
    if not fortschritt:
        return
    
    if initial_queue_len > 0:
        current_len = len(learning_queue)
        # Fortschritt = erledigter Anteil der ursprünglichen Last
        # Wenn Queue wächst (durch Fehler), kann Fortschritt sinken
        progress = max(0.0, min(1.0, (initial_queue_len - current_len) / initial_queue_len))
    else:
        progress = 1.0 if not learning_queue else 0.0
    
    fortschritt.set(progress)

# ========================== Quiz-Logik =======================================
def naechste_vokabel():
    global aktuelle_vokabel, feedback_active, weiter_button, current_question_direction
    feedback_active = False

    # Button zurück auf "Antwort prüfen" stellen
    try:
        btn_pruefen.configure(
            text="Antwort prüfen",
            command=pruefe_antwort,
            fg_color=BTN_COLOR,
            hover_color=BTN_HOVER_COLOR
        )
        # Im Auswahlmodus Startzustand: Prüf-Button ausblenden
        if training_settings.get('mode', 'input') == 'choice':
            try:
                btn_pruefen.grid_remove()
            except Exception:
                pass
        else:
            try:
                btn_pruefen.grid()
            except Exception:
                pass
    except Exception:
        pass
    weiter_button = None

    # Eingabe wieder aktivieren (nur im Eingabe-Modus)
    try:
        if training_settings.get('mode', 'input') == 'input':
            eingabe.configure(state='normal')
    except Exception:
        pass

    # Queue abarbeiten
    if not learning_queue:
        endbildschirm()
        return

    # Versuche, direkte Wiederholung zu vermeiden
    if len(learning_queue) > 1 and aktuelle_vokabel:
        next_v = learning_queue[0]
        # Vergleiche Inhalt, da es verschiedene Dict-Objekte sein könnten
        if (next_v['Deutsch'] == aktuelle_vokabel['Deutsch'] and 
            next_v['Englisch'] == aktuelle_vokabel['Englisch']):
            # Suche erstes Element, das anders ist
            for i in range(1, len(learning_queue)):
                other = learning_queue[i]
                if (other['Deutsch'] != aktuelle_vokabel['Deutsch'] or 
                    other['Englisch'] != aktuelle_vokabel['Englisch']):
                    # Tausche
                    learning_queue[0], learning_queue[i] = learning_queue[i], learning_queue[0]
                    break
    
    aktuelle_vokabel = learning_queue.pop(0)

    # Abfragerichtung berücksichtigen (inkl. Gemischt)
    set_dir = training_settings.get('direction', 'de_to_foreign')
    if set_dir == 'mixed':
        current_question_direction = random.choice(['de_to_foreign', 'foreign_to_de'])
    else:
        current_question_direction = set_dir
    if current_question_direction == 'foreign_to_de':
        frage_label.configure(text=f"Was heißt: {aktuelle_vokabel['Englisch']} auf Deutsch?")
    else:
        frage_label.configure(text=f"Was heißt: {aktuelle_vokabel['Deutsch']} auf {aktuelle_sprache}?")

    # UI je nach Modus aktualisieren
    update_trainer_mode_ui()

    # Eingabefeld zurücksetzen und Fokus setzen (nur im Eingabe-Modus)
    if training_settings.get('mode', 'input') == 'input':
        try:
            eingabe.delete(0, tk.END)
            feedback_label.configure(text="", text_color=TEXT_COLOR)
            update_fortschritt()
            eingabe.focus_set()
        except Exception:
            pass
    else:
        # Auswahl-Optionen füllen
        try:
            populate_choice_options()
            feedback_label.configure(text="", text_color=TEXT_COLOR)
            update_fortschritt()
            # Fokus auf erste Option
            if choice_buttons:
                app.after(50, lambda: choice_buttons[0].focus_set())
        except Exception:
            pass

def pruefe_antwort(event=None, user_answer=None):
    global xp_session, xp_woche, xp_gesamt, feedback_active

    # Falls schon Feedback aktiv ist, nicht nochmal prüfen
    if feedback_active:
        return

    if aktuelle_vokabel is None:
        return

    # Antwort beschaffen
    if user_answer is not None:
        ant = str(user_answer).strip()
    else:
        ant = eingabe.get().strip() if eingabe else ""
        # Eingabe deaktivieren nur im Eingabe-Modus
        try:
            if training_settings.get('mode', 'input') == 'input':
                eingabe.configure(state='disabled')
        except Exception:
            pass

    # Korrekte Antwort je nach Richtung wählen
    direction = current_question_direction or training_settings.get('direction', 'de_to_foreign')
    kor = (aktuelle_vokabel['Englisch'] if direction != 'foreign_to_de' else aktuelle_vokabel['Deutsch']).strip()
    key = (aktuelle_vokabel['Deutsch'], aktuelle_vokabel['Englisch'])

    # Auswahl-Buttons deaktivieren, wenn im Auswahlmodus
    if training_settings.get('mode', 'input') == 'choice':
        set_choice_buttons_state('disabled')

    global combo_streak
    xp_gewinn = 0
    is_correct = False
    mode = training_settings.get('mode', 'input')

    if ant.lower() == kor.lower():
        feedback_label.configure(text=f"{EMOJI_OK}Richtig!", text_color=SUCCESS_COLOR)
        vokabel_statistik[key]['richtig'] += 1
        vokabel_repetitions[key] = vokabel_repetitions.get(key, 0) + 1
        xp_gewinn = 10 if mode == 'input' else 2
        is_correct = True
    else:
        if mode == 'input' and is_typo(ant, kor):
            feedback_label.configure(
                text=f"{EMOJI_PART}Fast richtig! \nRichtig: {kor}",
                text_color=WARNING_COLOR
            )
            vokabel_statistik[key]['richtig'] += 1
            vokabel_repetitions[key] = vokabel_repetitions.get(key, 0) + 1
            xp_gewinn = 5
            is_correct = True
        else:
            feedback_label.configure(
                text=f"{EMOJI_BAD}Falsch! Richtig: {kor}",
                text_color=ERROR_COLOR
            )
            vokabel_statistik[key]['falsch'] += 1
            # Bei Fehler: Vokabel wieder hinten anstellen
            learning_queue.append(aktuelle_vokabel)

    # Combo-Bonus (nur Eingabe-Modus)
    combo_bonus = 0
    if mode == 'input':
        if is_correct:
            combo_streak += 1
            if combo_streak >= 5 and combo_streak % 5 == 0:
                level = combo_streak // 5
                combo_bonus = level * 2
                xp_gewinn += combo_bonus
                try:
                    combo_text = feedback_label.cget('text')
                    feedback_label.configure(
                        text=f"{combo_text}\n🔥 Combo x{combo_streak}! +{combo_bonus} Bonus-XP"
                    )
                except Exception:
                    pass
        else:
            combo_streak = 0

    # XP nur in der Session akkumulieren; Woche/Gesamt erst am Endbildschirm
    xp_session += xp_gewinn
    punktzahl_label.configure(text=f"XP: +{xp_session} XP")
    statistik_speichern()
    update_fortschritt()

    feedback_active = True

    # Statt separatem Weiter-Button: den vorhandenen Button temporär zum Weiter-Button umfunktionieren
    try:
        btn_pruefen.configure(
            text="Weiter",
            command=naechste_vokabel,
            fg_color=SUCCESS_COLOR,
            hover_color="#047857"
        )
        # Im Auswahlmodus Button sichtbar machen
        if training_settings.get('mode', 'input') == 'choice':
            try:
                btn_pruefen.grid()
            except Exception:
                pass
        # Fokus verzögert auf Button setzen, damit aktuelles Enter nicht überschießt
        app.after(50, lambda: btn_pruefen.focus_set())
    except Exception:
        # Fallback: wenn Button nicht existiert, mit Enter/Space über globale Handler weitermachen
        pass

# Helper für Auswahlmodus

def generate_distractors(correct: str, count: int = 2) -> list[str]:
    """Wählt 'count' falsche Übersetzungen aus allen Vokabeln aus."""
    direction = current_question_direction or training_settings.get('direction', 'de_to_foreign')
    field = 'Englisch' if direction != 'foreign_to_de' else 'Deutsch'
    pool = [v[field] for v in alle_vokabeln if v[field].strip().lower() != correct.strip().lower()]
    random.shuffle(pool)
    result = []
    for w in pool:
        if w not in result:
            result.append(w)
        if len(result) >= count:
            break
    return result

def populate_choice_options():
    """Befüllt die drei Auswahl-Buttons mit 1x richtig + 2x falsch (gemischt)."""
    if not aktuelle_vokabel or not choice_buttons:
        return
    direction = current_question_direction or training_settings.get('direction', 'de_to_foreign')
    kor = aktuelle_vokabel['Englisch'] if direction != 'foreign_to_de' else aktuelle_vokabel['Deutsch']
    distractors = generate_distractors(kor, 2)
    options = [kor] + distractors
    random.shuffle(options)

    # Buttons belegen/anzeigen
    for i, btn in enumerate(choice_buttons):
        try:
            btn.pack_forget()  # zunächst alle verstecken
        except Exception:
            pass
    for i, text in enumerate(options):
        if i < len(choice_buttons):
            btn = choice_buttons[i]
            btn.configure(text=text, command=lambda t=text: rate_limited_call(choice_selected_text, t), state="normal")
            try:
                btn.pack(pady=8, fill='x')
            except Exception:
                pass


def set_choice_buttons_state(state: str):
    for btn in choice_buttons:
        try:
            btn.configure(state=state)
        except Exception:
            pass


def choice_selected(index: int):
    # Fallback falls alter Command noch referenziert
    if 0 <= index < len(choice_buttons):
        txt = choice_buttons[index].cget('text')
        choice_selected_text(txt)


def choice_selected_text(answer_text: str):
    if feedback_active:
        return
    pruefe_antwort(user_answer=answer_text)


def update_trainer_mode_ui():
    """Blendet im Trainer je nach Modus Eingabefeld oder Auswahl-Buttons ein."""
    mode = training_settings.get('mode', 'input')
    if mode == 'choice':
        # Eingabe ausblenden, Auswahl einblenden
        try:
            input_frame.grid_remove()
        except Exception:
            pass
        try:
            choice_frame.grid(row=0, column=0)
            set_choice_buttons_state('normal')
        except Exception:
            pass
        try:
            btn_pruefen.grid_remove()  # erst nach Antwort zeigen
        except Exception:
            pass
    else:
        # Auswahl ausblenden, Eingabe einblenden
        try:
            choice_frame.grid_remove()
        except Exception:
            pass
        try:
            input_frame.grid(row=0, column=0)
        except Exception:
            pass
        try:
            btn_pruefen.grid()  # Prüfen-Button sichtbar
        except Exception:
            pass


def update_start_choice_access():
    """Aktualisiert den Zustand des Start-(Auswahl)-Buttons.
    Bedingung: Mindestens 5 Vokabeln in der aktuellen Sprache erforderlich.
    """
    try:
        count = len(alle_vokabeln)
    except Exception:
        count = 0

    if not start_choice_button:
        return

    if count >= 5:
        # Reaktivieren inkl. Original-Command
        try:
            if start_choice_command:
                start_choice_button.configure(command=start_choice_command)
            start_choice_button.configure(state="normal", fg_color=BTN_COLOR, hover_color=BTN_HOVER_COLOR)
            start_choice_button._keep_disabled = False
        except Exception:
            pass
    else:
        # Vollständig sperren: state disabled und command entfernen
        try:
            start_choice_button.configure(command=lambda: None, state="disabled", fg_color=DISABLED_COLOR, hover_color=DISABLED_HOVER)
            start_choice_button._keep_disabled = True
        except Exception:
            pass

    # Nach Anpassung alle drei Start-Container ausrichten
    try:
        app.after(0, update_start_buttons_alignment)
    except Exception:
        pass


def on_locked_choice_click(event=None):
    """Wenn der Auswahl-Button gesperrt ist und man klickt, leichtes Wackeln auslösen."""
    try:
        # Wenn keine 5 Vokabeln vorhanden sind, ist gesperrt
        if len(alle_vokabeln) >= 5:
            return
    except Exception:
        return

    # Nur auslösen, wenn Container existiert und nicht bereits in Animation
    if not start_choice_inner:
        return
    cont = start_choice_inner
    if not cont:
        return
    global _shake_in_progress
    if _shake_in_progress:
        return

    _shake_in_progress = True

    # Kleine Shake-Animation über Änderung des x-Offsets (place)
    A = START_CHOICE_SHAKE_AMPL
    seq = [0, +A, -A, int(0.7*A), int(-0.7*A), int(0.4*A), int(-0.4*A), 0]

    def step(i=0):
        global _shake_in_progress
        if i >= len(seq):
            _shake_in_progress = False
            return
        try:
            dx = seq[i]
            # place-Offset aktualisieren (nur innerer Frame wackelt)
            cont.place_configure(x=START_CHOICE_SHAKE_BASE_X + dx)
        except Exception:
            _shake_in_progress = False
            return
        # Nächsten Schritt planen
        try:
            cont.after(18, lambda: step(i + 1))
        except Exception:
            _shake_in_progress = False
            return

    step(0)
    

# ============================ Frame-Steuerung & UI-Updates ===================

def update_start_buttons_alignment():
    # Synchronisiert die Breite/Höhe beider Start-Container und richtet die Unterkanten aus.
    try:
        if not (start_input_container and start_choice_container and start_input_inner and start_choice_inner):
            return
        # benötigte Breiten ermitteln
        start_input_inner.update_idletasks(); start_choice_inner.update_idletasks()
        if start_input_button: start_input_button.update_idletasks()
        if start_choice_button: start_choice_button.update_idletasks()
        # Breite anhand der größeren inneren Breite
        need_w = max(start_input_inner.winfo_reqwidth(), start_choice_inner.winfo_reqwidth()) + 2 * START_CHOICE_SHAKE_AMPL + 12
        # gleiche Breite setzen
        start_input_container.configure(width=need_w)
        start_choice_container.configure(width=need_w)
        # gleiche (minimale) Höhe: Buttonhöhe + optional Labelhöhe + kleiner Abstand
        btn_h_left = start_input_button.winfo_reqheight() if start_input_button else 0
        btn_h_right = start_choice_button.winfo_reqheight() if start_choice_button else 0
        need_h = max(btn_h_left, btn_h_right)
        try:
            start_input_container.configure(height=need_h)
            start_choice_container.configure(height=need_h)
            # propagate aus, damit Höhe gehalten wird
            start_input_container.pack_propagate(False)
            start_choice_container.pack_propagate(False)
        except Exception:
            pass
    except Exception:
        pass

def zeige_frame(name: str):
    """Blendet den aktuellen Frame aus und zeigt den gewählten an (flackerarm)."""
    global current_visible_frame
    try:
        if current_visible_frame and current_visible_frame.winfo_ismapped():
            current_visible_frame.pack_forget()
    except Exception:
        pass

    if name in frames:
        try:
            frames[name].pack(fill='both', expand=True)
            current_visible_frame = frames[name]
            # Nach dem Anzeigen sanftes Resize
            if app:
                app.after_idle(update_font_sizes)
        except Exception:
            pass


# ====================== Reset- & Neustart-Funktionen ========================

def reset_for_new_attempt():
    global xp_session, vokabeln_zu_lernen, runde_status, vokabel_repetitions, learning_queue, initial_queue_len, combo_streak
    xp_session = 0
    combo_streak = 0
    vokabeln_zu_lernen = alle_vokabeln.copy()
    random.shuffle(vokabeln_zu_lernen)
    runde_status.clear()
    vokabel_repetitions.clear()

    # Queue basierend auf Statistik
    learning_queue = []
    for v in alle_vokabeln:
        key = (v['Deutsch'], v['Englisch'])
        stats = vokabel_statistik.get(key, {'richtig': 0, 'falsch': 0})
        richtig = stats.get('richtig', 0)
        falsch = stats.get('falsch', 0)
        total = richtig + falsch
        
        if total == 0:
            count = 1
        else:
            rate = richtig / total
            count = 1 + int(round(4 * (1.0 - rate)))
            
        learning_queue.extend([v] * count)
    
    random.shuffle(learning_queue)
    initial_queue_len = len(learning_queue)


def starte_neu():
    if not alle_vokabeln:
        messagebox.showerror("Fehler", "Keine Vokabeln verfügbar. Bitte fügen Sie zuerst Vokabeln hinzu.")
        return
    reset_for_new_attempt()
    zeige_frame('trainer')
    try:
        if punktzahl_label:
            punktzahl_label.configure(text=f"XP: +{xp_session} XP")
    except Exception:
        pass
    naechste_vokabel()

def endbildschirm():
    global xp_woche, xp_gesamt
    # XP der Session jetzt erst in Woche/Gesamt übertragen und speichern
    xp_woche  += xp_session
    xp_gesamt += xp_session
    speichere_xp()
    update_xp_label()

    if 'ende' in frames:
        try:
            frames['ende'].destroy()
        except Exception:
            pass
    frame = ctk.CTkFrame(app); frames['ende'] = frame

    outer = ctk.CTkFrame(frame)
    outer.pack(expand=True, fill='both')

    center = ctk.CTkFrame(outer, fg_color="transparent")
    center.pack(expand=True)

    title_lbl = ctk.CTkLabel(center, text=f"{EMOJI_PARTY}Fertig gelernt!", font=('Arial', 48))
    title_lbl.pack(pady=32)

    if aktuelle_sprache:
        lbl_lang = ctk.CTkLabel(center, text=aktuelle_sprache.capitalize(), font=('Segoe UI', 24, 'bold'), text_color=SPRACH_COLOR)
        lbl_lang.pack(pady=(0, 16))

    score_lbl = ctk.CTkLabel(center, text=f"{xp_session} XP diese Session", font=('Arial', 26))
    score_lbl.pack(pady=8)
    ctk.CTkLabel(center, text=f"XP diese Woche gesamt: {xp_woche}", font=('Arial', 18), text_color=LIGHT_TEXT).pack(pady=8)

    btn_wrap = ctk.CTkFrame(center, fg_color="transparent")
    btn_wrap.pack(fill='x', padx=40, pady=20)

    ctk.CTkButton(btn_wrap, text="Wiederholen", fg_color=BTN_COLOR, height=60, corner_radius=20,
                  command=starte_neu).pack(pady=8, fill='x')
    ctk.CTkButton(btn_wrap, text="Zum Startbildschirm", fg_color=BTN_COLOR, height=60, corner_radius=20,
                  command=lambda: zeige_frame('start')).pack(pady=8, fill='x')

    zeige_frame('ende')

# ========================== Main =============================================

# ============================ Hauptprogramm ==================================
if __name__ == "__main__":
    app = ctk.CTk()
    try:
        app.title("Vokabeltrainer")
    except Exception:
        pass
    try:
        if sys.platform == 'darwin':
            app.geometry(f"{app.winfo_screenwidth()}x{app.winfo_screenheight()}+0+0")
        else:
            app.state('zoomed')
    except (tk.TclError, Exception):
        pass
    if sys.platform != 'darwin':
        try:
            app.geometry("1200x800")
        except Exception:
            pass
    try:
        app.update_idletasks()
    except Exception:
        pass
    try:
        app.protocol("WM_DELETE_WINDOW", lambda: (statistik_speichern(), app.quit()))
    except Exception:
        pass

    # Resize-Handling
    try:
        app.bind("<Configure>", debounced_update_font_sizes)
    except Exception:
        pass

    # Icons laden
    try:
        haus_icon  = lade_icon("haus.png")
        birne_icon = lade_icon("birne.png")
        tipp_icon  = lade_icon("tipp.png")
    except Exception:
        pass

    # Sprache initialisieren
    try:
        initialisiere_sprache('englisch')
        update_fenstertitel()
    except Exception:
        pass

    # Benutzername laden / erfragen
    try:
        name = lade_benutzernamen()
        if name:
            aktueller_nutzer = name
        else:
            frage_benutzernamen()
    except Exception as e:
        print(f"[User] Fehler: {e}")

    # Firebase initialisieren
    try:
        init_firebase()
    except Exception as e:
        print(f"[Firebase] Init-Fehler: {e}")

    # XP laden & wöchentlichen Reset prüfen
    try:
        pruefe_weekly_reset()
        lade_xp()
    except Exception as e:
        print(f"[XP] Ladefehler: {e}")

    # Screens vorbereiten
    try:
        trainer()
        statistik()
        editor()
        einstellungen_screen()
        rangliste_screen()
    except Exception as _e:
        pass

    # Start-Screen wählen
    try:
        if aktuelle_sprache is None or not vorhandene_sprachen():
            sprache_verwalten_screen()
        else:
            startbildschirm()
    except Exception:
        # Fallback
        startbildschirm()

    # PIN-Check für aktiven Benutzer (Migration: bestehende Accounts ohne PIN)
    try:
        if aktueller_nutzer and firebase_db:
            hat_pin_status = _hat_pin(aktueller_nutzer)
            if hat_pin_status is False:
                # Account existiert, aber hat noch keinen PIN → Erstellung erzwingen
                pin_erstellen_dialog(aktueller_nutzer, callback=lambda: None, closable=False)
            # hat_pin_status is None → Firebase-Fehler, überspringen (nächstes Mal)
            # hat_pin_status is True → alles gut, kein PIN nötig beim normalen Start
    except Exception as e:
        print(f"[PIN] Startup-Check fehlgeschlagen: {e}")

    # Siegerehrung prüfen (nach Screens + PIN-Check, damit Hauptfenster existiert)
    try:
        ceremony_data = pruefe_siegerehrung()
        if ceremony_data:
            siegerehrung_anzeigen(ceremony_data)
    except Exception as e:
        print(f"[Siegerehrung] Startup-Check fehlgeschlagen: {e}")

    # Flask-Server im Hintergrund starten
    try:
        threading.Thread(target=start_flask_server, daemon=True).start()
    except Exception as e:
        print(f"[Flask] Server-Start fehlgeschlagen: {e}")

    # Global Keys
    try:
        app.bind_all('<KeyPress-Return>', handle_global_keys)
        app.bind_all('<KeyPress-KP_Enter>', handle_global_keys)
        app.bind_all('<KeyPress-space>', handle_global_keys)
        app.bind_all('<KeyPress-F1>', handle_global_keys)
    except Exception:
        pass

    app.mainloop()


