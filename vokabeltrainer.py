# ============================== Imports ======================================
import os
import sys
import json
import csv
import random
import re
import openai
import pytesseract
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image




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
    """
    Erkennt Tippfehler basierend auf Tastatur-Layout und Wahrscheinlichkeit
    Verwendet die globalen Training-Einstellungen für die Toleranz
    """
    # Wenn Tippfehler-Toleranz auf "Baby" (0) gesetzt ist, keine Tippfehler erlauben
    if training_settings['typo_tolerance'] == 0:
        return False
    
    # Nur lange Wörter (mehr als 6 Buchstaben) können als Tippfehler gewertet werden
    if len(correct_answer.strip()) <= 6:
        return False
    
    # Threshold basierend auf Tippfehler-Toleranz
    if threshold is None:
        thresholds = [0.9, 0.7, 0.6, 0.4, 0.2]  # Baby, Leicht, Mittel, Schwer, Profi
        threshold = thresholds[training_settings['typo_tolerance']]
    
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

# ======================= Pfade zu CSV und JSON ==============================
CSV_DATEI       = os.path.join(VOCAB_DIR, 'vokabeln.csv')
STATISTIK_DATEI = os.path.join(STAT_DIR, 'statistik.json')

# ======================= OpenAI API Key =======================================
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
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
punktzahl           = 100
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

statistik_frame     = None
stat_feedback_label = None
editor_frame        = None
editor_feedback     = None
neu_de              = None
neu_en              = None
aktuelle_sprache = None

# Sprachanzeige-Widgets
sprach_anzeige_label = None

# Training-Einstellungen
training_settings = {
    'typo_tolerance': 2,  # 0=Baby, 1=Leicht, 2=Mittel, 3=Schwer, 4=Profi
    'repetitions': 2      # 1-5 Wiederholungen
}

# Vokabel-Wiederholungs-Tracking
vokabel_repetitions = {}  # Speichert wie oft jede Vokabel richtig beantwortet wurde

# Feedback-System
feedback_active = False
weiter_button = None

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
    return os.path.join(VOCAB_DIR, f'vokabeln_{aktuelle_sprache}.csv')

def get_statistik_datei():
    return os.path.join(STAT_DIR, f'statistik_{aktuelle_sprache}.json')

def vorhandene_sprachen():
    if not os.path.exists(VOCAB_DIR):
        os.makedirs(VOCAB_DIR)
    files = [f for f in os.listdir(VOCAB_DIR) if f.startswith("vokabeln_") and f.endswith(".csv")]
    return [f[len("vokabeln_"):-len(".csv")] for f in files]

def initialisiere_sprache(sprache):
    global aktuelle_sprache, vokabel_statistik, alle_vokabeln, vokabeln_zu_lernen
    aktuelle_sprache = sprache.lower()

    # CSV-Datei anlegen falls nicht vorhanden:
    if not os.path.exists(VOCAB_DIR):
        os.makedirs(VOCAB_DIR)
    csv_path = get_csv_datei()
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['Deutsch', 'Englisch'], delimiter=';')
            writer.writeheader()

    # Statistik-Datei anlegen falls nicht vorhanden:
    if not os.path.exists(STAT_DIR):
        os.makedirs(STAT_DIR)
    stat_path = get_statistik_datei()
    if not os.path.exists(stat_path):
        with open(stat_path, 'w', encoding='utf-8') as f:
            json.dump({}, f)

    vokabel_statistik = statistik_laden()
    lade_vokabeln()
    statistik_bereinigen()
    
    # Fenstertitel aktualisieren
    update_fenstertitel()
    
    # Sprachanzeige aktualisieren falls vorhanden
    update_sprachanzeige()

# ========================= Datei-I/O & Statistik =============================
def statistik_laden():
    if not os.path.exists(STAT_DIR):
        os.makedirs(STAT_DIR)
    if not os.path.exists(get_statistik_datei()):
        with open(STATISTIK_DATEI, 'w', encoding='utf-8') as f:
            json.dump({}, f)
    with open(get_statistik_datei(), 'r', encoding='utf-8') as f:
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

def lade_vokabeln():
    global alle_vokabeln, vokabeln_zu_lernen
    alle_vokabeln.clear()

    if not os.path.exists(VOCAB_DIR):
        os.makedirs(VOCAB_DIR)

    if not os.path.exists(get_csv_datei()):

        messagebox.showerror(
            "Datei nicht gefunden",
            f"Die Datei {CSV_DATEI} existiert nicht."
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


def save_vokabeln_csv():
    with open(get_csv_datei(), 'w', newline='', encoding='utf-8') as f:

        writer = csv.DictWriter(f, fieldnames=['Deutsch', 'Englisch'], delimiter=';')
        writer.writeheader()
        for v in alle_vokabeln:
            writer.writerow({'Deutsch': v['Deutsch'], 'Englisch': v['Englisch']})
    lade_vokabeln()
    statistik_bereinigen()

# Initial load - wird nach der Definition der UI-Funktionen aufgerufen


# ====================== OCR + GPT-Filter ======================================
def extract_pairs_with_gpt(raw_text: str) -> list[dict]:
    system = (
    "Du arbeitest für ein Vokabeltrainer-Programm, das automatisch Vokabelpaare aus eingescannten Büchern und Arbeitsblättern importiert. "
    "Der Eingabetext stammt direkt von einer OCR-Software und enthält oft zusätzliche Elemente wie Überschriften, Beispiele, unstrukturierte Kommentare oder falsche Erkennungen. "
    "Deine Aufgabe ist es ausschließlich echte, vollständige und plausible Vokabelpaare zu extrahieren und zu liefern. "
    "Erkenne automatisch, welche Wörter Deutsch und welche aus einer Fremdsprache sind (z. B. Englisch oder Französisch) und stelle sie korrekt zu einem Schulbuch-Vokabelpaar zusammen. "
    "Achte darauf: "
    "- Gib nur Paare zurück, bei denen klar erkennbar ist, dass sie ein Vokabelpaar sind. "
    "- Ignoriere Überschriften, Beispiele, Sätze, Kommentare und unvollständige Einträge. "
    "- Bereinige OCR-Fehler wie falsch erkannte Buchstaben oder zeichen und liefere die plausibelste, korrekte Schreibweise. "
    "- Nutze dein Wissen über häufig vorkommende Vokabeln, um Fehler zu korrigieren, aber erfinde keine neuen Paare! "
    "- Die Reihenfolge ist immer: Deutsch links, Fremdsprache rechts. "
    "Gib die Paare nur im Format 'Deutsch;Fremdsprache' zurück. "
    "Beispiel: 'der Hund;the dog'. "
    "Schreibe genau eine Zeile pro Vokabelpaar und ansonsten nichts."
)


    resp = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": raw_text}
        ]
    )

    lines = resp.choices[0].message.content.strip().splitlines()
    pairs = []
    for line in lines:
        if not line.strip():
            continue
        parts = re.split(r"\s*[;,]\s*", line.strip(), maxsplit=1)
        if len(parts) != 2:
            print(f"⚠️ Ungültige Zeile übersprungen: {line!r}")
            continue

        deu, foreign = parts[0].strip(), parts[1].strip()

        # Hier wird direkt „umgedreht“:
        pairs.append({"Deutsch": deu, "Englisch": foreign})

    return pairs



def extract_pairs_from_image(path: str) -> list[dict]:
    img = Image.open(path)
    configure_tesseract_path()
    alle_sprachen = get_all_tesseract_langs()
    if not alle_sprachen:
        alle_sprachen = 'deu'
    raw = pytesseract.image_to_string(
        img,
        lang=alle_sprachen,
        config="--oem 1 --psm 6"
    )
    return extract_pairs_with_gpt(raw)

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
    bestehend = {(v['Deutsch'], v['Englisch']) for v in alle_vokabeln}

    for p in paths:
        try:
            neue = extract_pairs_from_image(p)
            for v in neue:
                key = (v['Deutsch'], v['Englisch'])
                if key not in bestehend:
                    alle_vokabeln.append(v)
                    bestehend.add(key)
                    imported += 1
        except Exception as e:
            editor_feedback.configure(text=f"Import fehlgeschlagen: {e}")
            return

    save_vokabeln_csv()
    show_editor()
    editor_feedback.configure(text=f"{imported} neue Vokabeln importiert ✅")

# =========================== UI Theme ========================================
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

# Moderne Farbpalette
BTN_COLOR       = "#6366f1"  # Indigo
BTN_HOVER_COLOR = "#4f46e5"  # Dunkleres Indigo
SPRACH_COLOR    = "#10b981"  # Emerald (für Sprachanzeige)
SUCCESS_COLOR   = "#059669"  # Dunkleres Emerald
WARNING_COLOR   = "#f59e0b"  # Amber
ERROR_COLOR     = "#dc2626"  # Rot
TEXT_COLOR      = "#374151"  # Grau für Text
LIGHT_TEXT      = "#6b7280"  # Helleres Grau

BUTTON_FONT     = ("Segoe UI", 14)
LABEL_COLOR     = "white"

# ======================= Hauptfenster & Setup ================================
app = ctk.CTk()
app.title("Vokabeltrainer")
try:
    app.state('zoomed')
except tk.TclError:
    app.state('normal')
app.protocol("WM_DELETE_WINDOW", lambda: (statistik_speichern(), app.quit()))

frames = {}
def zeige_frame(name):
    for f in frames.values():
        f.pack_forget()
    frames[name].pack(fill='both', expand=True)
    
    # Sprachanzeige aktualisieren wenn zum Startbildschirm gewechselt wird
    if name == 'start' and sprach_anzeige_label and aktuelle_sprache:
        sprach_anzeige_label.configure(text=aktuelle_sprache.capitalize())

def update_fenstertitel():
    """Aktualisiert den Fenstertitel mit der aktuellen Sprache"""
    if aktuelle_sprache:
        app.title(f"Vokabeltrainer - {aktuelle_sprache.capitalize()}")
    else:
        app.title("Vokabeltrainer")

def update_sprachanzeige():
    """Aktualisiert die Sprachanzeige in allen Frames"""
    if sprach_anzeige_label and aktuelle_sprache:
        sprach_anzeige_label.configure(text=aktuelle_sprache.capitalize())

app.geometry("1200x800")
app.update()
try:
    app.state('zoomed')
except tk.TclError:
    app.state('normal')

# ======================= Dynamische Schriftgrößen ============================
base_font_size = 20
def update_font_sizes(event=None):
    if not app.winfo_exists():
        return

    w, h = app.winfo_width(), app.winfo_height()
    if w < 300 or h < 300:
        return

    # Maßstab
    scale    = min(w/1200, h/800)
    sz_title = max(int(40 * scale), 18)
    sz_btn   = max(int(16 * scale), 10)
    sz_lbl   = max(int(20 * scale), 12)
    btn_w    = int(200 * scale)
    btn_h    = int(50 * scale)
    icon_w   = icon_h = int(60 * scale)

    def is_descendant(widget, ancestor):
        parent = widget
        while True:
            if parent == ancestor:
                return True
            par_name = parent.winfo_parent()
            if not par_name:
                return False
            parent = parent.nametowidget(par_name)

    def resize_widget(widget):
        # Statistik- und Editor-Frame überspringen
        if (statistik_frame and is_descendant(widget, statistik_frame)) or \
           (editor_frame    and is_descendant(widget, editor_frame)):
            return

        # Label anpassen
        if isinstance(widget, ctk.CTkLabel):
            text = widget.cget("text")
            if text and len(text) > 10:
                widget.configure(font=("Arial", sz_title))
            else:
                widget.configure(font=("Arial", sz_lbl))

        # Button anpassen (Icons ignorieren)
        elif isinstance(widget, ctk.CTkButton):
            if widget._image:
                # Icon-Button bleibt in der initialen Größe
                pass
            else:
                widget.configure(width=btn_w, height=btn_h, font=("Arial", sz_btn))

        # Entry anpassen
        elif isinstance(widget, ctk.CTkEntry):
            widget.configure(font=("Arial", sz_lbl))

        # ProgressBar anpassen
        elif isinstance(widget, ctk.CTkProgressBar):
            widget.configure(width=int(400 * scale))

        # Rekursion für alle Kinder
        for child in widget.winfo_children():
            resize_widget(child)

    # Starte Rekursion am Wurzel-Widget
    resize_widget(app)




resize_job = None
def debounced_update_font_sizes(event=None):
    global resize_job
    if resize_job:
        app.after_cancel(resize_job)
    resize_job = app.after(100, update_font_sizes)

app.bind("<Configure>", debounced_update_font_sizes)

# ======================== Light/Dark Mode ====================================
def wechsel_mode():
    ctk.set_appearance_mode("Light" if ctk.get_appearance_mode()=="Dark" else "Dark")

# ========================== Startbildschirm ==================================
def startbildschirm():
    frame = ctk.CTkFrame(app); frames['start'] = frame

    container = ctk.CTkFrame(frame)
    container.pack(expand=True, fill='both')

    ctk.CTkLabel(container, text="Vokabeltrainer", font=('Segoe UI', 48, 'bold')).pack(pady=40)

    # Sprachanzeige hinzufügen
    global sprach_anzeige_label
    sprach_anzeige_label = ctk.CTkLabel(
        container, 
        text=aktuelle_sprache.capitalize() if aktuelle_sprache else "Keine Sprache",
        font=('Segoe UI', 12),
        text_color=SPRACH_COLOR
    )
    sprach_anzeige_label.pack(pady=(0, 30))

    ctk.CTkButton(container, text="Start", fg_color=BTN_COLOR,
                  command=starte_neu,
                  width=220, height=55, corner_radius=25,
                  font=('Segoe UI', 16, 'bold')).pack(pady=12)

    ctk.CTkButton(container, text="Statistiken", fg_color=BTN_COLOR,
                  command=lambda: [zeige_frame('statistik'), zeige_statistik()],
                  width=220, height=55, corner_radius=25,
                  font=('Segoe UI', 16, 'bold')).pack(pady=12)

    ctk.CTkButton(container, text="Vokabeln bearbeiten", fg_color=BTN_COLOR,
                  command=lambda: [show_editor(), zeige_frame('editor')],
                  width=220, height=55, corner_radius=25,
                  font=('Segoe UI', 16, 'bold')).pack(pady=12)

    ctk.CTkButton(container, text="Einstellungen", fg_color=BTN_COLOR,
                  command=lambda: zeige_frame('einstellungen'),
                  width=220, height=55, corner_radius=25,
                  font=('Segoe UI', 16, 'bold')).pack(pady=12)

    btn_row = ctk.CTkFrame(container)
    btn_row.pack(pady=20)

    ctk.CTkButton(btn_row, image=birne_icon, text="",
                  width=50, height=50, corner_radius=25,
                  fg_color="transparent", hover_color="#f3f4f6",
                  command=wechsel_mode).pack(side='left', padx=8)

    ctk.CTkButton(btn_row, image=flagge_icon, text="",
                  width=50, height=50, corner_radius=25,
                  fg_color="transparent", hover_color="#f3f4f6",
                  command=sprache_verwalten_screen).pack(side='left', padx=8)



    zeige_frame('start')

def einstellungen_screen():
    """Einstellungsseite mit Schiebereglern für Training-Parameter"""
    frame = ctk.CTkFrame(app); frames['einstellungen'] = frame
    
    container = ctk.CTkFrame(frame)
    container.pack(expand=True, fill='both', padx=50, pady=50)
    
    # Header
    ctk.CTkLabel(
        container, 
        text="Training-Einstellungen", 
        font=('Arial', 32, 'bold')
    ).pack(pady=(0, 40))
    
    # Tippfehler-Toleranz
    typo_frame = ctk.CTkFrame(container)
    typo_frame.pack(fill='x', pady=20)
    
    ctk.CTkLabel(
        typo_frame, 
        text="Tippfehler-Toleranz", 
        font=('Segoe UI', 18, 'bold')
    ).pack(pady=(20, 10))
    
    typo_labels = ["Baby", "Leicht", "Mittel", "Schwer", "Profi"]
    typo_colors = ["#dc2626", "#f59e0b", "#eab308", "#22c55e", "#059669"]
    
    typo_slider = ctk.CTkSlider(
        typo_frame,
        from_=0,
        to=4,
        number_of_steps=4,
        command=lambda value: update_typo_label(int(value), typo_value_label, typo_labels, typo_colors)
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
    
    # Wiederholungen
    rep_frame = ctk.CTkFrame(container)
    rep_frame.pack(fill='x', pady=20)
    
    ctk.CTkLabel(
        rep_frame, 
        text="Wiederholungen pro Vokabel", 
        font=('Segoe UI', 18, 'bold')
    ).pack(pady=(20, 10))
    
    rep_slider = ctk.CTkSlider(
        rep_frame,
        from_=1,
        to=5,
        number_of_steps=4,
        command=lambda value: update_rep_label(int(value), rep_value_label)
    )
    rep_slider.set(training_settings['repetitions'])
    rep_slider.pack(pady=10)
    
    rep_value_label = ctk.CTkLabel(
        rep_frame,
        text=str(training_settings['repetitions']),
        font=('Segoe UI', 16),
        text_color=SUCCESS_COLOR
    )
    rep_value_label.pack(pady=(0, 20))
    
    # Buttons
    button_frame = ctk.CTkFrame(container)
    button_frame.pack(fill='x', pady=40)
    
    def start_training():
        # Einstellungen speichern
        training_settings['typo_tolerance'] = int(typo_slider.get())
        training_settings['repetitions'] = int(rep_slider.get())
        
        # Training starten
        reset_for_new_attempt()
        punktzahl_label.configure(text=f"Punktzahl: {punktzahl}")
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

def update_rep_label(value, label):
    """Aktualisiert das Label für die Wiederholungen"""
    label.configure(text=str(value))

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
        row = ctk.CTkFrame(frame)
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
    neu_frame = ctk.CTkFrame(frame)
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
def trainer():
    frame = ctk.CTkFrame(app); frames['trainer'] = frame

    container = ctk.CTkFrame(frame)
    container.pack(expand=True, fill='both')

    top = ctk.CTkFrame(container)
    top.pack(side='top', fill='x')

    ctk.CTkButton(top, image=haus_icon, text="", fg_color=BTN_COLOR,
                  width=40, height=40, corner_radius=20,
                  command=lambda: zeige_frame('start')).pack(side='left', padx=10, pady=10)

    # Sprachanzeige im Trainer
    if aktuelle_sprache:
        ctk.CTkLabel(
            top, 
            text=aktuelle_sprache.capitalize(), 
            font=('Segoe UI', 10),
            text_color=SPRACH_COLOR
        ).pack(side='left', padx=20, pady=10)

    ctk.CTkButton(top, image=tipp_icon, text="", fg_color=BTN_COLOR,
                  width=40, height=40, corner_radius=20,
                  command=zeige_tipp).pack(side='right', padx=10, pady=10)

    ctk.CTkButton(top, image=birne_icon, text="", fg_color=BTN_COLOR,
                  width=40, height=40, corner_radius=20,
                  command=wechsel_mode).pack(side='right', padx=10, pady=10)

    global frage_label, eingabe, feedback_label, punktzahl_label, fortschritt
    frage_label = ctk.CTkLabel(container, text="", font=('Arial', 30))
    frage_label.pack(pady=30)

    eingabe = ctk.CTkEntry(container, font=('Arial', 20), width=400)
    eingabe.pack(pady=10)
    eingabe.bind('<Return>', pruefe_antwort)

    ctk.CTkButton(container, text="Antwort prüfen", fg_color=BTN_COLOR,
                  command=pruefe_antwort, width=200, height=50, corner_radius=20).pack(pady=10)

    feedback_label = ctk.CTkLabel(container, text="", font=('Arial', 20))
    feedback_label.pack(pady=10)

    punktzahl_label = ctk.CTkLabel(container, text=f"Punktzahl: {punktzahl}", font=('Arial', 20))
    punktzahl_label.pack(pady=10)

    fortschritt = ctk.CTkProgressBar(container, width=400)
    fortschritt.set(0)
    fortschritt.pack(pady=10)

    # Vokabel Label für die Anzeige der nächsten Vokabel
    global vokabel_label
    vokabel_label = ctk.CTkLabel(container, text="", font=('Arial', 24, 'bold'))
    vokabel_label.pack(pady=20)

    

# =========================== Statistik-Screen =================================
def statistik():
    frame = ctk.CTkFrame(app); frames['statistik'] = frame

    container = ctk.CTkFrame(frame)
    container.pack(expand=True, fill='both')

    ctk.CTkButton(container, image=haus_icon, text="", fg_color=BTN_COLOR,
                  width=40, height=40, corner_radius=20,
                  command=lambda: zeige_frame('start')).pack(anchor='nw', padx=10, pady=10)

    # Sprachanzeige in der Statistik
    if aktuelle_sprache:
        ctk.CTkLabel(
            container, 
            text=f"Statistiken - {aktuelle_sprache.capitalize()}", 
            font=('Arial', 30)
        ).pack(pady=(20, 10))
    else:
        ctk.CTkLabel(container, text="Statistiken", font=('Arial', 30)).pack(pady=(20, 10))

    global stat_feedback_label, statistik_frame
    stat_feedback_label = ctk.CTkLabel(container, text="", font=('Arial', 16), text_color="green")
    stat_feedback_label.pack(pady=(0, 10))

    statistik_frame = ctk.CTkScrollableFrame(container, corner_radius=0)
    statistik_frame.pack(fill='both', expand=True, padx=20, pady=10)

    for col in range(4):
        statistik_frame.grid_columnconfigure(col, weight=1, uniform="stat")

    footer = ctk.CTkFrame(container)
    footer.pack(fill='x', side='bottom', pady=10)

    ctk.CTkButton(footer, text="Zurück", command=lambda: zeige_frame('start'), fg_color=BTN_COLOR,
                  width=150, height=40, corner_radius=20).pack(side='left', padx=20)

    ctk.CTkButton(footer, text="Alle löschen", command=alle_vokabeln_loeschen, fg_color=ERROR_COLOR,
                  width=150, height=40, corner_radius=20).pack(side='right', padx=20)

    ctk.CTkButton(footer, text="Speichern", command=save_editor, fg_color=BTN_COLOR,
                  width=150, height=40, corner_radius=20).pack(side='right', padx=20)



def zeige_statistik():
    stat_feedback_label.configure(text="")
    for w in statistik_frame.winfo_children():
        w.destroy()
    w, h = app.winfo_width(), app.winfo_height()
    scale = min(w/1200, h/800)
    hdr_font = ("Arial", 16, "bold")
    dt_font  = ("Arial", 14)
    headers  = ["Deutsch – Englisch", "Richtig", "Falsch", "% richtig"]
    for col, txt in enumerate(headers):
        ctk.CTkLabel(statistik_frame, text=txt, font=hdr_font).grid(row=0, column=col, padx=10, pady=(10,5), sticky='ew')
    r_tot = f_tot = 0
    for r, v in enumerate(alle_vokabeln, start=1):
        key = (v['Deutsch'], v['Englisch'])
        st  = vokabel_statistik.get(key, {'richtig': 0, 'falsch': 0})
        tot = st['richtig'] + st['falsch']
        pct = (st['richtig'] / tot * 100) if tot > 0 else 0
        fg  = "green" if pct >= 80 else "orange" if pct >= 50 else "red"
        vals = [f"{v['Deutsch']} – {v['Englisch']}", str(st['richtig']), str(st['falsch']), f"{pct:.2f}%"]
        for col, txt in enumerate(vals):
            ctk.CTkLabel(statistik_frame, text=txt, font=dt_font,
                         fg_color=fg if col == 3 else None,
                         corner_radius=(5 if col == 3 else 0)
            ).grid(row=r, column=col, padx=10, pady=5, sticky='ew')
        r_tot += st['richtig']; f_tot += st['falsch']
    row = len(alle_vokabeln) + 1
    ctk.CTkLabel(statistik_frame, text="GESAMT", font=hdr_font).grid(row=row, column=0, padx=10, pady=(10,5), sticky='ew')
    ctk.CTkLabel(statistik_frame, text=str(r_tot), font=hdr_font).grid(row=row, column=1, padx=10, pady=(10,5), sticky='ew')
    ctk.CTkLabel(statistik_frame, text=str(f_tot), font=hdr_font).grid(row=row, column=2, padx=10, pady=(10,5), sticky='ew')
    pct_tot = (r_tot / (r_tot + f_tot) * 100) if (r_tot + f_tot) > 0 else 0
    ctk.CTkLabel(statistik_frame, text=f"Prozent richtig: {pct_tot:.2f}%", font=dt_font
    ).grid(row=row+1, column=0, columnspan=4, padx=10, pady=(0,10), sticky='ew')

def statistik_zuruecksetzen():
    for k in vokabel_statistik:
        vokabel_statistik[k] = {'richtig': 0, 'falsch': 0}
    runde_status.clear()
    statistik_speichern()
    stat_feedback_label.configure(text="Statistik zurückgesetzt ✅")
    zeige_statistik()

# =========================== Vokabel-Editor ==================================
def editor():
    frame = ctk.CTkFrame(app); frames['editor'] = frame
    ctk.CTkButton(frame, image=haus_icon, text="", width=40, height=40, fg_color=BTN_COLOR,
                  corner_radius=20, command=lambda: zeige_frame('start')).pack(anchor='nw', padx=10, pady=10)
    
    # Sprachanzeige im Editor
    if aktuelle_sprache:
        ctk.CTkLabel(
            frame, 
            text=f"Vokabel-Editor - {aktuelle_sprache.capitalize()}", 
            font=('Arial',30)
        ).pack(pady=(20,10))
    else:
        ctk.CTkLabel(frame, text="Vokabel-Editor", font=('Arial',30)).pack(pady=(20,10))
    global editor_feedback
    editor_feedback = ctk.CTkLabel(frame, text="", font=('Arial',8), text_color="green"); editor_feedback.pack(pady=(0,10))
    ctk.CTkButton(frame, text="Bild importieren…", command=upload_and_import, fg_color=BTN_COLOR,
                  width=200, height=40, corner_radius=20).pack(pady=(0,10))
    global editor_frame
    editor_frame = ctk.CTkScrollableFrame(frame, corner_radius=0); editor_frame.pack(fill='both', expand=True, padx=20, pady=10)
    for col in range(3):
        editor_frame.grid_columnconfigure(col, weight=1, uniform="ed")
    footer = ctk.CTkFrame(frame); footer.pack(fill='x', side='bottom', pady=10)

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
            editor_frame, text="✖", width=30,
            fg_color=ERROR_COLOR, hover_color="#b91c1c",
            font=('Segoe UI', 10),
            command=lambda idx=i: (
                entferne_vokabel(idx),
                editor_feedback.configure(text="Vokabel gelöscht ✅")
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
                editor_feedback.configure(text="Änderung gespeichert ✅")
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
    placeholder_text=aktuelle_sprache.capitalize()
)
    # Grün gefärbter Hinzufügen-Button
    add_btn = ctk.CTkButton(
        editor_frame, text="+", width=30,
        fg_color=SUCCESS_COLOR, hover_color=SPRACH_COLOR,
        font=('Segoe UI', 12, 'bold'),
        command=lambda: (
            hinzufügen_vokabel(),
            editor_feedback.configure(text="Vokabel hinzugefügt ✅"),
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
                text="Vokabel hinzugefügt ✅",
                font=("Arial", 12),
                text_color="green"
            )
            save_editor()
            show_editor()
            # Statt editor_entries[-1][0].focus():
            neu_de.focus()

    neu_en.bind("<Return>", on_enter_new)

    zeige_frame('editor')


def save_editor():
    global alle_vokabeln, editor_entries
    for idx, (de_e, en_e) in enumerate(editor_entries):
        d = de_e.get().strip()
        e = en_e.get().strip()
        if d and e:
            alle_vokabeln[idx] = {'Deutsch': d, 'Englisch': e}
    save_vokabeln_csv()
    editor_feedback.configure(text="Änderungen gespeichert ✅")
    show_editor()

def alle_vokabeln_loeschen():
    global alle_vokabeln
    alle_vokabeln = []
    save_vokabeln_csv()
    editor_feedback.configure(text="Alle Vokabeln gelöscht ✅")
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
        editor_feedback.configure(text="Vokabel hinzugefügt ✅")

    show_editor()

# ============================ Tipp-Funktion ==================================
def zeige_tipp():
    if not aktuelle_vokabel:
        return

    artikel = [
        "to ", "a ", "an ", "the ",
        "le ", "la ", "les ", "l'",
        "un ", "une ",
        "des ", "du ",
        "de la ", "de l'"
    ]

    eng = aktuelle_vokabel['Englisch'].strip().lower()

    prefix = ""
    for art in artikel:
        if eng.startswith(art):
            prefix = art
            break

    core = eng[len(prefix):].strip()
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

        # Enter oder Leertaste
        if event.keysym in ['Return', 'space']:
            # Phase 2: Feedbackmodus -> nächste Vokabel
            if feedback_active:
                naechste_vokabel()
                return "break"

            # Phase 1: Eingabemodus -> Antwort prüfen
            elif eingabe and eingabe.focus_get() == eingabe:
                pruefe_antwort()
                return "break"

        # F1 für Tipp
        elif event.keysym == 'F1':
            zeige_tipp()
            return "break"

    return None
    
# ========================== Fortschritts-Anzeige =============================
def update_fortschritt():
    """Aktualisiert die Fortschrittsanzeige basierend auf den Wiederholungen"""
    if not alle_vokabeln or not fortschritt:
        return
    
    required_repetitions = training_settings['repetitions']
    
    # Berechne den Gesamtfortschritt basierend auf allen Wiederholungen
    total_possible_points = len(alle_vokabeln) * required_repetitions
    current_points = 0
    
    for v in alle_vokabeln:
        key = (v['Deutsch'], v['Englisch'])
        correct_count = vokabel_repetitions.get(key, 0)
        # Begrenze die Punkte auf die maximale Anzahl erforderlicher Wiederholungen
        current_points += min(correct_count, required_repetitions)
    
    # Fortschritt berechnen (0.0 bis 1.0)
    if total_possible_points > 0:
        progress = current_points / total_possible_points
    else:
        progress = 0.0
    
    fortschritt.set(progress)
    
    # Optional: Zusätzliche Anzeige der aktuellen Statistik
    completed_vocabs = sum(1 for v in alle_vokabeln 
                          if vokabel_repetitions.get((v['Deutsch'], v['Englisch']), 0) >= required_repetitions)
    
    # Du könntest hier auch ein Label hinzufügen, das den detaillierten Fortschritt anzeigt:
    # f"Fortschritt: {current_points}/{total_possible_points} | Fertig: {completed_vocabs}/{len(alle_vokabeln)}"
# ========================== Quiz-Logik =======================================
def naechste_vokabel():
    global aktuelle_vokabel, feedback_active, weiter_button
    feedback_active = False

    # Feedback-System zurücksetzen
    if weiter_button:
        weiter_button.pack_forget()

    # Eingabe wieder aktivieren
    eingabe.configure(state='normal')

    # Vokabeln filtern basierend auf Wiederholungs-Einstellung
    required_repetitions = training_settings['repetitions']

    # Vokabeln die noch nicht genug richtig beantwortet wurden
    unvollstaendig = [
        v for v in alle_vokabeln
        if vokabel_repetitions.get((v['Deutsch'], v['Englisch']), 0) < required_repetitions
    ]

    # Prüfen ob alle Vokabeln gelernt wurden
    if not unvollstaendig:
        endbildschirm()
        return

    # Intelligente Auswahl (nicht dieselbe wie zuvor)
    moegliche_vokabeln = unvollstaendig.copy()
    if len(moegliche_vokabeln) > 1 and aktuelle_vokabel:
        aktuelle_key = (aktuelle_vokabel['Deutsch'], aktuelle_vokabel['Englisch'])
        moegliche_vokabeln = [v for v in moegliche_vokabeln if (v['Deutsch'], v['Englisch']) != aktuelle_key]
    if not moegliche_vokabeln:
        moegliche_vokabeln = unvollstaendig

    # Zufällige Auswahl
    aktuelle_vokabel = random.choice(moegliche_vokabeln)

    frage_label.configure(text=f"Was heißt: {aktuelle_vokabel['Deutsch']} auf {aktuelle_sprache}?")
    eingabe.delete(0, tk.END)
    feedback_label.configure(text="")
    update_fortschritt()
    eingabe.focus_set()



def pruefe_antwort(event=None):
    global punktzahl, feedback_active

    # Falls schon Feedback aktiv ist, nicht nochmal prüfen
    if feedback_active:
        return

    if aktuelle_vokabel is None:
        return

    ant = eingabe.get().strip()
    kor = aktuelle_vokabel['Englisch'].strip()
    key = (aktuelle_vokabel['Deutsch'], aktuelle_vokabel['Englisch'])

    eingabe.configure(state='disabled')

    if ant.lower() == kor.lower():
        feedback_label.configure(text="✅ Richtig!", text_color=SUCCESS_COLOR)
        vokabel_statistik[key]['richtig'] += 1
        vokabel_repetitions[key] = vokabel_repetitions.get(key, 0) + 1
    else:
        if is_typo(ant, kor):
            feedback_label.configure(
                text=f"✅ Fast richtig! \nRichtig: {aktuelle_vokabel['Englisch']}",
                text_color=WARNING_COLOR
            )
            vokabel_statistik[key]['richtig'] += 1
            vokabel_repetitions[key] = vokabel_repetitions.get(key, 0) + 1
            punktzahl = max(0, punktzahl - 1)
        else:
            feedback_label.configure(
                text=f"❌ Falsch! Richtig: {aktuelle_vokabel['Englisch']}",
                text_color=ERROR_COLOR
            )
            punktzahl = max(0, punktzahl - 5)
            vokabel_statistik[key]['falsch'] += 1

    punktzahl_label.configure(text=f"Punktzahl: {punktzahl}")
    statistik_speichern()
    update_fortschritt()

    feedback_active = True
    show_weiter_button()

def show_weiter_button():
    """Zeigt den 'Weiter'-Button an"""
    global weiter_button
    
    # Weiter-Button erstellen falls noch nicht vorhanden
    if weiter_button is None:
        weiter_button = ctk.CTkButton(
            feedback_label.master,
            text="Weiter",
            command=naechste_vokabel,
            fg_color=SUCCESS_COLOR,
            hover_color="#047857",
            width=120,
            height=40,
            font=('Segoe UI', 14, 'bold')
        )
    
    weiter_button.pack(pady=10)
    app.focus_set()  # Fokus auf Button setzen

# =========================== Endbildschirm ===================================
def endbildschirm():
    frame = ctk.CTkFrame(app); frames['ende'] = frame

    container = ctk.CTkFrame(frame)
    container.pack(expand=True, fill='both')

    ctk.CTkLabel(container, text="🎉 Fertig gelernt!", font=('Arial', 40)).pack(pady=40)

    # Sprachanzeige im Endbildschirm
    if aktuelle_sprache:
        ctk.CTkLabel(
            container, 
            text=aktuelle_sprache.capitalize(), 
            font=('Segoe UI', 12),
            text_color=SPRACH_COLOR
        ).pack(pady=(0, 20))

    ctk.CTkLabel(container, text=f"Deine Punktzahl: {punktzahl}", font=('Arial', 20)).pack(pady=20)

    ctk.CTkButton(container, text="Wiederholen", command=starte_neu, fg_color=BTN_COLOR,
                  width=200, height=50, corner_radius=20).pack(pady=20)

    ctk.CTkButton(container, text="Zum Startbildschirm", command=zurücksetzen, fg_color=BTN_COLOR,
                  width=200, height=50, corner_radius=20).pack(pady=20)

    zeige_frame('ende')

# ====================== Reset- & Neustart-Funktionen ========================
def reset_for_new_attempt():
    global punktzahl, vokabeln_zu_lernen, runde_status, vokabel_repetitions
    punktzahl = 100
    vokabeln_zu_lernen = alle_vokabeln.copy()
    random.shuffle(vokabeln_zu_lernen)
    runde_status.clear()
    vokabel_repetitions.clear()  # Wiederholungen zurücksetzen

def zurücksetzen():
    reset_for_new_attempt()
    punktzahl_label.configure(text=f"Punktzahl: {punktzahl}")
    zeige_frame('start')

def starte_neu():
    global aktuelle_vokabel
    
    # Prüfen ob Vokabeln geladen sind
    if not alle_vokabeln:
        messagebox.showerror("Fehler", "Keine Vokabeln verfügbar. Bitte fügen Sie zuerst Vokabeln hinzu.")
        return
        
    # Trainer-Frame anzeigen und erste Vokabel laden
    zeige_frame('trainer')
    naechste_vokabel()

# ============================= Main Guard ====================================
if __name__ == "__main__":
    
    
    haus_icon  = lade_icon("haus.png")
    birne_icon = lade_icon("birne.png")
    tipp_icon  = lade_icon("tipp.png")
    flagge_icon = lade_icon("Flagge.png")

    # Initialisiere Standard-Sprache
    initialisiere_sprache('englisch')  # Oder 'deutsch' als Standard-Sprache

    # Nur einmal Screens vorbereiten (Frames initialisieren, nicht anzeigen):
    trainer()
    statistik()
    editor()
    einstellungen_screen()

    # Dann entscheiden welcher Frame zuerst gezeigt wird:
    if aktuelle_sprache is None or not vorhandene_sprachen():
        sprache_verwalten_screen()
    else:
        startbildschirm()

    # Fenstertitel initial setzen
    update_fenstertitel()
    
    update_font_sizes()

    app.bind_all('<KeyPress-Return>', handle_global_keys)
    app.bind_all('<KeyPress-space>', handle_global_keys) 
    app.bind_all('<KeyPress-F1>', handle_global_keys)

    app.mainloop()


