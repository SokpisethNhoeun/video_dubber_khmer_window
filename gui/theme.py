"""Theme system for the Video Dubber GUI.

Provides dark and light colour palettes and generates the full Qt stylesheet
from a palette dictionary so that the rest of the application never needs to
hard-code hex colours.
"""

from __future__ import annotations

from PyQt6.QtCore import QSettings

# ---------------------------------------------------------------------------
# Palette definitions
# ---------------------------------------------------------------------------

DARK_PALETTE: dict[str, str] = {
    # Backgrounds
    "bg_main":          "#0d0f14",
    "bg_panel":         "#13151c",
    "bg_control":       "#1a1d26",
    "bg_button":        "#22263a",
    "bg_button_hover":  "#2c3150",
    "bg_button_press":  "#181b28",
    "bg_button_dis":    "#14161e",
    "bg_secondary":     "#181b26",
    "bg_sec_hover":     "#232740",
    "bg_input_focus":   "#0f1219",
    "bg_console":       "#090b10",
    "bg_msg_box":       "#13151c",
    "bg_spin_btn":      "#1a1d26",
    "bg_spin_hover":    "#22263a",

    # Text
    "text_primary":     "#eef0f6",
    "text_input":       "#f0f3f8",
    "text_label":       "#c8cdd9",
    "text_secondary":   "#8e96aa",
    "text_muted":       "#6b7385",
    "text_disabled":    "#55596b",
    "text_progress":    "#b8c0cc",
    "text_progressbar": "#edf2f7",
    "text_console":     "#c8f7ed",

    # Accent – teal/cyan
    "accent":           "#22d3c8",
    "accent_dark":      "#0d9488",
    "accent_darker":    "#0a7a70",
    "accent_hover":     "#16b8ac",
    "accent_sel":       "#1a5e58",
    "accent_badge_bg":  "#0e3531",
    "accent_badge_txt": "#7df5e8",
    "accent_badge_bd":  "#1a6560",

    # Accent – amber
    "amber":            "#f5a623",
    "amber_badge_bg":   "#231d10",
    "amber_badge_txt":  "#ffd08a",
    "amber_badge_bd":   "#5a400e",

    # Accent – purple
    "purple":           "#8b5cf6",
    "purple_dark":      "#5b21b6",
    "purple_hover":     "#7c3aed",

    # Danger / cancel
    "danger_bg":        "#25141a",
    "danger_txt":       "#ffb3c0",
    "danger_bd":        "#a83840",
    "danger_hover_bg":  "#c03048",
    "danger_press":     "#8a1e30",
    "danger_record":    "#e53e3e",

    # Borders
    "border":           "#22263a",
    "border_input":     "#2c3050",
    "border_button":    "#32364e",
    "border_hover":     "#4a5070",
    "border_dis":       "#1e2030",
    "border_indicator": "#3e4462",
    "border_spin":      "#2c3050",
    "border_sec_hover": "#f5a623",

    # Eyebrow
    "eyebrow":          "#22d3c8",

    # Header
    "header_text":      "#ffffff",

    # Scrollbar
    "scrollbar_handle":       "#2e3452",
    "scrollbar_handle_hover": "#424878",

    # Misc
    "combo_arrow":      "#8e96aa",
    "combo_arrow_hover": "#ffffff",
    "white":            "#ffffff",

    # Cards / surfaces
    "bg_card":              "#161923",
    "bg_card_hover":        "#1c2030",
    "border_card":          "#262b40",
    "bg_card_selected":     "#16332f",
    "border_card_selected": "#22d3c8",

    # Status chips
    "status_waiting_bg":    "#231d10",
    "status_waiting_txt":   "#ffd08a",
    "status_waiting_bd":    "#5a400e",
    "status_active_bg":     "#0e3531",
    "status_active_txt":    "#7df5e8",
    "status_active_bd":     "#1a6560",
    "status_done_bg":       "#12321f",
    "status_done_txt":      "#8ff0b0",
    "status_done_bd":       "#1e5c38",
    "status_failed_bg":     "#25141a",
    "status_failed_txt":    "#ffb3c0",
    "status_failed_bd":     "#a83840",
}


LIGHT_PALETTE: dict[str, str] = {
    # Backgrounds
    "bg_main":          "#f0f2f7",
    "bg_panel":         "#ffffff",
    "bg_control":       "#e8eaf0",
    "bg_button":        "#dde0ea",
    "bg_button_hover":  "#ced1de",
    "bg_button_press":  "#c0c4d4",
    "bg_button_dis":    "#ebedf4",
    "bg_secondary":     "#e8eaf0",
    "bg_sec_hover":     "#ced1de",
    "bg_input_focus":   "#ffffff",
    "bg_console":       "#f5f7fc",
    "bg_msg_box":       "#ffffff",
    "bg_spin_btn":      "#dde0ea",
    "bg_spin_hover":    "#ced1de",

    # Text
    "text_primary":     "#1c1e26",
    "text_input":       "#1c1e26",
    "text_label":       "#2d3140",
    "text_secondary":   "#5a6278",
    "text_muted":       "#7a8295",
    "text_disabled":    "#a0a8bc",
    "text_progress":    "#3a4055",
    "text_progressbar": "#1c1e26",
    "text_console":     "#18362e",

    # Accent – teal
    "accent":           "#0d9488",
    "accent_dark":      "#0f766e",
    "accent_darker":    "#0a5c56",
    "accent_hover":     "#10b4a6",
    "accent_sel":       "#c0f0ea",
    "accent_badge_bg":  "#ccf0eb",
    "accent_badge_txt": "#085e54",
    "accent_badge_bd":  "#7ad8cc",

    # Accent – amber
    "amber":            "#d97706",
    "amber_badge_bg":   "#fef3c4",
    "amber_badge_txt":  "#92610a",
    "amber_badge_bd":   "#f0d070",

    # Accent – purple
    "purple":           "#7c3aed",
    "purple_dark":      "#5b21b6",
    "purple_hover":     "#6d28d9",

    # Danger / cancel
    "danger_bg":        "#fde8eb",
    "danger_txt":       "#9f1a2a",
    "danger_bd":        "#e57070",
    "danger_hover_bg":  "#ef4444",
    "danger_press":     "#c62020",
    "danger_record":    "#d9534f",

    # Borders
    "border":           "#ccd0de",
    "border_input":     "#bec4d4",
    "border_button":    "#bec4d4",
    "border_hover":     "#9298b0",
    "border_dis":       "#dce0ec",
    "border_indicator": "#aeb4c8",
    "border_spin":      "#bec4d4",
    "border_sec_hover": "#d97706",

    # Eyebrow
    "eyebrow":          "#0f766e",

    # Header
    "header_text":      "#1c1e26",

    # Scrollbar
    "scrollbar_handle":       "#bec4d4",
    "scrollbar_handle_hover": "#9298b0",

    # Misc
    "combo_arrow":      "#5a6278",
    "combo_arrow_hover": "#1c1e26",
    "white":            "#ffffff",

    # Cards / surfaces
    "bg_card":              "#ffffff",
    "bg_card_hover":        "#f5f7fc",
    "border_card":          "#dde0ea",
    "bg_card_selected":     "#e2f7f3",
    "border_card_selected": "#0d9488",

    # Status chips
    "status_waiting_bg":    "#fef3c4",
    "status_waiting_txt":   "#92610a",
    "status_waiting_bd":    "#f0d070",
    "status_active_bg":     "#ccf0eb",
    "status_active_txt":    "#085e54",
    "status_active_bd":     "#7ad8cc",
    "status_done_bg":       "#d4f5df",
    "status_done_txt":      "#146c34",
    "status_done_bd":       "#8fdba8",
    "status_failed_bg":     "#fde8eb",
    "status_failed_txt":    "#9f1a2a",
    "status_failed_bd":     "#e57070",
}


# ---------------------------------------------------------------------------
# Stylesheet template
# ---------------------------------------------------------------------------

_STYLESHEET_TEMPLATE = """
QMainWindow {{
    background-color: {bg_main};
}}
QWidget {{
    color: {text_primary};
    font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, "Roboto", "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
}}
QWidget#Root {{
    background: {bg_main};
}}
QWidget#HeaderPanel {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
        stop:0 {bg_panel}, stop:1 {bg_control});
    border: 1px solid {border};
    border-radius: 14px;
}}
QLabel#Eyebrow {{
    color: {eyebrow};
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 2px;
}}
QLabel#Header {{
    font-size: 24px;
    font-weight: 800;
    color: {header_text};
    padding: 0px;
}}
QLabel#Subheader {{
    color: {text_secondary};
    font-size: 12px;
}}
QLabel#StatusBadge,
QLabel#SoftBadge {{
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 11px;
    font-weight: 700;
}}
QLabel#StatusBadge {{
    background: {accent_badge_bg};
    color: {accent_badge_txt};
    border: 1px solid {accent_badge_bd};
}}
QLabel#SoftBadge {{
    background: {amber_badge_bg};
    color: {amber_badge_txt};
    border: 1px solid {amber_badge_bd};
}}
QLabel#SectionGuide {{
    color: {text_muted};
    font-size: 12px;
    font-style: italic;
    padding: 2px 0px 6px 0px;
}}
QScrollArea#SetupScroll {{
    border: 0px;
    background: transparent;
}}
QScrollArea#SetupScroll > QWidget > QWidget {{
    background: transparent;
}}
/* ── Tab Widget ─────────────────────────────────────────────── */
QTabWidget#SetupTabs {{
    background: transparent;
    border: none;
}}
QTabWidget#SetupTabs::pane {{
    border: 1px solid {border};
    border-radius: 0px 12px 12px 12px;
    background: {bg_panel};
    top: -1px;
    padding: 12px;
}}
QTabBar {{
    qproperty-drawBase: 0;
}}
QTabBar::tab {{
    background: {bg_control};
    color: {text_secondary};
    border: 1px solid {border};
    border-bottom: none;
    border-top-left-radius: 9px;
    border-top-right-radius: 9px;
    padding: 10px 18px;
    margin-right: 4px;
    font-size: 12px;
    font-weight: 600;
    min-width: 100px;
}}
QTabBar::tab:selected {{
    background: {bg_panel};
    color: {accent};
    border-color: {border};
    border-bottom-color: {bg_panel};
    font-weight: 800;
}}
QTabBar::tab:hover:!selected {{
    background: {bg_sec_hover};
    color: {text_primary};
}}
/* ── Group Boxes ────────────────────────────────────────────── */
QGroupBox {{
    border: 1px solid {border};
    border-radius: 12px;
    margin-top: 20px;
    background: transparent;
    font-weight: 700;
    font-size: 13px;
    color: {header_text};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    top: 5px;
    padding: 3px 10px;
    border-radius: 6px;
    color: {amber};
    background-color: {bg_panel};
}}
/* ── Tooltips ───────────────────────────────────────────────── */
QToolTip {{
    background: {bg_control};
    color: {text_primary};
    border: 1px solid {border_hover};
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
    font-weight: 500;
}}
/* ── Labels ─────────────────────────────────────────────────── */
QLabel {{
    color: {text_label};
}}
QLabel#Hint,
QLabel#ConsoleHint {{
    color: {text_muted};
    font-size: 12px;
    font-weight: 500;
}}
QLabel#ProgressLabel {{
    color: {text_progress};
    font-size: 12px;
    font-weight: 700;
}}
/* ── Inputs ─────────────────────────────────────────────────── */
QLineEdit {{
    background: {bg_main};
    border: 1px solid {border_input};
    border-radius: 8px;
    padding: 8px 11px;
    color: {text_input};
    selection-background-color: {accent};
    min-height: 28px;
}}
QLineEdit:focus {{
    border: 1.5px solid {accent};
    background: {bg_input_focus};
}}
QLineEdit:hover {{
    border-color: {border_hover};
}}
QComboBox {{
    background: {bg_main};
    border: 1px solid {border_input};
    border-radius: 8px;
    padding: 8px 30px 8px 11px;
    color: {text_input};
    selection-background-color: {accent};
    min-height: 28px;
}}
QComboBox:focus {{
    border: 1.5px solid {accent};
}}
QComboBox:hover {{
    border-color: {border_hover};
}}
QComboBox::drop-down {{
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 28px;
    border-left: 0px;
    border-top-right-radius: 8px;
    border-bottom-right-radius: 8px;
}}
QComboBox::down-arrow {{
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {combo_arrow};
    width: 0;
    height: 0;
    margin-top: 1px;
}}
QComboBox::down-arrow:hover {{
    border-top-color: {combo_arrow_hover};
}}
QComboBox QAbstractItemView {{
    background-color: {bg_panel};
    border: 1px solid {border_input};
    border-radius: 8px;
    selection-background-color: {accent_sel};
    selection-color: {white};
    outline: 0px;
    padding: 4px;
}}
QSpinBox, QDoubleSpinBox {{
    background: {bg_main};
    border: 1px solid {border_input};
    border-radius: 8px;
    padding: 8px 26px 8px 11px;
    color: {text_input};
    selection-background-color: {accent};
    min-height: 28px;
}}
QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 1.5px solid {accent};
}}
QSpinBox:hover, QDoubleSpinBox:hover {{
    border-color: {border_hover};
}}
QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 20px;
    border-left: 1px solid {border_spin};
    border-top-right-radius: 8px;
    background: {bg_spin_btn};
}}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {{
    background: {bg_spin_hover};
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
    border-left: 3px solid transparent;
    border-right: 3px solid transparent;
    border-bottom: 4px solid {combo_arrow};
    width: 0;
    height: 0;
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 20px;
    border-left: 1px solid {border_spin};
    border-bottom-right-radius: 8px;
    background: {bg_spin_btn};
}}
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: {bg_spin_hover};
}}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    border-left: 3px solid transparent;
    border-right: 3px solid transparent;
    border-top: 4px solid {combo_arrow};
    width: 0;
    height: 0;
}}
/* ── Buttons ────────────────────────────────────────────────── */
QPushButton {{
    background: {bg_button};
    color: {text_input};
    border: 1px solid {border_button};
    border-radius: 8px;
    padding: 8px 14px;
    font-weight: 700;
    min-height: 24px;
}}
QPushButton:hover {{
    background: {bg_button_hover};
    border-color: {border_hover};
}}
QPushButton:pressed {{
    background: {bg_button_press};
}}
QPushButton:disabled {{
    background: {bg_button_dis};
    color: {text_disabled};
    border-color: {border_dis};
}}
QPushButton#SecondaryButton {{
    background: {bg_secondary};
    color: {text_label};
    border: 1px solid {border_button};
}}
QPushButton#SecondaryButton:hover {{
    background: {bg_sec_hover};
    border-color: {border_sec_hover};
}}
QPushButton#CompactButton {{
    min-width: 54px;
    padding-left: 10px;
    padding-right: 10px;
}}
QPushButton#StartButton {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {accent_dark}, stop:1 {accent_darker});
    color: {white};
    border: 0px;
    border-radius: 8px;
    padding: 11px 22px;
    font-weight: 800;
    font-size: 14px;
    min-height: 28px;
}}
QPushButton#StartButton:hover {{
    background: {accent_hover};
}}
QPushButton#StartButton:pressed {{
    background: {accent_darker};
}}
QPushButton#CancelButton {{
    background: {danger_bg};
    color: {danger_txt};
    border: 1.5px solid {danger_bd};
    border-radius: 8px;
    padding: 11px 22px;
    font-weight: 800;
    font-size: 14px;
    min-height: 28px;
}}
QPushButton#CancelButton:hover {{
    background: {danger_hover_bg};
    color: {white};
}}
QPushButton#CancelButton:pressed {{
    background: {danger_press};
}}
QPushButton#CancelButton:disabled {{
    background: {bg_button_dis};
    color: {text_disabled};
    border-color: {border_dis};
}}
QPushButton#OpenButton {{
    background: {purple_dark};
    color: {white};
    border: 0px;
    border-radius: 8px;
    padding: 11px 22px;
    font-weight: 800;
    font-size: 14px;
    min-height: 28px;
}}
QPushButton#OpenButton:hover {{
    background: {purple_hover};
}}
QPushButton#ThemeToggle {{
    background: {bg_button};
    color: {text_input};
    border: 1px solid {border_button};
    border-radius: 8px;
    padding: 6px 12px;
    font-size: 16px;
    font-weight: 700;
    min-height: 20px;
    min-width: 40px;
}}
QPushButton#ThemeToggle:hover {{
    background: {bg_button_hover};
    border-color: {border_hover};
}}
/* ── Progress Bars ──────────────────────────────────────────── */
QProgressBar {{
    border: 1px solid {border_input};
    border-radius: 6px;
    background: {bg_main};
    height: 14px;
    text-align: center;
    font-weight: 700;
    font-size: 10px;
    color: {text_progressbar};
}}
QProgressBar::chunk {{
    border-radius: 5px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 {accent}, stop:0.5 {amber}, stop:1 {purple});
}}
/* ── Console ────────────────────────────────────────────────── */
QWidget#ConsoleWidget {{
    background: transparent;
}}
QWidget#ConsoleTitleBar {{
    background-color: {bg_control};
    border-top-left-radius: 12px;
    border-top-right-radius: 12px;
    border: 1px solid {border};
    border-bottom: 0px;
}}
QLabel#ConsoleTitle {{
    color: {header_text};
    font-size: 12px;
    font-weight: 800;
}}
QTextEdit#LogConsole {{
    background: {bg_console};
    color: {text_console};
    border: 1px solid {border};
    border-bottom-left-radius: 12px;
    border-bottom-right-radius: 12px;
    padding: 12px;
    font-family: "Cascadia Mono", "JetBrains Mono", "Consolas", monospace;
    min-height: 230px;
}}
/* ── CheckBoxes ─────────────────────────────────────────────── */
QCheckBox {{
    spacing: 8px;
    color: {text_label};
    font-weight: 600;
}}
QCheckBox::indicator {{
    width: 17px;
    height: 17px;
    border-radius: 5px;
    border: 1px solid {border_indicator};
    background: {bg_main};
}}
QCheckBox::indicator:hover {{
    border-color: {accent};
}}
QCheckBox::indicator:checked {{
    background: {accent_dark};
    border-color: {accent};
}}
/* ── Scrollbar ──────────────────────────────────────────────── */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 0px;
}}
QScrollBar::handle:vertical {{
    background: {scrollbar_handle};
    border-radius: 5px;
    min-height: 36px;
}}
QScrollBar::handle:vertical:hover {{
    background: {scrollbar_handle_hover};
}}
QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical {{
    height: 0px;
}}
QMessageBox {{
    background: {bg_msg_box};
}}

/* ------------------------------------------------------------------------ */
/* Wizard / Stepper components                                              */
/* ------------------------------------------------------------------------ */
QWidget#Stepper {{
    background: {bg_panel};
    border: 1px solid {border};
    border-radius: 12px;
}}
QWidget#StepperItem:focus {{
    outline: none;
    border: 1px solid {accent};
    border-radius: 8px;
}}
QLabel#StepperBadge {{
    background: {bg_button};
    color: {text_secondary};
    border: 1px solid {border};
    border-radius: 15px;
    font-weight: 600;
    font-size: 13px;
}}
QLabel#StepperBadge[state="current"] {{
    background: {accent};
    color: {bg_main};
    border-color: {accent};
}}
QLabel#StepperBadge[state="done"] {{
    background: {accent_dark};
    color: {text_primary};
    border-color: {accent_dark};
}}
QLabel#StepperTitle {{
    color: {text_secondary};
    font-size: 12px;
}}
QLabel#StepperTitle[state="current"] {{
    color: {text_primary};
    font-weight: 600;
}}
QLabel#StepperTitle[state="done"] {{
    color: {text_label};
}}
QWidget#StepperConnector {{
    background: {border};
    border-radius: 1px;
}}
QWidget#StepperConnector[active="true"] {{
    background: {accent_dark};
}}

QWidget#WizardNavBar {{
    background: transparent;
}}

/* ── Cards ──────────────────────────────────────────────────── */
QWidget#Card {{
    background: {bg_card};
    border: 1px solid {border_card};
    border-radius: 14px;
}}
QWidget#Card[hoverable="true"]:hover {{
    background: {bg_card_hover};
    border-color: {border_hover};
}}
QWidget#Card[selected="true"] {{
    background: {bg_card_selected};
    border: 2px solid {border_card_selected};
}}
QLabel#CardBadge {{
    background: {accent_badge_bg};
    color: {accent_badge_txt};
    border: 1px solid {accent_badge_bd};
    border-radius: 8px;
    padding: 3px 10px;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1px;
}}

/* ── Status chips (payment / download states) ─────────────────── */
QLabel#StatusChip {{
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 11px;
    font-weight: 700;
}}
QLabel#StatusChip[state="waiting"] {{
    background: {status_waiting_bg};
    color: {status_waiting_txt};
    border: 1px solid {status_waiting_bd};
}}
QLabel#StatusChip[state="active"] {{
    background: {status_active_bg};
    color: {status_active_txt};
    border: 1px solid {status_active_bd};
}}
QLabel#StatusChip[state="done"] {{
    background: {status_done_bg};
    color: {status_done_txt};
    border: 1px solid {status_done_bd};
}}
QLabel#StatusChip[state="failed"] {{
    background: {status_failed_bg};
    color: {status_failed_txt};
    border: 1px solid {status_failed_bd};
}}

/* ── Inline field feedback ──────────────────────────────────── */
QLabel#InlineFeedback {{
    font-size: 12px;
    font-weight: 600;
}}
QLabel#InlineFeedback[state="error"] {{
    color: {status_failed_txt};
}}
QLabel#InlineFeedback[state="success"] {{
    color: {status_done_txt};
}}
QLabel#InlineFeedback[state="neutral"] {{
    color: {text_muted};
}}

QWidget#FileDropZone {{
    background: {bg_control};
    border: 2px dashed {border};
    border-radius: 14px;
}}
QWidget#FileDropZone[dragOver="true"] {{
    border-color: {accent};
    background: {bg_input_focus};
}}
QLabel#DropZoneIcon {{
    font-size: 42px;
    color: {text_secondary};
}}
QLabel#DropZoneTitle {{
    color: {text_primary};
    font-size: 16px;
    font-weight: 600;
}}
QLabel#DropZoneSubtitle {{
    color: {text_secondary};
    font-size: 12px;
}}
QListWidget#DropZoneList {{
    background: {bg_panel};
    border: 1px solid {border};
    border-radius: 8px;
    color: {text_input};
    padding: 4px;
}}

QWidget#CollapsibleSection {{
    background: transparent;
}}
QPushButton#CollapsibleToggle {{
    background: {bg_secondary};
    color: {text_label};
    border: 1px solid {border};
    border-radius: 8px;
    padding: 8px 12px;
    text-align: left;
    font-weight: 600;
}}
QPushButton#CollapsibleToggle:hover {{
    background: {bg_sec_hover};
    color: {text_primary};
}}
QWidget#CollapsibleBody {{
    background: {bg_secondary};
    border: 1px solid {border};
    border-radius: 8px;
}}

QCheckBox#ExpertToggle {{
    color: {text_label};
    font-size: 12px;
    spacing: 6px;
}}

QWidget#WizardPage {{
    background: {bg_panel};
    border: 1px solid {border};
    border-radius: 12px;
}}

/* ── Sidebar ── */
QWidget#Sidebar {{
    background: {bg_panel};
    border-right: 1px solid {border};
}}
QLabel#SidebarTitle {{
    color: {text_muted};
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 2px;
    padding: 8px 8px 4px 8px;
}}
QPushButton#SidebarItem {{
    background: transparent;
    border: none;
    border-radius: 8px;
    color: {text_secondary};
    text-align: left;
    padding: 6px 12px;
    font-size: 13px;
}}
QPushButton#SidebarItem:hover {{
    background: {bg_control};
    color: {text_primary};
}}
QPushButton#SidebarItem:checked {{
    background: {accent};
    color: {bg_main};
    font-weight: 700;
}}
QFrame#SidebarSeparator {{
    color: {border};
    margin: 6px 12px;
}}
QPushButton#AdvancedToggle {{
    background: {bg_control};
    border: 1px solid {border};
    border-radius: 8px;
    color: {text_secondary};
    padding: 8px 12px;
    font-size: 12px;
}}
QPushButton#AdvancedToggle:checked {{
    background: {accent_dark};
    color: white;
    border-color: {accent};
}}

/* ── Header Bar ── */
QWidget#HeaderBar {{
    background: {bg_panel};
    border-bottom: 1px solid {border};
}}
QLabel#LogoLabel {{
    font-size: 18px;
    font-weight: 800;
    color: {header_text};
}}

/* ── Status Bar ── */
QWidget#StatusBar {{
    background: {bg_panel};
    border-top: 1px solid {border};
}}
QLabel#StatusLabel {{
    color: {text_secondary};
    font-size: 12px;
}}

/* ── Page headers ── */
QLabel#PageHeader {{
    font-size: 20px;
    font-weight: 800;
    color: {header_text};
    padding: 0;
}}
QLabel#PageDesc {{
    color: {text_secondary};
    font-size: 13px;
    padding-bottom: 4px;
}}
QLabel#SectionTitle {{
    color: {accent};
    font-size: 13px;
    font-weight: 700;
    padding-top: 4px;
}}
QLabel#HintLabel {{
    color: {text_muted};
    font-size: 11px;
}}
QLabel#ProgressLabel {{
    color: {text_secondary};
    font-size: 12px;
    padding: 0;
}}

/* ── Page scroll area ── */
QScrollArea#PageScroll {{
    background: {bg_main};
    border: none;
}}
QScrollArea#PageScroll > QWidget > QWidget {{
    background: {bg_main};
}}

/* ── Progress Panel ── */
QWidget#ProgressPanel {{
    background: transparent;
}}
"""


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

THEME_DARK = "dark"
THEME_LIGHT = "light"

_SETTINGS_KEY = "appearance/theme"


def get_saved_theme() -> str:
    """Return the persisted theme name, defaulting to dark."""
    settings = QSettings("VideoDubber", "KhmerVideoDubber")
    return settings.value(_SETTINGS_KEY, THEME_DARK)


def save_theme(theme_name: str) -> None:
    """Persist the current theme choice."""
    settings = QSettings("VideoDubber", "KhmerVideoDubber")
    settings.setValue(_SETTINGS_KEY, theme_name)


def get_palette(theme_name: str) -> dict[str, str]:
    """Return the palette dict for the given theme name."""
    if theme_name == THEME_LIGHT:
        return LIGHT_PALETTE
    return DARK_PALETTE


def build_stylesheet(theme_name: str) -> str:
    """Generate the full stylesheet string for a given theme."""
    palette = get_palette(theme_name)
    return _STYLESHEET_TEMPLATE.format(**palette)


def recording_style(theme_name: str) -> str:
    """Return the inline style for the record button in recording state."""
    palette = get_palette(theme_name)
    return f"background: {palette['danger_record']}; color: white;"
