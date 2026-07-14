from __future__ import annotations

import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

FEMALE_EDGE_VOICES = [
    "km-KH-SreymomNeural",
    "en-US-AriaNeural",
    "en-US-EmmaNeural",
    "zh-CN-XiaoxiaoNeural",
]
MALE_EDGE_VOICES = [
    "km-KH-PisethNeural",
    "en-US-GuyNeural",
    "en-US-BrianNeural",
    "zh-CN-YunxiNeural",
]


class VoicePage(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        header = QLabel("Voice & TTS Settings")
        header.setObjectName("PageHeader")
        layout.addWidget(header)

        desc = QLabel("Choose how the AI speaks in the dubbed video.")
        desc.setObjectName("PageDesc")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        form = QFormLayout()
        form.setSpacing(12)

        self.tts_provider = QComboBox()
        self.tts_provider.addItem("Edge Khmer TTS", "edge")
        self.tts_provider.addItem("Gemini expressive TTS", "gemini")
        form.addRow("TTS Provider", self.tts_provider)

        self.gemini_status = QLabel("")
        self.gemini_status.setObjectName("HintLabel")
        self.gemini_status.setWordWrap(True)
        form.addRow("", self.gemini_status)

        self.voice_gender = QComboBox()
        self.voice_gender.addItem("Female TTS (single voice)", "female")
        self.voice_gender.addItem("Male TTS (single voice)", "male")
        self.voice_gender.addItem("Auto male/female TTS with emotion", "auto")
        auto_tts_index = self.voice_gender.findData("auto")
        if auto_tts_index >= 0:
            self.voice_gender.setCurrentIndex(auto_tts_index)
        form.addRow("Voice Mode", self.voice_gender)

        self.mode_help = QLabel("")
        self.mode_help.setObjectName("HintLabel")
        self.mode_help.setWordWrap(True)
        form.addRow("", self.mode_help)

        self.simple_tts_flow = QCheckBox("Use automatic male/female voices with emotion")
        self.simple_tts_flow.setToolTip(
            "Auto-configures the easy flow:\n"
            "- Voice mode: Auto male/female TTS with emotion\n"
            "- Voice cloning: not included\n"
            "Use this for clear, emotional TTS voices."
        )
        self.simple_tts_flow.setChecked(False)
        form.addRow("", self.simple_tts_flow)

        self.detect_speakers_button = QPushButton("Detect Speakers / Map Voices")
        self.detect_speakers_button.setObjectName("SecondaryButton")
        form.addRow("", self.detect_speakers_button)

        female_row = QHBoxLayout()
        self.voice_female = QComboBox()
        self.voice_female.setEditable(True)
        for v in FEMALE_EDGE_VOICES:
            self.voice_female.addItem(v)
        self.test_female_button = QPushButton("Test")
        self.test_female_button.setObjectName("CompactButton")
        self.test_female_button.setFixedWidth(60)
        female_row.addWidget(self.voice_female, 1)
        female_row.addWidget(self.test_female_button)
        self._female_label = QLabel("Female Voice")
        form.addRow(self._female_label, female_row)

        male_row = QHBoxLayout()
        self.voice_male = QComboBox()
        self.voice_male.setEditable(True)
        for v in MALE_EDGE_VOICES:
            self.voice_male.addItem(v)
        self.test_male_button = QPushButton("Test")
        self.test_male_button.setObjectName("CompactButton")
        self.test_male_button.setFixedWidth(60)
        male_row.addWidget(self.voice_male, 1)
        male_row.addWidget(self.test_male_button)
        self._male_label = QLabel("Male Voice")
        form.addRow(self._male_label, male_row)

        rate_layout = QHBoxLayout()
        self.speech_rate_slider = QSlider(Qt.Orientation.Horizontal)
        self.speech_rate_slider.setRange(-50, 50)
        self.speech_rate_slider.setValue(10)
        self.speech_rate = QSpinBox()
        self.speech_rate.setRange(-50, 50)
        self.speech_rate.setValue(10)
        self.speech_rate.setSuffix("%")
        self.speech_rate_slider.valueChanged.connect(self.speech_rate.setValue)
        self.speech_rate.valueChanged.connect(self.speech_rate_slider.setValue)
        rate_layout.addWidget(self.speech_rate_slider, 1)
        rate_layout.addWidget(self.speech_rate)
        form.addRow("Speech rate", rate_layout)

        pitch_layout = QHBoxLayout()
        self.pitch_hz_slider = QSlider(Qt.Orientation.Horizontal)
        self.pitch_hz_slider.setRange(-20, 20)
        self.pitch_hz_slider.setValue(0)
        self.pitch_hz = QSpinBox()
        self.pitch_hz.setRange(-20, 20)
        self.pitch_hz.setValue(0)
        self.pitch_hz.setSuffix(" Hz")
        self.pitch_hz_slider.valueChanged.connect(self.pitch_hz.setValue)
        self.pitch_hz.valueChanged.connect(self.pitch_hz_slider.setValue)
        pitch_layout.addWidget(self.pitch_hz_slider, 1)
        pitch_layout.addWidget(self.pitch_hz)
        form.addRow("Pitch", pitch_layout)

        emotion_layout = QHBoxLayout()
        self.emotion_strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.emotion_strength_slider.setRange(0, 100)
        self.emotion_strength_slider.setValue(80)
        self.emotion_strength = QSpinBox()
        self.emotion_strength.setRange(0, 100)
        self.emotion_strength.setValue(80)
        self.emotion_strength.setSuffix("%")
        self.emotion_strength_slider.valueChanged.connect(self.emotion_strength.setValue)
        self.emotion_strength.valueChanged.connect(self.emotion_strength_slider.setValue)
        emotion_layout.addWidget(self.emotion_strength_slider, 1)
        emotion_layout.addWidget(self.emotion_strength)
        form.addRow("Emotion strength", emotion_layout)

        layout.addLayout(form)
        layout.addStretch(1)

        self.tts_provider.currentIndexChanged.connect(self._update_provider_status)
        self.voice_gender.currentIndexChanged.connect(
            lambda: self.set_voice_mode(str(self.voice_gender.currentData() or ""))
        )
        self.set_voice_mode(str(self.voice_gender.currentData() or ""))
        self._update_provider_status()

    def set_voice_mode(self, mode: str) -> None:
        is_per_person = mode in {"per_person", "per_person_auto"}
        is_per_speaker = mode == "per_speaker_auto"
        show_female = mode in {"female", "auto", "per_person", "per_person_auto", "per_speaker_auto"}
        show_male = mode in {"male", "auto", "per_person", "per_person_auto", "per_speaker_auto"}

        for widget in (self._female_label, self.voice_female, self.test_female_button):
            widget.setVisible(show_female)
        for widget in (self._male_label, self.voice_male, self.test_male_button):
            widget.setVisible(show_male)

        if is_per_person:
            no_clone_tip = (
                "Used when a speaker does not have enough clean speech to clone. "
                "Gender is detected automatically from the source audio."
            )
            self._female_label.setText("Female Voice (no clone)")
            self._male_label.setText("Male Voice (no clone)")
            self.voice_female.setToolTip(no_clone_tip)
            self.voice_male.setToolTip(no_clone_tip)
        elif is_per_speaker:
            self._female_label.setText("Female Voice (fallback)")
            self._male_label.setText("Male Voice (fallback)")
            self.voice_female.setToolTip("Used only with Edge TTS when a speaker line could not be assigned.")
            self.voice_male.setToolTip("Used only with Edge TTS when a speaker line could not be assigned.")
        else:
            self._female_label.setText("Female Voice")
            self._male_label.setText("Male Voice")
            self.voice_female.setToolTip("")
            self.voice_male.setToolTip("")

        self.detect_speakers_button.setVisible(mode == "per_person")
        help_text = {
            "female": (
                "TTS only. Every segment uses your selected female voice. "
                "Voice cloning is not used."
            ),
            "male": (
                "TTS only. Every segment uses your selected male voice. "
                "Voice cloning is not used."
            ),
            "auto": (
                "Recommended for simple dubbing: the app auto-detects male/female per segment, "
                "uses your selected TTS voices, and keeps emotion in the delivery. "
                "This does not require voice cloning."
            ),
            "per_speaker_auto": (
                "Detects speakers in the source video and keeps one consistent voice per person. "
                "With Gemini TTS, each speaker gets a fixed preset voice (Kore, Puck, etc.) — no cloning. "
                "With Edge TTS, male/female fallback is used only for unassigned lines."
            ),
            "per_person_auto": (
                "Clone workflow: the app tries to build a separate cloned voice for each speaker. "
                "Female/Male voice boxes are fallback TTS only when cloning is not possible."
            ),
            "per_person": (
                "Clone workflow with manual speaker mapping. Use Detect Speakers / Map Voices first, "
                "then assign references. Female/Male voice boxes are fallback TTS only."
            ),
        }.get(mode, "")
        self.mode_help.setText(help_text)
        self._update_provider_status()

    def save_state(self) -> dict:
        return {
            "tts_provider": self.tts_provider.currentData(),
            "voice_gender": self.voice_gender.currentText(),
            "voice_female": self.voice_female.currentText(),
            "voice_male": self.voice_male.currentText(),
            "speech_rate": self.speech_rate.value(),
            "pitch_hz": self.pitch_hz.value(),
            "emotion_strength": self.emotion_strength.value(),
            "simple_tts_flow": self.simple_tts_flow.isChecked(),
        }

    def load_state(self, config: dict) -> None:
        provider_index = self.tts_provider.findData(config.get("tts_provider", "edge"))
        if provider_index >= 0:
            self.tts_provider.setCurrentIndex(provider_index)
        voice_gender = config.get("voice_gender", "")
        if self.voice_gender.findText(voice_gender) < 0:
            voice_gender = "Auto male/female TTS with emotion"
        if voice_gender == "Auto per-person clone" and config.get("tts_provider") == "gemini":
            migrated = self.voice_gender.findData("per_speaker_auto")
            if migrated >= 0:
                self.voice_gender.setCurrentIndex(migrated)
        else:
            self.voice_gender.setCurrentText(voice_gender)
        self.voice_female.setCurrentText(config.get("voice_female", ""))
        self.voice_male.setCurrentText(config.get("voice_male", ""))
        self.speech_rate.setValue(config.get("speech_rate", 10))
        self.pitch_hz.setValue(config.get("pitch_hz", 0))
        self.emotion_strength.setValue(config.get("emotion_strength", 80))
        self.simple_tts_flow.setChecked(config.get("simple_tts_flow", False))
        self.set_voice_mode(str(self.voice_gender.currentData() or ""))

    def _update_provider_status(self) -> None:
        provider = self.tts_provider.currentData()
        mode = self.voice_gender.currentData()
        if provider == "gemini":
            from modules.gemini_tts_engine import resolve_gemini_api_keys

            model = os.getenv("GEMINI_TTS_MODEL", "gemini-3.1-flash-tts-preview")
            max_requests = os.getenv("GEMINI_TTS_MAX_REQUESTS", "10")
            api_keys = resolve_gemini_api_keys()
            base = ""
            if api_keys:
                fallback_note = (
                    f", {len(api_keys)} API key(s) configured"
                    if len(api_keys) > 1
                    else ""
                )
                base = (
                    f"Gemini TTS ready: {model}, max {max_requests} request(s) per video{fallback_note}."
                )
            else:
                base = (
                    "Gemini TTS selected. Set GEMINI_API_KEY "
                    "(or GEMINI_API_KEY_FALLBACK / GEMINI_API_KEYS) before running full video dubbing."
                )
            if mode not in {"per_speaker_auto", "per_person_auto", "per_person"}:
                base += (
                    " For one consistent voice per person, choose Voice Mode: "
                    "Auto per-speaker voices (no clone)."
                )
            elif mode == "per_speaker_auto":
                base += " Each detected speaker keeps the same Gemini preset voice for the whole video."
            self.gemini_status.setText(base)
        else:
            self.gemini_status.setText("Edge TTS uses the selected Khmer voice names below.")
