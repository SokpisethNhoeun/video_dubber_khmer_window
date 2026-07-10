from __future__ import annotations

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)


class AudioPage(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        header = QLabel("Audio Quality")
        header.setObjectName("PageHeader")
        layout.addWidget(header)

        desc = QLabel("Fine-tune audio processing, BGM handling, and loudness.")
        desc.setObjectName("PageDesc")
        desc.setWordWrap(True)
        layout.addWidget(desc)

        form = QFormLayout()
        form.setSpacing(12)

        self.audio_cleanup_check = QCheckBox("Clean extracted and reference audio")
        self.audio_cleanup_check.setChecked(True)
        form.addRow(self.audio_cleanup_check)

        self.final_mastering_check = QCheckBox("Master final dubbed audio")
        self.final_mastering_check.setChecked(True)
        form.addRow(self.final_mastering_check)

        self.per_speaker_prosody_check = QCheckBox("Per-speaker prosody (pace variation)")
        self.per_speaker_prosody_check.setChecked(True)
        form.addRow(self.per_speaker_prosody_check)

        self.preserve_bgm_check = QCheckBox("Preserve background music (Demucs)")
        self.preserve_bgm_check.setChecked(True)
        self.preserve_bgm_check.stateChanged.connect(self._update_bgm)
        form.addRow(self.preserve_bgm_check)

        self.bgm_ducking_check = QCheckBox("Duck BGM during speech")
        self.bgm_ducking_check.setChecked(True)
        form.addRow(self.bgm_ducking_check)

        self.voice_volume = QDoubleSpinBox()
        self.voice_volume.setRange(0.1, 3.0)
        self.voice_volume.setValue(1.0)
        self.voice_volume.setSingleStep(0.05)
        form.addRow("Voice volume", self.voice_volume)

        self.bgm_volume = QDoubleSpinBox()
        self.bgm_volume.setRange(0.0, 2.0)
        self.bgm_volume.setValue(0.85)
        self.bgm_volume.setSingleStep(0.05)
        form.addRow("BGM volume", self.bgm_volume)

        self.duck_depth = QDoubleSpinBox()
        self.duck_depth.setRange(0.0, 30.0)
        self.duck_depth.setValue(8.0)
        self.duck_depth.setSuffix(" dB")
        self.duck_depth.setSingleStep(1.0)
        form.addRow("Duck depth", self.duck_depth)

        self.publish_target = QComboBox()
        self.publish_target.addItem("YouTube (-14 LUFS)", "youtube")
        self.publish_target.addItem("TikTok/Reels (-12 LUFS)", "tiktok")
        self.publish_target.addItem("Podcast (-16 LUFS)", "podcast")
        self.publish_target.addItem("Custom", "custom")
        form.addRow("Loudness target", self.publish_target)

        self.custom_lufs = QDoubleSpinBox()
        self.custom_lufs.setRange(-30.0, -5.0)
        self.custom_lufs.setValue(-14.0)
        self.custom_lufs.setSuffix(" LUFS")
        self.custom_lufs.setSingleStep(1.0)
        form.addRow("Custom LUFS", self.custom_lufs)

        layout.addLayout(form)
        layout.addStretch(1)

    def save_state(self) -> dict:
        return {
            "audio_cleanup": self.audio_cleanup_check.isChecked(),
            "final_mastering": self.final_mastering_check.isChecked(),
            "per_speaker_prosody": self.per_speaker_prosody_check.isChecked(),
            "preserve_bgm": self.preserve_bgm_check.isChecked(),
            "bgm_ducking": self.bgm_ducking_check.isChecked(),
            "duck_depth_db": self.duck_depth.value(),
            "publish_target": self.publish_target.currentText(),
            "custom_lufs": self.custom_lufs.value(),
        }

    def load_state(self, config: dict) -> None:
        self.audio_cleanup_check.setChecked(config.get("audio_cleanup", True))
        self.final_mastering_check.setChecked(config.get("final_mastering", True))
        self.per_speaker_prosody_check.setChecked(config.get("per_speaker_prosody", True))
        self.preserve_bgm_check.setChecked(config.get("preserve_bgm", True))
        self.bgm_ducking_check.setChecked(config.get("bgm_ducking", True))
        self.duck_depth.setValue(config.get("duck_depth_db", 8.0))
        self.publish_target.setCurrentText(config.get("publish_target", ""))
        self.custom_lufs.setValue(config.get("custom_lufs", -14.0))

    def _update_bgm(self) -> None:
        enabled = self.preserve_bgm_check.isChecked()
        self.bgm_ducking_check.setEnabled(enabled)
        self.bgm_volume.setEnabled(enabled)
        self.duck_depth.setEnabled(enabled)
