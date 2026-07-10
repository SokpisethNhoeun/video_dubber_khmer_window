from __future__ import annotations

from pathlib import Path
from threading import Event
from typing import Callable

from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from modules.audio_quality import prepare_reference_audio, validate_reference_audio

FILE_DIALOG_OPTIONS = QFileDialog.Option(0)


class SpeakerMappingDialog(QDialog):
    def __init__(
        self,
        video_name: str,
        speaker_ids: list[str],
        work_dir: Path,
        min_reference_seconds: float,
        cleanup_enabled: bool,
        persistent_cache_dir: Path | None,
        voice_profiles: list,
        preview_callback: Callable[[str, dict[str, str]], None] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Map voices - {video_name}")
        self.resize(1180, 420)
        self.work_dir = work_dir
        self.min_reference_seconds = min_reference_seconds
        self.cleanup_enabled = cleanup_enabled
        self.persistent_cache_dir = persistent_cache_dir
        self.voice_profiles = voice_profiles
        self.preview_callback = preview_callback
        self.speaker_ids = speaker_ids
        self._profile_inputs: dict[str, QComboBox] = {}
        self._reference_inputs: dict[str, QLineEdit] = {}
        self._label_inputs: dict[str, QLineEdit] = {}
        self._status_labels: dict[str, QLabel] = {}
        self._cleaned_paths: dict[str, str] = {}
        self._merge_combos: dict[str, QComboBox] = {}

        layout = QVBoxLayout(self)
        intro = QLabel(f"Detected speakers in {video_name}")
        layout.addWidget(intro)

        quick_actions = QHBoxLayout()
        quick_actions.setContentsMargins(0, 0, 0, 0)
        quick_actions.setSpacing(8)
        fill_missing_button = QPushButton("Fill Missing From Library")
        fill_missing_button.clicked.connect(lambda: self._assign_profiles_to_speakers(only_missing=True))
        round_robin_button = QPushButton("Round-Robin Library")
        round_robin_button.clicked.connect(lambda: self._assign_profiles_to_speakers(only_missing=False))
        quick_actions.addWidget(fill_missing_button)
        quick_actions.addWidget(round_robin_button)
        quick_actions.addStretch(1)
        layout.addLayout(quick_actions)

        grid = QGridLayout()
        grid.addWidget(QLabel("Speaker"), 0, 0)
        grid.addWidget(QLabel("Label"), 0, 1)
        grid.addWidget(QLabel("Same as"), 0, 2)
        grid.addWidget(QLabel("Saved Voice"), 0, 3)
        grid.addWidget(QLabel("Reference MP3/WAV"), 0, 4)
        grid.addWidget(QLabel("Status"), 0, 6)
        for row, speaker_id in enumerate(speaker_ids, start=1):
            default_label = speaker_id.replace("_", " ").title()
            speaker_label = QLabel(speaker_id)
            label_input = QLineEdit(default_label)

            merge_combo = QComboBox()
            merge_combo.addItem("(unique)", "")
            for other_id in speaker_ids:
                if other_id != speaker_id:
                    merge_combo.addItem(other_id.replace("_", " ").title(), other_id)
            merge_combo.setToolTip("If this speaker is the same person as another, select them here")
            merge_combo.currentIndexChanged.connect(lambda _, sid=speaker_id: self._on_merge_changed(sid))
            self._merge_combos[speaker_id] = merge_combo

            profile_input = QComboBox()
            profile_input.addItem("Manual/reference", "")
            for profile in self.voice_profiles:
                profile_input.addItem(f"{profile.name} ({profile.gender})", str(profile.reference_audio_path))
            profile_input.currentIndexChanged.connect(lambda _, sid=speaker_id: self._select_profile(sid))
            reference_input = QLineEdit()
            reference_input.setPlaceholderText("Optional; missing references use normal TTS")
            browse_button = QPushButton("Browse")
            browse_button.clicked.connect(lambda _, ref=reference_input: self._select_reference(ref))
            prepare_button = QPushButton("Clean")
            prepare_button.clicked.connect(lambda _, sid=speaker_id: self._prepare_reference(sid))
            preview_button = QPushButton("Preview Voice")
            preview_button.clicked.connect(lambda _, sid=speaker_id: self._preview_voice(sid))
            status_label = QLabel("missing")

            self._label_inputs[speaker_id] = label_input
            self._profile_inputs[speaker_id] = profile_input
            self._reference_inputs[speaker_id] = reference_input
            self._status_labels[speaker_id] = status_label
            grid.addWidget(speaker_label, row, 0)
            grid.addWidget(label_input, row, 1)
            grid.addWidget(merge_combo, row, 2)
            grid.addWidget(profile_input, row, 3)
            grid.addWidget(reference_input, row, 4)
            grid.addWidget(browse_button, row, 5)
            grid.addWidget(status_label, row, 6)
            grid.addWidget(prepare_button, row, 7)
            grid.addWidget(preview_button, row, 8)

        layout.addLayout(grid)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _select_reference(self, input_widget: QLineEdit) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select speaker reference audio",
            str(Path.home()),
            "Audio files (*.mp3 *.wav);;All files (*)",
            options=FILE_DIALOG_OPTIONS,
        )
        if file_path:
            input_widget.setText(file_path)
            for speaker_id, widget in self._reference_inputs.items():
                if widget is input_widget:
                    self._profile_inputs[speaker_id].blockSignals(True)
                    self._profile_inputs[speaker_id].setCurrentIndex(0)
                    self._profile_inputs[speaker_id].blockSignals(False)
                    self._cleaned_paths.pop(speaker_id, None)
                    break
            self._refresh_status_for_input(input_widget)

    def _select_profile(self, speaker_id: str) -> None:
        profile_path = str(self._profile_inputs[speaker_id].currentData() or "").strip()
        if not profile_path:
            return
        self._reference_inputs[speaker_id].setText(profile_path)
        self._cleaned_paths.pop(speaker_id, None)
        self._refresh_status(speaker_id)

    def _assign_profiles_to_speakers(self, only_missing: bool) -> None:
        profile_paths = [str(profile.reference_audio_path) for profile in self.voice_profiles]
        if not profile_paths:
            QMessageBox.warning(self, "Voice Library", "Import or generate saved voices first.")
            return
        for index, speaker_id in enumerate(self.speaker_ids):
            if only_missing and self._reference_inputs[speaker_id].text().strip():
                continue
            profile_path = profile_paths[index % len(profile_paths)]
            combo = self._profile_inputs[speaker_id]
            combo_index = combo.findData(profile_path)
            if combo_index >= 0:
                combo.setCurrentIndex(combo_index)
            else:
                self._reference_inputs[speaker_id].setText(profile_path)
                self._refresh_status(speaker_id)

    def _refresh_status_for_input(self, input_widget: QLineEdit) -> None:
        for speaker_id, widget in self._reference_inputs.items():
            if widget is input_widget:
                self._refresh_status(speaker_id)
                return

    def _refresh_status(self, speaker_id: str) -> None:
        raw_path = self._reference_inputs[speaker_id].text().strip()
        if not raw_path:
            self._status_labels[speaker_id].setText("missing")
            return
        validation = validate_reference_audio(Path(raw_path).expanduser(), self.min_reference_seconds)
        self._status_labels[speaker_id].setText(validation.status)

    def _prepare_reference(self, speaker_id: str) -> None:
        raw_path = self._reference_inputs[speaker_id].text().strip()
        if not raw_path:
            self._status_labels[speaker_id].setText("missing")
            return
        if self.cleanup_enabled:
            try:
                cleaned_path, validation = prepare_reference_audio(
                    Path(raw_path).expanduser(),
                    self.work_dir,
                    self.min_reference_seconds,
                    Event(),
                    self.persistent_cache_dir,
                )
            except Exception as exc:
                validation = validate_reference_audio(Path(raw_path).expanduser(), self.min_reference_seconds)
                cleaned_path = validation.path if validation.exists and validation.supported else None
                validation.warnings.append(f"cleanup failed; using original audio: {exc}")
        else:
            validation = validate_reference_audio(Path(raw_path).expanduser(), self.min_reference_seconds)
            cleaned_path = validation.path if validation.exists and validation.supported else None
        self._status_labels[speaker_id].setText(validation.status)
        if cleaned_path:
            self._cleaned_paths[speaker_id] = str(cleaned_path)

    def _preview_voice(self, speaker_id: str) -> None:
        self._prepare_reference(speaker_id)
        if self.preview_callback:
            self.preview_callback(speaker_id, self._mapping_for(speaker_id))

    def _on_merge_changed(self, speaker_id: str) -> None:
        merge_target = self._merge_combos[speaker_id].currentData()
        if merge_target and merge_target in self._reference_inputs:
            ref = self._reference_inputs[merge_target].text().strip()
            if ref:
                self._reference_inputs[speaker_id].setText(ref)
                self._refresh_status(speaker_id)

    def _mapping_for(self, speaker_id: str) -> dict[str, str]:
        original = self._reference_inputs[speaker_id].text().strip()
        cleaned = self._cleaned_paths.get(speaker_id, "")
        profile_reference = str(self._profile_inputs[speaker_id].currentData() or "").strip()
        merge_target = self._merge_combos[speaker_id].currentData() if speaker_id in self._merge_combos else ""
        mapping = {
            "label": self._label_inputs[speaker_id].text().strip() or speaker_id.replace("_", " ").title(),
            "reference_audio_path": cleaned or original,
            "original_reference_audio_path": original,
            "cleaned_reference_audio_path": cleaned,
            "voice_profile_reference_audio_path": profile_reference,
            "reference_status": self._status_labels[speaker_id].text(),
        }
        if merge_target:
            mapping["merge_with"] = merge_target
        return mapping

    def mappings(self) -> dict[str, dict[str, str]]:
        result: dict[str, dict[str, str]] = {}
        for speaker_id, label_input in self._label_inputs.items():
            if self._reference_inputs[speaker_id].text().strip() and speaker_id not in self._cleaned_paths:
                self._prepare_reference(speaker_id)
            result[speaker_id] = self._mapping_for(speaker_id)
        return result
