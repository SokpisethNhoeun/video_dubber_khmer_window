from __future__ import annotations

import io
import os

import qrcode
from PyQt6.QtCore import QEasingCurve, QPropertyAnimation, QThread, QTimer, QUrl, Qt
from PyQt6.QtGui import QDesktopServices, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from config.user_secrets import load_user_secrets, save_user_secret
from gui.components import PlanCard, Stepper, WizardNavBar
from gui.icons import icon as themed_icon
from licensing.client import LicenseClient
from config.models import DEFAULT_WHISPER_MODEL
from config.paths import is_whisper_model_downloaded
from gui.workers import GeminiValidationWorker, ModelDownloadWorker
from modules import gemini_key_validator
from modules.model_downloader import ModelDownloadManager

STEP_TITLES = ["Activate License", "Verify Gmail", "Buy Subscription", "Gemini Key", "Download Models"]
MODEL_PRESETS = [
    ("tiny", "Tiny", "Ultra-fast (~75MB download, ~1GB RAM/VRAM, CPU-friendly)"),
    ("base", "Lightweight", "Best for low-end laptops (~140MB download, ~1.5GB RAM/VRAM)"),
    ("small", "Fast", "Best for quick drafts (~460MB download, ~2GB RAM/VRAM)"),
    ("medium", "Balanced", "Recommended quality (~1.5GB download, ~5GB RAM/VRAM)"),
    ("large-v3", "Best", "Highest accuracy (~3.0GB download, ~8GB RAM/VRAM, high-spec GPU)"),
]

PLAN_DEFINITIONS = [
    ("monthly", "Monthly", "$11.99", ["Full pipeline access", "Cancel anytime", "1 device license"], False),
    ("six_months", "6 Months", "$59.99", ["Full pipeline access", "~17% cheaper than monthly", "1 device license"], False),
    ("yearly", "Yearly", "$99.99", ["Full pipeline access", "Best value — ~30% cheaper than monthly", "1 device license"], True),
]


class StartupValidationDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Initializing Khmer Video Dubber")
        self.setModal(True)
        self.setFixedWidth(420)
        self.setFixedHeight(160)
        self.success = False
        self.message = ""

        # Apply app theme stylesheet if active
        try:
            from gui.theme import get_saved_theme, build_stylesheet
            self.setStyleSheet(build_stylesheet(get_saved_theme()))
        except Exception:
            pass

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)

        self.label = QLabel("Verifying licensing and API keys...")
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("font-size: 13px; font-weight: 600;")
        layout.addWidget(self.label)

        self.progress = QProgressBar()
        self.progress.setRange(0, 0)  # Indeterminate spinner
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(8)
        layout.addWidget(self.progress)

        self.thread = QThread()
        from gui.workers import StartupValidationWorker
        self.worker = StartupValidationWorker()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self._on_finished)
        self.thread.start()

    def _on_finished(self, success: bool, message: str) -> None:
        self.success = success
        self.message = message
        self.thread.quit()
        self.thread.wait()
        if success:
            self.accept()
        else:
            self.reject()

    def closeEvent(self, event) -> None:
        if self.thread.isRunning():
            self.thread.quit()
            self.thread.wait()
        super().closeEvent(event)


def validate_saved_startup_access() -> tuple[bool, str]:
    """Validate returning-customer access before constructing the main window."""
    dialog = StartupValidationDialog()
    if dialog.exec() == QDialog.DialogCode.Accepted:
        return True, dialog.message
    return False, dialog.message


class AccessOnboardingDialog(QDialog):
    def __init__(self, initial_message: str = "", parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Activate Khmer Video Dubber")
        self.setModal(True)
        self.setMinimumWidth(620)
        self.resize(700, 620)
        self._license_valid = False
        secrets = load_user_secrets()
        if secrets.get("LICENSE_ACTIVATION_TOKEN"):
            try:
                client = LicenseClient()
                if client.required:
                    client.timeout = 3.0
                    res = client.validate()
                    self._license_valid = res.valid
            except Exception:
                pass
        self._gemini_valid = False
        self._checkout_created = False
        self._payment_confirmed = False
        self._models_ready = is_whisper_model_downloaded(DEFAULT_WHISPER_MODEL)
        self._email_verification_token = ""
        self._selected_plan: str | None = None
        self._payment_reference_id = ""
        self._checkout_url = ""

        self._otp_timer = QTimer(self)
        self._otp_timer.setInterval(1000)
        self._otp_timer.timeout.connect(self._on_otp_tick)
        self._otp_cooldown_remaining = 0

        self._payment_poll_timer = QTimer(self)
        self._payment_poll_timer.setInterval(3000)
        self._payment_poll_timer.timeout.connect(self._poll_payment_status)

        root = QVBoxLayout(self)
        root.setContentsMargins(32, 32, 32, 32)
        root.setSpacing(20)

        # Apply app theme stylesheet if active
        try:
            from gui.theme import get_saved_theme, build_stylesheet
            self.setStyleSheet(build_stylesheet(get_saved_theme()))
        except Exception:
            pass

        title = QLabel("Khmer Video Dubber by NhoeunSokpiseth")
        title.setStyleSheet("font-size: 20px; font-weight: 700;")
        root.addWidget(title)
        description = QLabel(
            "Complete all five steps before the project can open. "
            "Each paid license works on one device."
        )
        description.setWordWrap(True)
        root.addWidget(description)

        self.stepper = Stepper(STEP_TITLES)
        self.stepper.step_clicked.connect(self._go_to_step)
        root.addWidget(self.stepper)

        self.page_stack = QStackedWidget()
        self.page_stack.addWidget(self._build_step_activate())
        self.page_stack.addWidget(self._build_step_gmail())
        self.page_stack.addWidget(self._build_step_buy())
        self.page_stack.addWidget(self._build_step_gemini())
        self.page_stack.addWidget(self._build_step_models())
        root.addWidget(self.page_stack, 1)

        self.nav_bar = WizardNavBar()
        self.nav_bar.back_clicked.connect(self._go_back)
        self.nav_bar.next_clicked.connect(self._go_next)
        root.addWidget(self.nav_bar)

        self.overall_status = QLabel(initial_message or "Complete the required steps above.")
        self.overall_status.setWordWrap(True)
        root.addWidget(self.overall_status)

        buttons = QHBoxLayout()
        cancel = QPushButton("Exit")
        cancel.clicked.connect(self.reject)
        self.open_button = QPushButton("Open Khmer Video Dubber")
        self.open_button.setEnabled(False)
        self.open_button.clicked.connect(self._open_app)
        buttons.addWidget(cancel)
        buttons.addStretch(1)
        buttons.addWidget(self.open_button)
        root.addLayout(buttons)

        self._go_to_step(0)

    # ------------------------------------------------------------------
    # Step page builders
    # ------------------------------------------------------------------
    def _build_step_gmail(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(14)
        layout.addWidget(self._heading("Verify your Gmail"))

        form_widget = QWidget()
        form = QFormLayout(form_widget)
        self.email = QLineEdit()
        self.email.setPlaceholderText("Email that will receive the license key")
        self.email.textChanged.connect(self._email_changed)
        self.email.returnPressed.connect(self._send_otp)
        self.send_otp_button = QPushButton("Send Gmail OTP")
        self.send_otp_button.setIcon(themed_icon("mdi.email-send-outline"))
        self.send_otp_button.clicked.connect(self._send_otp)
        form.addRow("Gmail", self.email)
        form.addRow("", self.send_otp_button)
        layout.addWidget(form_widget)

        otp_section = QVBoxLayout()
        otp_section.setContentsMargins(0, 4, 0, 0)
        otp_section.setSpacing(10)
        otp_section.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        self.otp_code = QLineEdit()
        self.otp_code.setPlaceholderText("6-digit code")
        self.otp_code.setMaxLength(6)
        self.otp_code.setFixedWidth(160)
        self.otp_code.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.otp_code.returnPressed.connect(self._verify_otp)
        otp_section.addWidget(self.otp_code, 0, Qt.AlignmentFlag.AlignHCenter)

        self.verify_otp_button = QPushButton("Verify")
        self.verify_otp_button.setIcon(themed_icon("mdi.check-decagram-outline"))
        self.verify_otp_button.setObjectName("StartButton")
        self.verify_otp_button.setFixedWidth(160)
        self.verify_otp_button.clicked.connect(self._verify_otp)
        otp_section.addWidget(self.verify_otp_button, 0, Qt.AlignmentFlag.AlignHCenter)

        self.email_status = QLabel("Verify Gmail before purchasing.")
        self.email_status.setObjectName("InlineFeedback")
        self.email_status.setProperty("state", "neutral")
        self.email_status.setWordWrap(True)
        self.email_status.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        otp_section.addWidget(self.email_status)

        layout.addLayout(otp_section)
        layout.addStretch(1)
        return page

    def _build_step_buy(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(14)
        layout.addWidget(self._heading("Choose your subscription"))

        cards_row = QHBoxLayout()
        cards_row.setSpacing(12)
        self._plan_cards: dict[str, PlanCard] = {}
        for plan_id, name, price, features, recommended in PLAN_DEFINITIONS:
            card = PlanCard(plan_id, name, price, features, recommended=recommended)
            card.clicked.connect(self._on_plan_selected)
            self._plan_cards[plan_id] = card
            cards_row.addWidget(card)
        layout.addLayout(cards_row)

        self.selected_plan_summary = QLabel("No plan selected")
        self.selected_plan_summary.setObjectName("InlineFeedback")
        self.selected_plan_summary.setProperty("state", "neutral")
        self.selected_plan_summary.setWordWrap(True)
        layout.addWidget(self.selected_plan_summary)

        promo_row = QHBoxLayout()
        promo_label = QLabel("Promo code (optional)")
        self.promo_code = QLineEdit()
        self.promo_code.setPlaceholderText("Example: SAVE20")
        self.promo_code.setMaxLength(30)
        self.promo_code.setClearButtonEnabled(True)
        self.promo_code.setAccessibleName("Optional promo code")
        self.promo_code.editingFinished.connect(
            lambda: self.promo_code.setText(self.promo_code.text().strip().upper())
        )
        promo_row.addWidget(promo_label)
        promo_row.addWidget(self.promo_code, 1)
        layout.addLayout(promo_row)

        promo_help = QLabel("Your promo code will be checked when checkout is created.")
        promo_help.setObjectName("PageDesc")
        promo_help.setWordWrap(True)
        layout.addWidget(promo_help)

        self.buy_button = QPushButton("Select a plan to continue")
        self.buy_button.setIcon(themed_icon("mdi.qrcode"))
        self.buy_button.setEnabled(False)
        self.buy_button.clicked.connect(self._buy)
        layout.addWidget(self.buy_button)

        payment_panel = QVBoxLayout()
        payment_panel.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        payment_panel.setSpacing(8)

        self.qr_label = QLabel()
        self.qr_label.setFixedSize(200, 200)
        self.qr_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.qr_label.setVisible(False)
        payment_panel.addWidget(self.qr_label, 0, Qt.AlignmentFlag.AlignHCenter)

        self.payment_status_chip = QLabel("")
        self.payment_status_chip.setObjectName("StatusChip")
        self.payment_status_chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.payment_status_chip.setVisible(False)
        payment_panel.addWidget(self.payment_status_chip, 0, Qt.AlignmentFlag.AlignHCenter)

        self.open_checkout_link = QPushButton("Having trouble? Open payment page in browser")
        self.open_checkout_link.setObjectName("SecondaryButton")
        self.open_checkout_link.setVisible(False)
        self.open_checkout_link.clicked.connect(self._open_checkout_in_browser)
        payment_panel.addWidget(self.open_checkout_link, 0, Qt.AlignmentFlag.AlignHCenter)

        layout.addLayout(payment_panel)

        self.buy_status = QLabel("Verify your Gmail on Step 1, then choose a plan.")
        self.buy_status.setObjectName("InlineFeedback")
        self.buy_status.setProperty("state", "neutral")
        self.buy_status.setWordWrap(True)
        layout.addWidget(self.buy_status)
        layout.addStretch(1)
        return page

    def _build_step_activate(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(self._heading("Activate the emailed license"))

        form_widget = QWidget()
        form = QFormLayout(form_widget)
        self.license_key = QLineEdit(load_user_secrets().get("LICENSE_KEY", ""))
        self.license_key.setPlaceholderText("KVD-XXXXXX-XXXXXX-XXXXXX-XXXXXX")
        self.license_key.returnPressed.connect(self._activate)
        self.activate_button = QPushButton("Activate on This Laptop")
        self.activate_button.setIcon(themed_icon("mdi.key-variant"))
        self.activate_button.clicked.connect(self._activate)
        paste_license = QPushButton("Paste")
        paste_license.setIcon(themed_icon("mdi.content-paste"))
        paste_license.clicked.connect(lambda: self.license_key.setText(QApplication.clipboard().text().strip()))
        self.license_status = QLabel("Waiting for license activation.")
        self.license_status.setObjectName("InlineFeedback")
        self.license_status.setProperty("state", "neutral")
        self.license_status.setWordWrap(True)
        key_row = QHBoxLayout()
        key_row.addWidget(self.license_key, 1)
        key_row.addWidget(paste_license)
        form.addRow("License key", key_row)
        form.addRow("", self.activate_button)
        self.register_button = QPushButton("Don't have a license key? Register & Buy here")
        self.register_button.setObjectName("SecondaryButton")
        self.register_button.setIcon(themed_icon("mdi.cart-outline"))
        self.register_button.clicked.connect(lambda: self._go_to_step(1))
        form.addRow("", self.register_button)
        form.addRow("", self.license_status)
        layout.addWidget(form_widget)
        layout.addStretch(1)
        return page

    def _build_step_gemini(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(12)
        layout.addWidget(self._heading("Add and validate your Gemini API key"))

        get_key_link = QLabel('<a href="https://aistudio.google.com/app/apikey" style="color:#22d3c8;">🔑 Get your free Gemini API key here →</a>')
        get_key_link.setOpenExternalLinks(True)
        get_key_link.setTextFormat(Qt.TextFormat.RichText)
        layout.addWidget(get_key_link)

        form_widget = QWidget()
        form = QFormLayout(form_widget)
        self.gemini_key = QLineEdit(os.getenv("GEMINI_API_KEY", ""))
        self.gemini_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.gemini_key.setPlaceholderText("Paste your Gemini API key here")
        self.gemini_key.returnPressed.connect(self._test_gemini)
        self.test_gemini_button = QPushButton("Save & Test Gemini Key")
        self.test_gemini_button.setIcon(themed_icon("mdi.robot-outline"))
        self.test_gemini_button.clicked.connect(self._test_gemini)
        paste_gemini = QPushButton("Paste")
        paste_gemini.setIcon(themed_icon("mdi.content-paste"))
        paste_gemini.clicked.connect(lambda: self.gemini_key.setText(QApplication.clipboard().text().strip()))
        self.gemini_status = QLabel("Waiting for Gemini validation.")
        self.gemini_status.setObjectName("InlineFeedback")
        self.gemini_status.setProperty("state", "neutral")
        self.gemini_status.setWordWrap(True)
        gemini_row = QHBoxLayout()
        gemini_row.addWidget(self.gemini_key, 1)
        gemini_row.addWidget(paste_gemini)
        form.addRow("Gemini key", gemini_row)
        form.addRow("", self.test_gemini_button)
        form.addRow("", self.gemini_status)
        layout.addWidget(form_widget)
        layout.addStretch(1)
        return page

    def _build_step_models(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setSpacing(12)
        layout.addWidget(self._heading("Download the Whisper speech model"))
        note = QLabel("Choose one preset. Downloads can be paused and resumed without losing completed bytes.")
        note.setWordWrap(True)
        layout.addWidget(note)
        self._model_rows = {}
        for model, label, description in MODEL_PRESETS:
            card = QWidget()
            card.setObjectName("Card")
            row = QHBoxLayout(card)
            title = QLabel(f"<b>{label}</b><br><span style='color:#8b93a7'>{model} · {description}</span>")
            title.setTextFormat(Qt.TextFormat.RichText)
            progress = QProgressBar()
            progress.setRange(0, 100)
            progress.setValue(100 if is_whisper_model_downloaded(model) else 0)
            progress.setMinimumWidth(180)
            detail = QLabel("Installed" if progress.value() == 100 else "Waiting")
            detail.setObjectName("StatusChip")
            detail.setProperty("state", "done" if progress.value() == 100 else "waiting")
            button = QPushButton("Installed" if progress.value() == 100 else "Download")
            button.setEnabled(progress.value() != 100)
            button.clicked.connect(lambda _checked=False, name=model: self._model_action(name))
            cancel = QPushButton("Cancel")
            cancel.setVisible(False)
            cancel.clicked.connect(lambda _checked=False, name=model: self._cancel_model(name))
            row.addWidget(title, 1)
            row.addWidget(progress)
            row.addWidget(detail)
            row.addWidget(button)
            row.addWidget(cancel)
            layout.addWidget(card)
            self._model_rows[model] = {"progress": progress, "detail": detail, "button": button, "cancel": cancel, "state": "done" if progress.value() == 100 else "waiting"}
        self.model_status = QLabel("A downloaded model is required before opening the app.")
        self.model_status.setObjectName("InlineFeedback")
        self.model_status.setProperty("state", "success" if self._models_ready else "neutral")
        layout.addWidget(self.model_status)
        layout.addStretch(1)
        return page

    @staticmethod
    def _heading(text: str) -> QLabel:
        label = QLabel(text)
        label.setStyleSheet("font-size: 15px; font-weight: 600;")
        return label

    # ------------------------------------------------------------------
    # Step navigation
    # ------------------------------------------------------------------
    def _completed_steps(self) -> set[int]:
        completed = set()
        if self._license_valid:
            completed.add(0)
            completed.add(1)
            completed.add(2)
        else:
            if self._email_verification_token:
                completed.add(1)
            if self._payment_confirmed:
                completed.add(2)
        if self._gemini_valid:
            completed.add(3)
        if self._models_ready:
            completed.add(4)
        return completed

    def _go_to_step(self, index: int) -> None:
        index = max(0, min(index, self.page_stack.count() - 1))
        if self._license_valid and (index == 1 or index == 2):
            if self.page_stack.currentIndex() == 0:
                index = 3
            else:
                index = 0
        if index != self.page_stack.currentIndex():
            self.page_stack.setCurrentIndex(index)
            effect = self.page_stack.currentWidget().graphicsEffect()
            if effect is None:
                from PyQt6.QtWidgets import QGraphicsOpacityEffect
                effect = QGraphicsOpacityEffect(self.page_stack.currentWidget())
                self.page_stack.currentWidget().setGraphicsEffect(effect)
            self._page_animation = QPropertyAnimation(effect, b"opacity", self)
            self._page_animation.setDuration(180)
            self._page_animation.setStartValue(0.2)
            self._page_animation.setEndValue(1.0)
            self._page_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
            self._page_animation.start()
        self.stepper.set_completed(self._completed_steps())
        self.stepper.set_current(index)
        self.nav_bar.set_state(index, self.page_stack.count())
        self._sync_nav_bar()

    def _sync_nav_bar(self) -> None:
        index = self.stepper.current()
        is_last = index == self.page_stack.count() - 1
        if is_last:
            ready = self._all_ready()
            self.nav_bar.set_next_enabled(
                ready,
                "" if ready else "Complete every onboarding step first.",
            )
        else:
            completed = index in self._completed_steps()
            self.nav_bar.set_next_enabled(
                completed,
                "" if completed else "Complete this step before continuing.",
            )

    def _go_back(self) -> None:
        index = self.stepper.current()
        if index == 3 and self._license_valid:
            self._go_to_step(0)
        else:
            self._go_to_step(index - 1)

    def _go_next(self) -> None:
        index = self.stepper.current()
        if index == self.page_stack.count() - 1:
            self._open_app()
        elif index == 0 and self._license_valid:
            self._go_to_step(3)
        else:
            self._go_to_step(index + 1)

    # ------------------------------------------------------------------
    # Actions (unchanged behavior from the previous single-page layout)
    # ------------------------------------------------------------------
    def _client(self) -> LicenseClient | None:
        client = LicenseClient()
        if not client.required:
            self.overall_status.setText("LICENSE_SERVER_URL is not configured. Contact the seller.")
            return None
        return client

    def _on_plan_selected(self, plan_id: str) -> None:
        self._selected_plan = plan_id
        for pid, card in self._plan_cards.items():
            card.set_selected(pid == plan_id)
        plan = next((item for item in PLAN_DEFINITIONS if item[0] == plan_id), None)
        if plan:
            _plan_id, name, price, _features, _recommended = plan
            self.selected_plan_summary.setText(f"Selected: {name} — {price}")
            self.selected_plan_summary.setProperty("state", "success")
            self._polish(self.selected_plan_summary)
        self._refresh_buy_button()

    def _refresh_buy_button(self) -> None:
        verified = bool(self._email_verification_token)
        selected = bool(self._selected_plan)
        ready = verified and selected and not self._checkout_created
        self.buy_button.setEnabled(ready)
        if self._checkout_created:
            self.buy_button.setText("Checkout created — complete payment below")
            return
        if not selected:
            self.buy_button.setText("Select a plan to continue")
            self.buy_status.setText("Choose one subscription plan above.")
            return
        plan = next((item for item in PLAN_DEFINITIONS if item[0] == self._selected_plan), None)
        plan_text = f"{plan[1]} — {plan[2]}" if plan else self._selected_plan
        if not verified:
            self.buy_button.setText("Verify Gmail to continue")
            self.buy_status.setText(f"{plan_text} selected. Verify your Gmail on Step 1 before purchasing.")
            return
        self.buy_button.setText(f"Continue with {plan_text}")
        self.buy_status.setText("Ready to create your Bakong KHQR checkout.")

    @staticmethod
    def _render_qr(data: str) -> QPixmap:
        image = qrcode.make(data)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        pixmap = QPixmap()
        pixmap.loadFromData(buffer.getvalue(), "PNG")
        return pixmap.scaled(
            200, 200, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )

    def _set_payment_status_chip(self, state: str, text: str) -> None:
        self.payment_status_chip.setText(text)
        self.payment_status_chip.setProperty("state", state)
        self._polish(self.payment_status_chip)

    def _open_checkout_in_browser(self) -> None:
        if self._checkout_url:
            QDesktopServices.openUrl(QUrl(self._checkout_url))

    def _buy(self) -> None:
        email = self.email.text().strip()
        if "@" not in email:
            QMessageBox.warning(self, "Purchase", "Enter a valid email address.")
            return
        if not self._selected_plan:
            QMessageBox.warning(self, "Purchase", "Choose a subscription plan first.")
            return
        client = self._client()
        if client is None:
            return
        if not self._email_verification_token:
            QMessageBox.warning(self, "Purchase", "Verify your Gmail OTP first.")
            return
        original_text = self.buy_button.text()
        checkout_error = ""
        self.buy_button.setEnabled(False)
        self.buy_button.setText("Creating secure checkout...")
        QApplication.processEvents()
        try:
            result = client.create_checkout(
                email,
                self._selected_plan,
                self._email_verification_token,
                self.promo_code.text(),
            )
        except Exception as exc:
            result = None
            checkout_error = str(exc)
            QMessageBox.warning(self, "Purchase", str(exc))
        finally:
            self.buy_button.setText(original_text)
        if result is None:
            self._refresh_buy_button()
            self._set_inline_feedback(self.buy_status, f"Could not create checkout: {checkout_error}", False)
            return
        if not result.created:
            self._refresh_buy_button()
            self._set_inline_feedback(self.buy_status, result.message, False)
            QMessageBox.warning(self, "Purchase", result.message)
            return
        self._checkout_url = result.checkout_url
        self._payment_reference_id = result.reference_id
        self._checkout_created = True
        self._refresh_buy_button()
        self.open_checkout_link.setVisible(True)
        if result.qr_string:
            self.qr_label.setPixmap(self._render_qr(result.qr_string))
            self.qr_label.setVisible(True)
        self.payment_status_chip.setVisible(True)
        self._set_payment_status_chip("waiting", "Waiting for payment")
        payment_help = (
            "Scan the QR code with your Bakong-enabled banking app. "
            if result.qr_string else
            "Open the payment page and follow the current payment instructions. "
        )
        self._set_inline_feedback(
            self.buy_status,
            payment_help + "This page updates automatically once payment is confirmed.",
            True,
        )
        self.stepper.set_completed(self._completed_steps())
        self._payment_poll_timer.start()

    def _poll_payment_status(self) -> None:
        if not self._payment_reference_id:
            return
        client = LicenseClient()
        result = client.check_payment_status(self._payment_reference_id)
        if not result.ok:
            return
        if result.status == "paid":
            self._payment_poll_timer.stop()
            self._payment_confirmed = True
            self._set_payment_status_chip("done", "Paid")
            self._set_inline_feedback(self.buy_status, result.message, True)
            self.stepper.set_completed(self._completed_steps())
            if self.stepper.current() == 2:
                self._go_to_step(0)
        else:
            self._set_payment_status_chip("waiting", "Waiting for payment")

    def _send_otp(self) -> None:
        email = self.email.text().strip()
        if "@" not in email:
            QMessageBox.warning(self, "Gmail verification", "Enter a valid Gmail address.")
            return
        client = self._client()
        if client is None:
            return
        self.send_otp_button.setEnabled(False)
        self.send_otp_button.setText("Sending OTP...")
        QApplication.processEvents()
        try:
            result = client.request_email_otp(email)
            self._set_inline_feedback(self.email_status, result.message, result.success)
            if result.success:
                self._start_otp_cooldown(result.resend_after_seconds or 60)
        finally:
            if not self._otp_timer.isActive():
                self.send_otp_button.setEnabled(True)
                self.send_otp_button.setText("Send Gmail OTP")

    def _start_otp_cooldown(self, seconds: int) -> None:
        self._otp_cooldown_remaining = seconds
        self.send_otp_button.setEnabled(False)
        self._update_otp_button_label()
        self._otp_timer.start()

    def _on_otp_tick(self) -> None:
        self._otp_cooldown_remaining -= 1
        if self._otp_cooldown_remaining <= 0:
            self._otp_timer.stop()
            self.send_otp_button.setEnabled(True)
            self.send_otp_button.setText("Resend OTP")
            return
        self._update_otp_button_label()

    def _update_otp_button_label(self) -> None:
        minutes, seconds = divmod(self._otp_cooldown_remaining, 60)
        self.send_otp_button.setText(f"Resend OTP ({minutes:02d}:{seconds:02d})")

    @staticmethod
    def _polish(widget: QWidget) -> None:
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def _set_inline_feedback(self, label: QLabel, message: str, success: bool) -> None:
        label.setText(message)
        label.setProperty("state", "success" if success else "error")
        self._polish(label)

    def _email_changed(self) -> None:
        self._email_verification_token = ""
        self.email_status.setText("Verify this Gmail address before purchasing.")
        self.email_status.setProperty("state", "neutral")
        self._polish(self.email_status)
        self._refresh_buy_button()
        self.stepper.set_completed(self._completed_steps())

    def _verify_otp(self) -> None:
        client = self._client()
        if client is None:
            return
        self.verify_otp_button.setEnabled(False)
        self.verify_otp_button.setText("Verifying...")
        QApplication.processEvents()
        try:
            result = client.verify_email_otp(self.email.text(), self.otp_code.text())
            self._set_inline_feedback(self.email_status, result.message, result.success)
            self._email_verification_token = result.verification_token if result.success else ""
            self._refresh_buy_button()
            if result.success:
                self._otp_timer.stop()
                if not self._selected_plan:
                    self.buy_status.setText("Gmail verified. Choose a plan and buy with Bakong KHQR.")
                self.buy_status.setStyleSheet("")
                self.stepper.set_completed(self._completed_steps())
                QTimer.singleShot(400, lambda: self._go_to_step(2))
            else:
                self.stepper.set_completed(self._completed_steps())
        finally:
            self.verify_otp_button.setEnabled(True)
            self.verify_otp_button.setText("Verify")

    def _activate(self) -> None:
        client = self._client()
        if client is None:
            return
        self.activate_button.setEnabled(False)
        self.activate_button.setText("Activating...")
        QApplication.processEvents()
        try:
            result = client.activate(self.license_key.text())
            self._license_valid = result.valid
            if result.valid:
                self._set_inline_feedback(self.license_status, f"Activated. Plan: {result.plan}; expires: {result.expires_at}", True)
                QMessageBox.information(
                    self,
                    "Activation Successful",
                    "License activated successfully! Gmail and purchase steps are skipped."
                )
                if self.stepper.current() == 0:
                    self._go_to_step(3)
            else:
                self._set_inline_feedback(self.license_status, result.message, False)
            self._refresh_open_button()
        finally:
            self.activate_button.setEnabled(True)
            self.activate_button.setText("Activate on This Laptop")

    def _test_gemini(self) -> None:
        key = self.gemini_key.text().strip()
        if hasattr(self, "_gemini_thread") and self._gemini_thread.isRunning():
            return
        self.test_gemini_button.setEnabled(False)
        self.test_gemini_button.setText("Validating...")
        self._gemini_thread = QThread(self)
        self._gemini_worker = GeminiValidationWorker(key)
        self._gemini_worker.moveToThread(self._gemini_thread)
        self._gemini_thread.started.connect(self._gemini_worker.run)
        self._gemini_worker.finished.connect(lambda valid, message: self._gemini_finished(key, valid, message))
        self._gemini_worker.finished.connect(self._gemini_thread.quit)
        self._gemini_thread.finished.connect(self._gemini_worker.deleteLater)
        self._gemini_thread.finished.connect(self._gemini_thread.deleteLater)
        self._gemini_thread.start()

    def _gemini_finished(self, key: str, valid: bool, message: str) -> None:
        self._gemini_valid = valid
        self._set_inline_feedback(self.gemini_status, message, valid)
        self.test_gemini_button.setEnabled(True)
        self.test_gemini_button.setText("Save & Test Gemini Key")
        if valid:
            save_user_secret("GEMINI_API_KEY", key)
            os.environ["GEMINI_API_KEY"] = key
        self._refresh_open_button()
        if valid and self.stepper.current() == 3:
            self._go_to_step(4)

    def _model_action(self, model: str) -> None:
        row = self._model_rows[model]
        if row["state"] == "downloading":
            self._model_manager.pause()
            return
        if row["state"] == "paused":
            self._model_manager.resume()
        else:
            self._model_manager = ModelDownloadManager(model)

        # Ensure any previous thread is finished before starting a new one
        old_thread = getattr(self, "_download_thread", None)
        if old_thread and old_thread.isRunning():
            old_thread.quit()
            old_thread.wait(3000)

        for name, other in self._model_rows.items():
            other["button"].setEnabled(name == model or other["state"] == "done")

        self._download_thread = QThread(self)
        self._download_worker = ModelDownloadWorker(self._model_manager)
        self._download_worker.moveToThread(self._download_thread)
        self._download_thread.started.connect(self._download_worker.run)
        self._download_worker.progress.connect(
            lambda filename, done, total, speed, eta: self._model_progress(model, filename, done, total, speed, eta)
        )
        self._download_worker.status.connect(lambda state: self._model_state(model, state))
        self._download_worker.finished.connect(lambda _path: self._model_complete(model))
        self._download_worker.failed.connect(lambda message: self._model_failed(model, message))
        # Only quit thread on terminal states — NOT on intermediate "downloading" state
        self._download_worker.finished.connect(self._download_thread.quit)
        self._download_worker.failed.connect(self._download_thread.quit)
        self._download_worker.status.connect(
            lambda state: self._download_thread.quit() if state in {"paused", "cancelled"} else None
        )
        self._download_thread.finished.connect(self._download_worker.deleteLater)
        self._download_thread.finished.connect(self._download_thread.deleteLater)
        self._download_thread.start()

    def _model_progress(self, model: str, filename: str, done: int, total: int, speed: float, eta: object) -> None:
        row = self._model_rows[model]
        if total > 0:
            row["progress"].setValue(int(done * 100 / total))
        else:
            # Unknown total — show indeterminate progress
            row["progress"].setRange(0, 0)
        eta_text = f" · ETA {int(float(eta))}s" if eta is not None else ""
        row["detail"].setText(f"{speed / 1024 / 1024:.1f} MB/s{eta_text}")

    def _model_state(self, model: str, state: str) -> None:
        row = self._model_rows[model]
        row["state"] = state if state != "connecting" else "downloading"
        if state == "connecting":
            # Show indeterminate spinner while fetching file list from HuggingFace
            row["progress"].setRange(0, 0)
            row["detail"].setText("Connecting...")
            row["button"].setText("Pause")
            row["cancel"].setVisible(True)
            row["detail"].setProperty("state", "active")
        else:
            # Reset to determinate progress bar unless actively downloading
            if state != "downloading":
                row["progress"].setRange(0, 100)
            row["button"].setText("Pause" if state == "downloading" else "Resume" if state == "paused" else "Download")
            row["cancel"].setVisible(state in {"downloading", "paused"})
            row["detail"].setProperty("state", "active" if state == "downloading" else "waiting")
        self._polish(row["detail"])

    def _cancel_model(self, model: str) -> None:
        if getattr(self, "_model_manager", None):
            self._model_manager.cancel()
        row = self._model_rows[model]
        row["state"] = "waiting"
        row["progress"].setValue(0)
        row["detail"].setText("Cancelled")
        row["button"].setText("Download")
        row["button"].setEnabled(True)
        row["cancel"].setVisible(False)

    def _model_complete(self, model: str) -> None:
        row = self._model_rows[model]
        row["state"] = "done"
        row["progress"].setValue(100)
        row["detail"].setText("Installed")
        row["detail"].setProperty("state", "done")
        row["button"].setText("Installed")
        row["button"].setEnabled(False)
        row["cancel"].setVisible(False)
        self._models_ready = True
        self._set_inline_feedback(self.model_status, f"Whisper {model} is ready.", True)
        self._refresh_open_button()

    def _model_failed(self, model: str, message: str) -> None:
        row = self._model_rows[model]
        row["state"] = "failed"
        row["detail"].setText("Failed")
        row["detail"].setProperty("state", "failed")
        row["button"].setText("Retry")
        row["button"].setEnabled(True)
        row["cancel"].setVisible(False)
        self._set_inline_feedback(self.model_status, message, False)

    def _all_ready(self) -> bool:
        if self._license_valid:
            return bool(self._gemini_valid and self._models_ready)
        return bool(self._email_verification_token and self._payment_confirmed and self._gemini_valid and self._models_ready)

    def _refresh_open_button(self) -> None:
        ready = self._all_ready()
        self.open_button.setEnabled(ready)
        if ready:
            self.overall_status.setText("All requirements passed. You can open the project.")
        else:
            if self._license_valid:
                self.overall_status.setText("License is active. Gemini API key and Whisper model download are required.")
            else:
                self.overall_status.setText("License activation (or Gmail verification + payment), Gemini API key, and Whisper model are required.")
        self.stepper.set_completed(self._completed_steps())
        self._sync_nav_bar()

    def _open_app(self) -> None:
        client = self._client()
        if client is None:
            return
        license_result = client.validate()
        self._license_valid = license_result.valid
        if not license_result.valid:
            self.license_status.setText(license_result.message)
            self._refresh_open_button()
            return
        if not self._all_ready():
            self._refresh_open_button()
            return
        self.accept()
