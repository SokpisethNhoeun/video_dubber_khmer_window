from types import SimpleNamespace

from gui.dialogs.onboarding import AccessOnboardingDialog


class _TextWidget:
    def __init__(self) -> None:
        self.text = ""
        self.enabled = None

    def setText(self, text: str) -> None:
        self.text = text

    def setEnabled(self, enabled: bool) -> None:
        self.enabled = enabled


def _dialog_state(*, verified: bool, plan: str = "", checkout_created: bool = False):
    return SimpleNamespace(
        _email_verification_token="token" if verified else "",
        _selected_plan=plan,
        _checkout_created=checkout_created,
        buy_button=_TextWidget(),
        buy_status=_TextWidget(),
    )


def test_purchase_button_explains_that_a_plan_must_be_selected() -> None:
    dialog = _dialog_state(verified=True)

    AccessOnboardingDialog._refresh_buy_button(dialog)

    assert dialog.buy_button.enabled is False
    assert dialog.buy_button.text == "Select a plan to continue"
    assert "Choose one subscription plan" in dialog.buy_status.text


def test_purchase_button_explains_that_email_must_be_verified() -> None:
    dialog = _dialog_state(verified=False, plan="monthly")

    AccessOnboardingDialog._refresh_buy_button(dialog)

    assert dialog.buy_button.enabled is False
    assert dialog.buy_button.text == "Verify Gmail to continue"
    assert "Monthly — $11.99 selected" in dialog.buy_status.text


def test_purchase_button_names_selected_plan_when_ready() -> None:
    dialog = _dialog_state(verified=True, plan="yearly")

    AccessOnboardingDialog._refresh_buy_button(dialog)

    assert dialog.buy_button.enabled is True
    assert dialog.buy_button.text == "Continue with Yearly — $99.99"
    assert dialog.buy_status.text == "Ready to create your Bakong KHQR checkout."
