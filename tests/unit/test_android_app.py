"""Unit tests for AndroidApp in PymordialDroid."""

import pytest
from pymordial.core.app import PymordialApp
from pymordial.core.screen import PymordialScreen
from pymordial.core.state_machine import AppState
from pymordial.ui.image import PymordialImage

from pymordialdroid.android_app import AndroidApp


class TestAndroidApp:
    def test_app_satisfies_contract(self):
        """AndroidApp must be a subclass of PymordialApp."""
        assert issubclass(AndroidApp, PymordialApp)
        app = AndroidApp(app_name="TestApp", package_name="com.test")
        assert app.app_name == "TestApp"
        assert app.package_name == "com.test"

    def test_instantiation_and_properties(self):
        app = AndroidApp(
            app_name="TestGame",
            package_name="com.test.game",
        )
        assert app.app_name == "TestGame"
        assert app.package_name == "com.test.game"
        assert app.screens == {}
        assert app.ready_element is None
        assert app.is_closed()
        assert not app.is_open()
        assert not app.is_loading()

    def test_validation_empty_app_name(self):
        with pytest.raises(ValueError, match="app_name must be a non-empty string"):
            AndroidApp(app_name="", package_name="com.test.game")

    def test_validation_empty_package_name(self):
        with pytest.raises(ValueError, match="package_name must be a non-empty string"):
            AndroidApp(app_name="TestGame", package_name="")

    def test_screens_and_ready_element(self):
        element = PymordialImage(
            label="start_btn", filepath="assets/start.png", confidence=0.8
        )
        screen = PymordialScreen(name="main_menu")
        screen.add_element(element)

        app = AndroidApp(
            app_name="TestGame",
            package_name="com.test.game",
            screens={"main_menu": screen},
            ready_element=element,
        )
        assert "main_menu" in app.screens
        assert app.ready_element == element

    def test_check_ready_transition(self, mocker):
        element = PymordialImage(
            label="start_btn", filepath="assets/start.png", confidence=0.8
        )
        app = AndroidApp(
            app_name="TestGame",
            package_name="com.test.game",
            ready_element=element,
        )

        mock_controller = mocker.MagicMock()
        mock_controller.is_element_visible.return_value = True
        app.pymordial_controller = mock_controller

        # Must be LOADING before transitioning to READY
        app.app_state.transition_to(AppState.LOADING)
        assert app.is_loading()

        res = app.check_ready()
        assert res is True
        assert app.is_open()

    def test_open_transitions_to_loading(self, mocker):
        """open() transitions to LOADING on success."""
        app = AndroidApp(app_name="TestApp", package_name="com.test.app")

        mock_controller = mocker.MagicMock()
        mock_controller.open_app.return_value = True
        app.pymordial_controller = mock_controller

        assert app.is_closed()
        result = app.open()
        assert result is True
        assert app.is_loading()
        assert not app.is_open()

    def test_open_transitions_to_ready_with_ready_element(self, mocker):
        """open() with ready_element transitions to READY via wait_until_ready."""
        element = PymordialImage(
            label="start_btn", filepath="assets/start.png", confidence=0.8
        )
        app = AndroidApp(
            app_name="TestApp",
            package_name="com.test.app",
            ready_element=element,
        )

        mock_controller = mocker.MagicMock()
        mock_controller.open_app.return_value = True
        mock_controller.is_element_visible.return_value = True
        app.pymordial_controller = mock_controller

        result = app.open()
        assert result is True
        assert app.is_open()
        assert not app.is_loading()

    def test_open_remains_loading_when_ready_element_not_found(self, mocker):
        """open() stays LOADING when ready_element never appears."""
        element = PymordialImage(
            label="start_btn", filepath="assets/start.png", confidence=0.8
        )
        app = AndroidApp(
            app_name="TestApp",
            package_name="com.test.app",
            ready_element=element,
        )

        mock_controller = mocker.MagicMock()
        mock_controller.open_app.return_value = True
        mock_controller.is_element_visible.return_value = False
        app.pymordial_controller = mock_controller

        result = app.open(timeout=1)
        assert result is False
        assert app.is_loading()
        assert not app.is_open()

    def test_close_transitions_to_closed(self, mocker):
        """close() transitions to CLOSED on success."""
        app = AndroidApp(app_name="TestApp", package_name="com.test.app")
        app.app_state.transition_to(AppState.LOADING)

        mock_controller = mocker.MagicMock()
        mock_controller.close_app.return_value = True
        app.pymordial_controller = mock_controller

        assert app.is_loading()
        result = app.close()
        assert result is True
        assert app.is_closed()

    def test_open_failure_stays_closed(self, mocker):
        """open() stays CLOSED when the controller reports failure."""
        app = AndroidApp(app_name="TestApp", package_name="com.test.app")

        mock_controller = mocker.MagicMock()
        mock_controller.open_app.return_value = False
        app.pymordial_controller = mock_controller

        assert app.is_closed()
        result = app.open()
        assert result is False
        assert app.is_closed()

    def test_wait_until_ready_transitions_to_ready(self, mocker):
        """wait_until_ready() polls and transitions to READY when element found."""
        element = PymordialImage(
            label="start_btn", filepath="assets/start.png", confidence=0.8
        )
        app = AndroidApp(
            app_name="TestApp",
            package_name="com.test.app",
            ready_element=element,
        )
        app.app_state.transition_to(AppState.LOADING)

        mock_controller = mocker.MagicMock()
        mock_controller.is_element_visible.return_value = True
        app.pymordial_controller = mock_controller

        assert app.is_loading()
        result = app.wait_until_ready(timeout=5)
        assert result is True
        assert app.is_open()

    def test_wait_until_ready_no_element(self, mocker):
        """wait_until_ready() returns False when no ready_element is set."""
        app = AndroidApp(app_name="TestApp", package_name="com.test.app")
        app.app_state.transition_to(AppState.LOADING)

        mock_controller = mocker.MagicMock()
        app.pymordial_controller = mock_controller

        result = app.wait_until_ready()
        assert result is False
        assert app.is_loading()

    def test_wait_until_ready_not_loading(self, mocker):
        """wait_until_ready() returns False when not in LOADING state."""
        app = AndroidApp(app_name="TestApp", package_name="com.test.app")
        # State starts as CLOSED; no need to transition

        mock_controller = mocker.MagicMock()
        app.pymordial_controller = mock_controller

        result = app.wait_until_ready()
        assert result is False
        assert app.is_closed()
