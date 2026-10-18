"""Tests for the runtime socket-level egress guard (Member 2 requirement).

These tests verify:
1. Attempting to connect to a public IP raises RuntimeError (egress blocked).
2. Loopback addresses (127.0.0.1, ::1, localhost) are allowed through.
3. install_egress_guard() is idempotent.
4. _is_loopback() correctly classifies addresses.
5. The static audit logic detects a missing egress guard.

No real TCP connections are made: for blocked addresses the guard raises before
the OS sees the connect(); for loopback addresses the OS will refuse on port 1
(which is expected and distinguishes "guard allowed it" from "guard blocked it").
"""
from __future__ import annotations

import re
import socket
import unittest

# Import the module under test.  Importing it also installs the guard.
from legal_triage.local_runtime import _is_loopback, install_egress_guard


class TestIsLoopback(unittest.TestCase):
    """Unit tests for the _is_loopback helper."""

    def test_localhost_string(self) -> None:
        self.assertTrue(_is_loopback("localhost"))

    def test_127_0_0_1(self) -> None:
        self.assertTrue(_is_loopback("127.0.0.1"))

    def test_127_x_x_x_range(self) -> None:
        # All 127.x.x.x addresses are loopback per RFC 5735.
        self.assertTrue(_is_loopback("127.100.200.50"))

    def test_ipv6_loopback(self) -> None:
        self.assertTrue(_is_loopback("::1"))

    def test_public_ipv4(self) -> None:
        self.assertFalse(_is_loopback("8.8.8.8"))

    def test_private_but_non_loopback(self) -> None:
        # 192.168.x.x is private but NOT loopback.
        self.assertFalse(_is_loopback("192.168.1.1"))

    def test_unknown_hostname_treated_as_non_loopback(self) -> None:
        # Unresolvable or non-IP hostnames must be treated as non-loopback (fail-closed).
        self.assertFalse(_is_loopback("example.com"))

    def test_empty_string(self) -> None:
        self.assertFalse(_is_loopback(""))


class TestEgressGuardBlocking(unittest.TestCase):
    """Verify that the installed guard blocks non-loopback connect() calls."""

    def test_public_ip_is_blocked(self) -> None:
        """Connecting to a public IP must raise RuntimeError, not OSError."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with self.assertRaises(RuntimeError) as ctx:
                # Port 80 – the guard fires before any real TCP handshake.
                sock.connect(("8.8.8.8", 80))
            self.assertIn("Egress blocked", str(ctx.exception))
            self.assertIn("8.8.8.8", str(ctx.exception))
        finally:
            sock.close()

    def test_private_non_loopback_ip_is_blocked(self) -> None:
        """192.168.x.x is private but not loopback – must be blocked."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with self.assertRaises(RuntimeError):
                sock.connect(("192.168.1.1", 8080))
        finally:
            sock.close()

    def test_loopback_127_passes_to_real_connect(self) -> None:
        """127.0.0.1 must pass through (OS may refuse – that shows guard allowed it)."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        try:
            # Port 1 is almost certainly not in use; we expect an OS-level refusal.
            with self.assertRaises((ConnectionRefusedError, OSError)):
                sock.connect(("127.0.0.1", 1))
        finally:
            sock.close()

    def test_localhost_passes_to_real_connect(self) -> None:
        """'localhost' should also be let through; OS will refuse on port 1."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        try:
            with self.assertRaises((ConnectionRefusedError, OSError)):
                sock.connect(("localhost", 1))
        finally:
            sock.close()

    def test_ipv6_loopback_passes(self) -> None:
        """::1 must not be blocked by the guard."""
        sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        try:
            with self.assertRaises((ConnectionRefusedError, OSError)):
                sock.connect(("::1", 1, 0, 0))
        finally:
            sock.close()


class TestEgressGuardIdempotency(unittest.TestCase):
    """install_egress_guard() must be safe to call multiple times."""

    def test_double_install_does_not_double_wrap(self) -> None:
        """Calling install_egress_guard() twice must not stack-wrap connect()."""
        import legal_triage.local_runtime as lrt

        connect_before = socket.socket.connect
        lrt.install_egress_guard()
        connect_after = socket.socket.connect
        self.assertIs(
            connect_before,
            connect_after,
            "install_egress_guard() applied a second time, wrapping connect() again.",
        )

    def test_guard_still_blocks_after_double_install(self) -> None:
        """Even after a second install call, public IPs must still be blocked."""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            with self.assertRaises(RuntimeError):
                sock.connect(("93.184.216.34", 80))  # example.com IP
        finally:
            sock.close()


class TestStaticAuditDetectsGuard(unittest.TestCase):
    """Verify that static_privacy_audit logic detects a missing egress guard."""

    # Regex matching a genuine module-level call (not a def/class definition line).
    # Kept in sync with scripts/static_privacy_audit.py.
    _MODULE_LEVEL_CALL_RE = re.compile(
        r"^(?![ \t]*(?:def|class)\s).*\binstall_egress_guard\(\)",
        re.MULTILINE,
    )

    def _run_egress_checks(self, runtime_source: str) -> list[str]:
        """Run only the egress-guard portion of the static audit on a given source string."""
        errors: list[str] = []
        if "def install_egress_guard" not in runtime_source:
            errors.append("Runtime egress guard function install_egress_guard is missing.")
        if not self._MODULE_LEVEL_CALL_RE.search(runtime_source):
            errors.append("install_egress_guard() is not called at module level.")
        if "socket.socket.connect" not in runtime_source:
            errors.append("Egress guard does not patch socket.socket.connect.")
        return errors

    def test_real_source_passes_audit(self) -> None:
        from pathlib import Path
        source = (
            Path(__file__).resolve().parents[1] / "legal_triage" / "local_runtime.py"
        ).read_text(encoding="utf-8")
        errors = self._run_egress_checks(source)
        self.assertEqual(errors, [], f"Static audit found issues in real source: {errors}")

    def test_missing_function_detected(self) -> None:
        fake_source = "# no egress guard here\nself.session.trust_env = False\n"
        errors = self._run_egress_checks(fake_source)
        self.assertTrue(
            any("install_egress_guard" in e for e in errors),
            "Audit did not flag missing install_egress_guard function.",
        )

    def test_missing_module_level_call_detected(self) -> None:
        fake_source = (
            "def install_egress_guard():\n    pass\n"
            "socket.socket.connect = lambda *a: None\n"
        )
        errors = self._run_egress_checks(fake_source)
        self.assertTrue(
            any("module level" in e for e in errors),
            "Audit did not flag missing module-level install_egress_guard() call.",
        )

    def test_missing_socket_patch_detected(self) -> None:
        fake_source = (
            "def install_egress_guard():\n    pass\n"
            "install_egress_guard()\n"
        )
        errors = self._run_egress_checks(fake_source)
        self.assertTrue(
            any("socket.socket.connect" in e for e in errors),
            "Audit did not flag missing socket.socket.connect patch.",
        )


if __name__ == "__main__":
    unittest.main()
