"""M-05: the startup script must be able to start the bot.

QuantDinger was removed and market data now comes straight from MT5, but
``startup.bat`` was left describing the old stack. Its step 4 polled
``http://localhost:8888/health`` for 60 seconds and then::

    echo ERROR: Backend did not start after 60s!
    pause
    exit /b 1

That endpoint belongs to a service the project no longer runs, so the check
could only ever fail, and step 5 — the one that launches ``main.py`` — was
unreachable. The documented way to start the bot could not start the bot.

Code being correct is not the same as the system being startable. These tests
cover the deployment surface, which no unit test touched.
"""

from __future__ import annotations

import os
import re

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STARTUP = os.path.join(REPO_ROOT, "startup.bat")


def _read(path: str) -> str:
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        return fh.read()


def _executable_lines(source: str):
    """Lines that actually run: no blanks, no `REM`/`::` comments."""
    for raw in source.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.upper().startswith("REM ") or line.startswith("::"):
            continue
        yield line


@pytest.fixture(scope="module")
def startup_source():
    assert os.path.exists(STARTUP), "startup.bat is missing"
    return _read(STARTUP)


@pytest.fixture(scope="module")
def startup_code(startup_source):
    return "\n".join(_executable_lines(startup_source))


class TestRemovedInfrastructureIsGone:
    """Nothing executable may depend on the deleted stack."""

    @pytest.mark.parametrize(
        "pattern,what",
        [
            (r"docker", "Docker"),
            (r"docker\s+compose", "docker compose"),
            (r"QuantDinger", "the QuantDinger directory"),
            (r"localhost:8888", "the QuantDinger health endpoint"),
            (r"backend_api_python", "the QuantDinger backend"),
        ],
    )
    def test_no_executable_reference(self, startup_code, pattern, what):
        hits = re.findall(pattern, startup_code, flags=re.IGNORECASE)
        assert not hits, (
            f"startup.bat still executes something depending on {what}: {hits}"
        )

    def test_the_comment_explaining_the_removal_survives(self, startup_source):
        """The history matters; it is why the script looks different now."""
        assert "QuantDinger" in startup_source, (
            "the explanatory header was dropped — someone will re-add the "
            "Docker steps without knowing why they went away"
        )

    def test_no_hardcoded_developer_paths(self, startup_code):
        """`C:\\Users\\ACER\\...` only worked on one machine."""
        hits = re.findall(r"C:\\Users\\[A-Za-z0-9_]+", startup_code)
        assert not hits, f"machine-specific paths in startup.bat: {hits}"


class TestTheBotActuallyGetsLaunched:
    def test_main_py_is_started(self, startup_code):
        assert "python main.py" in startup_code

    def test_nothing_can_exit_before_launching_main(self, startup_code):
        """The old failure mode: an `exit /b 1` gate that always tripped.

        Every remaining early exit must guard a real prerequisite (config, the
        interpreter, the terminal) — not an optional service.
        """
        lines = startup_code.splitlines()
        launch_index = next(
            i for i, line in enumerate(lines) if "python main.py" in line
        )
        gates_before_launch = [
            line for line in lines[:launch_index] if "exit /b 1" in line
        ]
        # There should be gates — a startup script with none is not checking
        # anything — but each must be reachable only on a genuine failure.
        assert gates_before_launch, "startup.bat validates nothing before launching"

        preceding = "\n".join(lines[:launch_index]).lower()
        for banned in ("8888", "docker", "quantdinger"):
            assert banned not in preceding, (
                f"an exit gate before launch still depends on {banned}"
            )


class TestPrerequisitesAreChecked:
    def test_env_file_presence_is_checked(self, startup_code):
        """A blank .env now stops the session, so catch it before MT5 starts."""
        assert ".env" in startup_code

    def test_python_availability_is_checked(self, startup_code):
        assert "python --version" in startup_code

    def test_mt5_is_started_and_waited_for(self, startup_code):
        assert "terminal64.exe" in startup_code
        assert "tasklist" in startup_code

    def test_the_mt5_wait_is_bounded(self, startup_code):
        """The Docker wait was an unbounded `goto` loop with no escape."""
        assert re.search(r"lss\s+\d+", startup_code), (
            "the MT5 wait loop has no attempt ceiling — it can hang forever"
        )

    def test_the_session_is_verified_before_the_bot_starts(self, startup_code):
        """A running terminal is not a logged-in terminal."""
        assert "ensure_session" in startup_code

        lines = startup_code.splitlines()
        verify = next(i for i, line in enumerate(lines) if "ensure_session" in line)
        launch = next(i for i, line in enumerate(lines) if "python main.py" in line)
        assert verify < launch, "the connection check runs after the bot starts"


class TestNoUnescapedPercentSigns:
    """A literal `%` in a .bat file starts a variable reference to cmd.exe.

    ``python -c "... print('...%s...' % (...)) ..."`` looked like ordinary
    Python string formatting, but cmd.exe strips `%s balance=%s` (or
    anything between two bare `%`s that isn't a real environment variable)
    before Python ever sees the line — the interpreter then received a
    mangled, often syntactically invalid one-liner. Depending on exactly how
    cmd.exe folded the string this either raised a confusing error inside the
    "MT5 connection failed" branch or, in some invocation contexts, closed
    without a readable message — reported as "the script just closes."

    Every batch variable reference has a recognizable shape: `%VAR%`,
    `%~dp0`-style modifiers, `%0`-`%9`/`%*` positional parameters, or an
    escaped `%%`. Anything left over after stripping those is an unescaped
    percent sign — usually printf-style Python formatting that leaked into a
    .bat file.
    """

    _VALID_PERCENT_FORMS = re.compile(
        r"%%"                       # escaped literal percent
        r"|%~[a-zA-Z0-9]*\d"        # %~dp0, %~nx1, etc.
        r"|%[A-Za-z_][A-Za-z0-9_]*%"  # %VAR%
        r"|%[0-9*]"                 # %0-%9, %*
    )

    def test_no_bare_percent_signs_survive_batch_variable_stripping(self, startup_code):
        offenders = []
        for line in startup_code.splitlines():
            stripped = self._VALID_PERCENT_FORMS.sub("", line)
            if "%" in stripped:
                offenders.append(line)
        assert not offenders, (
            "unescaped '%' in startup.bat — cmd.exe will silently mangle "
            "these before any embedded command (e.g. `python -c \"...\"`) "
            "runs:\n  " + "\n  ".join(offenders)
        )

    def test_the_checker_flags_the_original_defect(self):
        """Sanity check: the regex above must actually catch the bug it
        exists for, not just report a clean file by accident."""
        broken_line = (
            "python -c \"print('Connected: login=%s balance=%s' "
            "% (acc.login, acc.balance))\""
        )
        stripped = self._VALID_PERCENT_FORMS.sub("", broken_line)
        assert "%" in stripped

    def test_the_checker_does_not_flag_real_batch_variables(self):
        clean_line = 'echo %CD% %MT5_TRIES% %errorlevel% %~dp0 %1 %%literal%%'
        stripped = self._VALID_PERCENT_FORMS.sub("", clean_line)
        assert "%" not in stripped


class TestEnvExampleMatchesConfig:
    """A copied .env.example must name every variable config.py reads."""

    def test_every_required_variable_is_documented(self):
        import ast

        example = _read(os.path.join(REPO_ROOT, ".env.example"))
        documented = {
            line.split("=", 1)[0].strip()
            for line in example.splitlines()
            if "=" in line and not line.strip().startswith("#")
        }

        with open(os.path.join(REPO_ROOT, "config.py"), encoding="utf-8-sig") as fh:
            tree = ast.parse(fh.read())

        required = {
            node.args[0].value
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in {"_env_str", "_env_int", "_env_float"}
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value.startswith("MT5_")
            and node.args[0].value != "MT5_ORDER_TYPE_FILLING"
        }

        assert required, "the scan found no MT5_* variables — it is not working"
        assert required <= documented, (
            f"variables config.py needs but .env.example omits: "
            f"{sorted(required - documented)}"
        )

    def test_example_does_not_carry_removed_services(self):
        example = _read(os.path.join(REPO_ROOT, ".env.example"))
        assert "QUANTDINGER" not in example.upper(), (
            "QuantDinger settings linger in .env.example"
        )
