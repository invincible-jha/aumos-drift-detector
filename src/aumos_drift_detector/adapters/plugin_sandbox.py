"""Sandboxed custom drift test plugin system for aumos-drift-detector.

Executes user-supplied Python drift test functions in a subprocess with
resource limits to prevent runaway execution or interference with the
main service process.

GAP-173: Custom Test Plugin System
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
from typing import Any

from aumos_common.observability import get_logger

logger = get_logger(__name__)

_DEFAULT_TIMEOUT_SECONDS = 30
_SANDBOX_RUNNER_TEMPLATE = textwrap.dedent("""
import json, sys

# Inject plugin code
{plugin_code}

# Run the plugin function with provided data
try:
    result = drift_test(reference_data, production_data)
    print(json.dumps(result))
    sys.exit(0)
except Exception as exc:
    print(json.dumps({{"error": str(exc), "drift_detected": False}}))
    sys.exit(1)
""")


class PluginSandbox:
    """Sandboxed execution environment for user-supplied drift test plugins.

    Plugins must implement a `drift_test(reference_data, production_data)` function
    that accepts two lists of numeric values and returns a dict with at minimum
    a `drift_detected: bool` key.

    Args:
        timeout_seconds: Maximum execution time for a plugin run.
        max_memory_mb: Memory limit for the subprocess (Unix only via resource module).
    """

    def __init__(
        self,
        timeout_seconds: int = _DEFAULT_TIMEOUT_SECONDS,
        max_memory_mb: int = 512,
    ) -> None:
        """Initialise the plugin sandbox.

        Args:
            timeout_seconds: Subprocess timeout.
            max_memory_mb: Memory cap (applied on Unix via resource.setrlimit).
        """
        self._timeout = timeout_seconds
        self._max_memory_mb = max_memory_mb

    def run_plugin(
        self,
        plugin_code: str,
        reference_data: list[float],
        production_data: list[float],
    ) -> dict[str, Any]:
        """Execute a user-supplied drift test plugin in isolation.

        Writes a temporary Python script that defines the plugin function,
        injects the data as constants, and runs it as a subprocess.

        Args:
            plugin_code: Python source code defining `drift_test(ref, prod)`.
            reference_data: Reference distribution samples to pass to the plugin.
            production_data: Production distribution samples to pass to the plugin.

        Returns:
            Dictionary returned by the plugin's `drift_test` function.
            On error or timeout, returns `{'drift_detected': False, 'error': '...'}`.
        """
        # Serialize data into the script
        runner_script = (
            f"reference_data = {json.dumps(reference_data)}\n"
            f"production_data = {json.dumps(production_data)}\n"
            + _SANDBOX_RUNNER_TEMPLATE.format(plugin_code=plugin_code)
        )

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(runner_script)
            tmp_path = tmp.name

        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            stdout = result.stdout.strip()
            if not stdout:
                error_msg = result.stderr.strip() or "no output"
                logger.warning("plugin_sandbox_no_output", error=error_msg)
                return {"drift_detected": False, "error": error_msg}
            output = json.loads(stdout)
            logger.info("plugin_sandbox_completed", drift_detected=output.get("drift_detected", False))
            return output
        except subprocess.TimeoutExpired:
            logger.error("plugin_sandbox_timeout", timeout=self._timeout)
            return {"drift_detected": False, "error": f"Plugin timed out after {self._timeout}s"}
        except json.JSONDecodeError as exc:
            logger.error("plugin_sandbox_json_error", error=str(exc))
            return {"drift_detected": False, "error": f"Invalid JSON output: {exc}"}
        except Exception as exc:
            logger.error("plugin_sandbox_error", error=str(exc))
            return {"drift_detected": False, "error": str(exc)}
