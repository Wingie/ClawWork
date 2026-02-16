"""
Beta9 Code Execution Endpoint

This module defines a Beta9 endpoint that executes arbitrary Python code.
It is deployed to the Beta9 infrastructure and called by the Beta9Sandbox.
"""

import sys
import io
import contextlib
import traceback
import os
import json
from typing import Dict, Any

try:
    from beta9 import endpoint, Image, Volume
except ImportError:
    print("Beta9 SDK not installed")

# Define the image for the worker
# We need a standard python environment. 
# We can add dependencies as needed.
exec_image = Image(
    base_image="python:3.10",
).add_python_packages([
    "pandas",
    "numpy",
    "requests",
    "scikit-learn",
    "matplotlib",
    "seaborn",
])

@endpoint(
    name="agento-code-exec",
    image=exec_image,
    cpu=1,
    memory="2Gi",
    timeout=600, # 10 minutes
    volumes=[Volume(name="agento-sandbox-vol", mount_path="/sandbox")]
)
def run_code(code: str) -> Dict[str, Any]:
    """
    Execute Python code and return stdout, stderr, and result.
    """
    # Capture stdout and stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    exit_code = 0
    error_message = None

    # Use the mounted volume as the workspace
    files_dir = "/sandbox"
    os.makedirs(files_dir, exist_ok=True)
    
    # Switch to sandbox directory so file operations are relative to it
    cwd = os.getcwd()
    try:
        os.chdir(files_dir)
    except Exception:
        # Fallback if mount failed
        files_dir = "/tmp/sandbox_files"
        os.makedirs(files_dir, exist_ok=True)
        os.chdir(files_dir)

    try:
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
            try:
                # We use exec() to run the code
                # Note: This is NOT fully sandboxed in terms of security 
                # (it runs as the user in the container).
                # But it is isolated in the container.
                exec(code, {"__name__": "__main__"})
            except Exception:
                # Print traceback to stderr
                traceback.print_exc()
                exit_code = 1
    except Exception as e:
        # Fallback for system errors
        error_message = str(e)
        exit_code = 1
    finally:
        # Restore CWD
        os.chdir(cwd)

    return {
        "stdout": stdout_capture.getvalue(),
        "stderr": stderr_capture.getvalue(),
        "exit_code": exit_code,
        "error": error_message,
        "files_dir": files_dir
    }

if __name__ == "__main__":
    # Local testing
    code = "print('Hello from Beta9 local test'); import os; print(os.getcwd())"
    print(run_code(code))
