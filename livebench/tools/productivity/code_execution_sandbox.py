"""
Code execution tool with sandboxing
"""

from langchain_core.tools import tool
from typing import Dict, Any, Optional, List
try:
    from e2b_code_interpreter import Sandbox
except ImportError:
    Sandbox = None
import os
import re
from pathlib import Path
from dotenv import load_dotenv
try:
    import beta9
    # Check if we can import the deployed function pointer if needed, 
    # but usually we invoke via client/name.
    # For now we'll assume we use the beta9 client directly.
    from beta9 import inference
except ImportError:
    beta9 = None

load_dotenv()

# Import global state from parent module
def _get_global_state():
    """Get global state from parent module"""
    from livebench.tools.direct_tools import _global_state
    return _global_state




# Docker Sandbox Implementation for Fallback
class DockerSandbox:
    """
    Docker-based implementation of Sandbox interface.
    Runs code inside the 'django' service container using `docker compose exec`.
    WARNING: This does NOT provide session persistence for variables (each run is a new process).
    """
    def __init__(self, id: str = "docker-sandbox", timeout: int = 3600):
        self.id = id
        self.timeout = timeout
        self.files = self.DockerFiles(self)
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")) # FlowState root
        self.compose_file = os.path.join(self.project_root, "docker-compose.yml")
        
        # Verify docker compose file exists
        if not os.path.exists(self.compose_file):
            print(f"⚠️  WARNING: docker-compose.yml not found at {self.compose_file}")
            print("    Docker execution might fail.")
            
        print(f"🐳 Initialized Docker Sandbox (using {self.compose_file})")

    def kill(self):
        """Cleanup resources"""
        pass

    def run_code(self, code: str) -> Any:
        """Run code inside container using docker compose exec"""
        import subprocess
        
        # Helper class to mimic E2B execution result
        class ExecutionResult:
            def __init__(self, stdout, stderr, error=None):
                self.logs = type('Logs', (), {'stdout': stdout, 'stderr': stderr})()
                self.error = error

        try:
            # Command: docker compose exec -T django python -c "..."
            # We use -T to disable pseudo-tty allocation which is better for automation
            cmd = [
                "docker", "compose", 
                "-f", self.compose_file, 
                "exec", "-T", "django", 
                "python", "-c", code
            ]
            
            # Execute command
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=self.project_root,
                timeout=self.timeout
            )
            
            return ExecutionResult(
                stdout=result.stdout,
                stderr=result.stderr,
                error=None if result.returncode == 0 else f"Process exited with code {result.returncode}"
            )
                
        except Exception as e:
            return ExecutionResult(stdout="", stderr="", error=str(e))

    class DockerFiles:
        """Mimics E2B files API using docker commands"""
        def __init__(self, sandbox):
            self.sandbox = sandbox

        def list(self, path: str):
            """List files in container"""
            import subprocess
            cmd = [
                "docker", "compose", 
                "-f", self.sandbox.compose_file, 
                "exec", "-T", "django", 
                "ls", "-1", path
            ]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.sandbox.project_root)
                if result.returncode == 0:
                    return [
                        type('File', (), {'name': f.strip(), 'is_dir': False})() 
                        for f in result.stdout.splitlines() if f.strip()
                    ]
            except:
                pass
            return []

        def write(self, path: str, content: Any):
            """Write file to container using docker exec and cat"""
            import subprocess
            
            # Create directory first
            dir_path = os.path.dirname(path)
            if dir_path and dir_path != "/":
                subprocess.run(
                    ["docker", "compose", "-f", self.sandbox.compose_file, "exec", "-T", "django", "mkdir", "-p", dir_path],
                    cwd=self.sandbox.project_root,
                    check=False
                )
            
            # Write content
            cmd = [
                "docker", "compose", 
                "-f", self.sandbox.compose_file, 
                "exec", "-T", "django", 
                "sh", "-c", f"cat > '{path}'"
            ]
            
            input_bytes = content if isinstance(content, (bytes, bytearray)) else content.encode('utf-8')
            
            subprocess.run(
                cmd,
                input=input_bytes,
                cwd=self.sandbox.project_root,
                check=True
            )

        def read(self, path: str, format: str = "text"):
            """Read file from container"""
            import subprocess
            cmd = [
                "docker", "compose", 
                "-f", self.sandbox.compose_file, 
                "exec", "-T", "django", 
                "cat", path
            ]
            
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                cwd=self.sandbox.project_root
            )
            
            if result.returncode != 0:
                raise FileNotFoundError(f"File not found: {path}")
                
            if format == 'bytes':
                return result.stdout.encode('utf-8') # docker exec returns bytes mixed with text usually, need careful handling if strictly binary is needed
            return result.stdout

        def _resolve_path(self, path: str) -> str:
            """Return path as is (assumed absolute in container)"""
            return path



# Beta9 Sandbox Implementation
class Beta9Sandbox:
    """
    Beta9-based implementation of Sandbox interface.
    Runs code on Beta9 serverless infrastructure execution endpoint.
    Uses 'agento-sandbox-vol' for file persistence.
    """
    def __init__(self, id: str = "beta9-sandbox", timeout: int = 3600):
        self.id = id
        self.timeout = timeout
        self.files = self.Beta9Files(self)
        self.endpoint_name = "agento-code-exec"
        self.volume_name = "agento-sandbox-vol"
        print(f"🚀 Initialized Beta9 Sandbox (endpoint: {self.endpoint_name}, volume: {self.volume_name})")

    def kill(self):
        """Cleanup resources"""
        pass

    def run_code(self, code: str) -> Any:
        """Run code on Beta9 endpoint"""
        import requests
        import json
        from pathlib import Path

        # Helper class to mimic E2B execution result
        class ExecutionResult:
            def __init__(self, stdout, stderr, error=None):
                self.logs = type('Logs', (), {'stdout': stdout, 'stderr': stderr})()
                self.error = error

        try:
            # 1. Get Gateway URL and Token from ~/.beta9/config.ini
            config_path = Path.home() / ".beta9" / "config.ini"
            if not config_path.exists():
                # Fallback to contexts.json just in case
                config_path_json = Path.home() / ".beta9" / "contexts.json"
                if config_path_json.exists():
                     return ExecutionResult(stdout="", stderr="", error="Please update Beta9Sandbox to support contexts.json (found but not using)")
                return ExecutionResult(stdout="", stderr="", error="Beta9 config not found (~/.beta9/config.ini)")
            
            import configparser
            config = configparser.ConfigParser()
            config.read(config_path)
            
            # Use 'default' section
            if 'default' not in config:
                 return ExecutionResult(stdout="", stderr="", error="No [default] section in Beta9 config")

            gateway_host = config['default'].get('gateway_host', 'localhost')
            # detailed debug info
            # print(f"DEBUG: gateway_host={gateway_host}")
            
            # config.ini usually has 1993 (grpc), we need 1994 (http)
            gateway_port = config['default'].get('gateway_port', '1993')
            if gateway_port == '1993':
                gateway_port = '1994'
            
            token = config['default'].get('token', '')

            # Construct URL
            # If host doesn't start with http, add it
            if not gateway_host.startswith('http'):
                gateway_url = f"http://{gateway_host}:{gateway_port}"
            else:
                gateway_url = f"{gateway_host}:{gateway_port}"

            # 2. Call Endpoint
            endpoint_url = f"{gateway_url}/endpoint/{self.endpoint_name}/v2"
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}"
            }
            
            payload = {"code": code}
            
            try:
                # print(f"DEBUG: Posting to {endpoint_url}")
                response = requests.post(endpoint_url, json=payload, headers=headers, timeout=self.timeout)
                response.raise_for_status()
                result = response.json()
                
                return ExecutionResult(
                    stdout=result.get("stdout", ""),
                    stderr=result.get("stderr", ""),
                    error=result.get("error")
                )
            except requests.exceptions.RequestException as e:
                 return ExecutionResult(stdout="", stderr="", error=f"Beta9 API Error: {str(e)}")
                 
        except Exception as e:
            return ExecutionResult(stdout="", stderr="", error=str(e))

    class Beta9Files:
        """Mimics E2B files API using beta9 CLI"""
        def __init__(self, sandbox):
            self.sandbox = sandbox

        def list(self, path: str):
            """List files in volume"""
            import subprocess
            # beta9 ls volume_name:path
            cmd = ["beta9", "ls", f"{self.sandbox.volume_name}:{path}"]
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                     return [
                        type('File', (), {'name': f.strip(), 'is_dir': False})() 
                        for f in result.stdout.splitlines() if f.strip()
                    ]
            except:
                pass
            return []

        def write(self, path: str, content: Any):
            """Write file to volume using beta9 cp"""
            import subprocess
            import tempfile
            
            # path is relative to volume root if not absolute?
            # volume path: volume_name:/path/to/file
            
            # Write to temp file first
            with tempfile.NamedTemporaryFile(delete=False) as tf:
                if isinstance(content, str):
                    tf.write(content.encode('utf-8'))
                else:
                    tf.write(content)
                temp_path = tf.name
            
            try:
                # beta9 cp local volume:/remote
                # We strip leading slash from path for relative to volume root?
                # or use absolute path in volume.
                remote_path = path if path.startswith('/') else f"/{path}"
                
                cmd = ["beta9", "cp", temp_path, f"{self.sandbox.volume_name}:{remote_path}"]
                subprocess.run(cmd, check=True, capture_output=True)
            except Exception as e:
                print(f"Error writing to Beta9 volume: {e}")
            finally:
                os.unlink(temp_path)

        def read(self, path: str, format: str = "text"):
            """Read file from volume"""
            import subprocess
            import tempfile
            
            remote_path = path if path.startswith('/') else f"/{path}"
            
            with tempfile.NamedTemporaryFile(delete=False) as tf:
                local_temp = tf.name
            
            try:
                cmd = ["beta9", "cp", f"{self.sandbox.volume_name}:{remote_path}", local_temp]
                subprocess.run(cmd, check=True, capture_output=True)
                
                with open(local_temp, 'rb') as f:
                    content = f.read()
                    
                if format == 'bytes':
                    return content
                return content.decode('utf-8')
            except Exception:
                raise FileNotFoundError(f"File not found: {path}")
            finally:
                if os.path.exists(local_temp):
                    os.unlink(local_temp)

        def _resolve_path(self, path: str) -> str:
            return path

# Session-level sandbox manager
class SessionSandbox:
    """
    Manages a persistent E2B sandbox for an agent session.
    This ensures files created in one execute_code call are accessible in subsequent calls.
    """
    _instance: Optional['SessionSandbox'] = None
    
    def __init__(self):
        self.sandbox: Optional[Any] = None  # Union[Sandbox, LocalSandbox]
        self.sandbox_id: Optional[str] = None
        self.uploaded_reference_files: Dict[str, str] = {}  # local_path -> remote_path
        self.use_local_fallback = False
    
    @classmethod
    def get_instance(cls) -> 'SessionSandbox':
        """Get or create the singleton session sandbox instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    @classmethod
    def reset(cls):
        """Reset the session sandbox (for new sessions/days)"""
        if cls._instance and cls._instance.sandbox:
            try:
                cls._instance.sandbox.kill()  # Use kill() for immediate termination
            except:
                pass
        cls._instance = None
    
    def get_or_create_sandbox(self, timeout: int = 3600) -> Any:  # Default 1 hour for task duration
        """Get existing sandbox or create a new one, with health check"""
        
        # Health check existing sandbox
        if self.sandbox is not None:
            try:
                # Quick health check - list root directory
                self.sandbox.files.list("/")
                return self.sandbox  # Sandbox is healthy
            except Exception as e:
                # Sandbox is dead, clean up and recreate
                print(f"⚠️ Sandbox {self.sandbox_id} died ({e}), recreating...")

                try:
                    self.sandbox.kill()  # Use kill() for immediate termination
                except:
                    pass
                
                self.sandbox = None
                self.sandbox_id = None
                self.uploaded_reference_files = {}
        
        # Create new sandbox if needed
        if self.sandbox is None:
            e2b_key = os.getenv("E2B_API_KEY")
            
            # Check if E2B is configured AND available
            if Sandbox and e2b_key and e2b_key.strip() and e2b_key != "your-e2b-api-key-here":
                # Try to use E2B
                try:
                    self.sandbox = Sandbox.create("gdpval-workspace", timeout=timeout)
                    self.sandbox_id = getattr(self.sandbox, "id", None)
                    self.use_local_fallback = False
                    print(f"🔧 Created persistent E2B sandbox: {self.sandbox_id}")
                except Exception as e:
                    print(f"❌ Failed to create E2B sandbox: {str(e)}")
                    print("⚠️ Falling back to BETA9 sandbox")
                    self.use_local_fallback = True # Beta9 behaves like local fallback in terms of file path resolution usually
                    self.sandbox = Beta9Sandbox()
                    self.sandbox_id = "beta9-fallback"
            else:
                # Use Beta9 as primary if E2B is not configured
                if beta9:
                     print("🔧 Using BETA9 sandbox as primary execution engine.")
                     self.sandbox = Beta9Sandbox()
                     self.sandbox_id = "beta9-primary"
                     self.use_local_fallback = True
                else:
                    # Use local fallback directly
                    if not Sandbox:
                        print("⚠️ E2B SDK not installed. Using DOCKER sandbox.")
                    else:
                        print("⚠️ No E2B_API_KEY found (or is default). Using DOCKER sandbox.")
                    
                    self.use_local_fallback = True
                    self.sandbox = DockerSandbox()
                    self.sandbox_id = "docker-fallback"
        
        return self.sandbox
    
    def upload_reference_file(self, local_path: str, remote_dir: str = "/home/user/reference_files") -> str:
        """
        Upload a reference file to the sandbox
        
        Args:
            local_path: Path to local file
            remote_dir: Directory in sandbox to upload to
            
        Returns:
            Remote path in sandbox
        """
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Reference file not found: {local_path}")
        
        # Check if already uploaded
        if local_path in self.uploaded_reference_files:
            print(f"♻️ Reference file already uploaded: {os.path.basename(local_path)}")
            return self.uploaded_reference_files[local_path]
        
        sandbox = self.get_or_create_sandbox()
        
        # Ensure remote directory exists by creating parent directories
        # E2B will create the directory structure if it doesn't exist
        print(f"📁 Ensuring directory exists: {remote_dir}")
        
        # Read file content
        with open(local_path, 'rb') as f:
            content = f.read()
        
        # Create remote path
        filename = os.path.basename(local_path)
        remote_path = f"{remote_dir}/{filename}"
        
        # Upload file - E2B will create parent directories automatically
        try:
            sandbox.files.write(remote_path, content)
            self.uploaded_reference_files[local_path] = remote_path
            print(f"✅ Uploaded reference file: {filename} -> {remote_path}")
            
            if self.use_local_fallback:
                 print(f"   📍 Local Sandbox path: {sandbox.files._resolve_path(remote_path)}")
            else:
                print(f"   📍 E2B Sandbox path: {remote_path}")
            
            print(f"   📦 File size: {len(content)} bytes")
            return remote_path
        except Exception as e:
            error_msg = f"Failed to upload file {local_path} to {remote_path}: {str(e)}"
            print(f"❌ {error_msg}")
            raise RuntimeError(error_msg)
    
    def download_artifact(self, remote_path: str, local_dir: str) -> str:
        """
        Download an artifact file from the sandbox to local storage
        
        Args:
            remote_path: Path in sandbox
            local_dir: Local directory to save to
            
        Returns:
            Local path of downloaded file
        """
        if not self.sandbox:
            raise RuntimeError("No active sandbox")
        
        try:
            # Read file content as bytes to prevent corruption of binary files (PNG, DOCX, XLSX, etc.)
            # E2B SDK: format="bytes" returns bytearray, format="text" returns str
            content_bytes = self.sandbox.files.read(remote_path, format="bytes")
            
            # Create local path
            os.makedirs(local_dir, exist_ok=True)
            filename = os.path.basename(remote_path)
            local_path = os.path.join(local_dir, filename)
            
            # Write content as binary
            with open(local_path, 'wb') as f:
                f.write(content_bytes)
            
            print(f"📥 Downloaded artifact: {remote_path} -> {local_path}")
            return local_path
        except Exception as e:
            raise RuntimeError(f"Failed to download {remote_path}: {str(e)}")
    
    def cleanup(self):
        """Kill the sandbox and clean up resources"""
        if self.sandbox:
            try:
                self.sandbox.kill()  # Use kill() for immediate termination
                print(f"🧹 Killed sandbox: {self.sandbox_id}")
            except:
                pass
            self.sandbox = None
            self.sandbox_id = None
            self.uploaded_reference_files = {}


@tool
def execute_code(code: str, language: str = "python") -> Dict[str, Any]:
    """
    Execute code in a persistent cloud sandbox (E2B) or local fallback with artifact download support.

    FEATURES:
    - Code runs in an isolated E2B Sandbox VM (separate from LiveBench host) OR locally if E2B is not configured.
    - Uses persistent sandbox per session (files persist across calls)
    - Currently restricted to Python code via E2B Python template
    - No direct access to LiveBench host filesystem (unless using local fallback, then standard os permissions apply)
    - API key based access control via E2B (requires E2B_API_KEY)
    - Automatically downloads files marked with ARTIFACT_PATH: prefix

    ARTIFACT DOWNLOAD:
    - To make files accessible to submit_work, include in your code:
      print("ARTIFACT_PATH:/path/to/file.ext")
    - Files will be automatically downloaded to the agent's sandbox directory
    - The result will include a 'downloaded_artifacts' list with the local paths
    - ALWAYS use the paths from 'downloaded_artifacts' for submit_work, NOT the /tmp/ paths
    - Example:
      result = execute_code('print("ARTIFACT_PATH:/tmp/report.pdf")')
      # Use result['downloaded_artifacts'] for submit_work!

    Args:
        code: Code to execute
        language: Programming language - currently only "python" supported

    Returns:
        Dictionary with execution result (stdout, stderr, exit_code, downloaded_artifacts)
    """
    # Validate inputs
    if not code or len(code) < 1:
        return {"error": "Code cannot be empty"}

    language = language.lower().strip()
    if language != "python":
        return {
            "error": f"Language '{language}' not supported",
            "supported_languages": ["python"]
        }

    # Get global state for sandbox directory
    global_state = {}
    try:
        global_state = _get_global_state()
    except Exception:
        pass

    # Get or create persistent session sandbox
    session_sandbox = SessionSandbox.get_instance()
    
    try:
        sandbox = session_sandbox.get_or_create_sandbox(timeout=3600)  # 1 hour to match max task duration
        
        # Execute code
        try:
            execution = sandbox.run_code(code)
        except Exception as e:
            return {
                "success": False,
                "error": f"Sandbox execution failed: {str(e)}"
            }

        logs = getattr(execution, "logs", "")
        error = getattr(execution, "error", None)
        success = error is None
        
        # Extract stdout properly for artifact path detection
        if hasattr(logs, 'stdout'):
            stdout_str = '\n'.join(logs.stdout) if isinstance(logs.stdout, list) else str(logs.stdout)
        else:
            stdout_str = str(logs)
        
        # Parse ARTIFACT_PATH markers and download files
        downloaded_artifacts = []
        if success and "ARTIFACT_PATH:" in stdout_str:
            artifact_paths = re.findall(r'ARTIFACT_PATH:(\S+)', stdout_str)
            
            if artifact_paths and global_state.get("data_path"):
                # Determine local download directory
                current_date = global_state.get("current_date", "unknown")
                sandbox_dir = os.path.join(
                    global_state["data_path"], 
                    "sandbox", 
                    current_date
                )
                os.makedirs(sandbox_dir, exist_ok=True)
                
                # Download each artifact
                for remote_path in artifact_paths:
                    try:
                        local_path = session_sandbox.download_artifact(remote_path, sandbox_dir)
                        downloaded_artifacts.append(local_path)
                    except Exception as e:
                        print(f"⚠️ Warning: Could not download {remote_path}: {e}")
        
        result = {
            "success": success,
            "exit_code": 0 if success else 1,
            "stdout": logs if success else "",
            "stderr": str(error) if error else "",
            "sandbox_id": session_sandbox.sandbox_id,
            "message": f"✅ Code executed in {session_sandbox.sandbox_id}" if success else f"❌ {session_sandbox.sandbox_id} execution reported an error",
        }
        
        # Add reference files info if available
        if session_sandbox.uploaded_reference_files:
            result["message"] += f"\n\n📎 REFERENCE FILES AVAILABLE in sandbox at /home/user/reference_files/:"
            for local_path, remote_path in session_sandbox.uploaded_reference_files.items():
                filename = os.path.basename(remote_path)
                result["message"] += f"\n  • {filename} at {remote_path}"
        
        # Add downloaded artifacts info
        if downloaded_artifacts:
            result["downloaded_artifacts"] = downloaded_artifacts
            result["message"] += f"\n\n📥 DOWNLOADED {len(downloaded_artifacts)} ARTIFACT(S) - Use these paths for submit_work:"
            for path in downloaded_artifacts:
                result["message"] += f"\n  ✅ {path}"
            result["message"] += f"\n\n⚠️ IMPORTANT: Use the paths above (not /tmp/ paths) when calling submit_work!"
        
        return result
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Unexpected error during sandbox execution: {str(e)}"
        }


def upload_task_reference_files(reference_file_paths: List[str]) -> List[str]:
    """
    Upload reference files to the persistent sandbox.
    This should be called when a task is assigned to make reference files available.
    
    Args:
        reference_file_paths: List of local file paths to upload
        
    Returns:
        List of remote paths in the sandbox
    """
    if not reference_file_paths:
        return []
    
    print(f"\n📤 Uploading {len(reference_file_paths)} reference file(s) to sandbox...")
    
    session_sandbox = SessionSandbox.get_instance()
    
    # Ensure sandbox is created before uploading
    sandbox = session_sandbox.get_or_create_sandbox()
    print(f"✅ Sandbox ready (ID: {session_sandbox.sandbox_id})")
    
    remote_paths = []
    
    for i, local_path in enumerate(reference_file_paths, 1):
        try:
            print(f"\n[{i}/{len(reference_file_paths)}] Uploading: {os.path.basename(local_path)}")
            remote_path = session_sandbox.upload_reference_file(local_path)
            remote_paths.append(remote_path)
        except Exception as e:
            print(f"❌ Failed to upload {local_path}: {e}")
    
    if remote_paths:
        print(f"\n✅ Successfully uploaded {len(remote_paths)}/{len(reference_file_paths)} files to sandbox")
        print(f"📍 All files are accessible at: /home/user/reference_files/")
        print(f"   Files uploaded:")
        for path in remote_paths:
            print(f"     • {path}")
    else:
        print(f"\n⚠️ No files were successfully uploaded")
    
    return remote_paths


def cleanup_session_sandbox():
    """
    Clean up the session sandbox.
    Should be called at the end of each agent session/day.
    """
    SessionSandbox.reset()


if __name__ == "__main__":
    """
    Test the persistent sandbox functionality
    """
    def test1():
        # Test basic code execution
        test_code = """
print("Hello from sandbox!")
for i in range(3):
    print("Number:", i)
        """

        result = execute_code.func(test_code, language="python")

        print("=== Sandbox Execution Result ===")
        for k, v in result.items():
            print(f"{k}: {v}")
            
    def test2():
        # Test file persistence across calls
        test_code1 = """
with open("/tmp/test.txt", "w") as f:
    f.write("Hello from first call!")
print("ARTIFACT_PATH:/tmp/test.txt")
        """
        
        result1 = execute_code.func(test_code1, language="python")
        print("=== First Call Result ===")
        print(result1.get("message"))
        
        # Second call should be able to read the file
        test_code2 = """
with open("/tmp/test.txt", "r") as f:
    content = f.read()
print(f"File content: {content}")
        """
        
        result2 = execute_code.func(test_code2, language="python")
        print("\n=== Second Call Result ===")
        print(result2.get("stdout"))

    print("Running test 1: Basic execution")
    test1()
    
    print("\n" + "="*50)
    print("Running test 2: File persistence")
    test2()
    
    # Cleanup
    cleanup_session_sandbox()