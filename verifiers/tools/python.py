import ast
import contextlib
import io
import resource
import signal
import time
import traceback
from typing import Dict, Any, Optional, List, Set


def python_raw(code: str, timeout: int = 30) -> str:
    """Evaluates a block of Python code and returns output of print() statements. Allowed libraries: astropy, biopython, networkx, numpy, scipy, sympy.

    Args:
        code: A block of Python code

    Returns:
        The output of the code (truncated to 1000 chars) or an error message

    Examples:
        {"code": "import numpy as np; print(np.array([1, 2, 3]) + np.array([4, 5, 6]))"} -> "[5 7 9]"
        {"code": "import scipy; print(scipy.linalg.inv(np.array([[1, 2], [3, 4]])))"} -> "[[-2.   1. ] [ 1.5 -0.5]]"
        {"code": "import sympy; x, y = sympy.symbols('x y'); print(sympy.integrate(x**2, x))"} -> "x**3/3"
    """

    import subprocess

    try:
        # Run the code block in subprocess with 10-second timeout
        result = subprocess.run(
            ["python", "-c", code],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            text=True,
        )
        if result.stderr:
            return f"Error: {result.stderr.strip()}"
        output = result.stdout.strip() if result.stdout else ""
        if len(output) > 1000:
            output = f"{output[:1000]}... (truncated to 1000 chars)"
        return output
    except subprocess.TimeoutExpired:
        return f"Error: Code execution timed out after {timeout} seconds"


class CodeExecutionError(Exception):
    """Custom exception for code execution errors"""

    pass


class TimeoutError(Exception):
    """Custom exception for timeout errors"""

    pass


class MemoryLimitExceededError(Exception):
    """Custom exception for memory limit errors"""

    pass


def timeout_handler(signum, frame):
    """Signal handler for timeouts"""
    raise TimeoutError("Code execution timed out")


class RestrictedNodeVisitor(ast.NodeVisitor):
    """AST visitor to detect potentially harmful operations"""

    def __init__(self):
        self.issues: List[str] = []
        self.allowed_imports: Set[str] = {
            # Whitelist of safe standard libraries
            "math",
            "random",
            "datetime",
            "collections",
            "itertools",
            "functools",
            "statistics",
            "json",
            "csv",
            "re",
            "string",
            "typing",
            "enum",
            "dataclasses",
            "abc",
            "array",
            "bisect",
            "heapq",
            "decimal",
            "fractions",
            "time",
            "numpy",
            "pandas",
            "matplotlib",
            "scipy",
        }

    def visit_Import(self, node):
        for name in node.names:
            if name.name not in self.allowed_imports:
                self.issues.append(f"Potentially unsafe import: {name.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module not in self.allowed_imports:
            self.issues.append(f"Potentially unsafe import: {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node):
        # Check for dangerous built-in functions
        if isinstance(node.func, ast.Name):
            dangerous_builtins = {
                "eval",
                "exec",
                "compile",
                "open",
                "globals",
                "locals",
                "vars",
                "getattr",
                "setattr",
                "delattr",
                "dir",
                "input",
            }
            if node.func.id in dangerous_builtins:
                self.issues.append(
                    f"Potentially unsafe function call: {node.func.id}()"
                )

        # Check for attribute access that could be risky
        elif isinstance(node.func, ast.Attribute):
            dangerous_attributes = {
                "os": {
                    "system",
                    "popen",
                    "spawn",
                    "exec",
                    "unlink",
                    "remove",
                    "rmdir",
                    "listdir",
                },
                "sys": {"exit", "_exit", "modules"},
                "subprocess": {"run", "call", "check_call", "check_output", "Popen"},
                "shutil": {"rmtree"},
                "": {"read", "write", "delete"},  # Generic dangerous file operations
            }

            if isinstance(node.func.value, ast.Name):
                module = node.func.value.id
                if (
                    module in dangerous_attributes
                    and node.func.attr in dangerous_attributes[module]
                ):
                    self.issues.append(
                        f"Potentially unsafe operation: {module}.{node.func.attr}()"
                    )

            # Check for any file operations
            if node.func.attr in dangerous_attributes[""]:
                self.issues.append(
                    f"Potentially unsafe file operation: {node.func.attr}()"
                )

        self.generic_visit(node)

    def visit_Attribute(self, node):
        # Check for access to dunder methods
        if (
            isinstance(node.attr, str)
            and node.attr.startswith("__")
            and node.attr.endswith("__")
        ):
            self.issues.append(f"Access to dunder method: {node.attr}")
        self.generic_visit(node)


def secure_execute_python(
    code: str,
    time_limit: int = 30,  # seconds
    memory_limit: int = 500 * 1024 * 1024,  # 500MB
    allowed_imports: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Safely execute Python code with restrictions.

    Args:
        code: Python code to execute
        time_limit: Maximum execution time in seconds
        memory_limit: Maximum memory usage in bytes
        allowed_imports: Optional list of additional allowed imports

    Returns:
        Dictionary containing execution results and metadata
    """
    start_time = time.time()
    result = {
        "status": "error",
        "output": "",
        "error": "",
        "execution_time": 0,
        "peak_memory": 0,
        "security_warnings": [],
    }

    # Sanitize and validate code
    try:
        # Parse the code to check for potentially harmful operations
        parsed_code = ast.parse(code)

        # Perform security analysis
        visitor = RestrictedNodeVisitor()
        if allowed_imports:
            visitor.allowed_imports.update(allowed_imports)
        visitor.visit(parsed_code)

        if visitor.issues:
            result["security_warnings"] = visitor.issues
            result["error"] = "Potentially unsafe code detected"
            return result

    except SyntaxError as e:
        result["error"] = f"Syntax error: {str(e)}"
        return result

    # Set up redirects for stdout and stderr
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    # Set resource limits (for Unix-like systems)
    def limit_resources():
        try:
            # Set CPU time limit
            resource.setrlimit(resource.RLIMIT_CPU, (time_limit, time_limit + 1))
            # Set memory limit
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        except (ValueError, resource.error) as e:
            print(f"Warning: Could not set resource limits: {e}")

    # Set up timeout handler
    try:
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(time_limit)
    except (ValueError, AttributeError):
        # SIGALRM might not be available on all platforms (e.g., Windows)
        old_handler = None

    try:
        # Create a secure import function
        def secure_importer(name, globals=None, locals=None, fromlist=(), level=0):
            if name not in visitor.allowed_imports:
                raise ImportError(f"Import of '{name}' is not allowed")
            return __import__(name, globals, locals, fromlist, level)

        # Create a restricted builtins dictionary
        restricted_builtins = {}
        for name in __builtins__:
            if name not in {
                "eval",
                "exec",
                "compile",
                "open",
                "globals",
                "locals",
                "vars",
                "input",
                "breakpoint",
                "exit",
                "quit",
                "help",
                "copyright",
                "credits",
                "license",
            }:
                if isinstance(__builtins__, dict):
                    if name in __builtins__:
                        restricted_builtins[name] = __builtins__[name]
                else:
                    restricted_builtins[name] = getattr(__builtins__, name)

        # Add back __import__ but use our secure version
        restricted_builtins["__import__"] = secure_importer

        # Set up execution namespace
        execution_namespace = {"__builtins__": restricted_builtins}

        # Pre-import allowed modules to the namespace
        for module_name in visitor.allowed_imports:
            if allowed_imports is None or module_name in allowed_imports:
                try:
                    execution_namespace[module_name] = __import__(module_name)
                except ImportError:
                    pass

        with (
            contextlib.redirect_stdout(stdout_capture),
            contextlib.redirect_stderr(stderr_capture),
        ):
            # Apply resource limits if possible
            limit_resources()

            # Execute the code with a shared namespace
            # This allows recursive functions to find themselves
            exec(code, execution_namespace)

        result["status"] = "success"

    except TimeoutError:
        result["error"] = f"Code execution timed out after {time_limit} seconds"
    except MemoryError:
        result["error"] = (
            f"Code exceeded memory limit of {memory_limit / (1024 * 1024):.1f} MB"
        )
    except ImportError as e:
        result["error"] = f"Import error: {str(e)}"
    except Exception as e:
        result["error"] = f"Runtime error: {str(e)}\n{traceback.format_exc()}"
    finally:
        # Restore signal handler and cancel alarm
        if old_handler is not None:
            signal.signal(signal.SIGALRM, old_handler)
            signal.alarm(0)

        # Capture outputs
        result["output"] = stdout_capture.getvalue()
        if stderr_capture.getvalue() and not result["error"]:
            result["error"] = stderr_capture.getvalue()

        # Measure execution time
        result["execution_time"] = time.time() - start_time

        # Get peak memory usage (if available on this platform)
        try:
            result["peak_memory"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        except (AttributeError, resource.error):
            result["peak_memory"] = -1

    return result
