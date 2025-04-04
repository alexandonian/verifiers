import tracemalloc
import ast
import contextlib
import io
import resource
import signal
import time
import traceback
from typing import Dict, Any, Optional, List


def python(code: str, timeout: int = 30) -> str:
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


class SecurityChecker:
    """Class to check code for security issues"""

    def __init__(self, allowed_imports=None):
        self.allowed_imports = {
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
            "scipy",
            "sympy",
        }

        if allowed_imports:
            self.allowed_imports.update(allowed_imports)

        self.dangerous_builtins = {
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

        self.dangerous_attributes = {
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
        self.forbidden_builtins = {
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
        }

    def check_code(self, code: str) -> List[str]:
        """Check code for security issues and return a list of warnings"""
        try:
            parsed_code = ast.parse(code)
            visitor = self._create_node_visitor()
            visitor.visit(parsed_code)
            return visitor.issues
        except SyntaxError as e:
            return [f"Syntax error: {str(e)}"]

    def _create_node_visitor(self):
        """Create and return a NodeVisitor configured with our security settings"""
        return RestrictedNodeVisitor(
            self.allowed_imports, self.dangerous_builtins, self.dangerous_attributes
        )


class RestrictedNodeVisitor(ast.NodeVisitor):
    """AST visitor to detect potentially harmful operations"""

    def __init__(self, allowed_imports, dangerous_builtins, dangerous_attributes):
        self.issues: List[str] = []
        self.allowed_imports = allowed_imports
        self.dangerous_builtins = dangerous_builtins
        self.dangerous_attributes = dangerous_attributes

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
        self._check_dangerous_call(node)
        self.generic_visit(node)

    def _check_dangerous_call(self, node):
        # Check for dangerous built-in functions
        if isinstance(node.func, ast.Name) and node.func.id in self.dangerous_builtins:
            self.issues.append(f"Potentially unsafe function call: {node.func.id}()")

        # Check for attribute access that could be risky
        elif isinstance(node.func, ast.Attribute):
            self._check_attribute_call(node.func)

    def _check_attribute_call(self, func):
        if isinstance(func.value, ast.Name):
            module = func.value.id
            if (
                module in self.dangerous_attributes
                and func.attr in self.dangerous_attributes[module]
            ):
                self.issues.append(
                    f"Potentially unsafe operation: {module}.{func.attr}()"
                )

        # Check for any file operations
        if func.attr in self.dangerous_attributes[""]:
            self.issues.append(f"Potentially unsafe file operation: {func.attr}()")

    def visit_Attribute(self, node):
        # Check for access to dunder methods
        if (
            isinstance(node.attr, str)
            and node.attr.startswith("__")
            and node.attr.endswith("__")
        ):
            self.issues.append(f"Access to dunder method: {node.attr}")
        self.generic_visit(node)


class ResourceLimiter:
    """Class to manage resource limits for code execution"""

    def __init__(self, time_limit=5, memory_limit=100 * 1024 * 1024):
        self.time_limit = time_limit
        self.memory_limit = memory_limit
        self.old_handler = None

    def setup(self):
        """Set up resource limits and timeout handler"""
        self._setup_memory_limits()
        return self._setup_timeout_handler()

    def _setup_memory_limits(self):
        """Set up memory limits if possible"""
        try:
            resource.setrlimit(
                resource.RLIMIT_AS, (self.memory_limit, self.memory_limit)
            )
        except (ValueError, resource.error, AttributeError) as e:
            print(f"Warning: Could not set memory limit: {e}")

    def _setup_timeout_handler(self):
        """Set up timeout handler and return the old handler"""
        try:
            self.old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(self.time_limit)
            return self.old_handler
        except (ValueError, AttributeError):
            return None

    def cleanup(self):
        """Clean up resources and restore handlers"""
        if self.old_handler is not None:
            signal.signal(signal.SIGALRM, self.old_handler)
            signal.alarm(0)


class ExecutionEnvironment:
    """Class to manage the execution environment for code"""

    def __init__(self, allowed_imports=None):
        self.allowed_imports = allowed_imports
        self.security_checker = SecurityChecker(allowed_imports)

    def create_namespace(self):
        """Create a safe execution namespace"""

        # Create a secure import function
        def secure_importer(name, globals=None, locals=None, fromlist=(), level=0):
            if name not in self.security_checker.allowed_imports:
                raise ImportError(f"Import of '{name}' is not allowed")
            return __import__(name, globals, locals, fromlist, level)

        # Create a restricted builtins dictionary
        restricted_builtins = self._create_restricted_builtins()

        # Add back __import__ but use our secure version
        restricted_builtins["__import__"] = secure_importer

        # Set up execution namespace
        namespace = {"__builtins__": restricted_builtins}

        # Pre-import allowed modules to the namespace
        self._preimport_modules(namespace)

        return namespace

    def _create_restricted_builtins(self):
        """Create a restricted builtins dictionary"""
        restricted_builtins = {}

        for name in __builtins__:
            if name not in self.security_checker.forbidden_builtins:
                if isinstance(__builtins__, dict):
                    if name in __builtins__:
                        restricted_builtins[name] = __builtins__[name]
                else:
                    restricted_builtins[name] = getattr(__builtins__, name)

        return restricted_builtins

    def _preimport_modules(self, namespace):
        """Pre-import allowed modules to the namespace"""
        for module_name in self.security_checker.allowed_imports:
            with contextlib.suppress(ImportError):
                namespace[module_name] = __import__(module_name)


def collect_execution_results(
    result: Dict[str, Any],
    stdout_capture: io.StringIO,
    stderr_capture: io.StringIO,
    start_time: float,
) -> Dict[str, Any]:
    """Collect execution results and update the result dictionary"""
    # Capture outputs
    result["output"] = stdout_capture.getvalue()
    if stderr_capture.getvalue() and not result["error"]:
        result["error"] = stderr_capture.getvalue()

    # Measure execution time
    result["execution_time"] = time.time() - start_time

    # Get peak memory usage (if available on this platform)
    try:
        result["alt_peak_memory"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (AttributeError, resource.error):
        result["alt_peak_memory"] = -1

    return result


def measure_memory_usage(func, *args, **kwargs):
    tracemalloc.start()
    start_snapshot = tracemalloc.take_snapshot()

    result = func(*args, **kwargs)

    current_snapshot = tracemalloc.take_snapshot()
    tracemalloc.stop()

    stats = current_snapshot.compare_to(start_snapshot, "lineno")
    total_diff = sum(stat.size_diff for stat in stats)
    peak_size = (
        current_snapshot.statistics("filename")[0].size
        if current_snapshot.statistics("filename")
        else 0
    )

    return result, total_diff, peak_size


def secure_execute_python(
    code: str,
    time_limit: int = 5,  # seconds
    memory_limit: int = 100 * 1024 * 1024,  # 100MB
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
    # Initialize result structure
    result = {
        "status": "error",
        "output": "",
        "error": "",
        "execution_time": 0,
        "peak_memory": 0,
        "security_warnings": [],
    }
    start_time = time.time()

    # Check code for security issues
    security_checker = SecurityChecker(allowed_imports)
    security_warnings = security_checker.check_code(code)

    if security_warnings:
        result["security_warnings"] = security_warnings
        result["error"] = "Potentially unsafe code detected"
        result["output"] = "".join(security_warnings)
        return result

    # Set up execution environment
    stdout_capture, stderr_capture = io.StringIO(), io.StringIO()
    resource_limiter = ResourceLimiter(time_limit, memory_limit)
    execution_env = ExecutionEnvironment(security_checker.allowed_imports)

    try:
        # Set up resource limits and redirects
        _ = resource_limiter.setup()
        namespace = execution_env.create_namespace()

        with (
            contextlib.redirect_stdout(stdout_capture),
            contextlib.redirect_stderr(stderr_capture),
        ):
            # TODO: Convert to context manager
            # Use tracemalloc to measure memory usage
            tracemalloc.start()
            baseline = tracemalloc.take_snapshot()

            exec(code, namespace)

            # Take final snapshot
            final_snapshot = tracemalloc.take_snapshot()
            tracemalloc.stop()

            # Calculate memory usage
            stats = final_snapshot.compare_to(baseline, "lineno")
            memory_used = sum(stat.size_diff for stat in stats)
            peak_memory = (
                max(stat.size for stat in final_snapshot.statistics("filename"))
                if final_snapshot.statistics("filename")
                else 0
            )

            result["memory_used"] = memory_used
            result["peak_memory"] = peak_memory
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
        # Clean up resources
        resource_limiter.cleanup()

        # Collect execution results
        result["output"] = stdout_capture.getvalue()
        if stderr_capture.getvalue() and not result["error"]:
            result["error"] = stderr_capture.getvalue()

        # Measure execution time
        result["execution_time"] = time.time() - start_time

    return result
