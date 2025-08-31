# Project Utilities

Shared utilities for the LLM Computational Graph project.

## Logging Configuration

Centralized logging configuration for consistent logging across all project modules.

### Usage

#### In main scripts (with CLI arguments):

```python
from scripts.utils.logging_config import setup_logging

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    # Setup logging for the entire application
    logger = setup_logging(verbose=args.verbose, module_name="my_script")

    logger.info("Application started")
    # ... rest of your code
```

#### In library modules:

```python
from scripts.utils.logging_config import get_logger

class MyClass:
    def __init__(self):
        self.logger = get_logger(__name__)

    def do_something(self):
        self.logger.info("Doing something...")
        try:
            # External API call
            result = external_api_call()
        except Exception as e:
            self.logger.error(f"API call failed: {e}")
            return None

        self.logger.debug(f"API returned: {result}")
        return result
```

### Features

- **Consistent formatting**: All logs use the same timestamp and format
- **Multiple handlers**: Console output + file logging (`logs/project.log`)
- **Level control**: `--verbose` flag enables DEBUG level
- **Centralized**: One configuration for the entire project
- **Module-specific**: Each module gets its own named logger

### Log Levels

- `DEBUG`: Detailed information for debugging (only with --verbose)
- `INFO`: General information about program execution
- `WARNING`: Something unexpected happened but the program continues
- `ERROR`: A serious problem occurred

### File Output

All logs are written to `logs/project.log` with DEBUG level, regardless of console level.
