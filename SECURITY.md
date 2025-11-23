# Security Policy

## Critical Issues to Fix Before Open-Sourcing

### 1. Hardcoded API Keys (CRITICAL)

The following files contain hardcoded API keys that **MUST** be removed before making this repository public:

#### `shapetofunction.py`
- **Line 51**: Remove `VALID_API_KEYS` hardcoded values
- **Line 554**: Remove hardcoded `DEFAULT_MAPBOX_TOKEN`
- **Line 555**: Remove hardcoded `DEFAULT_API_KEY_OPENTOPO`
- **Line 552**: Remove hardcoded personal address

#### `apicalldisplay.py`
- **Line 28**: Remove hardcoded `api_key_opentopo`
- **Line 29**: Remove hardcoded `api_key_mapbox`
- **Line 421**: Remove hardcoded default address

#### `Docker/servercode.py`
- **Line 317**: Remove hardcoded `api_key_opentopo`
- **Line 318**: Remove hardcoded `style`
- **Line 319**: Remove hardcoded `mapbox_token`

#### `TileVector.py`
- **Line 33**: Remove hardcoded `access_token`

### 2. Recommended Changes

#### Use Environment Variables
Replace all hardcoded credentials with environment variable loading:

```python
import os
from dotenv import load_dotenv

load_dotenv()

MAPBOX_ACCESS_TOKEN = os.getenv('MAPBOX_ACCESS_TOKEN')
OPENTOPO_API_KEY = os.getenv('OPENTOPO_API_KEY')
VALID_API_KEYS = set(os.getenv('VALID_API_KEYS', '').split(','))
```

#### Disable Debug Mode in Production
In `apicalldisplay.py` line 435:
```python
# Change from:
app.run(debug=True)

# To:
app.run(debug=os.getenv('FLASK_DEBUG', 'False') == 'True')
```

#### Add Input Validation
Add validation for address inputs to prevent injection attacks:
```python
import re

def validate_address(address):
    # Basic validation - adjust as needed
    if not address or len(address) > 200:
        raise ValueError("Invalid address")
    # Remove potentially dangerous characters
    return re.sub(r'[^\w\s,.-]', '', address)
```

### 3. Files to Never Commit

Ensure these are in `.gitignore`:
- `.env` files
- `*.pem` files (SSL certificates)
- `area3.json` (may contain location data)
- `voxelgridtest.json`
- Any files with personal addresses or location data

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it by:

1. **DO NOT** open a public issue
2. Email the maintainers directly at [your-email@example.com]
3. Provide details about the vulnerability and steps to reproduce
4. Allow reasonable time for a fix before public disclosure

## Security Best Practices

When using this project:

1. **Never commit API keys** to version control
2. **Use environment variables** for all sensitive configuration
3. **Rotate API keys** regularly
4. **Monitor API usage** for unusual activity
5. **Use HTTPS** for all API communications
6. **Implement rate limiting** on public endpoints
7. **Validate all user inputs** before processing

## Pre-Release Checklist

Before making this repository public, complete these steps:

- [ ] Remove all hardcoded API keys from all files
- [ ] Remove all hardcoded personal addresses
- [ ] Implement environment variable configuration
- [ ] Update all Python files to use environment variables
- [ ] Test with `.env` file configuration
- [ ] Verify `.gitignore` includes all sensitive files
- [ ] Review all JSON files for personal/sensitive data
- [ ] Set Flask debug mode to False by default
- [ ] Add input validation to all user-facing endpoints
- [ ] Review and test Docker configuration
- [ ] Add rate limiting to API endpoints
- [ ] Document all required environment variables
- [ ] Add security headers to Flask app
- [ ] Review all commit history for accidentally committed secrets
- [ ] Consider using git-secrets or similar tools

## Recommended Tools

- `git-secrets`: Prevents committing secrets
- `python-dotenv`: Environment variable management
- `bandit`: Python security linter
- `safety`: Checks dependencies for known vulnerabilities

Run security checks:
```bash
# Install security tools
pip install bandit safety

# Check for security issues
bandit -r .

# Check dependencies
safety check
```
