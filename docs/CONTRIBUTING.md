# Contributing Guide

Thank you for your interest in contributing to the Sentinel Environment! This guide will help you set up your development environment and understand our contribution workflow.

## Table of Contents

- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Code Standards](#code-standards)
- [Testing](#testing)
- [Workflow](#workflow)
- [Submitting Changes](#submitting-changes)
- [Documentation](#documentation)

---

## Development Setup

### Prerequisites

- **Python:** 3.11 or higher
- **Package Manager:** `uv` (recommended) or `pip`
- **Git:** For version control
- **Optional:** CUDA toolkit (for GPU training)

### Quick Start

```bash
# Clone repository
git clone <repository-url>
cd "E:\OpenENV RL Challenge"

# Create virtual environment with uv
uv venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
uv pip install -e ".[dev,observability,ml]"

# Install pre-commit hooks
pre-commit install
```

### Install All Dependencies

```bash
# Full installation (all features)
uv pip install -e ".[dev,observability,ml,llm-security,llm-providers,translation,graph]"

# Minimal development setup
uv pip install -e ".[dev]"
```

### Verify Installation

```bash
# Run tests
pytest tests/ -v

# Type checking
mypy server/ client.py models.py

# Linting
ruff check server/ client.py models.py

# Security scan
bandit -r server/ -c pyproject.toml
```

---

## Project Structure

```
E:\OpenENV RL Challenge\
├── server/                      # FastAPI server
│   ├── app.py                   # Main application entry point
│   ├── sentinel_environment.py  # Core RL environment logic
│   ├── grader.py                # Step/episode grading
│   ├── reward_shaper.py         # Reward computation
│   ├── episode_manager.py       # Episode lifecycle management
│   ├── attack_provider.py       # Attack sequence generation
│   ├── resilience_profile.py    # Per-attack-type diagnostics
│   ├── batch_api.py             # v1 batch endpoints
│   ├── middleware.py            # Production middleware
│   ├── rate_limiter.py          # Per-IP rate limiting
│   ├── dependencies.py          # Shared singletons
│   ├── text_embedder.py         # Sentence-transformers embeddings
│   ├── hyperion_policy_network.py  # SoftMoE policy network
│   ├── mcts_reasoning.py        # MCTS reasoning engine
│   └── sentinel_gym_env.py      # Gymnasium wrapper
├── client.py                    # Python HTTP client
├── models.py                    # Pydantic data models
├── inference.py                 # LLM-powered agent evaluation
├── inference_logging.py         # Structured logging helpers
├── train_hyperion.py            # HyperionRL trainer (2590 lines)
├── visualize_dashboard.py       # Real-time training dashboard
├── test_hyperion_e2e.py         # End-to-end test suite
├── validate_submission.py       # Hackathon validation script
├── deploy-hf.py                 # Hugging Face Space deployment
├── fix-hf-space.py              # HF Space utility
├── tests/                       # Test suite (312 tests)
├── docs/                        # Documentation
├── pyproject.toml               # Project configuration
├── Dockerfile                   # Container image definition
├── openenv.yaml                 # OpenEnv SDK configuration
└── .pre-commit-config.yaml      # Pre-commit hooks
```

---

## Code Standards

### Python Style

We use **Ruff** for linting and formatting:

```bash
# Check for linting issues
ruff check server/ client.py models.py

# Auto-fix fixable issues
ruff check --fix server/

# Format code
ruff format server/ client.py models.py
```

**Key Style Rules:**
- Line length: 120 characters
- Double quotes for strings
- 4-space indentation
- Follow PEP 8 naming conventions
- Type hints required for all public APIs

### Type Checking

We use **MyPy** for static type checking:

```bash
# Run type checker
mypy server/ client.py models.py

# Strict mode (new code should aim for this)
mypy --strict server/app.py
```

**Type Hint Requirements:**
- All function signatures must have type hints
- Use `typing.Any` sparingly (prefer specific types)
- Return types are mandatory
- Use Pydantic models for data structures

### Security Scanning

We use **Bandit** for security analysis:

```bash
# Run security scan
bandit -r server/ -c pyproject.toml
```

**Common Skipped Rules:**
- `B101`: Assert statements (used in tests)
- `B104`: Binding to 0.0.0.0 (required for containers)
- `B110`: Try/except/pass (graceful degradation)
- `B311`: Random module (non-crypto usage)

### Pre-Commit Hooks

Pre-commit hooks run automatically on `git commit`:

```yaml
# .pre-commit-config.yaml
repos:
  - repo: local
    hooks:
      - id: ruff
        name: ruff
        entry: ruff check
        language: system
        types: [python]
      
      - id: ruff-format
        name: ruff-format
        entry: ruff format
        language: system
        types: [python]
      
      - id: mypy
        name: mypy
        entry: mypy
        language: system
        types: [python]
      
      - id: pytest
        name: pytest
        entry: pytest tests/
        language: system
        types: [python]
        pass_filenames: false
```

**Install hooks:**
```bash
pre-commit install
```

**Run manually:**
```bash
pre-commit run --all-files
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=server --cov=client --cov=models --cov-report=html

# Run specific test file
pytest tests/test_grader.py -v

# Run specific test
pytest tests/test_grader.py::test_grade_step_correct -v

# Run async tests
pytest tests/ -v --asyncio-mode=strict
```

### Writing Tests

**Test Structure:**
```python
import pytest
from models import SentinelAction, ThreatCategory, RecommendedAction

class TestGrader:
    """Tests for grader.py"""
    
    def test_grade_step_correct_classification(self):
        """Test correct detection returns full reward."""
        # Arrange
        action = SentinelAction(
            classification=ThreatCategory.INJECTION,
            reasoning="Clear injection pattern detected",
            recommended_action=RecommendedAction.BLOCK
        )
        ground_truth = ThreatCategory.INJECTION
        
        # Act
        result = grade_step(action, ground_truth)
        
        # Assert
        assert result.correct is True
        assert result.reward == pytest.approx(1.0)
    
    @pytest.mark.asyncio
    async def test_async_endpoint(self):
        """Test async API endpoint."""
        # Arrange
        async with AsyncClient(app=app, base_url="http://test") as ac:
            # Act
            response = await ac.post("/reset", params={"task_name": "basic-injection"})
            
            # Assert
            assert response.status_code == 200
            assert "episode_id" in response.json()
```

**Test Categories:**
- **Unit Tests:** Individual functions/classes
- **Integration Tests:** Multi-component interactions
- **Property-Based Tests:** Hypothesis-generated inputs
- **Load Tests:** Performance under stress
- **E2E Tests:** Full pipeline validation

### Test Coverage Requirements

| Component | Minimum Coverage |
|-----------|-----------------|
| `server/grader.py` | 95% |
| `server/reward_shaper.py` | 90% |
| `server/episode_manager.py` | 85% |
| `server/rate_limiter.py` | 90% |
| `server/middleware.py` | 80% |
| `client.py` | 85% |
| `models.py` | 100% |

**Check coverage:**
```bash
pytest tests/ --cov=server --cov=client --cov=models --cov-report=term-missing
```

---

## Workflow

### Branch Strategy

```
main
  └── develop (current working branch)
        ├── feature/add-new-attack-type
        ├── fix/grading-edge-case
        ├── refactor/middleware-ordering
        └── docs/api-reference
```

**Branch Naming:**
- `feature/` — New functionality
- `fix/` — Bug fixes
- `refactor/` — Code improvements (no behavior change)
- `docs/` — Documentation updates
- `test/` — Test additions/improvements
- `chore/` — Maintenance tasks

### Development Workflow

1. **Create Feature Branch:**
   ```bash
   git checkout develop
   git pull origin develop
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes:**
   ```bash
   # Edit code
   # Run tests frequently
   pytest tests/ -v --tb=short
   
   # Run linters
   ruff check server/ client.py models.py
   ruff format server/ client.py models.py
   mypy server/ client.py models.py
   ```

3. **Commit Changes:**
   ```bash
   git add .
   git commit -m "feat: add support for custom attack templates

   - Add AttackTemplateProvider interface
   - Implement JSON-based template loading
   - Add validation for template schema
   - Update tests for new functionality
   
   Closes #123"
   ```

4. **Push and Create PR:**
   ```bash
   git push origin feature/your-feature-name
   # Create Pull Request on GitHub
   ```

### Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting, no logic change)
- `refactor:` Code refactoring
- `test:` Test additions/changes
- `chore:` Maintenance tasks

**Examples:**
```
feat: add WebSocket streaming for real-time observations
fix: resolve race condition in episode cleanup
docs: update API reference with batch endpoints
refactor: extract grader logic into separate module
test: add property-based tests for attack generation
chore: update dependencies to latest versions
```

---

## Submitting Changes

### Pull Request Checklist

Before submitting a PR, ensure:

- [ ] All tests pass: `pytest tests/ -v`
- [ ] Code is formatted: `ruff format server/ client.py models.py`
- [ ] No linting errors: `ruff check server/ client.py models.py`
- [ ] Type checks pass: `mypy server/ client.py models.py`
- [ ] Security scan passes: `bandit -r server/ -c pyproject.toml`
- [ ] Pre-commit hooks pass: `pre-commit run --all-files`
- [ ] Coverage hasn't decreased: `pytest tests/ --cov=server --cov-report=term-missing`
- [ ] Documentation updated (if API changed)
- [ ] Commit messages follow Conventional Commits format

### PR Template

```markdown
## Description

Brief description of changes (2-3 sentences)

## Type of Change

- [ ] Bug fix (non-breaking change)
- [ ] New feature (non-breaking change)
- [ ] Breaking change (fix or feature causing existing functionality to break)
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing

Describe how you tested these changes:

```bash
pytest tests/test_your_feature.py -v
```

## Screenshots/Logs (if applicable)

Add relevant output or screenshots

## Checklist

- [ ] I have read the CONTRIBUTING.md guide
- [ ] My code follows the project's style guidelines
- [ ] I have performed a self-review of my code
- [ ] I have added tests that prove my fix/feature works
- [ ] All tests pass locally
- [ ] I have updated documentation (if needed)
```

### Review Process

1. **Automated Checks:** CI runs tests, linting, type checking
2. **Code Review:** At least one maintainer reviews the PR
3. **Address Feedback:** Make requested changes and push commits
4. **Approval:** Maintainer approves and merges
5. **Cleanup:** Delete feature branch after merge

---

## Documentation

### Documentation Structure

```
docs/
├── API_REFERENCE.md           # REST API documentation
├── CLIENT_GUIDE.md            # Client library usage
├── HYPERIONRL_TRAINING_GUIDE.md  # HyperionRL training
├── MIDDLEWARE_ARCHITECTURE.md  # Middleware design
├── DEPLOYMENT_GUIDE.md        # Deployment & troubleshooting
├── SECURITY_MODEL.md          # Security & threat model
└── README.md                  # Main documentation index
```

### Writing Documentation

**Guidelines:**
- Use markdown formatting
- Include code examples for all features
- Document edge cases and error handling
- Link to related documentation
- Keep examples practical and complete

**API Documentation:**
- Document all endpoints with request/response schemas
- Include curl examples and Python examples
- Document error responses and status codes

**Code Documentation:**
- Docstrings for all public functions/classes
- Parameters and return values typed
- Brief description of purpose
- Example usage for complex functions

**Example Docstring:**
```python
async def reset(
    request: Request,
    task_name: str = "basic-injection",
    seed: int = 42,
    api_key: str = Depends(verify_api_key),
) -> ResetResponse:
    """Start a new episode.
    
    Generates a seed-deterministic attack sequence based on the 
    specified task difficulty. Each episode supports up to max_steps 
    interactions before automatic completion.
    
    Args:
        request: FastAPI request object.
        task_name: Task difficulty (basic-injection, social-engineering, 
                   stealth-exfiltration).
        seed: Random seed for reproducible attack sequences.
        api_key: Verified API key from authentication middleware.
    
    Returns:
        ResetResponse with episode_id and first observation.
    
    Raises:
        HTTPException: 401 if API key invalid
        HTTPException: 500 if episode generation fails
    """
```

---

## Common Tasks

### Adding a New Attack Type

1. Add to `ThreatCategory` enum in `models.py`:
   ```python
   class ThreatCategory(str, Enum):
       # ... existing types ...
       NEW_ATTACK_TYPE = "new_attack_type"
   ```

2. Add to appropriate superclass in `models.py`:
   ```python
   THREAT_SUPERCLASSES = MappingProxyType({
       # ... existing superclasses ...
       "new_superclass": frozenset({ThreatCategory.NEW_ATTACK_TYPE})
   })
   ```

3. Update `attack_provider.py` with attack templates

4. Add tests in `tests/test_grader.py`

5. Update documentation in `docs/API_REFERENCE.md`

### Adding a New Endpoint

1. Define request/response models in `server/app.py`:
   ```python
   class NewEndpointResponse(BaseModel):
       field1: str
       field2: int
   ```

2. Add endpoint handler:
   ```python
   @app.get("/new-endpoint", response_model=NewEndpointResponse)
   async def new_endpoint(
       api_key: str = Depends(verify_api_key),
   ):
       # Implementation
       return NewEndpointResponse(field1="value", field2=42)
   ```

3. Add authentication and rate limiting if needed

4. Write tests in `tests/test_endpoints.py`

5. Update `docs/API_REFERENCE.md`

### Adding a New Middleware

1. Create middleware class in `server/middleware.py`:
   ```python
   class NewMiddleware(BaseHTTPMiddleware):
       async def dispatch(self, request: Request, call_next):
           # Pre-processing
           response = await call_next(request)
           # Post-processing
           return response
   ```

2. Add to `setup_production_middleware()`:
   ```python
   def setup_production_middleware(app: FastAPI):
       # Order matters!
       app.add_middleware(ErrorHandlingMiddleware)  # Outermost
       app.add_middleware(NewMiddleware)           # Your middleware
       app.add_middleware(PrometheusMetricsMiddleware)
       # ... rest of middleware
   ```

3. Document middleware order in `docs/MIDDLEWARE_ARCHITECTURE.md`

4. Write tests for middleware behavior

---

## Getting Help

### Communication

- **Issues:** GitHub Issues for bugs and feature requests
- **Discussions:** GitHub Discussions for questions and ideas
- **Email:** Contact maintainers directly

### Common Questions

**Q: How do I run a single test?**
```bash
pytest tests/test_file.py::test_function -v
```

**Q: Why is pre-commit failing?**
```bash
# See what's failing
pre-commit run --all-files

# Fix issues and commit again
```

**Q: How do I update dependencies?**
```bash
uv pip install -e ".[dev]" --upgrade
git add pyproject.toml
git commit -m "chore: update dependencies"
```

**Q: Where are checkpoints saved?**
```
model_checkpoints_hyperion/
├── checkpoint_ep_*.pt
└── best_model.pt
```

---

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Assume positive intent
- Help others learn and grow
- Follow the project's coding standards

---

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

---

**Last Updated:** April 12, 2026  
**Version:** 1.1.0
