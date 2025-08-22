# 🧪 Testing Directory

This directory contains all test scripts for the Vivum Backend project.

## 📁 Directory Structure

### 🔬 Unit Tests (`unit/`)
Individual component tests:
- `test_auto_tuning_system.py` - Auto-tuning system unit tests
- `test_subgraph_extractor.py` - Subgraph extractor unit tests

### 🔗 Integration Tests (`integration/`)
System integration and end-to-end tests:
- `test_lego_async_pipeline.py` - LEGO framework with async pipeline
- `test_lego_integration_simple.py` - Simplified LEGO integration tests

### ⚡ Performance Tests (`performance/`)
Performance and benchmarking tests:
- `test_quantization_and_spacy.py` - Quantization and spaCy performance tests
- `test_quantization_integration.py` - Quantization integration tests

## 🚀 Running Tests

### Unit Tests
```bash
cd testing/unit
python test_auto_tuning_system.py
python test_subgraph_extractor.py
```

### Integration Tests
```bash
cd testing/integration
python test_lego_async_pipeline.py
python test_lego_integration_simple.py
```

### Performance Tests
```bash
cd testing/performance
python test_quantization_and_spacy.py
python test_quantization_integration.py
```

### All Tests
```bash
# From project root
find testing -name "test_*.py" -exec python {} \;
```

## 📊 Test Categories

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions and system behavior
- **Performance Tests**: Test system performance, memory usage, and optimization

## 🔧 Test Requirements

- Python 3.8+
- Required packages: see `requirements.txt`
- Environment variables: see `documents/integration/INTEGRATION_REQUIREMENTS.md`

## �� Adding New Tests

When adding new test files:
- Place unit tests in `unit/`
- Place integration tests in `integration/`
- Place performance tests in `performance/`
- Update this README with new entries
