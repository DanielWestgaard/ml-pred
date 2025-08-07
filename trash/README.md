# Deprecated Files

This folder contains deprecated files that have been replaced by cleaner, more modular implementations.

## Deprecated Feature Files

- `target_hypothesis_legacy.py` - Original target hypothesis implementation with backward compatibility cruft
- `target_hypothesis_v2.py` - Intermediate version with some improvements
- `target_hypothesis_original.py` - Original backup before modernization

## Replacement

These files have been replaced by the clean, streamlined implementation in:
- `data/hypothesis/target_generators.py` - Clean hypothesis implementations
- `data/hypothesis/base_target_generator.py` - Base class for all target generators

The new implementation provides:
- Better organization (hypothesis-specific logic in dedicated folder)
- No backward compatibility complexity
- Cleaner error handling
- Easier to extend with new hypotheses