# Cleanup Notes

## Files to Clean Up

### Temporary Files
- `src/figure_1.pdf` and `src/figure_1.png` - Should be moved to `src/figures/` or removed
- `src/online_experiment/perlin_numpy/__pycache__/` - Python cache, can be removed

### Scripts
- `fix_scipy_env.sh` - Temporary fix script. Can be removed if environment is stable.

## Commands to Run

```bash
# Remove Python caches
find . -type d -name "__pycache__" -exec rm -r {} +

# Move figures to proper location (if needed)
mv src/figure_1.pdf src/figures/ 2>/dev/null || true
mv src/figure_1.png src/figures/ 2>/dev/null || true

# Or use the Makefile
make clean
```

## Using the Makefile

The Makefile includes a `clean` target that handles most cleanup automatically:

```bash
make clean          # Remove caches and temporary files
make clean-figures  # Remove generated figures
make clean-all      # Deep clean (everything except data)
```

