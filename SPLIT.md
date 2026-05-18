# Splitting this branch into a standalone repo

This tree (`claude/refactor-pretraining-pipelines-CXanw`) is the cleaned-up
extract of the parent `mft-cpea` repo, containing only the **enhanced
pretraining** and **cosine few-shot evaluation** pipelines. Below are three
ways to publish it as its own repository.

## Option A — Fresh repo, no history (simplest)

```bash
git clone --single-branch --branch claude/refactor-pretraining-pipelines-CXanw \
    https://github.com/nMaric-0/mft-cpea.git mft-cpea-cosine
cd mft-cpea-cosine
rm -rf .git
git init
git add .
git commit -m "Initial commit: enhanced pretraining + cosine eval"
git remote add origin <new-repo-url>
git push -u origin main
```

Use this if you don't need to preserve the per-file commit history.

## Option B — `git filter-repo` (preserves history for surviving paths)

`git filter-repo` (https://github.com/newren/git-filter-repo) rewrites history
to drop everything outside an allow-list while keeping the commits that
touched the surviving files.

```bash
git clone https://github.com/nMaric-0/mft-cpea.git mft-cpea-cosine
cd mft-cpea-cosine
git checkout claude/refactor-pretraining-pipelines-CXanw

# Allow-list of paths to keep. Anything else is purged from every commit.
cat > /tmp/keep-paths.txt <<'EOF'
.gitignore
LICENSE
README.md
SPLIT.md
pyproject.toml
requirements.txt
setup.py
configs/pretrain/base.yaml
configs/pretrain/houston_pretrain_enhanced.yaml
configs/pretrain/trento_pretrain_enhanced.yaml
data/
docs/COSINE_VARIANT.md
docs/ENHANCED_PRETRAINING.md
models/__init__.py
models/mft_cpea_cosine.py
models/backbones/
models/components/__init__.py
models/components/tokenizers.py
models/components/transformer.py
models/components/projection.py
models/components/mft_blocks.py
models/wrappers/__init__.py
models/wrappers/pretrain_wrapper.py
pretrain/__init__.py
pretrain/masked_modeling.py
pretrain/masked_modeling_enhanced.py
pretrain/decoders.py
scripts/download_data.sh
scripts/pretrain_enhanced.py
scripts/evaluate_cosine.py
scripts/run_cosine_eval.sh
scripts/run_trento_cosine_eval.sh
tests/__init__.py
tests/test_data.py
tests/test_models.py
tests/test_pretrain_enhanced.py
tests/test_spatial_weights.py
trainers/__init__.py
trainers/pretrain_trainer.py
utils/
EOF

git filter-repo --paths-from-file /tmp/keep-paths.txt --force

git remote add origin <new-repo-url>
git push -u origin claude/refactor-pretraining-pipelines-CXanw:main
```

## Option C — `git subtree split` (preserves history, branch-based)

If you'd rather not rewrite history globally and the surviving paths fit a
single prefix, `git subtree split` works. Since this cleanup is spread across
the root rather than a single subdirectory, **Option A or B is preferred**.

## After the split

- Update `<new-repo-url>` references in CI / docs.
- Add a top-level pointer in the parent repo (`README.md`) linking to the new
  repo for users who came looking for the cosine variant.
- Verify the new repo with:

  ```bash
  pip install -e .
  pytest tests/ -q
  bash -n scripts/run_cosine_eval.sh
  ```
