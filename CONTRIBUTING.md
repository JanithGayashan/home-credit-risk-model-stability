# Contribution workflow

All project changes go through a working branch and a pull request into `main`.

1. Start a feature, fix, or chore branch from the latest `main`.
2. Make the changes and run the checks relevant to those changes.
3. Review the staged files before committing. Dataset files, credentials, and
   generated model artifacts must remain excluded by `.gitignore`.
4. Commit and push the working branch, then open a pull request targeting `main`.
5. Review the pull request and its checks before merging it into `main`.

Do not push project changes directly to `main`. Creating or pushing a working
branch or opening a pull request does not authorize automatically merging it.

For a completely empty repository, a one-time empty base commit is needed to
establish `main` before the first pull request. Publishing that base commit
requires the repository owner's approval; project files belong in the first PR.

Keep pull requests focused and describe the behavior changed, the reason for the
change, and how it was verified. For notebook changes, check that the notebook is
valid and inspect the relevant execution outputs. Do not rerun expensive model
training solely for documentation or Git configuration changes.
