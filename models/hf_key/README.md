# Hugging Face Access Key

This folder is a small local utility directory used by the joint LLM model download scripts in the VEX release.

Its purpose is to provide a simple filesystem location for a Hugging Face access token when downloading gated model weights, in particular the Gemma-based joint grading models under:

- [../joint_models/gemma/prepare_gemma.py](C:/Git/Bachelor/VEX/models/joint_models/gemma/prepare_gemma.py:1)
- [../joint_models/llama/prepare_gemma.py](C:/Git/Bachelor/VEX/models/joint_models/llama/prepare_gemma.py:1)

## Files

This directory contains:

- [hf_api_key.txt](C:/Git/Bachelor/VEX/models/hf_key/hf_api_key.txt:1): local text file expected to contain a Hugging Face token.
- [README.md](C:/Git/Bachelor/VEX/models/hf_key/README.md:1): this documentation file.

## Expected Format

The download scripts read the token with:

```python
HF_TOKEN_FILE = Path("../hf_key/hf_api_key.txt")
HF_TOKEN = HF_TOKEN_FILE.read_text(encoding="utf-8").strip()
```

Accordingly, `hf_api_key.txt` should contain exactly one token string, for example:

```text
hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

No JSON, YAML, or key-value wrapper is required.

## When This File Is Needed

This file is only needed for scripts that download gated Hugging Face models. In the current release, that primarily applies to the Gemma preparation scripts, which attempt to download:

```text
google/gemma-4-E4B-it
```

or an alternative Unsloth-hosted variant if the script is edited accordingly.

The token is not required for the lightweight baselines and is not used by all model families in this repository.

## Access Requirements

Having a token alone may not be sufficient. For gated models, you may also need to:

1. Be logged into a Hugging Face account with access to the requested model.
2. Accept the model license or usage terms on Hugging Face.
3. Use a token with permission to read model repositories.

If download fails, the most common causes are:

- the token is missing or invalid,
- the model access request has not been approved,
- the model usage conditions have not been accepted,
- the script is pointing to a different model than expected.

## Security and Release Notes

This folder should be treated as a **local-only credential helper**.

- Do not store a real access token in the public release.
- Do not include a real token in commits, archives, or screenshots.
- If you are preparing a public fork or mirror, leave `hf_api_key.txt` empty or replace it with a local untracked file.

The file [hf_api_key.txt](C:/Git/Bachelor/VEX/models/hf_key/hf_api_key.txt:1) is included here as a placeholder path expected by the scripts, not as a recommended mechanism for long-term secret management.

## Recommended Usage

For local reproduction:

1. Place your Hugging Face token into [hf_api_key.txt](C:/Git/Bachelor/VEX/models/hf_key/hf_api_key.txt:1).
2. Run the relevant model preparation script from its model directory.
3. Remove or replace the token before sharing the repository.

For cleaner setups, users may prefer adapting the scripts to read from an environment variable such as `HF_TOKEN` instead of a local plaintext file.

## Scope in This Release

This folder is intentionally minimal. It exists only to support reproducible access to gated model downloads used by the released joint-model experiments, while keeping the credential interface explicit and easy to inspect.
