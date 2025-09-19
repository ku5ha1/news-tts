import os
from huggingface_hub import scan_cache_dir

def main():
    cache_dir = "/app/.cache/huggingface/hub"
    print(f"Verifying cache in production stage: {cache_dir}")

    if not os.path.exists(cache_dir):
        print("⚠️ Cache directory not found, models may download at runtime.")
    else:
        expected_repos = [
            "ai4bharat/indictrans2-en-indic-dist-200M",
            "ai4bharat/indictrans2-indic-en-dist-200M",
        ]
        repos = [repo.repo_id for repo in scan_cache_dir(cache_dir).repos]
        for repo in expected_repos:
            if repo not in repos:
                print(f"{repo} not found in cache, will be fetched at runtime if needed.")
            else:
                print(f"Found {repo} in cache")
    print("Production stage model verification complete.")

if __name__ == "__main__":
    main()