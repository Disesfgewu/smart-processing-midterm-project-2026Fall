"""Read-only GitHub check; execute the trusted base branch version in CI."""
import json
import os
import urllib.request


def api(path):
    request = urllib.request.Request(
        f"https://api.github.com/repos/{os.environ['GH_REPO']}/{path}",
        headers={"Authorization": f"Bearer {os.environ['GH_TOKEN']}",
                 "Accept": "application/vnd.github+json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def pages(path):
    page = 1
    while True:
        batch = api(f"{path}?per_page=100&page={page}")
        yield from batch
        if len(batch) < 100:
            return
        page += 1


def violations(pr, commits):
    problems = []
    total = pr['additions'] + pr['deletions']
    if total > 300:
        problems.append(f"PR changes {total} lines (limit 300)")
    for commit in commits:
        stats = commit['stats']
        count = stats['additions'] + stats['deletions']
        if count > 300:
            problems.append(f"Commit {commit['sha']} changes {count} lines (limit 300)")
    return problems


def main():
    number = int(os.environ['PR_NUMBER'])
    pr = api(f"pulls/{number}")
    head = pr['head']['sha']
    refs = list(pages(f"pulls/{number}/commits"))
    if len(refs) != pr['commits']:
        raise SystemExit("Cannot verify every commit; split this PR.")
    commits = [api(f"commits/{ref['sha']}") for ref in refs]
    problems = violations(pr, commits)
    approved = any(
        comment['user']['login'] == 'Disesfgewu'
        and comment['body'].strip() == f'/approve-size {head}'
        for comment in pages(f"issues/{number}/comments")
    ) if problems else False
    if api(f"pulls/{number}")['head']['sha'] != head:
        raise SystemExit("PR changed during verification; rerun check.")
    for problem in problems:
        print(problem)
    if problems and not approved:
        raise SystemExit("Split changes or obtain owner approval for this exact head.")
    print(f"PASS head {head}" + (" with owner size exception" if approved else ""))


if __name__ == '__main__':
    main()
