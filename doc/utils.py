import subprocess


def get_git_branch() -> str:
    """Gets the current git branch.

    Returns:
        str: Current branch.
    """
    # Getting the current branch
    branch = (
        subprocess.check_output(
            ["git", "branch", "--show-current"],
            stdin=None,
            stderr=None,
            shell=False,
            universal_newlines=False,
        )
        .decode("utf-8")
        .strip()
    )
    return branch
