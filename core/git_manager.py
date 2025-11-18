"""
Git Manager
Handles automatic git operations for code synchronization
"""

import logging
import subprocess
from pathlib import Path
from typing import Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class GitManager:
    """
    Manages git operations for automatic code synchronization

    Features:
    - Auto-commit with timestamp
    - Auto-push to remote
    - Status checking
    - Commit hash retrieval
    """

    def __init__(self, repo_path: Optional[Path] = None):
        """
        Initialize Git Manager

        Args:
            repo_path: Path to git repository (default: project root)
        """
        if repo_path is None:
            # Default to project root (3 levels up from this file)
            repo_path = Path(__file__).parent.parent.parent

        self.repo_path = Path(repo_path)

        if not self._is_git_repo():
            raise ValueError(f"Not a git repository: {self.repo_path}")

    def _is_git_repo(self) -> bool:
        """Check if directory is a git repository"""
        git_dir = self.repo_path / ".git"
        return git_dir.exists() and git_dir.is_dir()

    def _run_git_command(
        self,
        command: list[str],
        timeout: int = 30
    ) -> Tuple[bool, str, str]:
        """
        Run a git command

        Args:
            command: Git command as list (e.g., ['git', 'status'])
            timeout: Command timeout in seconds

        Returns:
            Tuple of (success, stdout, stderr)
        """
        try:
            result = subprocess.run(
                command,
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=timeout
            )

            success = result.returncode == 0
            return success, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            return False, "", "Command timeout"
        except Exception as e:
            return False, "", str(e)

    def has_changes(self) -> bool:
        """
        Check if there are uncommitted changes

        Returns:
            True if there are changes, False otherwise
        """
        success, stdout, _ = self._run_git_command(['git', 'status', '--porcelain'])

        if not success:
            return False

        # If stdout is not empty, there are changes
        return bool(stdout.strip())

    def get_current_branch(self) -> Optional[str]:
        """
        Get current git branch

        Returns:
            Branch name or None if error
        """
        success, stdout, _ = self._run_git_command(['git', 'branch', '--show-current'])

        if success:
            return stdout.strip()

        return None

    def get_latest_commit_hash(self, short: bool = True) -> Optional[str]:
        """
        Get latest commit hash

        Args:
            short: Return short hash (7 chars) if True

        Returns:
            Commit hash or None if error
        """
        format_str = '%h' if short else '%H'
        success, stdout, _ = self._run_git_command(['git', 'log', '-1', f'--format={format_str}'])

        if success:
            return stdout.strip()

        return None

    def auto_commit_and_push(
        self,
        message: Optional[str] = None,
        include_untracked: bool = True
    ) -> Tuple[bool, Optional[str]]:
        """
        Automatically commit and push changes

        Args:
            message: Commit message (default: auto-generated with timestamp)
            include_untracked: Include untracked files (git add -A)

        Returns:
            Tuple of (success, commit_hash or error_message)
        """
        # Check if there are changes
        if not self.has_changes():
            logger.info("✅ No changes to commit")
            return True, self.get_latest_commit_hash()

        # Generate commit message if not provided
        if message is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            message = f"auto-deploy: {timestamp}"

        logger.info(f"🔄 Committing changes: {message}")

        # Stage changes
        add_command = ['git', 'add', '-A'] if include_untracked else ['git', 'add', '-u']
        success, _, stderr = self._run_git_command(add_command)

        if not success:
            logger.error(f"❌ Failed to stage changes: {stderr}")
            return False, f"Failed to stage: {stderr}"

        # Commit
        success, _, stderr = self._run_git_command(['git', 'commit', '-m', message])

        if not success:
            # Check if it's "nothing to commit"
            if "nothing to commit" in stderr.lower():
                logger.info("✅ Nothing to commit (already up to date)")
                return True, self.get_latest_commit_hash()

            logger.error(f"❌ Failed to commit: {stderr}")
            return False, f"Failed to commit: {stderr}"

        # Get commit hash
        commit_hash = self.get_latest_commit_hash()
        logger.info(f"✅ Committed: {commit_hash}")

        # Push
        logger.info("📤 Pushing to remote...")
        branch = self.get_current_branch()

        if not branch:
            logger.error("❌ Failed to get current branch")
            return False, "Failed to get current branch"

        success, _, stderr = self._run_git_command(['git', 'push', 'origin', branch], timeout=60)

        if not success:
            logger.error(f"❌ Failed to push: {stderr}")
            return False, f"Failed to push: {stderr}"

        logger.info(f"✅ Pushed to origin/{branch}")

        return True, commit_hash

    def get_remote_url(self) -> Optional[str]:
        """
        Get remote origin URL

        Returns:
            Remote URL or None if error
        """
        success, stdout, _ = self._run_git_command(['git', 'remote', 'get-url', 'origin'])

        if success:
            return stdout.strip()

        return None

    def get_status_summary(self) -> dict:
        """
        Get git status summary

        Returns:
            Dict with status information
        """
        return {
            'has_changes': self.has_changes(),
            'current_branch': self.get_current_branch(),
            'latest_commit': self.get_latest_commit_hash(),
            'remote_url': self.get_remote_url(),
            'repo_path': str(self.repo_path)
        }


# Singleton instance
_git_manager_instance: Optional[GitManager] = None


def get_git_manager(repo_path: Optional[Path] = None) -> GitManager:
    """
    Get global GitManager instance

    Args:
        repo_path: Path to git repository (default: project root)

    Returns:
        GitManager instance
    """
    global _git_manager_instance

    if _git_manager_instance is None:
        _git_manager_instance = GitManager(repo_path)

    return _git_manager_instance
