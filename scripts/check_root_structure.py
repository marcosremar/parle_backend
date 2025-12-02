from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Entradas permitidas explicitamente na raiz do repositório.
# Baseado em docs/PROJECT_ORGANIZATION.md e configurações existentes.
ALLOWED_ROOT_ENTRIES = {
    # Metadados e controle de versão
    ".git",
    ".github",
    ".cursor",
    ".cursorrules",
    ".windsurfrules",
    ".gitignore",
    ".gitmodules",
    ".pre-commit-config.yaml",
    ".coveragerc",
    ".dockerignore",
    ".secrets.baseline",

    # Variáveis de ambiente (mantidas fora do versionamento)
    ".env",
    ".env.example",

    # Documentação principal e licenças
    "README.md",
    "LICENSE.txt",
    "CONTRIBUTING.md",

    # Scripts e configuração principais
    "main.sh",
    "pyproject.toml",
    "environment.yml",
    "requirements.txt",
    "requirements-dev.txt",
    "requirements-test.txt",

    # Diretórios principais da organização
    "docker",
    "config",
    "data",
    "docs",
    "scripts",
    "src",
    "tests",
    "vendor",
}


def is_git_ignored(path: Path) -> bool:
    """Retorna True se o caminho for ignorado pelo Git (segundo .gitignore)."""

    try:
        result = subprocess.run(
            ["git", "check-ignore", str(path)],
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        # Se algo der errado, não bloquear por causa disso
        return False

    return result.returncode == 0


def main() -> None:
    unexpected: list[str] = []

    for entry in PROJECT_ROOT.iterdir():
        name = entry.name

        # Sempre permitir diretório .git internamente
        if name in ALLOWED_ROOT_ENTRIES:
            continue

        # Ignorar tudo que for gitignored (cache, venv, tmp, etc.)
        if is_git_ignored(entry):
            continue

        # Qualquer outra coisa na raiz é considerada fora do padrão
        unexpected.append(name)

    if unexpected:
        print("[check-root-structure] Estrutura inválida na raiz do repositório.\n")
        print("Foram encontrados arquivos/diretórios não permitidos na raiz:")
        for name in sorted(unexpected):
            print(f"  - {name}")

        print("\nDe acordo com docs/PROJECT_ORGANIZATION.md, a raiz deve conter apenas arquivos "
              "essenciais (README, requirements, pyproject, main.sh, etc.) e diretórios "
              "principais (docker, config, docs, scripts, src, tests).")
        print(
            "\nAção sugerida:\n"
            "  - Mover documentação para 'docs/' (por exemplo, 'docs/status/' ou 'docs/deploy/').\n"
            "  - Mover scripts .sh/.py para 'scripts/' (por exemplo, 'scripts/deploy/').\n"
            "  - Ou, se realmente precisar manter algo na raiz, adicione explicitamente o nome em\n"
            "    'ALLOWED_ROOT_ENTRIES' em 'scripts/check_root_structure.py'.\n"
        )

        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":  # pragma: no cover - script utilitário
    main()
