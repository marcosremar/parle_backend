#!/usr/bin/env python3
"""
Script de Validação da Migração Services → Modules

Valida que:
1. Todos os módulos podem ser criados
2. Nenhum BasicWrapper está sendo usado
3. Imports estão corretos
4. Módulos podem ser inicializados
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from loguru import logger
from src.modules import module_factory

# Configure logger
logger.remove()
logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}", level="INFO")

def test_module_creation():
    """Testa criação de todos os módulos"""
    print("\n" + "="*70)
    print("TESTE 1: Criação de Módulos")
    print("="*70)
    
    modules = [
        'stt', 'tts', 'llm', 'user', 'orchestrator', 'session', 
        'scenarios', 'conversation_store', 'conversation_history',
        'file_storage', 'database', 'student_model', 'diagnostic_module',
        'pedagogical_policy', 'learning_path', 'rest_polling'
    ]
    
    success = 0
    failed = 0
    wrappers = 0
    errors = []
    
    for module_name in modules:
        try:
            module = module_factory.create(module_name)
            module_type = type(module).__name__
            is_wrapper = 'Wrapper' in module_type and 'DisabledTutoring' not in module_type
            
            if is_wrapper:
                wrappers += 1
                status = '⚠️  WRAPPER'
                errors.append(f"{module_name}: Using BasicWrapper (should be real module)")
            elif 'DisabledTutoring' in module_type:
                success += 1
                status = '✅ MÓDULO (disabled)'
            else:
                success += 1
                status = '✅ MÓDULO'
            
            print(f"{status:20s} {module_name:25s} -> {module_type}")
        except Exception as e:
            failed += 1
            error_msg = str(e)[:60]
            errors.append(f"{module_name}: {error_msg}")
            print(f"❌ ERRO{'':15s} {module_name:25s} -> {error_msg}")
    
    print("="*70)
    print(f"Resultado: {success} módulos, {wrappers} wrappers, {failed} erros")
    
    if wrappers > 0 or failed > 0:
        print("\n⚠️  Problemas encontrados:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True


def test_imports():
    """Testa imports críticos"""
    print("\n" + "="*70)
    print("TESTE 2: Imports Críticos")
    print("="*70)
    
    import_tests = [
        ("skill_registry", "src.modules.tutoring.student_model.skill_registry", "SKILL_CEFR_MAP"),
        ("session models", "src.modules.conversation.session.models", "LLMType"),
        ("orchestrator engine", "src.modules.conversation.orchestrator.engine", "ConversationOrchestrator"),
    ]
    
    success = 0
    failed = 0
    
    for name, module_path, attr in import_tests:
        try:
            module = __import__(module_path, fromlist=[attr])
            if hasattr(module, attr):
                success += 1
                print(f"✅ {name:25s} -> {module_path}")
            else:
                failed += 1
                print(f"❌ {name:25s} -> {attr} not found in {module_path}")
        except ImportError as e:
            failed += 1
            print(f"❌ {name:25s} -> ImportError: {str(e)[:50]}")
    
    print("="*70)
    print(f"Resultado: {success} sucessos, {failed} falhas")
    
    return failed == 0


def test_module_initialization():
    """Testa inicialização de módulos principais"""
    print("\n" + "="*70)
    print("TESTE 3: Inicialização de Módulos")
    print("="*70)
    
    import asyncio
    
    # Módulos principais (não tutoring, pois podem estar desativados)
    main_modules = [
        'stt', 'tts', 'llm', 'user', 'orchestrator', 'session',
        'scenarios', 'conversation_store', 'file_storage', 'database'
    ]
    
    async def test_init():
        success = 0
        failed = 0
        
        for module_name in main_modules:
            try:
                module = module_factory.create(module_name)
                
                # Check if module has initialize method
                if hasattr(module, 'initialize'):
                    # Check if already initialized
                    is_initialized = getattr(module, 'initialized', False) or getattr(module, '_initialized', False)
                    
                    if not is_initialized:
                        result = await module.initialize()
                        if result is False:
                            failed += 1
                            print(f"⚠️  {module_name:25s} -> Initialization returned False")
                        else:
                            success += 1
                            print(f"✅ {module_name:25s} -> Initialized successfully")
                    else:
                        success += 1
                        print(f"✅ {module_name:25s} -> Already initialized")
                else:
                    success += 1
                    print(f"✅ {module_name:25s} -> No initialization needed")
                    
            except Exception as e:
                failed += 1
                error_msg = str(e)[:50]
                print(f"❌ {module_name:25s} -> {error_msg}")
        
        print("="*70)
        print(f"Resultado: {success} sucessos, {failed} falhas")
        return failed == 0
    
    return asyncio.run(test_init())


def test_no_services_imports():
    """Verifica se módulos não dependem de services (exceto fallbacks)"""
    print("\n" + "="*70)
    print("TESTE 4: Verificação de Dependências de Services")
    print("="*70)
    
    import re
    from pathlib import Path
    
    modules_dir = project_root / "src" / "modules"
    services_imports = []
    
    # Padrão para encontrar imports de services
    pattern = re.compile(r'from\s+src\.services\.|import\s+.*src\.services\.')
    
    # Arquivos Python em modules
    for py_file in modules_dir.rglob("*.py"):
        # Ignorar __pycache__
        if "__pycache__" in str(py_file):
            continue
        
        try:
            content = py_file.read_text()
            lines = content.split('\n')
            
            for i, line in enumerate(lines, 1):
                if pattern.search(line):
                    # Verificar se é um fallback (try/except)
                    is_fallback = False
                    # Verificar linhas anteriores para try
                    for j in range(max(0, i-5), i):
                        if 'try:' in lines[j] or 'except ImportError:' in lines[j]:
                            is_fallback = True
                            break
                    
                    if not is_fallback:
                        rel_path = py_file.relative_to(project_root)
                        services_imports.append(f"{rel_path}:{i} - {line.strip()}")
        except Exception as e:
            print(f"⚠️  Erro ao ler {py_file}: {e}")
    
    if services_imports:
        print("⚠️  Imports diretos de services encontrados (sem fallback):")
        for imp in services_imports[:10]:  # Mostrar apenas os primeiros 10
            print(f"  - {imp}")
        if len(services_imports) > 10:
            print(f"  ... e mais {len(services_imports) - 10} imports")
        print("="*70)
        print(f"Total: {len(services_imports)} imports diretos encontrados")
        return False
    else:
        print("✅ Nenhum import direto de services encontrado (todos com fallback)")
        print("="*70)
        return True


def main():
    """Executa todos os testes"""
    print("\n" + "="*70)
    print("VALIDAÇÃO DA MIGRAÇÃO: Services → Modules")
    print("="*70)
    
    results = []
    
    # Teste 1: Criação de módulos
    results.append(("Criação de Módulos", test_module_creation()))
    
    # Teste 2: Imports
    results.append(("Imports Críticos", test_imports()))
    
    # Teste 3: Inicialização
    results.append(("Inicialização", test_module_initialization()))
    
    # Teste 4: Dependências
    results.append(("Dependências de Services", test_no_services_imports()))
    
    # Resumo final
    print("\n" + "="*70)
    print("RESUMO FINAL")
    print("="*70)
    
    all_passed = True
    for test_name, passed in results:
        status = "✅ PASSOU" if passed else "❌ FALHOU"
        print(f"{status:15s} {test_name}")
        if not passed:
            all_passed = False
    
    print("="*70)
    
    if all_passed:
        print("✅ TODOS OS TESTES PASSARAM!")
        return 0
    else:
        print("❌ ALGUNS TESTES FALHARAM")
        return 1


if __name__ == "__main__":
    sys.exit(main())
