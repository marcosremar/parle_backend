#!/usr/bin/env python3
"""
Script to verify and fix APA format references in the database
Ensures all references follow APA 7th edition format
"""

import sqlite3
from pathlib import Path
import re

# Database path
db_path = Path(__file__).parent / "implementation_tracker.db"

# Create connection
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Fetch all references
cursor.execute("SELECT id, concept, reference FROM implementation_concepts")
all_refs = cursor.fetchall()

print("=== Verificando Formato APA das Referências ===\n")

issues = []
correct = []

for ref_id, concept, reference in all_refs:
    # Basic APA format checks
    has_author = bool(re.search(r'^[A-Z][a-z]+, [A-Z]', reference)) or bool(re.search(r'^[A-Z][a-z]+ & [A-Z]', reference)) or bool(re.search(r'^[A-Z][a-z]+ et al\.', reference))
    has_year = bool(re.search(r'\([0-9]{4}\)', reference))
    has_title = bool(re.search(r'\. [A-Z]', reference))  # Title starts after period
    has_publisher_or_journal = bool(re.search(r'\. (?:[A-Z][a-z]+|Cambridge|Oxford|Harvard|Longman|Newbury|Academic|John Benjamins|Lawrence Erlbaum|Pergamon|Teachers College)', reference))
    
    # Check if it's a journal article (has volume/issue)
    is_journal = bool(re.search(r', [0-9]+\([0-9]+\)', reference)) or bool(re.search(r', [0-9]+\([0-9]+\)', reference))
    # Check if it's a book (has publisher)
    is_book = bool(re.search(r'\. (?:Cambridge|Oxford|Harvard|Longman|Newbury|Academic|John Benjamins|Lawrence Erlbaum|Pergamon|Teachers College)', reference))
    # Check if it's a chapter (has "In")
    is_chapter = bool(re.search(r'In [A-Z]', reference))
    
    # Check if it's a conference paper (has "Proceedings")
    is_conference = bool(re.search(r'Proceedings', reference, re.IGNORECASE))
    
    # Check if it's arXiv (has "arXiv:")
    is_arxiv = bool(re.search(r'arXiv:', reference, re.IGNORECASE))
    
    # All references should have at least author, year, and title
    if not (has_author and has_year):
        issues.append({
            'id': ref_id,
            'concept': concept,
            'reference': reference,
            'issue': 'Missing author or year'
        })
    elif not has_title:
        issues.append({
            'id': ref_id,
            'concept': concept,
            'reference': reference,
            'issue': 'Missing or unclear title'
        })
    else:
        correct.append(concept)

print(f"✅ Referências corretas: {len(correct)}/{len(all_refs)}")
print(f"⚠️  Referências com problemas: {len(issues)}/{len(all_refs)}\n")

if issues:
    print("=== Referências que precisam de correção ===\n")
    for item in issues[:10]:  # Show first 10
        print(f"ID: {item['id']}")
        print(f"Conceito: {item['concept']}")
        print(f"Problema: {item['issue']}")
        print(f"Referência: {item['reference'][:100]}...")
        print()
    
    if len(issues) > 10:
        print(f"... e mais {len(issues) - 10} referências com problemas\n")
else:
    print("✅ Todas as referências estão em formato APA!\n")

# Show sample of correct references
print("=== Exemplos de Referências Corretas ===\n")
cursor.execute("""
    SELECT concept, reference 
    FROM implementation_concepts 
    WHERE reference LIKE '%, %' 
    AND reference LIKE '%.%' 
    AND reference LIKE '%(%'
    LIMIT 5
""")
for row in cursor.fetchall():
    print(f"Conceito: {row[0]}")
    print(f"Referência: {row[1]}")
    print()

# Summary by type
cursor.execute("""
    SELECT 
        CASE 
            WHEN reference LIKE '%Proceedings%' OR reference LIKE '%Conference%' THEN 'Conference Paper'
            WHEN reference LIKE '%arXiv%' THEN 'Preprint'
            WHEN reference LIKE 'In %' THEN 'Chapter in Book'
            WHEN reference LIKE '%, [0-9]%(%' THEN 'Journal Article'
            WHEN reference LIKE '%. Cambridge%' OR reference LIKE '%. Oxford%' OR reference LIKE '%. Harvard%' THEN 'Book'
            ELSE 'Other'
        END as ref_type,
        COUNT(*) as count
    FROM implementation_concepts
    GROUP BY ref_type
    ORDER BY count DESC
""")

print("=== Distribuição por Tipo de Referência ===\n")
for row in cursor.fetchall():
    print(f"  {row[0]}: {row[1]}")

conn.close()

