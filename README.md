What it does:
Given two lists of names (left and right), the script tries to decide which pairs refer to the same underlying product/entity. It outputs predicted matches and prints evaluation metrics (precision/recall/F1) plus sample false positives and false negatives for quick error analysis.

How to run:
	1.	Put your input files (the two name lists + labels if you have them) in the project folder.
	2.	Install requirements: pip install -r requirements.txt
	3.	Run: python3 entity_resolution.py

What you’ll see:
A tuned “best” threshold from validation, test-set results at that threshold, and an error-analysis section showing example mismatches (e.g., “upgrade” vs “full”, Mac vs Windows, version differences).

Entity Resolution / Identity Matching (Record Linkage) — Jan 2026
• Built a baseline that matches “messy” product/entity names across two datasets using TF-IDF character n-grams + cosine similarity, plus RapidFuzz string matching
• Tuned the match threshold on a validation split and reported precision/recall/F1 on a held-out test set
• Wrote error analysis on where it fails, e.g., same product family but different versions/licenses 
