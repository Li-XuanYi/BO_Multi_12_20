---
name: battery-bo-literature-search
description: Search academic literature for battery charging protocols, Bayesian optimization, multi-objective optimization, and related fields. Use when the user needs to find papers, cite references, or research related work for LLAMBO-MO project.
---

# Battery BO Literature Search

Search academic papers related to battery charging protocols, Bayesian optimization, and multi-objective optimization.

## Overview

This skill helps you find relevant academic literature for the LLAMBO-MO project, which focuses on:
- Battery fast-charging protocol optimization
- Multi-objective Bayesian optimization
- Physics-informed Gaussian Processes
- LLM-augmented optimization methods

## Key Research Areas

### 1. Battery Charging Optimization
- Search terms: "battery fast charging", "charging protocol optimization", "lithium-ion charging strategy"
- Key venues: Journal of Power Sources, Electrochimica Acta, Nature Energy, Joule

### 2. Bayesian Optimization (BO)
- Search terms: "Bayesian optimization", "multi-objective BO", "ParEGO", "EHVI"
- Key venues: NeurIPS, ICML, AISTATS, Journal of Machine Learning Research

### 3. Multi-Objective Optimization
- Search terms: "multi-objective optimization", "Pareto front", "scalarization", "Tchebycheff"
- Key venues: IEEE Transactions on Evolutionary Computation, Evolutionary Multi-Criterion Optimization (EMO)

### 4. Physics-Informed Machine Learning
- Search terms: "physics-informed neural networks", "physics-informed GP", "battery degradation modeling"
- Key venues: Nature Computational Science, Computer Methods in Applied Mechanics and Engineering

### 5. LLM for Scientific Discovery
- Search terms: "LLM for optimization", "large language model science", "AI for battery design"
- Key venues: arXiv, Science, Nature

## Search Workflow

### Step 1: Search Papers

Use web search to find relevant papers:

```
Search for: "battery charging protocol optimization Bayesian optimization"
Search for: "multi-objective Bayesian optimization lithium-ion battery"
Search for: "physics-informed Gaussian process battery degradation"
```

### Step 2: Download PDFs

For papers found, attempt to download PDFs from:
- arXiv (direct PDF download)
- Open access repositories
- Publisher websites (if you have access)

### Step 3: Extract Key Information

For each relevant paper, extract:
- **Citation**: Full bibliographic information
- **Problem**: What optimization problem does it address?
- **Method**: What BO/acquisition function approach is used?
- **Objectives**: What are the conflicting objectives?
- **Physics**: How is physical knowledge incorporated?
- **Results**: Key findings and performance metrics

## Recommended Databases

| Database | URL | Best For |
|----------|-----|----------|
| arXiv | arxiv.org | Preprints, CS/ML papers |
| Google Scholar | scholar.google.com | Broad search, citations |
| Web of Science | webofknowledge.com | High-quality journals |
| Scopus | scopus.com | Comprehensive coverage |
| Semantic Scholar | semanticscholar.org | AI/ML papers, recommendations |
| PubMed | pubmed.ncbi.nlm.nih.gov | Biomedical aspects |

## Key Papers to Know

### Battery Charging
- **Dubarry et al.** - Battery degradation modeling
- **Lu et al.** - Fast charging strategies review
- **Shen & Ouyang** - Optimal charging current profiles

### Multi-Objective BO
- **Knowles (2006)** - ParEGO (original algorithm)
- **Emmerich et al.** - EHVI acquisition
- **Daulton et al.** - qEHVI for batch BO

### Physics-Informed ML
- **Karniadakis et al.** - Physics-informed machine learning review
- **Raissi et al.** - Physics-informed neural networks
- **Mcgreyer et al.** - GP with physical constraints

## Citation Format

Use IEEE format for citations:
```
[1] A. Author, B. Author, and C. Author, "Title of the paper," Journal Name, vol. X, no. Y, pp. Z, Month Year.

[2] A. Author et al., "Conference paper title," in Proc. Conference Name, City, Country, Year, pp. X-Y.
```

## Tips for Effective Searching

1. **Use Boolean operators**: "battery AND (charging OR charging protocol) AND optimization"
2. **Phrase search**: Use quotes for exact phrases: "Bayesian optimization"
3. **Author search**: "author:Knowles multi-objective"
4. **Year filter**: Focus on papers from 2018-2025 for cutting-edge methods
5. **Citation tracking**: Check papers that cite key foundational works

## Example Searches

```
# Find recent multi-objective BO papers
"multi-objective Bayesian optimization" 2020-2025

# Find battery charging optimization with constraints
"battery charging" optimization constraints degradation

# Find physics-informed surrogate models
"physics-informed" surrogate model battery electrochemical

# Find LLM applications in optimization
"large language model" optimization OR "LLM" optimization
```

## Output Format

When reporting search results, provide:

```markdown
## Search Results: [Topic]

### Paper 1: [Title]
- **Authors**: 
- **Venue**: 
- **Year**: 
- **DOI/URL**: 
- **Relevance**: How this relates to LLAMBO-MO
- **Key Finding**: Most important result
- **Citable Quote**: Relevant excerpt

### Paper 2: [Title]
...
```
