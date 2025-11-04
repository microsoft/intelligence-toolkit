# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

### Unreleased Template
```md
## [Unreleased]
### Added
### Changed
### Deprecated
### Removed
### Fixed
### Security
```

## [Unreleased]

### Added
- Added automatic schema normalization for imported schemas to ensure OpenAI structured outputs compatibility
- Added unit tests for detect entity networks prepare_model functionality
- Added Docker validation to CI workflow
- Added comprehensive test suite for `build_network_from_entities` with coverage for trimmed attributes, integrated flags, and inferred links

### Changed
- Migrated build system from hatchling to setuptools for better package management
- Updated import paths to use `intelligence_toolkit.AI.metaprompts` instead of `app.workflows.security.metaprompts`
- Added unit tests for coverage
- Changed local embedding default to false in Detect Entity Networks (DEN)
- Updated example notebooks for better consistency and reduced output verbosity
- Updated `detect_entity_networks/explore_networks.py` to handle `trimmed_attributes` as a DataFrame instead of list of tuples for better data consistency
- Added permission on gh workflow

### Removed
- Removed deprecated `app/workflows/security/` module (metaprompts moved to `intelligence_toolkit.AI.metaprompts`)
- Removed Required and Additional fields UI controls from schema builder (now automatically managed for OpenAI structured outputs)

### Fixed
- Fixed import paths in Match Entity Records workflow prompts
- Fixed import paths in Query Text Data workflow prompts
- Updated `.gitignore` to exclude `**.egg-info/` directories
- Fixed Generate Mock Data (GMD) workflow issues in data generation logic
- Fixed Compare Case Groups (CCG) API and dataframe building functionality
- Fixed Detect Case Patterns (DCP) workflow processing
- Fixed Query Text Data (QTD) answer builder and notebook examples
- Fixed torch dependency version constraints in pyproject.toml
- Fixed plotly installation in dependencies
- Fixed detect entity networks prepare_model edge case handling
- Fixed CSV file upload persistence in Extract Record Data workflow: uploaded CSV files and dataframes now persist when navigating between tabs by caching files and dataframes in session state and preserving the extraction mode selection
- Fixed Detect Case Patterns to gracefully handle cases with no converging patterns, returning empty DataFrame instead of single NaN row
- Added comprehensive unit tests for empty pattern detection scenario
- Fixed Detect Case Patterns `compute_attribute_counts()` to handle missing columns gracefully with warning message
- Fixed Detect Case Patterns `detect_patterns()` normalization to handle empty pattern DataFrames without division errors
- Fixed CSV error in detect entity networks report functionality
- Fixed entity network exploration to properly handle DataFrame-based trimmed attributes

## [0.1.2] - 2024-10-15

### Added
- Live analysis of relevant chunks and thematic commentary in Query Text Data (QTD)
- Hierarchical community expansion for network analysis
- Download option for synthetic aggregates in Anonymize Case Data (ACD)
- Fabrication mode to ACD for enhanced data generation
- Local embedding model support per workflow
- Azure Marketplace deployment templates
- AWS deployment instructions
- Option to change embedding models for both local and OpenAI
- Async embedding operations for improved performance
- LanceDB integration for vector embeddings storage
- Retry decorator for LanceDB operations
- New workflows: Extract Record Data and Generate Mock Data
- Example outputs and tutorial structure for all workflows

### Changed
- Migrated from Poetry to UV for package management
- Updated OpenAI client to version >=1.37.1,<2.0.0
- Improved QTD query expansion and citation format
- Enhanced schema builder for OpenAI structured output compliance
- Text splitting now uses semchunk by token for better chunking
- Concurrent coroutines optimization for QTD embedding
- Updated torch installation (2.4.1 for non-macOS, 2.5.1 for macOS)
- Improved prompting to prevent content item duplication
- Enhanced data preparation and preprocessing steps
- Better error handling for QTD multi-file upload
- Replaced pdfplumber with pypdf for PDF processing

### Fixed
- Analysis fix for Query Text Data workflow (QTD)
- Entity matching algorithm bug fixes
- Build issues on Azure DevOps
- Local embedding and model compatibility issues
- Tokenizer issues on newer models
- Required field validation in schema builder
- Embedding shift bug
- Source bug in QTD
- Multi-file upload error handling
- CSV loading for raw files in QTD
- Null attribute value connections in Detect Entity Networks (DEN)
- Anonymize index bug (None to 0)
- Datetime quantization bug in data preparation
- Azure client authentication bug
- Delta count in Compare Case Groups (CCG) for temporal data
- Type mismatches and unbound variable errors
- Docker image size optimization
- Executable (.exe) generation issues

### Security
- Bumped Jinja2 from 3.1.4 to 3.1.5 (CVE fix)
- Bumped Tornado from 6.4.1 to 6.4.2 (security update)
- Bumped Transformers from 4.46.1 to 4.48.0
- Bumped Cryptography from 43.0.0 to 43.0.1 (vulnerability fix)
