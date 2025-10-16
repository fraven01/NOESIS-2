# Data Contracts

The `noesis_docs` directory contains the data contracts for the NOESIS 2 platform. These contracts are defined as Pydantic models and are used to ensure data consistency and validation across the application.

## Purpose

The primary purpose of these contracts is to provide a single source of truth for the data structures used in the application. By using Pydantic models, we can ensure that all data exchanged between different parts of the application, as well as with external clients, adheres to a predefined schema. This helps to prevent data-related errors and makes the application more robust and reliable.

## Structure

The contracts are organized into subdirectories based on their domain. For example, the `noesis_docs/contracts` directory contains the contracts for the "Collection" feature.

### `collection_v1.py`

This file defines the Pydantic models for a `CollectionRef` and `CollectionLink`. These models are used to validate and serialize data related to collection references, as described in the [RAG Overview documentation](rag/overview.md#collection-scopes).
