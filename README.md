# BNS Legal Assistant (BharatBrick_Hac)

An AI-powered legal assistant designed to bridge the gap between layman language and the **Bharatiya Nyaya Sanhita (BNS)**. Describe a crime or legal situation in plain English, and the assistant will navigate complex legal codes to identify the exact BNS Section, provide a description, and summarize relevant real-world case law precedents.

Built for deployment on **Hugging Face Spaces (CPU)** and accessible via **Web UI (Gradio)** and **WhatsApp (Twilio)**.

---

## Features
* **Layman-to-Legal Translation:** Uses an LLM to rewrite casual descriptions into formal legal terminology.
* **Hybrid Search Engine:** Combines BM25 (keyword matching) and SBERT (semantic similarity) to find the correct legal chapter and subtype.
* **Zero-Cost Decision Trees:** Uses pre-generated JSON decision trees for precise section matching without relying on expensive LLM tokens.
* **RAG Case Law Summaries:** Automatically searches IndianKanoon/Casemine via DuckDuckGo and uses the LLM to summarize past Supreme Court/High Court judgments.
* **Omnichannel Access:** Run both a Gradio Web App and a Flask Twilio Webhook from a single Hugging Face Space.

---

## System Architecture

```mermaid
graph TD
    %% Define Styles
    classDef user fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#000000;
    classDef model fill:#f3e5f5,stroke:#4a148c,stroke-width:2px,color:#000000;
    classDef process fill:#fff3e0,stroke:#e65100,stroke-width:2px,color:#000000;
    classDef data fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px,color:#000000;
    classDef output fill:#ffebee,stroke:#b71c1c,stroke-width:3px,color:#000000;

    %% Flowchart Nodes
    Input([User Query: Layman Description]):::user
    
    subgraph Stage_1 [Stage 1: Translation]
        LLM1[Phi-3.5-mini GGUF]:::model
        Rewrite[Generate 3 Formal Legal Variations]:::process
    end

    subgraph Stage_2 [Stage 2: Hybrid Search]
        BM25[BM25 Index Search]:::process
        SBERT[SBERT Semantic Search]:::process
        DB[(BNS Sections CSV)]:::data
        Group[Identify Best Chapter_Subtype Group]:::process
    end

    subgraph Stage_3 [Stage 3: Decision Tree]
        JSON[(Pre-generated JSON Trees)]:::data
        Walk[Auto-walk / Interactive Walk]:::process
        Section[Exact BNS Section Identified]:::data
    end

    subgraph Stage_4 [Stage 4: RAG & Summary]
        DDG[DuckDuckGo Web Search]:::process
        LLM2[Phi-3.5-mini GGUF]:::model
        Summ[Summarize Case Law Precedents]:::process
    end

    Output((Final WhatsApp / Web UI Response)):::output

    %% Connections
    Input --> LLM1
    LLM1 --> Rewrite
    Rewrite --> BM25
    Rewrite --> SBERT
    BM25 --> DB
    SBERT --> DB
    DB --> Group
    Group --> JSON
    JSON --> Walk
    Walk --> Section
    Section --> DDG
    DDG --> LLM2
    LLM2 --> Summ
    Section --> Output
    Summ --> Output
