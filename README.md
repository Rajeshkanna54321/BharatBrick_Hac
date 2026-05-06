graph TD
    %% Define Styles
    classDef user fill:#e1f5fe,stroke:#01579b,stroke-width:2px;
    classDef model fill:#f3e5f5,stroke:#4a148c,stroke-width:2px;
    classDef process fill:#fff3e0,stroke:#e65100,stroke-width:2px;
    classDef data fill:#e8f5e9,stroke:#1b5e20,stroke-width:2px;
    classDef output fill:#ffebee,stroke:#b71c1c,stroke-width:3px;

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
