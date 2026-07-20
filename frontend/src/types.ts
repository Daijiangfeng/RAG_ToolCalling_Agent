// Shared types mirroring the backend API contracts.

export interface Source {
  text: string;
  score: number;
  metadata: Record<string, any>;
}

export interface TraceStep {
  step: string;
  summary: string;
  tool?: string | null;
  data?: Record<string, any>;
}

export interface ToolCall {
  tool: string;
  input: Record<string, any>;
  output: any;
}

export interface ChatResponse {
  answer: string;
  sources: Source[];
  confidence: number;
  tools: ToolCall[];
  trace: TraceStep[];
  intent: string;
}

export interface UploadResponse {
  filename: string;
  pages: number;
  chunks: number;
  status: string;
}

export interface DocumentInfo {
  id: number;
  file_name: string;
  pages: number;
  chunks: number;
  status: string;
  created_time?: string | null;
}

export interface EvaluationResponse {
  total: number;
  retrieval: { precision_at_k: number; recall_at_k: number };
  generation: {
    answer_relevance: number;
    context_relevance: number;
    faithfulness: number;
  };
  safety: { hallucination_rate: number };
  per_type: Record<string, { count: number; passed: number; faithfulness: number }>;
  generated_at?: string | null;
}
