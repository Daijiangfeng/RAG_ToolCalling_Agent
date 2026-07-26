// Shared types mirroring the backend API contracts.

// 归一化后的 API 错误：始终携带一个可展示给用户的 message，
// 尽量还原后端返回的 detail（如 502 ProviderError 的友好文案）。
export interface ApiError {
  message: string;
  status?: number;
}

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
  // 同名文档被覆盖替换时为 true，首次入库为 false/缺省。
  replaced?: boolean;
}

export interface DocumentInfo {
  id: number;
  file_name: string;
  pages: number;
  chunks: number;
  status: string;
  created_time?: string | null;
}

export interface HealthResponse {
  status: string;
  llm_mode: string;
  vector_count: number;
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
