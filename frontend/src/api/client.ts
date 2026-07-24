import axios, { AxiosError } from "axios";
import type {
  ApiError,
  ChatResponse,
  DocumentInfo,
  EvaluationResponse,
  TraceStep,
  Source,
  ToolCall,
  UploadResponse,
} from "../types";

/** 所有请求共用的基础路径（axios 与原生 fetch 保持一致）。 */
export const API_BASE = "/api";

const api = axios.create({ baseURL: API_BASE, timeout: 60000 });

/** 将任意错误归一化为 ApiError，尽量还原后端 detail。 */
export function toApiError(err: unknown): ApiError {
  if (axios.isAxiosError(err)) {
    const ax = err as AxiosError<{ detail?: string }>;
    const detail = ax.response?.data?.detail;
    return {
      message: detail || ax.message || "请求失败",
      status: ax.response?.status,
    };
  }
  if (err instanceof Error) return { message: err.message };
  return { message: "未知错误" };
}

// 响应拦截器：统一把 axios 错误转换为带可读 message 的 ApiError 后抛出。
api.interceptors.response.use(
  (resp) => resp,
  (error) => Promise.reject(toApiError(error))
);

export async function chat(question: string, sessionId = "default"): Promise<ChatResponse> {
  const { data } = await api.post<ChatResponse>("/chat", { question, session_id: sessionId });
  return data;
}

export async function uploadFile(file: File): Promise<UploadResponse> {
  const form = new FormData();
  form.append("file", file);
  const { data } = await api.post<UploadResponse>("/upload", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

export async function getDocuments(): Promise<DocumentInfo[]> {
  const { data } = await api.get<DocumentInfo[]>("/documents");
  return data;
}

export async function getEvaluation(): Promise<EvaluationResponse> {
  const { data } = await api.get<EvaluationResponse>("/evaluation");
  return data;
}

export async function runEvaluation(): Promise<EvaluationResponse> {
  const { data } = await api.post<EvaluationResponse>("/evaluation/run");
  return data;
}

export interface StreamCallbacks {
  onToken: (token: string) => void;
  onDone: (payload: {
    answer: string;
    sources: Source[];
    confidence: number;
    tools: ToolCall[];
    trace: TraceStep[];
    intent: string;
  }) => void;
  onError?: (err: ApiError) => void;
}

// Stream tokens via fetch + ReadableStream (SSE-style `data:` lines).
export async function streamChat(
  question: string,
  cb: StreamCallbacks,
  sessionId = "default",
  signal?: AbortSignal
): Promise<void> {
  try {
    const resp = await fetch(`${API_BASE}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_id: sessionId }),
      signal,
    });
    // 后端 500 / 502(ProviderError) 时 body 不是 SSE，若不校验会一直卡在流式解析。
    if (!resp.ok) {
      let detail = `请求失败 (HTTP ${resp.status})`;
      try {
        const body = await resp.json();
        if (body?.detail) detail = body.detail;
      } catch {
        /* 非 JSON 响应，沿用默认文案。 */
      }
      cb.onError?.({ message: detail, status: resp.status });
      return;
    }
    if (!resp.body) throw new Error("No response body");
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop() ?? "";
      for (const part of parts) {
        const line = part.trim();
        if (!line.startsWith("data:")) continue;
        try {
          const json = JSON.parse(line.slice(5).trim());
          if (json.type === "token") cb.onToken(json.content);
          else if (json.type === "done") cb.onDone(json);
        } catch {
          // 跳过单条损坏的 SSE 帧，不中断整条流。
          continue;
        }
      }
    }
  } catch (err) {
    // 主动取消（AbortController）不视为错误。
    if (err instanceof DOMException && err.name === "AbortError") return;
    cb.onError?.(toApiError(err));
  }
}

export default api;
