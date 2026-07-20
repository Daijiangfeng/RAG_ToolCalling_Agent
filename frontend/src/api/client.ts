import axios from "axios";
import type {
  ChatResponse,
  DocumentInfo,
  EvaluationResponse,
  TraceStep,
  Source,
  ToolCall,
  UploadResponse,
} from "../types";

const api = axios.create({ baseURL: "/api", timeout: 60000 });

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
  onError?: (err: unknown) => void;
}

// Stream tokens via fetch + ReadableStream (SSE-style `data:` lines).
export async function streamChat(
  question: string,
  cb: StreamCallbacks,
  sessionId = "default"
): Promise<void> {
  try {
    const resp = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_id: sessionId }),
    });
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
        const json = JSON.parse(line.slice(5).trim());
        if (json.type === "token") cb.onToken(json.content);
        else if (json.type === "done") cb.onDone(json);
      }
    }
  } catch (err) {
    cb.onError?.(err);
  }
}

export default api;
