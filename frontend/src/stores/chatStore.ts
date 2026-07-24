import { create } from "zustand";
import type { Source, ToolCall, TraceStep } from "../types";
import { streamChat } from "../api/client";

export interface Message {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  tools?: ToolCall[];
  confidence?: number;
  intent?: string;
  streaming?: boolean;
  /** 标记为错误气泡（渲染为告警样式）。 */
  error?: boolean;
}

interface ChatState {
  messages: Message[];
  lastTrace: TraceStep[];
  loading: boolean;
  send: (question: string) => Promise<void>;
  stop: () => void;
  clear: () => void;
}

const TRACE_KEY = "lastTrace";

/** 从 sessionStorage 恢复上次轨迹，避免刷新后 Trace 页空白。 */
function loadTrace(): TraceStep[] {
  try {
    const raw = sessionStorage.getItem(TRACE_KEY);
    return raw ? (JSON.parse(raw) as TraceStep[]) : [];
  } catch {
    return [];
  }
}

function saveTrace(trace: TraceStep[]): void {
  try {
    sessionStorage.setItem(TRACE_KEY, JSON.stringify(trace));
  } catch {
    /* 忽略存储配额等异常。 */
  }
}

// 当前流的 AbortController（不放入 state，避免无谓的重渲染）。
let controller: AbortController | null = null;

export const useChatStore = create<ChatState>((set, get) => ({
  messages: [],
  lastTrace: loadTrace(),
  loading: false,

  send: async (question: string) => {
    // 单飞守卫：上一次请求未结束时忽略新的发送。
    if (get().loading) return;

    const userMsg: Message = { role: "user", content: question };
    const assistantMsg: Message = { role: "assistant", content: "", streaming: true };
    const baseMessages = [...get().messages, userMsg, assistantMsg];
    // 发送时固定助手消息下标，后续更新只作用于该槽位（避免用 length-1 竞态）。
    const assistantIdx = baseMessages.length - 1;
    set({ messages: baseMessages, loading: true });

    const patchAssistant = (patch: Partial<Message>) => {
      const msgs = [...get().messages];
      msgs[assistantIdx] = { ...msgs[assistantIdx], ...patch };
      set({ messages: msgs });
    };

    controller = new AbortController();

    await streamChat(
      question,
      {
        onToken: (token) => {
          const current = get().messages[assistantIdx]?.content ?? "";
          patchAssistant({ content: current + token });
        },
        onDone: (payload) => {
          patchAssistant({
            content: payload.answer,
            sources: payload.sources,
            tools: payload.tools,
            confidence: payload.confidence,
            intent: payload.intent,
            streaming: false,
          });
          set({ lastTrace: payload.trace, loading: false });
          saveTrace(payload.trace);
        },
        onError: (err) => {
          // 保留已流式产出的部分内容，仅在其后追加错误提示。
          const partial = get().messages[assistantIdx]?.content ?? "";
          const note = err.message || "请求失败,请检查后端服务是否启动。";
          patchAssistant({
            content: partial ? `${partial}\n\n${note}` : note,
            streaming: false,
            error: !partial,
          });
          set({ loading: false });
        },
      },
      "default",
      controller.signal
    );

    controller = null;
  },

  stop: () => {
    controller?.abort();
    controller = null;
    set({ loading: false });
  },

  clear: () => {
    set({ messages: [], lastTrace: [] });
    saveTrace([]);
  },
}));
