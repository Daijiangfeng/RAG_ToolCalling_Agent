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
}

interface ChatState {
  messages: Message[];
  lastTrace: TraceStep[];
  loading: boolean;
  send: (question: string) => Promise<void>;
  clear: () => void;
}

export const useChatStore = create<ChatState>((set, get) => ({
  messages: [],
  lastTrace: [],
  loading: false,

  send: async (question: string) => {
    const userMsg: Message = { role: "user", content: question };
    const assistantMsg: Message = { role: "assistant", content: "", streaming: true };
    set({ messages: [...get().messages, userMsg, assistantMsg], loading: true });

    const updateLast = (patch: Partial<Message>) => {
      const msgs = [...get().messages];
      const idx = msgs.length - 1;
      msgs[idx] = { ...msgs[idx], ...patch };
      set({ messages: msgs });
    };

    await streamChat(question, {
      onToken: (token) => {
        const msgs = get().messages;
        const idx = msgs.length - 1;
        updateLast({ content: msgs[idx].content + token });
      },
      onDone: (payload) => {
        updateLast({
          content: payload.answer,
          sources: payload.sources,
          tools: payload.tools,
          confidence: payload.confidence,
          intent: payload.intent,
          streaming: false,
        });
        set({ lastTrace: payload.trace, loading: false });
      },
      onError: () => {
        updateLast({ content: "请求失败,请检查后端服务是否启动。", streaming: false });
        set({ loading: false });
      },
    });
  },

  clear: () => set({ messages: [], lastTrace: [] }),
}));
