import { useEffect, useRef, useState } from "react";
import { Button, Input } from "antd";
import {
  SendOutlined,
  StopOutlined,
  LayoutOutlined,
  ClearOutlined,
} from "@ant-design/icons";
import { useChatStore } from "../stores/chatStore";
import MessageItem from "../components/MessageItem";
import AgentTimeline from "../components/AgentTimeline";

/** 空态示例 prompt（点击即发送）。 */
const SAMPLE_PROMPTS = [
  "RAG 的优势是什么？",
  "12345 * 678 等于多少？",
  "最新的 AI 新闻有哪些？",
];

/**
 * AI 工作区（核心页）：左侧消息流 + 底部 Composer + 右侧可折叠 Inspector
 * 面板（Agent 执行时间线）。
 */
export default function ChatPage() {
  // 使用细粒度 selector，避免每个 token 触发整页重渲染。
  const messages = useChatStore((s) => s.messages);
  const loading = useChatStore((s) => s.loading);
  const send = useChatStore((s) => s.send);
  const stop = useChatStore((s) => s.stop);
  const clear = useChatStore((s) => s.clear);
  const [input, setInput] = useState("");
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const bottomRef = useRef<HTMLDivElement>(null);

  // 新消息或流式追加时自动滚动到底部。
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const onSend = async (text?: string) => {
    const q = (text ?? input).trim();
    if (!q || loading) return;
    setInput("");
    await send(q);
  };

  return (
    <div style={{ flex: 1, minHeight: 0, display: "flex" }}>
      {/* 主区：消息流 + Composer */}
      <section
        style={{
          flex: 1,
          minWidth: 0,
          display: "flex",
          flexDirection: "column",
          height: "calc(100vh - 44px)",
        }}
      >
        {/* 消息流 */}
        <div style={{ flex: 1, overflowY: "auto" }}>
          <div style={{ maxWidth: 820, margin: "0 auto", padding: "8px 24px" }}>
            {messages.length === 0 ? (
              // 空态：Cursor 式居中排版（display 400 字重 + 示例 prompt 卡片）。
              <div style={{ textAlign: "center", paddingTop: "18vh" }}>
                <h1 className="display-lg">有什么可以帮你？</h1>
                <p style={{ color: "var(--text-secondary)", margin: "12px 0 32px", fontSize: 14 }}>
                  基于知识库检索、工具调用与 LangGraph Agent 的智能问答
                </p>
                <div
                  style={{
                    display: "grid",
                    gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
                    gap: 12,
                    maxWidth: 640,
                    margin: "0 auto",
                  }}
                >
                  {SAMPLE_PROMPTS.map((p) => (
                    <button key={p} className="prompt-card" onClick={() => onSend(p)}>
                      {p}
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              messages.map((m, i) => <MessageItem key={i} message={m} />)
            )}
            <div ref={bottomRef} />
          </div>
        </div>

        {/* Composer：固定底部输入区 */}
        <div className="hairline-top" style={{ flex: "none", background: "var(--app-bg)" }}>
          <div style={{ maxWidth: 820, margin: "0 auto", padding: "12px 24px 16px" }}>
            <div
              className="hairline"
              style={{ display: "flex", alignItems: "flex-end", gap: 8, padding: 8 }}
            >
              <Input.TextArea
                aria-label="输入问题"
                placeholder="输入你的问题...（Enter 发送，Shift+Enter 换行）"
                autoSize={{ minRows: 1, maxRows: 6 }}
                variant="borderless"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  // 避免中文输入法回车确认候选词时误触发送；Shift+Enter 换行。
                  if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                    e.preventDefault();
                    onSend();
                  }
                }}
                disabled={loading}
              />
              {loading ? (
                <Button danger icon={<StopOutlined />} onClick={stop}>
                  停止
                </Button>
              ) : (
                <Button type="primary" icon={<SendOutlined />} onClick={() => onSend()}>
                  发送
                </Button>
              )}
            </div>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginTop: 6,
              }}
            >
              <Button
                type="text"
                size="small"
                icon={<ClearOutlined />}
                style={{ color: "var(--text-muted)", fontSize: 12 }}
                onClick={clear}
                disabled={loading || messages.length === 0}
              >
                清空对话
              </Button>
              <Button
                type="text"
                size="small"
                icon={<LayoutOutlined />}
                style={{ color: "var(--text-muted)", fontSize: 12 }}
                onClick={() => setInspectorOpen((v) => !v)}
              >
                {inspectorOpen ? "收起轨迹面板" : "展开轨迹面板"}
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* 右侧 Inspector 面板：Agent 执行时间线 */}
      {inspectorOpen && (
        <aside
          style={{
            width: 320,
            flex: "none",
            borderLeft: "1px solid var(--border)",
            background: "var(--panel)",
            height: "calc(100vh - 44px)",
            overflowY: "auto",
            boxSizing: "border-box",
            padding: 16,
          }}
        >
          <div className="caption-upper" style={{ color: "var(--text-muted)", marginBottom: 16 }}>
            Agent 执行轨迹
          </div>
          <AgentTimeline />
        </aside>
      )}
    </div>
  );
}
