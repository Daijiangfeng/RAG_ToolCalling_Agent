import { useEffect, useRef, useState } from "react";
import { Button, Input, Space, Empty } from "antd";
import { SendOutlined } from "@ant-design/icons";
import { useChatStore } from "../stores/chatStore";
import PageContainer from "../components/PageContainer";
import SectionCard from "../components/SectionCard";
import MessageItem from "../components/MessageItem";

export default function ChatPage() {
  // 使用细粒度 selector，避免每个 token 触发整页重渲染。
  const messages = useChatStore((s) => s.messages);
  const loading = useChatStore((s) => s.loading);
  const send = useChatStore((s) => s.send);
  const stop = useChatStore((s) => s.stop);
  const [input, setInput] = useState("");
  const bottomRef = useRef<HTMLDivElement>(null);

  // 新消息或流式追加时自动滚动到底部。
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const onSend = async () => {
    const q = input.trim();
    if (!q || loading) return;
    setInput("");
    await send(q);
  };

  return (
    <PageContainer maxWidth={900}>
      <SectionCard title="智能问答 (RAG + Tool Calling)" style={{ marginBottom: 16, minHeight: 420 }}>
        {messages.length === 0 && (
          <Empty description="试试:RAG 的优势是什么?  /  12345*678  /  最新的 AI 新闻" />
        )}
        <Space direction="vertical" style={{ width: "100%" }} size="middle">
          {messages.map((m, i) => (
            <MessageItem key={i} message={m} />
          ))}
          <div ref={bottomRef} />
        </Space>
      </SectionCard>

      <Space.Compact style={{ width: "100%" }}>
        <Input
          size="large"
          aria-label="输入问题"
          placeholder="输入你的问题..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            // 避免中文输入法回车确认候选词时误触发送。
            if (e.key === "Enter" && !e.nativeEvent.isComposing) {
              e.preventDefault();
              onSend();
            }
          }}
          disabled={loading}
        />
        {loading ? (
          <Button size="large" danger onClick={stop}>
            停止
          </Button>
        ) : (
          <Button type="primary" size="large" icon={<SendOutlined />} onClick={onSend}>
            发送
          </Button>
        )}
      </Space.Compact>
    </PageContainer>
  );
}
