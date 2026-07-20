import { useState } from "react";
import {
  Button,
  Card,
  Input,
  Space,
  Tag,
  Typography,
  Progress,
  Empty,
  List,
} from "antd";
import { SendOutlined, ToolOutlined } from "@ant-design/icons";
import ReactMarkdown from "react-markdown";
import { useChatStore } from "../stores/chatStore";

const { Text, Paragraph } = Typography;

function confidenceColor(c: number): string {
  if (c >= 0.6) return "#52c41a";
  if (c >= 0.3) return "#faad14";
  return "#ff4d4f";
}

export default function ChatPage() {
  const { messages, loading, send } = useChatStore();
  const [input, setInput] = useState("");

  const onSend = async () => {
    const q = input.trim();
    if (!q || loading) return;
    setInput("");
    await send(q);
  };

  return (
    <div style={{ maxWidth: 900, margin: "0 auto" }}>
      <Card
        title="智能问答 (RAG + Tool Calling)"
        style={{ marginBottom: 16, minHeight: 420 }}
      >
        {messages.length === 0 && (
          <Empty description="试试:RAG 的优势是什么?  /  12345*678  /  最新的 AI 新闻" />
        )}
        <Space direction="vertical" style={{ width: "100%" }} size="middle">
          {messages.map((m, i) => (
            <div key={i} style={{ textAlign: m.role === "user" ? "right" : "left" }}>
              <Card
                size="small"
                style={{
                  display: "inline-block",
                  maxWidth: "88%",
                  textAlign: "left",
                  background: m.role === "user" ? "#e6f4ff" : "#fafafa",
                }}
              >
                {m.role === "assistant" ? (
                  <>
                    <ReactMarkdown>{m.content || (m.streaming ? "▍" : "")}</ReactMarkdown>

                    {typeof m.confidence === "number" && !m.streaming && (
                      <div style={{ marginTop: 8 }}>
                        <Text type="secondary">置信度</Text>
                        <Progress
                          percent={Math.round(m.confidence * 100)}
                          size="small"
                          strokeColor={confidenceColor(m.confidence)}
                        />
                        {m.intent && <Tag color="blue">意图: {m.intent}</Tag>}
                      </div>
                    )}

                    {m.tools && m.tools.length > 0 && (
                      <Card
                        size="small"
                        title={<><ToolOutlined /> 工具调用</>}
                        style={{ marginTop: 8 }}
                      >
                        {m.tools.map((t, ti) => (
                          <div key={ti}>
                            <Tag color="geekblue">{t.tool}</Tag>
                            <Text code>{JSON.stringify(t.input)}</Text>
                            <Paragraph type="secondary" style={{ margin: "4px 0" }}>
                              {JSON.stringify(t.output)}
                            </Paragraph>
                          </div>
                        ))}
                      </Card>
                    )}

                    {m.sources && m.sources.length > 0 && (
                      <Card size="small" title="来源引用" style={{ marginTop: 8 }}>
                        <List
                          size="small"
                          dataSource={m.sources}
                          renderItem={(s, si) => (
                            <List.Item>
                              <div>
                                <Tag color="green">[来源 {si + 1}] 相关度 {s.score}</Tag>
                                <Text type="secondary">
                                  {s.metadata?.file_name} · 第 {s.metadata?.page_number} 页
                                </Text>
                                <Paragraph style={{ marginTop: 4 }} ellipsis={{ rows: 2 }}>
                                  {s.text}
                                </Paragraph>
                              </div>
                            </List.Item>
                          )}
                        />
                      </Card>
                    )}
                  </>
                ) : (
                  <Text>{m.content}</Text>
                )}
              </Card>
            </div>
          ))}
        </Space>
      </Card>

      <Space.Compact style={{ width: "100%" }}>
        <Input
          size="large"
          placeholder="输入你的问题..."
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onPressEnter={onSend}
          disabled={loading}
        />
        <Button
          type="primary"
          size="large"
          icon={<SendOutlined />}
          onClick={onSend}
          loading={loading}
        >
          发送
        </Button>
      </Space.Compact>
    </div>
  );
}
