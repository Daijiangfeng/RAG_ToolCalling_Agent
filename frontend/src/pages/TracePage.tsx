import { Timeline, Tag, Typography, Empty } from "antd";
import {
  BranchesOutlined,
  SearchOutlined,
  SortAscendingOutlined,
  ToolOutlined,
  EditOutlined,
  StopOutlined,
  BulbOutlined,
  CheckCircleOutlined,
} from "@ant-design/icons";
import { useChatStore } from "../stores/chatStore";
import type { TraceStep } from "../types";
import PageContainer from "../components/PageContainer";
import SectionCard from "../components/SectionCard";

const { Text, Paragraph } = Typography;

// Map each agent step to a Chinese display label.
const STEP_LABELS: Record<string, string> = {
  intent_router: "意图路由",
  rewrite: "查询改写",
  retrieve: "检索",
  rerank: "重排序",
  tool: "工具调用",
  reject: "拒答",
  generate: "生成",
  critique: "自我校验",
};

// Map each agent step to an icon + color for the timeline.
function stepStyle(step: string): { icon: React.ReactNode; color: string } {
  switch (step) {
    case "intent_router":
      return { icon: <BranchesOutlined />, color: "blue" };
    case "rewrite":
      return { icon: <EditOutlined />, color: "purple" };
    case "retrieve":
      return { icon: <SearchOutlined />, color: "cyan" };
    case "rerank":
      return { icon: <SortAscendingOutlined />, color: "geekblue" };
    case "tool":
      return { icon: <ToolOutlined />, color: "orange" };
    case "reject":
      return { icon: <StopOutlined />, color: "red" };
    case "generate":
      return { icon: <BulbOutlined />, color: "green" };
    case "critique":
      return { icon: <CheckCircleOutlined />, color: "green" };
    default:
      return { icon: <BranchesOutlined />, color: "gray" };
  }
}

export default function TracePage() {
  const lastTrace = useChatStore((s) => s.lastTrace);

  return (
    <PageContainer maxWidth={900}>
      <SectionCard title="Agent 执行轨迹 (LangGraph State → Node → Edge)">
        {lastTrace.length === 0 ? (
          <Empty description="尚无轨迹,请先在对话页发起一次提问" />
        ) : (
          <Timeline
            items={lastTrace.map((t: TraceStep) => {
              const s = stepStyle(t.step);
              return {
                color: s.color,
                dot: s.icon,
                children: (
                  <div>
                    <Text strong>{STEP_LABELS[t.step] ?? t.step}</Text>
                    {t.tool && (
                      <Tag color="orange" style={{ marginLeft: 8 }}>
                        {t.tool}
                      </Tag>
                    )}
                    <Paragraph type="secondary" style={{ margin: "4px 0" }}>
                      {t.summary}
                    </Paragraph>
                    {t.data && Object.keys(t.data).length > 0 && (
                      <Text code style={{ fontSize: 12 }}>
                        {JSON.stringify(t.data)}
                      </Text>
                    )}
                  </div>
                ),
              };
            })}
          />
        )}
      </SectionCard>
    </PageContainer>
  );
}
