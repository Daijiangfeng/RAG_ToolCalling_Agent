import { useEffect, useState } from "react";
import { Button, Table, App, Typography, Progress, Space, Empty } from "antd";
import { PlayCircleOutlined, ReloadOutlined } from "@ant-design/icons";
import { getEvaluation, runEvaluation, toApiError } from "../api/client";
import type { EvaluationResponse } from "../types";
import { formatDateTime } from "../utils/format";
import PageContainer from "../components/PageContainer";
import SectionCard from "../components/SectionCard";

const { Text } = Typography;

function pct(v: number): number {
  return Math.round((v ?? 0) * 100);
}

/** hairline 指标卡：小型大写标签 + mono 数值。danger 时数值用语义色。 */
function MetricCard({
  label,
  value,
  danger,
}: {
  label: string;
  value: string;
  danger?: boolean;
}) {
  return (
    <div className="hairline" style={{ padding: 16 }}>
      <div className="caption-upper" style={{ color: "var(--text-muted)" }}>
        {label}
      </div>
      <div
        className="mono"
        style={{
          fontSize: 24,
          marginTop: 6,
          color: danger ? "var(--danger)" : "var(--text-primary)",
        }}
      >
        {value}
      </div>
    </div>
  );
}

export default function EvaluationPage() {
  const { message } = App.useApp();
  const [data, setData] = useState<EvaluationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);

  const refresh = async () => {
    setLoading(true);
    try {
      setData(await getEvaluation());
    } catch (err) {
      // 区分“暂无数据”与“真实错误”：仅在确有错误时告警。
      message.error(toApiError(err).message || "获取评估结果失败,请确认后端已启动");
    } finally {
      setLoading(false);
    }
  };

  const run = async () => {
    setRunning(true);
    try {
      setData(await runEvaluation());
      message.success("评估完成");
    } catch (err) {
      message.error(toApiError(err).message || "运行评估失败,请确认后端已启动");
    } finally {
      setRunning(false);
    }
  };

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const perTypeRows = data
    ? Object.entries(data.per_type).map(([type, v]) => ({
        key: type,
        type,
        count: v.count,
        passed: v.passed,
        faithfulness: v.faithfulness,
      }))
    : [];

  const columns = [
    { title: "问题类型", dataIndex: "type" },
    { title: "数量", dataIndex: "count", width: 90, render: (v: number) => <span className="mono">{v}</span> },
    { title: "通过数", dataIndex: "passed", width: 90, render: (v: number) => <span className="mono">{v}</span> },
    {
      title: "忠实度",
      dataIndex: "faithfulness",
      width: 180,
      render: (v: number) => (
        <Progress percent={pct(v)} size="small" strokeColor="var(--accent)" />
      ),
    },
  ];

  const metricGrid: React.CSSProperties = {
    display: "grid",
    gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
    gap: 12,
  };

  return (
    <div style={{ overflowY: "auto" }}>
      <PageContainer maxWidth={1000}>
        <div
          style={{
            padding: "40px 0 24px",
            display: "flex",
            alignItems: "flex-end",
            justifyContent: "space-between",
            flexWrap: "wrap",
            gap: 12,
          }}
        >
          <div>
            <h1 className="display-lg">评估</h1>
            <p style={{ color: "var(--text-secondary)", margin: "10px 0 0", fontSize: 14 }}>
              RAG 检索与生成质量的自动评估看板
            </p>
          </div>
          <Space>
            <Button icon={<ReloadOutlined />} onClick={refresh} loading={loading}>
              刷新
            </Button>
            <Button type="primary" icon={<PlayCircleOutlined />} onClick={run} loading={running}>
              运行评估
            </Button>
          </Space>
        </div>

        {data ? (
          <>
            <div style={metricGrid}>
              <MetricCard label="Precision@K" value={`${pct(data.retrieval.precision_at_k)}%`} />
              <MetricCard label="Recall@K" value={`${pct(data.retrieval.recall_at_k)}%`} />
              <MetricCard label="Faithfulness" value={`${pct(data.generation.faithfulness)}%`} />
              <MetricCard
                label="幻觉率"
                value={`${pct(data.safety.hallucination_rate)}%`}
                danger
              />
            </div>

            <div style={{ ...metricGrid, marginTop: 12 }}>
              <MetricCard
                label="Answer Relevance"
                value={`${pct(data.generation.answer_relevance)}%`}
              />
              <MetricCard
                label="Context Relevance"
                value={`${pct(data.generation.context_relevance)}%`}
              />
              <MetricCard label="测试样本数" value={String(data.total)} />
            </div>

            <SectionCard title="分类别指标" style={{ marginTop: 16 }}>
              <Table
                columns={columns}
                dataSource={perTypeRows}
                pagination={false}
                size="small"
                scroll={{ x: "max-content" }}
              />
            </SectionCard>

            {data.generated_at && (
              <Text
                type="secondary"
                className="mono"
                style={{ display: "block", marginTop: 12, fontSize: 12 }}
              >
                生成时间:{formatDateTime(data.generated_at)}
              </Text>
            )}
          </>
        ) : (
          <SectionCard>
            <Empty description="暂无评估数据,请点击“运行评估”生成" />
          </SectionCard>
        )}
      </PageContainer>
    </div>
  );
}
