import { useEffect, useState } from "react";
import {
  Card,
  Row,
  Col,
  Statistic,
  Button,
  Table,
  message,
  Typography,
  Progress,
  Space,
} from "antd";
import { PlayCircleOutlined, ReloadOutlined } from "@ant-design/icons";
import { getEvaluation, runEvaluation } from "../api/client";
import type { EvaluationResponse } from "../types";

const { Text } = Typography;

function pct(v: number): number {
  return Math.round((v ?? 0) * 100);
}

export default function EvaluationPage() {
  const [data, setData] = useState<EvaluationResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);

  const refresh = async () => {
    setLoading(true);
    try {
      setData(await getEvaluation());
    } catch {
      message.info("暂无评估结果,请点击“运行评估”生成");
    } finally {
      setLoading(false);
    }
  };

  const run = async () => {
    setRunning(true);
    try {
      setData(await runEvaluation());
      message.success("评估完成");
    } catch {
      message.error("运行评估失败,请确认后端已启动");
    } finally {
      setRunning(false);
    }
  };

  useEffect(() => {
    refresh();
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
    { title: "数量", dataIndex: "count", width: 90 },
    { title: "通过数", dataIndex: "passed", width: 90 },
    {
      title: "忠实度",
      dataIndex: "faithfulness",
      width: 160,
      render: (v: number) => <Progress percent={pct(v)} size="small" />,
    },
  ];

  return (
    <div style={{ maxWidth: 1000, margin: "0 auto" }}>
      <Card
        title="RAG 自动评估看板"
        style={{ marginBottom: 16 }}
        extra={
          <Space>
            <Button icon={<ReloadOutlined />} onClick={refresh} loading={loading}>
              刷新
            </Button>
            <Button
              type="primary"
              icon={<PlayCircleOutlined />}
              onClick={run}
              loading={running}
            >
              运行评估
            </Button>
          </Space>
        }
      >
        {data ? (
          <>
            <Row gutter={16}>
              <Col span={6}>
                <Card size="small">
                  <Statistic title="检索精确率 Precision@K" value={pct(data.retrieval.precision_at_k)} suffix="%" />
                </Card>
              </Col>
              <Col span={6}>
                <Card size="small">
                  <Statistic title="检索召回率 Recall@K" value={pct(data.retrieval.recall_at_k)} suffix="%" />
                </Card>
              </Col>
              <Col span={6}>
                <Card size="small">
                  <Statistic title="忠实度 Faithfulness" value={pct(data.generation.faithfulness)} suffix="%" />
                </Card>
              </Col>
              <Col span={6}>
                <Card size="small">
                  <Statistic
                    title="幻觉率"
                    value={pct(data.safety.hallucination_rate)}
                    suffix="%"
                    valueStyle={{ color: "#cf1322" }}
                  />
                </Card>
              </Col>
            </Row>

            <Row gutter={16} style={{ marginTop: 16 }}>
              <Col span={8}>
                <Card size="small">
                  <Statistic title="答案相关性 Answer Relevance" value={pct(data.generation.answer_relevance)} suffix="%" />
                </Card>
              </Col>
              <Col span={8}>
                <Card size="small">
                  <Statistic title="上下文相关性 Context Relevance" value={pct(data.generation.context_relevance)} suffix="%" />
                </Card>
              </Col>
              <Col span={8}>
                <Card size="small">
                  <Statistic title="测试样本数" value={data.total} />
                </Card>
              </Col>
            </Row>

            <Card size="small" title="分类别指标" style={{ marginTop: 16 }}>
              <Table
                columns={columns}
                dataSource={perTypeRows}
                pagination={false}
                size="small"
              />
            </Card>

            {data.generated_at && (
              <Text type="secondary" style={{ display: "block", marginTop: 12 }}>
                生成时间:{data.generated_at}
              </Text>
            )}
          </>
        ) : (
          <Text type="secondary">暂无评估数据,请点击“运行评估”。</Text>
        )}
      </Card>
    </div>
  );
}
