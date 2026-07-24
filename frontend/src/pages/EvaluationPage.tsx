import { useEffect, useState } from "react";
import { Row, Col, Statistic, Button, Table, App, Typography, Progress, Space, Empty } from "antd";
import { PlayCircleOutlined, ReloadOutlined } from "@ant-design/icons";
import { getEvaluation, runEvaluation, toApiError } from "../api/client";
import type { EvaluationResponse } from "../types";
import { STATUS } from "../theme/tokens";
import { formatDateTime } from "../utils/format";
import PageContainer from "../components/PageContainer";
import SectionCard from "../components/SectionCard";

const { Text } = Typography;

function pct(v: number): number {
  return Math.round((v ?? 0) * 100);
}

// 统计卡在小屏下堆叠、大屏均分。
const STAT_COL = { xs: 24, sm: 12, md: 6 };
const STAT_COL_3 = { xs: 24, sm: 12, md: 8 };

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
    <PageContainer maxWidth={1000}>
      <SectionCard
        title="RAG 自动评估看板"
        style={{ marginBottom: 16 }}
        extra={
          <Space>
            <Button icon={<ReloadOutlined />} onClick={refresh} loading={loading}>
              刷新
            </Button>
            <Button type="primary" icon={<PlayCircleOutlined />} onClick={run} loading={running}>
              运行评估
            </Button>
          </Space>
        }
      >
        {data ? (
          <>
            <Row gutter={[16, 16]}>
              <Col {...STAT_COL}>
                <SectionCard size="small">
                  <Statistic
                    title="检索精确率 Precision@K"
                    value={pct(data.retrieval.precision_at_k)}
                    suffix="%"
                  />
                </SectionCard>
              </Col>
              <Col {...STAT_COL}>
                <SectionCard size="small">
                  <Statistic
                    title="检索召回率 Recall@K"
                    value={pct(data.retrieval.recall_at_k)}
                    suffix="%"
                  />
                </SectionCard>
              </Col>
              <Col {...STAT_COL}>
                <SectionCard size="small">
                  <Statistic
                    title="忠实度 Faithfulness"
                    value={pct(data.generation.faithfulness)}
                    suffix="%"
                  />
                </SectionCard>
              </Col>
              <Col {...STAT_COL}>
                <SectionCard size="small">
                  <Statistic
                    title="幻觉率"
                    value={pct(data.safety.hallucination_rate)}
                    suffix="%"
                    valueStyle={{ color: STATUS.danger }}
                  />
                </SectionCard>
              </Col>
            </Row>

            <Row gutter={[16, 16]} style={{ marginTop: 16 }}>
              <Col {...STAT_COL_3}>
                <SectionCard size="small">
                  <Statistic
                    title="答案相关性 Answer Relevance"
                    value={pct(data.generation.answer_relevance)}
                    suffix="%"
                  />
                </SectionCard>
              </Col>
              <Col {...STAT_COL_3}>
                <SectionCard size="small">
                  <Statistic
                    title="上下文相关性 Context Relevance"
                    value={pct(data.generation.context_relevance)}
                    suffix="%"
                  />
                </SectionCard>
              </Col>
              <Col {...STAT_COL_3}>
                <SectionCard size="small">
                  <Statistic title="测试样本数" value={data.total} />
                </SectionCard>
              </Col>
            </Row>

            <SectionCard size="small" title="分类别指标" style={{ marginTop: 16 }}>
              <Table
                columns={columns}
                dataSource={perTypeRows}
                pagination={false}
                size="small"
                scroll={{ x: "max-content" }}
              />
            </SectionCard>

            {data.generated_at && (
              <Text type="secondary" style={{ display: "block", marginTop: 12 }}>
                生成时间:{formatDateTime(data.generated_at)}
              </Text>
            )}
          </>
        ) : (
          <Empty description="暂无评估数据,请点击“运行评估”生成" />
        )}
      </SectionCard>
    </PageContainer>
  );
}
