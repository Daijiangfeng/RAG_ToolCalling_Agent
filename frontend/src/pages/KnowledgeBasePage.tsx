import { useEffect, useState } from "react";
import { Upload, Table, App, Typography, Button, Space } from "antd";
import { InboxOutlined, ReloadOutlined } from "@ant-design/icons";
import type { UploadProps } from "antd";
import { getDocuments, uploadFile, toApiError } from "../api/client";
import type { DocumentInfo } from "../types";
import { formatDateTime } from "../utils/format";
import PageContainer from "../components/PageContainer";
import SectionCard from "../components/SectionCard";

const { Dragger } = Upload;
const { Text } = Typography;

const ACCEPTED = ".pdf,.md,.markdown,.txt";

/** 文档状态 → badge-pill 语义色（颜色 + 文本双重表达）。 */
function statusDotColor(status: string): string {
  if (status === "indexed" || status === "ready" || status === "processed") return "var(--success)";
  if (status === "processing") return "var(--warning)";
  return "var(--text-muted)";
}

export default function KnowledgeBasePage() {
  const { message } = App.useApp();
  const [docs, setDocs] = useState<DocumentInfo[]>([]);
  const [loading, setLoading] = useState(false);

  const refresh = async () => {
    setLoading(true);
    try {
      setDocs(await getDocuments());
    } catch (err) {
      message.error(toApiError(err).message || "获取知识库列表失败,请确认后端已启动");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const draggerProps: UploadProps = {
    name: "file",
    multiple: false,
    accept: ACCEPTED,
    showUploadList: false,
    customRequest: async (options) => {
      const file = options.file as File;
      try {
        const res = await uploadFile(file);
        const action = res.replaced ? "已更新" : "已入库";
        message.success(`${action} ${res.filename}:${res.pages} 页 / ${res.chunks} chunks`);
        options.onSuccess?.(res);
        refresh();
      } catch (err) {
        message.error(toApiError(err).message || "上传失败");
        options.onError?.(err as Error);
      }
    },
  };

  const columns = [
    { title: "ID", dataIndex: "id", width: 70, render: (v: number) => <span className="mono">{v}</span> },
    { title: "文件名", dataIndex: "file_name" },
    { title: "页数", dataIndex: "pages", width: 90, render: (v: number) => <span className="mono">{v}</span> },
    { title: "分块数", dataIndex: "chunks", width: 100, render: (v: number) => <span className="mono">{v}</span> },
    {
      title: "状态",
      dataIndex: "status",
      width: 130,
      // 颜色 + 文本双重表达状态，避免仅靠颜色传达信息。
      render: (s: string) => (
        <span className="badge-pill mono">
          <span className="status-dot" style={{ background: statusDotColor(s) }} />
          {s}
        </span>
      ),
    },
    {
      title: "创建时间",
      dataIndex: "created_time",
      render: (t?: string | null) => (
        <Text type="secondary" className="mono" style={{ fontSize: 12 }}>
          {formatDateTime(t)}
        </Text>
      ),
    },
  ];

  return (
    <div style={{ overflowY: "auto" }}>
      <PageContainer maxWidth={1000}>
        <div style={{ padding: "40px 0 24px" }}>
          <h1 className="display-lg">知识库</h1>
          <p style={{ color: "var(--text-secondary)", margin: "10px 0 0", fontSize: 14 }}>
            上传后自动解析 → 切分 → 向量化 → 入库
          </p>
        </div>
        <SectionCard title="上传文档 (PDF / Markdown / 文本)" style={{ marginBottom: 16 }}>
          <Dragger {...draggerProps}>
            <p className="ant-upload-drag-icon">
              <InboxOutlined style={{ color: "var(--accent)" }} />
            </p>
            <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
            <p className="ant-upload-hint">
              支持 PDF / Markdown / TXT，上传后自动解析 → 切分 → 向量化 → 入库
            </p>
          </Dragger>
        </SectionCard>

        <SectionCard
          title="知识库文档"
          extra={
            <Space>
              <Button icon={<ReloadOutlined />} onClick={refresh} loading={loading}>
                刷新
              </Button>
            </Space>
          }
        >
          <Table
            rowKey="id"
            dataSource={docs}
            columns={columns}
            loading={loading}
            pagination={{ pageSize: 8 }}
            scroll={{ x: "max-content" }}
          />
        </SectionCard>
      </PageContainer>
    </div>
  );
}
