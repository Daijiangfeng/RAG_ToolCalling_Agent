import { useEffect, useState } from "react";
import {
  Card,
  Upload,
  Table,
  Tag,
  message,
  Typography,
  Button,
  Space,
} from "antd";
import { InboxOutlined, ReloadOutlined } from "@ant-design/icons";
import type { UploadProps } from "antd";
import { getDocuments, uploadFile } from "../api/client";
import type { DocumentInfo } from "../types";

const { Dragger } = Upload;
const { Text } = Typography;

function statusColor(status: string): string {
  if (status === "indexed" || status === "ready") return "green";
  if (status === "processing") return "blue";
  return "default";
}

export default function KnowledgeBasePage() {
  const [docs, setDocs] = useState<DocumentInfo[]>([]);
  const [loading, setLoading] = useState(false);

  const refresh = async () => {
    setLoading(true);
    try {
      setDocs(await getDocuments());
    } catch {
      message.error("获取知识库列表失败,请确认后端已启动");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    refresh();
  }, []);

  const draggerProps: UploadProps = {
    name: "file",
    multiple: false,
    accept: ".pdf,.md,.markdown,.txt",
    showUploadList: false,
    customRequest: async (options) => {
      const file = options.file as File;
      try {
        const res = await uploadFile(file);
        message.success(
          `已入库 ${res.filename}:${res.pages} 页 / ${res.chunks} chunks`
        );
        options.onSuccess?.(res);
        refresh();
      } catch (err) {
        message.error("上传失败");
        options.onError?.(err as Error);
      }
    },
  };

  const columns = [
    { title: "ID", dataIndex: "id", width: 70 },
    { title: "文件名", dataIndex: "file_name" },
    { title: "页数", dataIndex: "pages", width: 90 },
    { title: "分块数", dataIndex: "chunks", width: 100 },
    {
      title: "状态",
      dataIndex: "status",
      width: 120,
      render: (s: string) => <Tag color={statusColor(s)}>{s}</Tag>,
    },
    {
      title: "创建时间",
      dataIndex: "created_time",
      render: (t?: string | null) => <Text type="secondary">{t ?? "-"}</Text>,
    },
  ];

  return (
    <div style={{ maxWidth: 1000, margin: "0 auto" }}>
      <Card title="上传文档 (PDF / Markdown)" style={{ marginBottom: 16 }}>
        <Dragger {...draggerProps}>
          <p className="ant-upload-drag-icon">
            <InboxOutlined />
          </p>
          <p className="ant-upload-text">点击或拖拽文件到此区域上传</p>
          <p className="ant-upload-hint">
            支持 PDF / Markdown,上传后自动解析 → 切分 → 向量化 → 入库
          </p>
        </Dragger>
      </Card>

      <Card
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
        />
      </Card>
    </div>
  );
}
