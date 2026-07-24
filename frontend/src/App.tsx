import { lazy, Suspense } from "react";
import { Layout, Menu, Spin, Typography } from "antd";
import {
  MessageOutlined,
  NodeIndexOutlined,
  DatabaseOutlined,
  BarChartOutlined,
} from "@ant-design/icons";
import { Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";

// 路由级代码分割：仅按需加载各页面，减小首屏（对话页）体积。
const ChatPage = lazy(() => import("./pages/ChatPage"));
const TracePage = lazy(() => import("./pages/TracePage"));
const KnowledgeBasePage = lazy(() => import("./pages/KnowledgeBasePage"));
const EvaluationPage = lazy(() => import("./pages/EvaluationPage"));

const { Header, Sider, Content } = Layout;

const items = [
  { key: "/chat", icon: <MessageOutlined />, label: "对话" },
  { key: "/trace", icon: <NodeIndexOutlined />, label: "Agent 执行轨迹" },
  { key: "/kb", icon: <DatabaseOutlined />, label: "知识库" },
  { key: "/evaluation", icon: <BarChartOutlined />, label: "评估" },
];

export default function App() {
  const navigate = useNavigate();
  const location = useLocation();
  const selected = items.find((i) => location.pathname.startsWith(i.key))?.key ?? "/chat";

  return (
    <Layout style={{ minHeight: "100vh" }}>
      <Sider
        theme="light"
        breakpoint="lg"
        collapsedWidth="0"
        className="app-sider"
      >
        <div
          style={{
            padding: "18px 16px",
            fontWeight: 600,
            fontSize: 16,
            color: "var(--text-primary)",
          }}
        >
          知识智能体
        </div>
        <Menu
          theme="light"
          mode="inline"
          style={{ background: "transparent", borderInlineEnd: "none" }}
          selectedKeys={[selected]}
          items={items}
          onClick={({ key }) => navigate(key)}
        />
      </Sider>
      <Layout>
        <Header
          className="app-glass"
          style={{ position: "sticky", top: 0, zIndex: 10, paddingInline: 24 }}
        >
          <Typography.Title level={4} style={{ margin: "16px 0" }}>
            智能知识 Agent 平台
          </Typography.Title>
        </Header>
        <Content style={{ margin: "24px 0" }}>
          <Suspense
            fallback={
              <div style={{ textAlign: "center", padding: 64 }}>
                <Spin size="large" />
              </div>
            }
          >
            <Routes>
              <Route path="/" element={<Navigate to="/chat" replace />} />
              <Route path="/chat" element={<ChatPage />} />
              <Route path="/trace" element={<TracePage />} />
              <Route path="/kb" element={<KnowledgeBasePage />} />
              <Route path="/evaluation" element={<EvaluationPage />} />
            </Routes>
          </Suspense>
        </Content>
      </Layout>
    </Layout>
  );
}
