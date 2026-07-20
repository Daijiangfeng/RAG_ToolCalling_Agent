import { Layout, Menu, Typography } from "antd";
import {
  MessageOutlined,
  NodeIndexOutlined,
  DatabaseOutlined,
  BarChartOutlined,
} from "@ant-design/icons";
import { Navigate, Route, Routes, useLocation, useNavigate } from "react-router-dom";
import ChatPage from "./pages/ChatPage";
import TracePage from "./pages/TracePage";
import KnowledgeBasePage from "./pages/KnowledgeBasePage";
import EvaluationPage from "./pages/EvaluationPage";

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
      <Sider theme="dark" breakpoint="lg" collapsedWidth="0">
        <div style={{ color: "#fff", padding: 16, fontWeight: 600, fontSize: 15 }}>
          知识智能体
        </div>
        <Menu
          theme="dark"
          mode="inline"
          selectedKeys={[selected]}
          items={items}
          onClick={({ key }) => navigate(key)}
        />
      </Sider>
      <Layout>
        <Header style={{ background: "#fff", paddingInline: 24 }}>
          <Typography.Title level={4} style={{ margin: "16px 0" }}>
            智能知识 Agent 平台
          </Typography.Title>
        </Header>
        <Content style={{ margin: 16 }}>
          <Routes>
            <Route path="/" element={<Navigate to="/chat" replace />} />
            <Route path="/chat" element={<ChatPage />} />
            <Route path="/trace" element={<TracePage />} />
            <Route path="/kb" element={<KnowledgeBasePage />} />
            <Route path="/evaluation" element={<EvaluationPage />} />
          </Routes>
        </Content>
      </Layout>
    </Layout>
  );
}
