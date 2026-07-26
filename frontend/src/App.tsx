import { lazy, Suspense } from "react";
import { Spin } from "antd";
import { Navigate, Route, Routes } from "react-router-dom";
import AppShell from "./components/layout/AppShell";

// 路由级代码分割：仅按需加载各页面，减小首屏体积。
const DashboardPage = lazy(() => import("./pages/DashboardPage"));
const ChatPage = lazy(() => import("./pages/ChatPage"));
const KnowledgeBasePage = lazy(() => import("./pages/KnowledgeBasePage"));
const EvaluationPage = lazy(() => import("./pages/EvaluationPage"));
const SettingsPage = lazy(() => import("./pages/SettingsPage"));

export default function App() {
  return (
    <AppShell>
      <Suspense
        fallback={
          <div style={{ textAlign: "center", padding: 64 }}>
            <Spin size="large" />
          </div>
        }
      >
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/chat" element={<ChatPage />} />
          {/* 旧轨迹页已并入对话页右侧 Inspector 面板。 */}
          <Route path="/trace" element={<Navigate to="/chat" replace />} />
          <Route path="/kb" element={<KnowledgeBasePage />} />
          <Route path="/evaluation" element={<EvaluationPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Suspense>
    </AppShell>
  );
}
