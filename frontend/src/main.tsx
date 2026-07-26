import React from "react";
import ReactDOM from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { ConfigProvider, App as AntdApp } from "antd";
import zhCN from "antd/locale/zh_CN";
import App from "./App";
import ErrorBoundary from "./components/ErrorBoundary";
import { buildTheme } from "./theme/tokens";
import { useThemeStore } from "./stores/themeStore";
import "antd/dist/reset.css";
import "./styles/global.css";

/** 根组件：订阅 themeStore，主题切换时同步 antd ConfigProvider。 */
function Root() {
  const mode = useThemeStore((s) => s.mode);
  return (
    <ConfigProvider locale={zhCN} theme={buildTheme(mode)}>
      <AntdApp>
        <ErrorBoundary>
          <BrowserRouter>
            <App />
          </BrowserRouter>
        </ErrorBoundary>
      </AntdApp>
    </ConfigProvider>
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>
);
