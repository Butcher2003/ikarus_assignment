import { Navigate, Route, Routes } from "react-router-dom";

import Layout from "./components/Layout";
import AnalyticsPage from "./pages/AnalyticsPage";
import ChatPage from "./pages/ChatPage";
import DescriptionPage from "./pages/DescriptionPage";

function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<ChatPage />} />
        <Route path="/analytics" element={<AnalyticsPage />} />
        <Route path="/descriptions" element={<DescriptionPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Layout>
  );
}

export default App;
